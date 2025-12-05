
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
import operator
import httpx
import json
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:4000/mcp")


class AgentState(TypedDict):
    """Agent state tracking"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    volvox_token: str
    project_tokens: dict  

def call_mcp_tool(tool_name: str, arguments: dict) -> dict:
    """
    Call a tool through the Python MCP server
    
    Args:
        tool_name: Name of the tool to call
        arguments: Tool arguments
    
    Returns:
        Tool execution result
    """
    try:
        mcp_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        with httpx.Client(timeout=300.0) as client:
            response = client.post(
                MCP_SERVER_URL,
                json=mcp_request,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract content from MCP response
            if "result" in result and "content" in result["result"]:
                content = result["result"]["content"]
                if isinstance(content, list) and len(content) > 0:
                    text_content = content[0].get("text", "")
                    try:
                        return json.loads(text_content)
                    except:
                        return {"result": text_content}
            
            return result
            
    except Exception as e:
        return {"error": f"MCP call failed: {str(e)}"}


@tool
def volvox_signup(email:str, password: str, fullName: str)->dict:
    """
    Sinup to Volvox and get JWT token.
    ALWAYS use this first before other Volvox operations.
    
    Args:
        email: User email address
        password: User password
        fullName: User Name
    
    Returns:
        Authentication response with access_token
    """
    return call_mcp_tool("volvox_auth_signup", {
        "email": email,
        "password": password,
        "fullName": fullName
    })

@tool
def volvox_login(email: str, password: str) -> dict:
    """
    Login to Volvox and get JWT token.
    ALWAYS use this first before other Volvox operations.
    
    Args:
        email: User email address
        password: User password
    
    Returns:
        Authentication response with access_token
    """
    return call_mcp_tool("volvox_auth_login", {
        "email": email,
        "password": password
    })

@tool
def volvox_search_documents(token: str, search: str = "", limit: int = 20,
                           start_date: str = "", end_date: str = "") -> dict:
    """
    Search and list user's research documents.
    Can filter by keywords, date range, and limit results.
    
    Args:
        token: JWT token from login
        search: Search query for document/file names
        limit: Maximum number of results (default 20)
        start_date: Filter start date (ISO format YYYY-MM-DD)
        end_date: Filter end date (ISO format YYYY-MM-DD)
    
    Returns:
        List of research documents with metadata
    """
    args = {"token": token, "limit": limit}
    if search:
        args["search"] = search
    if start_date:
        args["start"] = start_date
    if end_date:
        args["end"] = end_date
    
    return call_mcp_tool("volvox_research_list", args)

@tool
def volvox_ask_document(token: str, question: str, 
                       document_id: str = "", chat_id: str = "") -> dict:
    """
    Ask AI assistant a question about documents using RAG.
    Can query specific documents or ask general questions.
    
    Args:
        token: JWT token
        question: The question to ask
        document_id: Optional - ID of specific document to query
        chat_id: Optional - continue existing conversation
    
    Returns:
        AI response with answer based on document content
    """
    args = {"token": token, "question": question}
    if document_id:
        args["document_id"] = document_id
    if chat_id:
        args["chat_id"] = chat_id
    
    return call_mcp_tool("volvox_chat_ask", args)

@tool
def volvox_summarize_documents(token: str, document_ids: list[str]) -> dict:
    """
    Generate AI summary of multiple research documents.
    Useful for getting overview of several papers at once.
    
    Args:
        token: JWT token
        document_ids: List of document IDs to summarize (as strings)
    
    Returns:
        Comprehensive summary of all documents
    """
    return call_mcp_tool("volvox_summarize_research", {
        "token": token,
        "document_ids": document_ids
    })

@tool
def volvox_summarize_video(token: str, video_url:str)->dict:
    """
    Generate AI summary of video transcripts.
    Useful for getting overview of things discussed in a video.
    
    Args:
        token: JWT token
        video_url: URL of youtube video which we have to summarize.
    
    Returns:
        Comprehensive summary video whose url is added
    """
    return call_mcp_tool("volvox_summarize_video",{
        "token": token,
        "video_url": video_url
    })

@tool
def volvox_get_chat_history(token: str, chat_id: str = "") -> dict:
    """
    Get chat conversation history.
    If chat_id provided, gets specific conversation.
    Otherwise lists all conversations.
    
    Args:
        token: JWT token
        chat_id: Optional - specific chat ID
    
    Returns:
        Chat history or list of chats
    """
    if chat_id:
        return call_mcp_tool("volvox_chat_history_get", {
            "token": token,
            "chat_id": chat_id
        })
    else:
        return call_mcp_tool("volvox_chat_history_list", {
            "token": token
        })

@tool
def volvox_delete_chat_history(token: str, chat_id: str)->str:
    """
    Delete a specific chat history session messages
    
    Args:
        token: JWT token
        chat_id: specific chat ID
    
    Returns:
        Successful Deletion Message
    """
    return call_mcp_tool("volvox_chat_history_delete", {
            "token": token,
            "chat_id": chat_id
        })


SYSTEM_PROMPT = """You are an expert automation assistant.
You have access to external tools via the Model Context Protocol (MCP).

RULES:
- If the user asks you to sign up, log in, create a profile, list research, ask a document question → YOU MUST use the appropriate tool.
- Never say "I can't do that" or "you need to do it manually".
- Never make up fake responses.
- Always call the real tools with correct arguments.
- Tool calling is mandatory when applicable.

Available tools and when to use them:
- Use `auth_signup` to register new users
- Use `auth_login` to log in
- Use `profile_create` to create company profiles
- Use `profile_list` to show existing profiles
- Use `research_list` to list uploaded documents
- Use `chat_ask` to ask questions about documents
"""


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Determine if agent should continue or finish"""
    messages = state["messages"]
    last_message = messages[-1]
    
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return "end"
    return "tools"

def call_model(state: AgentState):
    """Let LLM decide next action"""
    messages = state["messages"]
    
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    if state.get("volvox_token"):
        context = SystemMessage(
            content=f"Available Volvox token: {state['volvox_token']}"
        )
        messages = [messages[0], context] + messages[1:]
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",                    
        temperature=0,
        convert_system_message_to_human=True,
    )

    llm_with_tools = llm.bind_tools([
        volvox_signup,
        volvox_login,
        volvox_search_documents,
        volvox_ask_document,
        volvox_summarize_documents,
        volvox_summarize_video,
        volvox_get_chat_history,
        volvox_delete_chat_history
    ])
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def extract_tokens(state: AgentState):
    """Extract and store tokens from tool results"""
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, "content"):
        try:
            content = json.loads(last_message.content) if isinstance(last_message.content, str) else last_message.content
            if isinstance(content, dict):
                if "access_token" in content:
                    return {"volvox_token": content["access_token"]}
        except:
            pass
    
    return {}

def create_agent():
    """Build the LangGraph agent"""

    tool_node = ToolNode([
        volvox_signup,
        volvox_login,
        volvox_search_documents,
        volvox_ask_document,
        volvox_summarize_documents,
        volvox_summarize_video,
        volvox_get_chat_history,
        volvox_delete_chat_history
    ])

    workflow = StateGraph(AgentState)

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_node("extract_tokens", extract_tokens)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    workflow.add_edge("tools", "extract_tokens")
    workflow.add_edge("extract_tokens", "agent")

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def run_agent(query: str, token: str = "", thread_id: str = None):
    """
    Run the agent with a user query
    
    Args:
        query: Natural language query
        token: Optional JWT token if already authenticated
        thread_id: Optional conversation thread ID
    
    Returns:
        Agent response with results
    """
    agent = create_agent()
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "volvox_token": token,
        "project_tokens": {}
    }
    
    config = {
        "configurable": {
            "thread_id": thread_id or f"thread_{datetime.now().timestamp()}"
        }
    }
    
    result = agent.invoke(initial_state, config)
    
    final_message = result["messages"][-1]
    
    return {
        "response": final_message if isinstance(final_message, str) else final_message.content,
        "token": result.get("volvox_token", ""),
        "thread_id": config["configurable"]["thread_id"],
        "tool_calls_count": len([
            m for m in result["messages"] 
            if hasattr(m, "tool_calls") and m.tool_calls
        ])
    }

def stream_agent(query: str, token: str = "", thread_id: str = None):
    """Stream agent execution for real-time updates"""
    agent = create_agent()
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "volvox_token": token,
        "project_tokens": {}
    }
    
    config = {
        "configurable": {
            "thread_id": thread_id or f"thread_{datetime.now().timestamp()}"
        }
    }
    
    for event in agent.stream(initial_state, config):
        for value in event.values():
            if "messages" in value:
                last_message = value["messages"][-1]
                yield {
                    "type": "update",
                    "content": last_message.content if hasattr(last_message, "content") else str(last_message),
                    "timestamp": datetime.now().isoformat()
                }


if __name__ == "__main__":
    
    print(" LangGraph Agent + Python MCP Server")
    
    print(f" MCP Server: {MCP_SERVER_URL}")

    print("\n Example: Simple Query")
    result = run_agent(
        query="Login with email mobeen@gmail.com password Mobeen123, then list my research documents",
        token=""
    )
    print(f" Response: {result['response'][:200]}...")
    print(f" Token extracted: {'Yes' if result['token'] else 'No'}")
    print(f" Tool calls made: {result['tool_calls_count']}")