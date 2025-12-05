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
import anyio
from typing import Optional

from agentic_tools import (
    direct_signup, direct_login, direct_get_user,
    direct_research_list, direct_chat_ask,
    direct_summarize_research, direct_summarize_video,
    direct_chat_history_list, direct_chat_history_get, direct_chat_history_delete
)


load_dotenv()



class AgentState(TypedDict):
    """Agent state tracking"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    volvox_token: str
    project_tokens: dict  


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
    return anyio.run(direct_signup, email, password, fullName)

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
    return anyio.run(direct_login, email, password)

@tool
def volvox_auth_get_user(token: str):
    """Get current user info"""
    return anyio.run(direct_get_user, token)

@tool
def volvox_search_documents(
    token: str,
    limit: int = 20,
    offset: int = 0,
    search: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """List research documents with filters"""
    return anyio.run(
        direct_research_list,
        token,
        limit,
        offset,
        search,
        start_date,
        end_date
    )
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
    return anyio.run(direct_chat_ask, token, question, document_id, chat_id)

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
    return anyio.run(direct_summarize_research, token, document_ids)

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
    return anyio.run(direct_summarize_video, token, video_url)

@tool
def volvox_chat_history_list(token: str):
    """List all conversations"""
    return anyio.run(direct_chat_history_list, token)

@tool
def volvox_chat_history_get(token: str, chat_id: str):
    """Get full conversation with messages of a chat"""
    return anyio.run(direct_chat_history_get, token, chat_id)

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
    return anyio.run(direct_chat_history_delete, token, chat_id)


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
        volvox_auth_get_user,
        volvox_search_documents,
        volvox_ask_document,
        volvox_summarize_documents,
        volvox_summarize_video,
        volvox_chat_history_list,
        volvox_chat_history_get,
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
        volvox_auth_get_user,
        volvox_search_documents,
        volvox_ask_document,
        volvox_summarize_documents,
        volvox_summarize_video,
        volvox_chat_history_list,
        volvox_chat_history_get,
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

    print("\n Example: Simple Query")
    result = run_agent(
        query="Login with email mobeen@gmail.com password Mobeen123, then list my research documents",
        token=""
    )
    print(f" Response: {result['response'][:200]}...")
    print(f" Token extracted: {'Yes' if result['token'] else 'No'}")
    print(f" Tool calls made: {result['tool_calls_count']}")