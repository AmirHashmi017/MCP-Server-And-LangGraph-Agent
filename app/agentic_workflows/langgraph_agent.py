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

from app.agentic_tools.agentic_tools import (
    direct_research_list, direct_chat_ask,
    direct_summarize_research, direct_summarize_video,
    direct_chat_history_list, direct_chat_history_get, direct_chat_history_delete
)


load_dotenv()



class AgentState(TypedDict):
    """Agent state tracking"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_id: str 


@tool
def volvox_search_documents(
    user_id: str,
    limit: int = 20,
    offset: int = 0,
    search: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """List research documents with filters"""
    return anyio.run(
        direct_research_list,
        user_id,
        limit,
        offset,
        search,
        start_date,
        end_date
    )
@tool
def volvox_ask_document(user_id: str, question: str, 
                       document_id: str = "", chat_id: str = "") -> dict:
    """
    Ask AI assistant a question about documents using RAG.
    Can query specific documents or ask general questions.
    
    Args:
        user_id: UserID
        question: The question to ask
        document_id: Optional - ID of specific document to query
        chat_id: Optional - continue existing conversation
    
    Returns:
        AI response with answer based on document content
    """
    return anyio.run(direct_chat_ask, user_id, question, document_id, chat_id)

@tool
def volvox_summarize_documents(document_ids: list[str]) -> dict:
    """
    Generate AI summary of multiple research documents.
    Useful for getting overview of several papers at once.
    
    Args:
        user_id: UserID
        document_ids: List of document IDs to summarize (as strings)
    
    Returns:
        Comprehensive summary of all documents
    """
    return anyio.run(direct_summarize_research, document_ids)

@tool
def volvox_summarize_video(video_url:str)->dict:
    """
    Generate AI summary of video transcripts.
    Useful for getting overview of things discussed in a video.
    
    Args:
        video_url: URL of youtube video which we have to summarize.
    
    Returns:
        Comprehensive summary video whose url is added
    """
    return anyio.run(direct_summarize_video, video_url)

@tool
def volvox_chat_history_list(user_id: str):
    """
    List all conversations
    Args:
        user_id: UserID
    
    Returns:
        All conversations of a specific user
    """
    return anyio.run(direct_chat_history_list, user_id)

@tool
def volvox_chat_history_get(user_id: str, chat_id: str):
    """
    Get full conversation with messages of a chat

    Args:
        user_id: UserID
        chat_id: specific chat ID
    
    Returns:
        All messages of a specific chat
    """
    return anyio.run(direct_chat_history_get, user_id, chat_id)

@tool
def volvox_delete_chat_history(user_id: str, chat_id: str)->str:
    """
    Delete a specific chat history session messages
    
    Args:
        user_id: UserID
        chat_id: specific chat ID
    
    Returns:
        Successful Deletion Message
    """
    return anyio.run(direct_chat_history_delete, user_id, chat_id)


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
    messages = state["messages"]
    user_id = state["user_id"]

    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
    )

    def create_tool_with_user_id(original_tool):
        import functools
        @functools.wraps(original_tool)
        def wrapper(**kwargs):
            kwargs["user_id"] = user_id
            return original_tool(**kwargs)

        wrapper.__name__ = original_tool.__name__
        wrapper.__doc__ = original_tool.__doc__
        return wrapper

    tools_with_user = [
        create_tool_with_user_id(tool) for tool in [
            volvox_search_documents,
            volvox_ask_document,
            volvox_chat_history_list,
            volvox_chat_history_get,
            volvox_delete_chat_history
        ]
    ] + [
        volvox_summarize_documents,
        volvox_summarize_video
    ]

    llm_with_tools = llm.bind_tools(tools_with_user)
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def create_agent():
    """Build the LangGraph agent"""

    tool_node = ToolNode([
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

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    workflow.add_edge("tools", "agent")

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def run_agent(query: str, user_id: str, thread_id: str = None):
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
        "user_id": user_id
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
        "thread_id": config["configurable"]["thread_id"],
        "tool_calls_count": len([
            m for m in result["messages"] 
            if hasattr(m, "tool_calls") and m.tool_calls
        ])
    }

def stream_agent(query: str, user_id: str = "", thread_id: str = None):
    """Stream agent execution for real-time updates"""
    agent = create_agent()
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "user_id": user_id
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