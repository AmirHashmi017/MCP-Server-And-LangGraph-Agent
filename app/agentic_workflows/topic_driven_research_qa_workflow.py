from typing import TypedDict, Annotated, Sequence, Literal, Optional
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
import operator
from datetime import datetime
import json

from app.agentic_tools.agentic_tools import (
    direct_research_list, direct_chat_ask,
    direct_summarize_research, direct_summarize_video,
    direct_chat_history_list, direct_chat_history_get, direct_chat_history_delete, 
    direct_deep_answer, direct_summarize_content
)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_id: str  


@tool
async def volvox_search_documents(
    user_id: str,
    limit: int = 20,
    offset: int = 0,
    search: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """List user's research documents"""
    return await direct_research_list(user_id, limit, offset, search, start_date, end_date)

@tool
async def volvox_ask_document(user_id: str, question: str, document_id: str = "", chat_id: str = "", 
                              web_search: bool= False):
    """Ask a question about documents or content using RAG, also aware of the chat history using 
    specific chat_id for a chat search and also enable web_search"""
    return await direct_chat_ask(user_id, question, document_id, chat_id, web_search)

@tool
async def volvox_summarize_documents(document_ids: list[str]):
    """Summarize multiple research documents"""
    return await direct_summarize_research(document_ids)

@tool
async def volvox_summarize_content(content: str):
    """Summarize long Content Text"""
    return await direct_summarize_content(content)

@tool
async def volvox_summarize_video(video_url: str):
    """Summarize a YouTube video"""
    return await direct_summarize_video(video_url)

@tool
async def volvox_chat_history_list(user_id: str):
    """List all chat conversations for the user"""
    return await direct_chat_history_list (user_id)

@tool
async def volvox_chat_history_get(user_id: str, chat_id: str):
    """Get full chat history"""
    return await direct_chat_history_get(user_id, chat_id)

@tool
async def volvox_delete_chat_history(user_id: str, chat_id: str) -> str:
    """Delete a chat"""
    return await direct_chat_history_delete(user_id, chat_id)

@tool
async def smart_deep_search(question: str, mode: str = "deep") -> str:
    """"Ask AI assistant a question about topic and it will search
        from the knowledge base"""
    return await direct_deep_answer(question,mode)




SYSTEM_PROMPT = """You are Volvox AI — an expert research assistant.

You have full access to the user's personal research library and chat history.
The user_id is already known and automatically passed to all tools — NEVER ask for it.

Your available tools:
• smart_deep_search — Ask AI assistant a question about topic and it will search from the knowledge base
• volvox_search_documents — list and search research papers
• volvox_ask_document — Ask a question about documents or content using RAG, also aware of the chat history using specific chat_id for a chat search and also enable web_search
• volvox_summarize_documents — summarize multiple papers
• volvox_summarize_content — Summarize long Content Text
• volvox_summarize_video — summarize YouTube videos
• volvox_chat_history_list — show past conversations
• volvox_chat_history_get — retrieve a full chat
• volvox_delete_chat_history — delete a chat

RULES:
- Always use tools when the user wants to see, search, or ask about their documents
- Never say "I don't have access" or "please provide your user ID"
- Always respond helpfully and directly
- If no documents exist, say: "You haven't uploaded any research yet."

You are ready. Begin helping the user immediately.
"""


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    last = state["messages"][-1]
    return "tools" if hasattr(last, "tool_calls") and last.tool_calls else "end"


def call_model(state: AgentState):
    messages = state["messages"]
    user_id = state["user_id"]

    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    llm_with_tools = llm.bind_tools([
        volvox_search_documents,
        volvox_ask_document,
        volvox_summarize_documents,
        volvox_summarize_content,
        volvox_summarize_video,
        volvox_chat_history_list,
        volvox_chat_history_get,
        volvox_delete_chat_history,
        smart_deep_search
    ])

    response = llm_with_tools.invoke(messages)

    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            tc["args"]["user_id"] = user_id

    return {"messages": [response]}


def create_agent():
    """Build the LangGraph agent with FULL async support"""

    async def execute_tools(state: AgentState):
        user_id = state["user_id"]
        tool_calls = state["messages"][-1].tool_calls
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            args = tool_call["args"].copy()
            args["user_id"] = user_id  

            tool_map = {
                "volvox_search_documents": volvox_search_documents,
                "volvox_ask_document": volvox_ask_document,
                "volvox_summarize_documents": volvox_summarize_documents,
                "volvox_summarize_content": volvox_summarize_content,
                "volvox_summarize_video": volvox_summarize_video,
                "volvox_chat_history_list": volvox_chat_history_list,
                "volvox_chat_history_get": volvox_chat_history_get,
                "volvox_delete_chat_history": volvox_delete_chat_history,
                "smart_deep_search": smart_deep_search
            }

            tool_func = tool_map.get(tool_name)
            if not tool_func:
                result = {"error": f"Unknown tool: {tool_name}"}
            else:
                try:
                    result = await tool_func.ainvoke(args) 
                except Exception as e:
                    result = {"error": str(e)}

            results.append(ToolMessage(
                content=json.dumps(result, indent=2),
                tool_call_id=tool_call["id"]
            ))

        return {"messages": results}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", execute_tools)  

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=MemorySaver())

async def run_agent_smart_qa(query: str, user_id: str, thread_id: str = None):
    agent = create_agent()
    config = {"configurable": {"thread_id": thread_id or f"thread_{datetime.now().timestamp()}"}}

    result = await agent.ainvoke({
        "messages": [HumanMessage(content=query)],
        "user_id": user_id
    }, config)

    final_msg = result["messages"][-1]
    response_text = final_msg.content if hasattr(final_msg, "content") else str(final_msg)

    return {
        "response": response_text,
        "thread_id": config["configurable"]["thread_id"],
        "tool_calls_count": len([m for m in result["messages"] if hasattr(m, "tool_calls") and m.tool_calls])
    }