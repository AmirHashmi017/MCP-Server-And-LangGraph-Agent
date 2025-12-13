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
from fastapi import WebSocket

from app.agentic_tools.agentic_tools import (
    direct_deep_answer, direct_summarize_content
)

_send_stream_update = None

def set_send_stream_update(func):
    global _send_stream_update
    _send_stream_update = func


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_id: str  


@tool
async def volvox_summarize_content(content: str):
    """Summarize long Content Text"""
    return await direct_summarize_content(content)


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
• volvox_summarize_content — Summarize long Content Text

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
        volvox_summarize_content,
        smart_deep_search
    ])

    response = llm_with_tools.invoke(messages)

    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            tc["args"]["user_id"] = user_id

    return {"messages": [response]}


def create_agent():
    """Build the LangGraph agent with FULL async support"""

    async def execute_tools(state: AgentState, config: dict):
        user_id = state["user_id"]
        tool_calls = state["messages"][-1].tool_calls
        results = []

        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id")
        print(f"[DEBUG] thread_id: {thread_id}")
        print(f"[DEBUG] _send_stream_update is set: {_send_stream_update is not None}")

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            args = tool_call["args"].copy()
            args["user_id"] = user_id  

            tool_map = {
                "volvox_summarize_content": volvox_summarize_content,
                "smart_deep_search": smart_deep_search,
            }

            tool_func = tool_map.get(tool_name)
            if not tool_func:
                result = {"error": f"Unknown tool: {tool_name}"}
            else:
                try:
                    
                    if thread_id and _send_stream_update:
                        print(f"[DEBUG] Sending tool_start for {tool_name}")
                        await _send_stream_update(thread_id, {
                            "type": "tool_start",
                            "tool_name": tool_name,
                            "input": args
                        })
                    
                    result = await tool_func.ainvoke(args) 
                    
                   
                    if thread_id and _send_stream_update:
                        await _send_stream_update(thread_id, {
                            "type": "tool_end",
                            "tool_name": tool_name,
                            "response": result
                        })
                except Exception as e:
                    result = {"error": str(e)}
                    
                    if thread_id and _send_stream_update:
                        await _send_stream_update(thread_id, {
                            "type": "tool_error",
                            "tool_name": tool_name,
                            "error": str(e)
                        })

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

async def run_agent_smart_search(query: str, user_id: str, 
                                 thread_id: str = None):
    agent = create_agent()
    thread_id = thread_id or f"thread_{datetime.now().timestamp()}"
    config = {"configurable": {"thread_id": thread_id}}

    result = await agent.ainvoke({
        "messages": [HumanMessage(content=query)],
        "user_id": user_id
    }, config)

    final_msg = result["messages"][-1]
    response_text = final_msg.content if hasattr(final_msg, "content") else str(final_msg)

    final_result = {
        "response": response_text,
        "thread_id": thread_id,
        "tool_calls_count": len([m for m in result["messages"] if hasattr(m, "tool_calls") and m.tool_calls])
    }
    
    # Stream final result through WebSocket
    if thread_id and _send_stream_update:
        await _send_stream_update(thread_id, {
            "type": "workflow_complete",
            "result": final_result
        })
    
    return final_result