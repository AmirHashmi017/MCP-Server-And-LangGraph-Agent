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
    direct_research_list,
    direct_summarize_research,
    direct_access_feasibility, direct_access_roadmap,
    direct_generate_proposal
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
async def volvox_summarize_documents(document_ids: list[str]):
    """Summarize multiple research documents"""
    return await direct_summarize_research(document_ids)



@tool
async def generate_feasibility(summary: str) -> str:
    """
    Generates feasibility content by calling the feasibility streaming API.
    
    Args:
        summary (str): The project summary text.
    
    Returns:
        str: The final combined feasibility output as a single string.
    """
    return await direct_access_feasibility(summary)

@tool
async def generate_roadmap(summary: str) -> str:
    """
    Generates a roadmap from a project summary by calling the streaming roadmap API.
    
    Args:
        summary (str): The project summary text.
    
    Returns:
        str: The final combined roadmap output as a single string.
    """
    return await direct_access_roadmap(summary)

@tool
async def generate_proposal_from_text(report_text: str) -> bytes:
    """
    Generates a funding proposal PDF from a feasibility report text.
    
    Args:
        report_text (str): The feasibility report content.
    
    Returns:
        bytes: PDF file content as bytes.
    """
    return await direct_generate_proposal(report_text)




SYSTEM_PROMPT = """You are Volvox AI — an expert research assistant.

You have full access to the user's personal research library and chat history.
The user_id is already known and automatically passed to all tools — NEVER ask for it.

Your available tools:
• volvox_search_documents — list and search research papers
• volvox_summarize_documents — summarize multiple papers
• generate_feasibility — Generate feasibility output from a project summary
• generate_roadmap — Generate a roadmap from a project summary
• generate_proposal_from_text — Generate a funding proposal PDF from a feasibility report text

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
        volvox_summarize_documents,
        generate_roadmap,
        generate_feasibility,
        generate_proposal_from_text
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
                "volvox_summarize_documents": volvox_summarize_documents,
                "generate_roadmap": generate_roadmap,
                "generate_feasibility": generate_feasibility,
                "generate_proposal_from_text": generate_proposal_from_text
            }

            tool_func = tool_map.get(tool_name)
            if not tool_func:
                result = {"error": f"Unknown tool: {tool_name}"}
            else:
                try:
                    result = await tool_func.ainvoke(args) 
                except Exception as e:
                    result = {"error": str(e)}

            if isinstance(result, bytes):
                results.append(ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"],
                    additional_kwargs={"raw_binary": True} 
                ))
            else:
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

async def run_agent_business_proposal(query: str, user_id: str, thread_id: str = None):
    agent = create_agent()
    config = {"configurable": {"thread_id": thread_id or f"thread_{datetime.now().timestamp()}"}}

    result = await agent.ainvoke({
        "messages": [HumanMessage(content=query)],
        "user_id": user_id
    }, config)

    for msg in reversed(result["messages"]):
        if isinstance(msg, ToolMessage) and "generate_proposal_from_text" in msg.content:
            try:
                data = json.loads(msg.content)
                if isinstance(data, dict) and "result" in data:
                    return data["result"]  
                elif isinstance(data, bytes):
                    return data
            except:
                pass

        if hasattr(msg, "content") and isinstance(msg.content, bytes):
            return msg.content