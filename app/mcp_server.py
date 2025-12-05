from langgraph_agent import run_agent
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Literal
import httpx
import os
from datetime import datetime
from dotenv import load_dotenv
import json
import traceback

load_dotenv()

VOLVOX_API = os.getenv("VOLVOX_API_URL", "http://localhost:8000/api/v1")

app = FastAPI(title="Unified MCP Server", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: int
    method: str
    params: Optional[Dict[str, Any]] = None

class MCPToolContent(BaseModel):
    type: Literal["text"] = "text"
    text: str

class MCPToolResult(BaseModel):
    content: List[MCPToolContent]
    isError: Optional[bool] = None

TOOLS = [
    {
        "name": "run_agent",
        "description": "Run agent to achieve goal through agentic workflow",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "User Query to perform tasks"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "volvox_auth_signup",
        "description": "Signup to Volvox resulting in creation of user account",
        "inputSchema": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "User Email"},
                "password": {"type": "string", "description": "User password"},
                "fullName": {"type": "string", "description": "User Full Name"},
            },
            "required": ["email", "password","fullName"]
        }
    },
    {
        "name": "volvox_auth_login",
        "description": "Login to Volvox and get JWT token. Use this first before other Volvox operations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "User email"},
                "password": {"type": "string", "description": "User password"}
            },
            "required": ["email", "password"]
        }
    },
    {
        "name": "volvox_auth_get_user",
        "description": "Get current user information using JWT token",
        "inputSchema": {
            "type": "object",
            "properties": {
                "token": {"type": "string", "description": "JWT token from login"}
            },
            "required": ["token"]
        }
    },
    {
        "name": "volvox_research_list",
        "description": "List user's research documents with optional filters",
        "inputSchema": {
            "type": "object",
            "properties": {
                "token": {"type": "string"},
                "limit": {"type": "number", "default": 20},
                "offset": {"type": "number", "default": 0},
                "search": {"type": "string", "description": "Search query"},
                "start_date": {"type": "string", "description": "ISO date string"},
                "end_date": {"type": "string", "description": "ISO date string"}
            },
            "required": ["token"]
        }
    },
    {
        "name": "volvox_chat_ask",
        "description": "Ask AI assistant a question about documents using RAG",
        "inputSchema": {
            "type": "object",
            "properties": {
                "token": {"type": "string"},
                "question": {"type": "string"},
                "document_id": {"type": "string", "description": "Optional: specific document"},
                "chat_id": {"type": "string", "description": "Optional: continue conversation"}
            },
            "required": ["token", "question"]
        }
    },
    {
        "name": "volvox_summarize_research",
        "description": "Generate AI summary of multiple research documents",
        "inputSchema": {
            "type": "object",
            "properties": {
                "token": {"type": "string"},
                "document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Array of document IDs"
                }
            },
            "required": ["token", "document_ids"]
        }
    },
    {
        "name": "volvox_summarize_video",
        "description": "Generate AI summary of video transcripts",
        "inputSchema": {
            "type": "object",
            "properties": {
                "token": {"type": "string"},
                "video_url": {"type": "string"}
            },
            "required": ["token", "video_url"]
        }
    },
    {
        "name": "volvox_chat_history_list",
        "description": "Get list of all chat conversations",
        "inputSchema": {
            "type": "object",
            "properties": {
                "token": {"type": "string"}
            },
            "required": ["token"]
        }
    },
    {
        "name": "volvox_chat_history_get",
        "description": "Get full chat history for specific conversation",
        "inputSchema": {
            "type": "object",
            "properties": {
                "token": {"type": "string"},
                "chat_id": {"type": "string"}
            },
            "required": ["token", "chat_id"]
        }
    },
    {
        "name": "volvox_chat_history_delete",
        "description": "Delete chat history for specific conversation",
        "inputSchema": {
            "type": "object",
            "properties": {
                "token": {"type": "string"},
                "chat_id": {"type": "string"}
            },
            "required": ["token", "chat_id"]
        }
    },
]

def log_tool(tool_name: str, args: dict, url: str, method: str = "POST"):
    print("\n" + "═" * 100)
    print(f"TOOL → {tool_name.upper()} @ {datetime.now().isoformat()}")
    print(f"{method} {url}")
    print(f"Arguments:\n{json.dumps(args, indent=2)}")
    print("─" * 100)

def safe_return(data: Any, is_error: bool = False):
    text = json.dumps(data) if not isinstance(data, str) else data
    status = "ERROR" if is_error else "SUCCESS"
    print(f"Result → {status} | {text[:300]}{'...' if len(text)>300 else ''}")
    print("═" * 100 + "\n")
    return MCPToolResult(content=[MCPToolContent(text=text)], isError=is_error or None)

async def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> MCPToolResult:
    try:
        async with httpx.AsyncClient(
    timeout=300.0,
    follow_redirects=True,
    transport=httpx.AsyncHTTPTransport(retries=3)
) as client:

            if tool_name== "run_agent":
                query= arguments["query"]
                resp= run_agent(query,token="")
                return safe_return(resp)

            elif tool_name== "volvox_auth_signup":
                log_tool(tool_name, arguments, f"{VOLVOX_API}/auth/signup")
                resp= await client.post(f"{VOLVOX_API}/auth/signup",json=arguments)
                print(f"Status: {resp.status_code} | Body: {resp.text}")
                if resp.status_code != 201:
                    return safe_return({"error": "Signup failed", "detail": resp.text}, True)
                return safe_return(resp.json())
            
            elif tool_name == "volvox_auth_login":
                log_tool(tool_name, arguments, f"{VOLVOX_API}/auth/login")
                resp = await client.post(f"{VOLVOX_API}/auth/login", json=arguments)
                print(f"Status: {resp.status_code} | Body: {resp.text}")
                if resp.status_code != 200:
                    return safe_return({"error": "Login failed", "detail": resp.text}, True)
                return safe_return(resp.json())

            elif tool_name == "volvox_auth_get_user":
                log_tool(tool_name, arguments, f"{VOLVOX_API}/auth/me", "GET")
                resp = await client.get(f"{VOLVOX_API}/auth/me", headers={"Authorization": f"Bearer {arguments['token']}"})
                print(f"Status: {resp.status_code} | Body: {resp.text}")
                return safe_return(resp.json() if resp.status_code == 200 else {"error": resp.text}, resp.status_code != 200)

            elif tool_name == "volvox_research_list":
                token = arguments["token"]
                limit = arguments.get("limit", 20)
                offset = arguments.get("offset", 0)
                search = arguments.get("search")
                start_date = arguments.get("start_date")   
                end_date = arguments.get("end_date")        

                params = {
                    "limit": limit,
                    "offset": offset
                }
                if search:
                    params["search"] = search
                if start_date:
                    params["start"] = start_date
                if end_date:
                    params["end"] = end_date

                log_tool(tool_name, arguments, f"{VOLVOX_API}/research", "GET")
                print(f"GET → {VOLVOX_API}/research?{httpx.QueryParams(params)}")

                resp = await client.get(
                    f"{VOLVOX_API}/research/",
                    params=params,
                    headers={"Authorization": f"Bearer {token}"}
                )

                print(f"Status: {resp.status_code} | Body: {resp.text}")

                if resp.status_code != 200:
                    return safe_return({
                        "error": f"HTTP {resp.status_code}",
                        "detail": resp.text
                    }, True)

                data = resp.json()
                return safe_return(data)

            elif tool_name == "volvox_chat_ask":
                params = {"question": arguments["question"]}
                if arguments.get("document_id"):
                    params["document_id"] = arguments["document_id"]
                if arguments.get("chat_id"):
                    params["chat_id"] = arguments["chat_id"]

                log_tool(tool_name, arguments, f"{VOLVOX_API}/chat/ask")
                print(f"Sending as QUERY PARAMS → ?{httpx.QueryParams(params)}")

                resp = await client.post(
                    f"{VOLVOX_API}/chat/ask",
                    params=params,           
                    json={},                 
                    headers={"Authorization": f"Bearer {arguments['token']}"}
                )
                print(f"Status: {resp.status_code} | Body: {resp.text}")
                if resp.status_code != 200:
                    return safe_return({"error": f"HTTP {resp.status_code}", "detail": resp.text}, True)
                return safe_return(resp.json())

            elif tool_name == "volvox_summarize_research":
                log_tool(tool_name, arguments, f"{VOLVOX_API}/chat/summarize-research")
                resp = await client.post(
                    f"{VOLVOX_API}/chat/summarize-research",
                    json={"documents": arguments["document_ids"]},
                    headers={"Authorization": f"Bearer {arguments['token']}"}
                )
                print(f"Status: {resp.status_code} | Body: {resp.text}")
                return safe_return(resp.json() if resp.status_code == 200 else {"error": resp.text}, resp.status_code != 200)

            elif tool_name == "volvox_summarize_video":
                token = arguments["token"]
                video_url = arguments["video_url"]

                log_tool(tool_name, arguments, f"{VOLVOX_API}/chat/summarize-video")

                resp = await client.post(
                    f"{VOLVOX_API}/chat/summarize-video?video_url={video_url}",
                    headers={"Authorization": f"Bearer {token}"}
                )

                print(f"Status: {resp.status_code} | Body: {resp.text}")

                if resp.status_code != 200:
                    return safe_return({"error": f"HTTP {resp.status_code}", "detail": resp.text}, True)

                return safe_return({"summary": resp.text.strip()})

            elif tool_name == "volvox_chat_history_list":
                log_tool(tool_name, arguments, f"{VOLVOX_API}/chat/chatHistory", "GET")
                resp = await client.get(f"{VOLVOX_API}/chat/chatHistory", headers={"Authorization": f"Bearer {arguments['token']}"})
                print(f"Status: {resp.status_code} | Body: {resp.text}")
                return safe_return(resp.json() if resp.status_code == 200 else {"error": resp.text}, resp.status_code != 200)

            elif tool_name == "volvox_chat_history_get":
                url = f"{VOLVOX_API}/chat/chatHistory/{arguments['chat_id']}"
                log_tool(tool_name, arguments, url, "GET")
                resp = await client.get(url, headers={"Authorization": f"Bearer {arguments['token']}"})
                print(f"Status: {resp.status_code} | Body: {resp.text}")
                return safe_return(resp.json() if resp.status_code == 200 else {"error": resp.text}, resp.status_code != 200)
            
            elif tool_name == "volvox_chat_history_delete":
                url = f"{VOLVOX_API}/chat/deleteChat/{arguments['chat_id']}"
                log_tool(tool_name, arguments, url, "DELETE")
                resp = await client.delete(url, headers={"Authorization": f"Bearer {arguments['token']}"})
                print(f"Status: {resp.status_code} | Body: {resp.text}")
                return safe_return(resp.json() if resp.status_code == 200 else {"error": resp.text}, resp.status_code != 200)

            else:
                return safe_return({"error": f"Unknown tool: {tool_name}"}, True)

    except Exception as e:
        print(f"EXCEPTION in {tool_name}: {e}")
        traceback.print_exc()
        return safe_return({"error": str(e)}, True)

@app.post("/mcp")
async def mcp_endpoint(request: Request):
    try:
        body = await request.json()
        req = MCPRequest(**body)

        if req.method == "tools/list":
            return {"jsonrpc": "2.0", "id": req.id, "result": {"tools": TOOLS}}

        if req.method == "tools/call":
            result = await execute_tool(req.params["name"], req.params.get("arguments", {}))
            return {"jsonrpc": "2.0", "id": req.id, "result": result.dict()}

        if req.method == "initialize":
            return {"jsonrpc": "2.0", "id": req.id, "result": {
                "protocolVersion": "1.0", "serverInfo": {"name": "Unified MCP Server", "version": "1.0.0"},
                "capabilities": {"tools": {}}
            }}

        return {"jsonrpc": "2.0", "id": req.id, "error": {"code": -32601, "message": "Method not found"}}

    except Exception as e:
        return JSONResponse(status_code=500, content={"jsonrpc": "2.0", "id": 0, "error": {"code": -32603, "message": str(e)}})

@app.get("/")
async def root(): return {"service": "Unified MCP Server", "status": "running", "tools": len(TOOLS)}

if __name__ == "__main__":
    import uvicorn
    print("Unified MCP Server with FULL DEBUG LOGGING STARTED")
    print(f"MCP Endpoint → http://localhost:4000/mcp")
    uvicorn.run("mcp_server:app", host="0.0.0.0", port=4000, reload=True)