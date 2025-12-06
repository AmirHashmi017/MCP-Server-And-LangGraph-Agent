from app.agentic_workflows.langgraph_agent import run_agent
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
from app.database import connect_to_mongo, close_mongo_connection
from app.unified_auth.routes.user import signup,login
from app.unified_auth.schemas.user import UserSignupRequest,UserLoginRequest, UserResponse
from app.unified_auth.middleware.auth import get_current_user
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.agentic_tools.agentic_tools import (
    direct_research_list, direct_chat_ask,
    direct_summarize_research, direct_summarize_video,
    direct_chat_history_list, direct_chat_history_get, direct_chat_history_delete,
    direct_research_create,direct_research_update,direct_research_delete
)
from fastapi import File, UploadFile

async def get_user_from_token(token: str):
    if not token:
        raise HTTPException(status_code=401, detail="Token missing")
    clean_token = token.replace("Bearer ", "")
    credentials = HTTPAuthorizationCredentials(scheme="bearer", credentials=clean_token)
    return await get_current_user(credentials)

load_dotenv()

VOLVOX_API = os.getenv("VOLVOX_API_URL", "https://volvox-backend-integrated-production.up.railway.app/api/v1")

app = FastAPI(title="Unified MCP Server", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def startup_event():
    await connect_to_mongo()

@app.on_event("shutdown")
async def shutdown_event():
    await close_mongo_connection()

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
                "token": {"type": "string"},
                "query": {"type": "string", "description": "User Query to perform tasks"}
            },
            "required": ["token","query"]
        }
    },
    {
        "name": "run_file_agent",
        "description": "Run agent to achieve goal through agentic workflow",
        "inputSchema": {
            "type": "object",
            "properties": {
                "token": {"type": "string"},
                "researchName": {"type": "string"}
            },
            "required": ["token","researchName"]
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
    "name": "volvox_research_create",
    "description": "Upload a new research document (PDF, DOCX, etc.) for the user. Send as multipart/form-data.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "token": {"type": "string"},
            "researchName": {"type": "string"}
        },
        "required": ["token", "researchName"] 
    }
    },
    {
        "name": "volvox_research_update",
        "description": "Update research name and/or replace file. Use multipart/form-data if uploading new file.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "token": {"type": "string"},
                "research_id": {"type": "string"},
                "researchName": {"type": "string"}
            },
            "required": ["token", "research_id"]
        }
    },
    {
        "name": "volvox_research_delete",  
        "description": "Delete research document",
        "inputSchema": {
            "type": "object",
            "properties": {
                "token": {"type": "string"},
                "research_id": {"type": "string"}
            },
            "required": ["token", "research_id"]
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
    """
    Safely convert any object (including Pydantic models with datetime) to JSON string
    """
    def default_serializer(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    text = json.dumps(data, default=default_serializer, indent=2)
    
    status = "ERROR" if is_error else "SUCCESS"
    print(f"Result → {status} | {text[:300]}{'...' if len(text)>300 else ''}")
    print("═" * 100 + "\n")
    
    return MCPToolResult(content=[MCPToolContent(text=text)], isError=is_error)

async def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> MCPToolResult:
    try:
        async with httpx.AsyncClient(
    timeout=300.0,
    follow_redirects=True,
    transport=httpx.AsyncHTTPTransport(retries=3)
) as client:
            
            if tool_name in ["volvox_auth_signup", "volvox_auth_login"]:
                if tool_name == "volvox_auth_signup":
                    data = UserSignupRequest(**arguments)
                    result = await signup(data)
                    return safe_return(result.dict())
                elif tool_name == "volvox_auth_login":
                    data = UserLoginRequest(**arguments)
                    result = await login(data)
                    return safe_return(result.dict())
            
            token = arguments.get("token", "")
            try:
                current_user = await get_user_from_token(token) 
            except HTTPException as e:
                return safe_return({"error": e.detail}, True)

            if tool_name== "run_agent":
                query= arguments["query"]
                resp= await run_agent(query,user_id=str(current_user.id))
                return safe_return(resp)
            
            if tool_name== "run_file_agent":
                file = arguments.get("uploaded_file")
                if not file:
                    return safe_return({"error": "File is required for research creation"}, True)

                result = await direct_research_create(
                    user_id=str(current_user.id),
                    researchName=arguments["researchName"],
                    file=file 
                )
                if isinstance(result, dict) and "_id" in result:
                    research_id = result["_id"]   
                else:
                    return safe_return({"error": "Failed to create research – no ID returned"}, True)
                
                query= f"Summarize the content of the Research having ResearchID={research_id}"
                resp= await run_agent(query,user_id=str(current_user.id))
                return safe_return(resp)

            elif tool_name == "volvox_auth_get_user":
                return UserResponse(
                    _id=str(current_user.id),
                    email=current_user.email,
                    fullName= current_user.fullName,
                    created_at=current_user.created_at
                )

            elif tool_name == "volvox_research_list":
                result = await direct_research_list(
                user_id=current_user.id,
                limit=arguments.get("limit", 20),
                offset=arguments.get("offset", 0),
                search=arguments.get("search"),
                start_date=arguments.get("start_date"),
                end_date=arguments.get("end_date")
                )
                return safe_return(result)

            elif tool_name == "volvox_chat_ask":
                result = await direct_chat_ask(
                user_id=current_user.id,
                question=arguments["question"],
                document_id=arguments.get("document_id"),
                chat_id=arguments.get("chat_id")
                )
                return safe_return(result)

            elif tool_name == "volvox_summarize_research":
                result = await direct_summarize_research(arguments["document_ids"])
                return safe_return(result)

            elif tool_name == "volvox_summarize_video":
                result = await direct_summarize_video(arguments["video_url"])
                return safe_return(result)

            elif tool_name == "volvox_chat_history_list":
                result = await direct_chat_history_list(current_user.id)
                return safe_return(result)

            elif tool_name == "volvox_chat_history_get":
                result = await direct_chat_history_get(current_user.id, arguments["chat_id"])
                return safe_return(result)
            
            elif tool_name == "volvox_chat_history_delete":
                result = await direct_chat_history_delete(current_user.id, arguments["chat_id"])
                return safe_return(result)
            
            elif tool_name == "volvox_research_create":
                file = arguments.get("uploaded_file")
                if not file:
                    return safe_return({"error": "File is required for research creation"}, True)

                result = await direct_research_create(
                    user_id=str(current_user.id),
                    researchName=arguments["researchName"],
                    file=file 
                )
                return safe_return(result)

            elif tool_name == "volvox_research_update":
                result = await direct_research_update(
                    user_id=str(current_user.id),
                    research_id=arguments["research_id"],
                    researchName=arguments.get("researchName"),
                    file=arguments.get("uploaded_file")
                )
                return safe_return(result)

            elif tool_name == "volvox_research_delete":
                result = await direct_research_delete(
                    user_id=str(current_user.id),
                    research_id=arguments["research_id"]
                )
                return safe_return(result)

            else:
                return safe_return({"error": f"Unknown tool: {tool_name}"}, True)

    except Exception as e:
        print(f"EXCEPTION in {tool_name}: {e}")
        traceback.print_exc()
        return safe_return({"error": str(e)}, True)

@app.post("/mcp")
async def mcp_endpoint(request: Request):

    content_type = request.headers.get("content-type", "")
    uploaded_file: Optional[UploadFile] = None

    if "multipart/form-data" in content_type:
        form = await request.form()
        jsonrpc_field = form.get("jsonrpc")
        if not jsonrpc_field:
            return JSONResponse(status_code=400, content={"error": "Missing jsonrpc field"})
        try:
            body = json.loads(jsonrpc_field)
        except json.JSONDecodeError:
            return JSONResponse(status_code=400, content={"error": "Invalid jsonrpc JSON"})
        req = MCPRequest(**body)
        uploaded_file = form.get("file")

    else: 
        body = await request.json()
        req = MCPRequest(**body)
    
    try:
        if req.method == "tools/list":
            return {"jsonrpc": "2.0", "id": req.id, "result": {"tools": TOOLS}}

        if req.method == "tools/call":
            arguments = req.params.get("arguments", {})
            if uploaded_file is not None:
                arguments["uploaded_file"] = uploaded_file
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
async def root(): 
    return {"service": "Unified MCP Server", "status": "running", "tools": len(TOOLS)}
