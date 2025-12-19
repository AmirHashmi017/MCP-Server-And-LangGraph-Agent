import httpx
from typing import Dict, Any, Optional
import os

SMART_API = os.getenv("SMART_API_URL", "https://smart-research-answering-backend.up.railway.app")
DEFAULT_TIMEOUT = 300.0

async def smart_new_chat(user_id: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.post(
            f"{SMART_API}/chat/new",
            params={"userId": user_id}
        )
        return r.json() if r.status_code == 200 else {"error": r.text}

async def smart_send_message(
    session_id: int,
    message: str,
    user_id: str,
    mode: str = "simple"
) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.post(
            f"{SMART_API}/chat/message",
            json={
                "session_id": session_id,
                "message": message,
                "user_id": user_id,
                "mode": mode
            }
        )
        return r.json() if r.status_code == 200 else {"error": r.text}

async def smart_message_query(
    message: str,
    mode: str = "simple"
) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.post(
            f"{SMART_API}/chat/messageQuery",
            json={
                "message": message,
                "mode": mode
            }
        )
        return r.json() if r.status_code == 200 else {"error": r.text}

async def smart_get_history(session_id: int) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.get(f"{SMART_API}/chat/history/{session_id}")
        return r.json() if r.status_code == 200 else {"error": r.text}

async def smart_get_history_titles(user_id: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.get(f"{SMART_API}/chat/getHistoryTitle/{user_id}")
        return r.json() if r.status_code == 200 else {"error": r.text}