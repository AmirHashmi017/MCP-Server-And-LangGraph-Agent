import httpx
from typing import Dict, Any, List, Optional

VOLVOX_API = "http://localhost:8000/api/v1"

async def direct_signup(email: str, password: str, fullName: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{VOLVOX_API}/auth/signup", json={"email": email, "password": password, "fullName": fullName})
        return r.json() if r.status_code in (200, 201) else {"error": r.text, "status": r.status_code}

async def direct_login(email: str, password: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{VOLVOX_API}/auth/login", json={"email": email, "password": password})
        return r.json() if r.status_code == 200 else {"error": r.text, "status": r.status_code}

async def direct_get_user(token: str) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{VOLVOX_API}/auth/me", headers={"Authorization": f"Bearer {token}"})
        return r.json() if r.status_code == 200 else {"error": r.text}

async def direct_research_list(
    token: str,
    limit: int = 20,
    offset: int = 0,
    search: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    params = {"limit": limit, "offset": offset}
    if search: params["search"] = search
    if start_date: params["start"] = start_date
    if end_date: params["end"] = end_date

    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{VOLVOX_API}/research/",
            params=params,
            headers={"Authorization": f"Bearer {token}"}
        )
        return r.json() if r.status_code == 200 else {"error": r.text}

async def direct_chat_ask(
    token: str,
    question: str,
    document_id: Optional[str] = None,
    chat_id: Optional[str] = None
) -> Dict[str, Any]:
    params = {"question": question}
    if document_id: params["document_id"] = document_id
    if chat_id: params["chat_id"] = chat_id

    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{VOLVOX_API}/chat/ask",
            params=params,
            json={},
            headers={"Authorization": f"Bearer {token}"}
        )
        return r.json() if r.status_code == 200 else {"error": r.text}

async def direct_summarize_research(token: str, document_ids: List[str]) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{VOLVOX_API}/chat/summarize-research",
            json={"documents": document_ids},
            headers={"Authorization": f"Bearer {token}"}
        )
        return r.json() if r.status_code == 200 else {"error": r.text}

async def direct_summarize_video(token: str, video_url: str) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{VOLVOX_API}/chat/summarize-video",
            params={"video_url": video_url},
            headers={"Authorization": f"Bearer {token}"}
        )
        return {"summary": r.text.strip()} if r.status_code == 200 else {"error": r.text}

async def direct_chat_history_list(token: str) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{VOLVOX_API}/chat/chatHistory", headers={"Authorization": f"Bearer {token}"})
        return r.json() if r.status_code == 200 else {"error": r.text}

async def direct_chat_history_get(token: str, chat_id: str) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{VOLVOX_API}/chat/chatHistory/{chat_id}", headers={"Authorization": f"Bearer {token}"})
        return r.json() if r.status_code == 200 else {"error": r.text}

async def direct_chat_history_delete(token: str, chat_id: str) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        r = await client.delete(f"{VOLVOX_API}/chat/deleteChat/{chat_id}", headers={"Authorization": f"Bearer {token}"})
        return r.json() if r.status_code == 200 else {"error": r.text}