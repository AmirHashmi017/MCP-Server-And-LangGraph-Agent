import httpx
from typing import Dict, Any, List, Optional

VOLVOX_API = "http://localhost:8000/api/v1"

async def direct_research_list(
    user_id: str,
    limit: int = 20,
    offset: int = 0,
    search: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    params = {"user_id":user_id, "limit": limit, "offset": offset}
    if search: params["search"] = search
    if start_date: params["start"] = start_date
    if end_date: params["end"] = end_date

    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{VOLVOX_API}/research/",
            params=params
        )
        return r.json() if r.status_code == 200 else {"error": r.text}

async def direct_chat_ask(
    user_id: str,
    question: str,
    document_id: Optional[str] = None,
    chat_id: Optional[str] = None
) -> Dict[str, Any]:
    params = {"user_id":user_id, "question": question}
    if document_id: params["document_id"] = document_id
    if chat_id: params["chat_id"] = chat_id

    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{VOLVOX_API}/chat/ask",
            params=params,
            json={}
        )
        return r.json() if r.status_code == 200 else {"error": r.text}

async def direct_summarize_research(document_ids: List[str]) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{VOLVOX_API}/chat/summarize-research",
            json={"documents": document_ids}
        )
        return r.json() if r.status_code == 200 else {"error": r.text}

async def direct_summarize_video(video_url: str) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{VOLVOX_API}/chat/summarize-video",
            params={"video_url": video_url}
        )
        return {"summary": r.text.strip()} if r.status_code == 200 else {"error": r.text}

async def direct_chat_history_list(user_id: str) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{VOLVOX_API}/chat/chatHistory", params={"user_id": user_id})
        return r.json() if r.status_code == 200 else {"error": r.text}

async def direct_chat_history_get(user_id: str, chat_id: str) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{VOLVOX_API}/chat/chatHistory/{chat_id}", params={"user_id": user_id})
        return r.json() if r.status_code == 200 else {"error": r.text}

async def direct_chat_history_delete(user_id: str, chat_id: str) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        r = await client.delete(f"{VOLVOX_API}/chat/deleteChat/{chat_id}", params={"user_id": user_id})
        return r.json() if r.status_code == 200 else {"error": r.text}