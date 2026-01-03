import httpx
from typing import Dict, Any, List, Optional
from fastapi import UploadFile
import os


VOLVOX_API = os.getenv("VOLVOX_API_URL", "https://amirhashmi017-volvox-backend.hf.space/api/v1")
SMART_API = os.getenv("SMART_API_URL", "https://smart-research-answering-backend.up.railway.app")
INNOSCOPE_API= os.getenv("INNOSCOPE_API_URL", "https://mustafanoor-innoscope-backend.hf.space")
KICKSTART_API= os.getenv("KICKSTART_API_URL", "https://proposal-generation-for-funding-production.up.railway.app")

DEFAULT_TIMEOUT = 300.0   

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

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.get(
            f"{VOLVOX_API}/research/",
            params=params
        )
        return r.json() if r.status_code == 200 else {"error": r.text}

async def direct_chat_ask(
    user_id: str,
    question: str,
    document_id: Optional[str] = None,
    chat_id: Optional[str] = None,
    web_search: Optional[bool]= False
) -> Dict[str, Any]:
    params = {"user_id":user_id, "question": question}
    if document_id: params["document_id"] = document_id
    if chat_id: params["chat_id"] = chat_id
    if web_search: params["web_search"]= web_search

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.post(
            f"{VOLVOX_API}/chat/ask",
            params=params,
            json={}
        )
        return r.json() if r.status_code == 200 else {"error": r.text}

async def direct_summarize_research(document_ids: List[str]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.post(
            f"{VOLVOX_API}/chat/summarize-research",
            json={"documents": document_ids}
        )
        return r.json() if r.status_code == 200 else {"error": r.text}

async def direct_summarize_content(content: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.post(
            f"{VOLVOX_API}/chat/summarize-research-text",
            json={"content": content}
        )
        return r.json() if r.status_code == 200 else {"error": r.text}

async def direct_summarize_video(video_url: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.post(
            f"{VOLVOX_API}/chat/summarize-video",
            params={"video_url": video_url}
        )
        return {"summary": r.text.strip()} if r.status_code == 200 else {"error": r.text}

async def direct_chat_history_list(user_id: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.get(f"{VOLVOX_API}/chat/chatHistory", params={"user_id": user_id})
        return r.json() if r.status_code == 200 else {"error": r.text}

async def direct_chat_history_get(user_id: str, chat_id: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.get(f"{VOLVOX_API}/chat/chatHistory/{chat_id}", params={"user_id": user_id})
        return r.json() if r.status_code == 200 else {"error": r.text}

async def direct_chat_history_delete(user_id: str, chat_id: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.delete(f"{VOLVOX_API}/chat/deleteChat/{chat_id}", params={"user_id": user_id})
        return r.json() if r.status_code == 200 else {"error": r.text}
    
async def direct_research_create(
    user_id: str,
    researchName: str,
    file: UploadFile
) -> Dict[str, Any]:

    query_params = {"user_id": user_id}
    data = {"researchName": researchName}  
    files = {
        "file": (file.filename, await file.read(), file.content_type or "application/octet-stream")
    }

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.post(
            f"{VOLVOX_API}/research/addResearch",
            params=query_params,  
            data=data,
            files=files
        )
    return r.json() if r.status_code in (200, 201) else {"error": r.text, "status_code": r.status_code}

async def direct_research_update(
    user_id: str,
    research_id: str,
    researchName: Optional[str] = None,
    file: Optional[UploadFile] = None
) -> Dict[str, Any]:
    query_params = {"user_id": user_id}
    data = {}
    files = None

    if researchName:
        data["researchName"] = researchName
    if file:
        files = {
            "file": (file.filename, await file.read(), file.content_type or "application/octet-stream")
        }

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.patch(
            f"{VOLVOX_API}/research/updateResearch/{research_id}",
            params=query_params,  
            data=data,
            files=files
        )
    return r.json() if r.status_code == 200 else {"error": r.text, "status_code": r.status_code}

async def direct_research_delete(user_id: str, research_id: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.delete(
            f"{VOLVOX_API}/research/deleteResearch/{research_id}",
            params={"user_id": user_id}
        )
        return r.json() if r.status_code == 200 else {"error": r.text}
    
async def direct_deep_answer(question: str, mode:str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.post(
            f"{SMART_API}/chat/messageQuery",
            json={
                "message": question,
                "mode": mode
            }
        )
        return r.json() if r.status_code == 200 else {"error": r.text}

async def direct_access_feasibility(summary: str):
    url = f"{INNOSCOPE_API}/feasibility/assess-from-summary-stream"
    collected = [] 

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        async with client.stream(
            "POST",
            url,
            json={"summary": summary},
            headers={"Accept": "text/event-stream"},
        ) as response:

            if response.status_code != 200:
                text = await response.aread()
                raise Exception(f"Error {response.status_code}: {text.decode()}")

            async for line in response.aiter_lines():
                if line.strip():
                    collected.append(line)

        return "\n".join(collected)

async def direct_access_roadmap(summary: str) -> str:
    url = f"{INNOSCOPE_API}/roadmap/generate-from-summary-stream"
    collected = []

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        async with client.stream(
            "POST",
            url,
            json={"summary": summary},
            headers={"Accept": "text/event-stream"},
        ) as response:

            if response.status_code != 200:
                text = await response.aread()
                raise Exception(f"Error {response.status_code}: {text.decode()}")

            async for line in response.aiter_lines():
                if line.strip():
                    collected.append(line)

    return "\n".join(collected)

async def direct_generate_proposal(report_text: str) -> bytes:
    url = f"{KICKSTART_API}/generate-proposal/"

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        response = await client.post(
            url,
            params={"report_text": report_text},
        )

        if response.status_code != 200:
            raise Exception(f"Error {response.status_code}: {response.text}")

        return response.content  



 