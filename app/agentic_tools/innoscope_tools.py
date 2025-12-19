import httpx
from typing import Dict, Any, Optional, List
from fastapi import UploadFile
import os

INNOSCOPE_API = os.getenv("INNOSCOPE_API_URL", "https://mustafanoor-innoscope-backend.hf.space")
DEFAULT_TIMEOUT = 300.0

async def innoscope_send_message(
    user_id: int,
    message: str,
    session_id: Optional[int] = None
) -> Dict[str, Any]:
    
    payload = {
        "user_id": user_id,
        "message": message
    }
    if session_id:
        payload["session_id"] = session_id

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.post(
            f"{INNOSCOPE_API}/chat/send-message",
            json=payload
        )
        return r.json() if r.status_code == 200 else {"error": r.text}

async def innoscope_get_chat_sessions(user_id: int) -> Dict[str, Any]:
    
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.get(
            f"{INNOSCOPE_API}/chat/sessions",
            params={"user_id": user_id}
        )
        return r.json() if r.status_code == 200 else {"error": r.text}

async def innoscope_get_session_messages(
    session_id: int,
    user_id: int
) -> Dict[str, Any]:
    
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.get(
            f"{INNOSCOPE_API}/chat/sessions/{session_id}/messages",
            params={"user_id": user_id}
        )
        return r.json() if r.status_code == 200 else {"error": r.text}

async def innoscope_assess_feasibility_from_chat_stream(
    session_id: int
) -> str:
    collected = []
    
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        async with client.stream(
            "POST",
            f"{INNOSCOPE_API}/feasibility/from-chat/{session_id}/stream",
            headers={"Accept": "text/event-stream"}
        ) as response:
            if response.status_code != 200:
                text = await response.aread()
                raise Exception(f"Error {response.status_code}: {text.decode()}")

            async for line in response.aiter_lines():
                if line.strip():
                    collected.append(line)

    return "\n".join(collected)

async def innoscope_assess_feasibility_from_file_stream(
    file: UploadFile
) -> str:
    collected = []
    
    files = {
        "file": (file.filename, await file.read(), file.content_type or "application/octet-stream")
    }

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        async with client.stream(
            "POST",
            f"{INNOSCOPE_API}/feasibility/generate-stream",
            files=files,
            headers={"Accept": "text/event-stream"}
        ) as response:
            if response.status_code != 200:
                text = await response.aread()
                raise Exception(f"Error {response.status_code}: {text.decode()}")

            async for line in response.aiter_lines():
                if line.strip():
                    collected.append(line)

    return "\n".join(collected)

async def innoscope_assess_feasibility_from_summary_stream(
    summary: str
) -> str:
    collected = []

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        async with client.stream(
            "POST",
            f"{INNOSCOPE_API}/feasibility/assess-from-summary-stream",
            json={"summary": summary},
            headers={"Accept": "text/event-stream"}
        ) as response:
            if response.status_code != 200:
                text = await response.aread()
                raise Exception(f"Error {response.status_code}: {text.decode()}")

            async for line in response.aiter_lines():
                if line.strip():
                    collected.append(line)

    return "\n".join(collected)

async def innoscope_generate_roadmap_from_file(
    file: UploadFile
) -> Dict[str, Any]:
    files = {
        "file": (file.filename, await file.read(), file.content_type or "application/octet-stream")
    }

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.post(
            f"{INNOSCOPE_API}/roadmap/generate",
            files=files
        )
        return r.json() if r.status_code == 200 else {"error": r.text}

async def innoscope_generate_roadmap_from_chat(
    session_id: int
) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.post(f"{INNOSCOPE_API}/roadmap/from-chat/{session_id}")
        return r.json() if r.status_code == 200 else {"error": r.text}

async def innoscope_generate_roadmap_from_file_stream(
    file: UploadFile
) -> str:
    """Generate roadmap from file with streaming"""
    collected = []
    
    files = {
        "file": (file.filename, await file.read(), file.content_type or "application/octet-stream")
    }

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        async with client.stream(
            "POST",
            f"{INNOSCOPE_API}/roadmap/generate-stream",
            files=files,
            headers={"Accept": "text/event-stream"}
        ) as response:
            if response.status_code != 200:
                text = await response.aread()
                raise Exception(f"Error {response.status_code}: {text.decode()}")

            async for line in response.aiter_lines():
                if line.strip():
                    collected.append(line)

    return "\n".join(collected)

async def innoscope_generate_roadmap_from_summary_stream(
    summary: str
) -> str:
    
    collected = []

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        async with client.stream(
            "POST",
            f"{INNOSCOPE_API}/roadmap/generate-from-summary-stream",
            json={"summary": summary},
            headers={"Accept": "text/event-stream"}
        ) as response:
            if response.status_code != 200:
                text = await response.aread()
                raise Exception(f"Error {response.status_code}: {text.decode()}")

            async for line in response.aiter_lines():
                if line.strip():
                    collected.append(line)

    return "\n".join(collected)


async def innoscope_summarize_text(text: str) -> Dict[str, Any]:
    
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.post(
            f"{INNOSCOPE_API}/summarize/text",
            json={"text": text}
        )
        return r.json() if r.status_code == 200 else {"error": r.text}

async def innoscope_summarize_file(file: UploadFile) -> Dict[str, Any]:
    
    files = {
        "file": (file.filename, await file.read(), file.content_type or "application/octet-stream")
    }

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.post(
            f"{INNOSCOPE_API}/summarize/file",
            files=files
        )
        return r.json() if r.status_code == 200 else {"error": r.text}