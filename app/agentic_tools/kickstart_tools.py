import httpx
from typing import Dict, Any, Optional
import os

# KICKSTART_API = os.getenv("KICKSTART_JS_API_URL", "https://software-project-management-pwnl.vercel.app/")
KICKSTART_API= "http://localhost:5000"
DEFAULT_TIMEOUT = 300.0

async def kickstart_create_proposal(
    userid: str,
    proposal_data: Dict[str, Any]
) -> Dict[str, Any]:
    payload = {
        "userid": userid,
        **proposal_data
    }
    
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.post(
            f"{KICKSTART_API}/api/proposals/",
            json=payload
        )
        return r.json() if r.status_code == 201 else {"error": r.text}

async def kickstart_get_proposals(userid: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.get(f"{KICKSTART_API}/api/proposals/{userid}")
        return r.json() if r.status_code == 200 else {"error": r.text}

async def kickstart_get_proposal(
    userid: str,
    proposal_id: str
) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.get(f"{KICKSTART_API}/api/proposals/{userid}/{proposal_id}")
        return r.json() if r.status_code == 200 else {"error": r.text}

async def kickstart_update_proposal(
    proposal_id: str,
    userid: str,
    update_data: Dict[str, Any]
) -> Dict[str, Any]:
    payload = {
        "userid": userid,
        **update_data
    }
    
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.put(
            f"{KICKSTART_API}/api/proposals/{proposal_id}",
            json=payload
        )
        return r.json() if r.status_code == 200 else {"error": r.text}

async def kickstart_delete_proposal(
    proposal_id: str,
    userid: str
) -> Dict[str, Any]:
    """Delete a proposal"""
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.delete(
            f"{KICKSTART_API}/api/proposals/{userid}/{proposal_id}"
        )
        return r.json() if r.status_code == 200 else {"error": r.text}

async def kickstart_generate_proposal_ai(
    proposal_id: str,
    generation_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate AI proposal content for existing proposal"""
    payload = generation_data or {}
    
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.post(
            f"{KICKSTART_API}/api/proposals/{proposal_id}/generate",
            json=payload
        )
        return r.json() if r.status_code == 200 else {"error": r.text}

async def kickstart_edit_proposal_ai(
    proposal_id: str,
    edit_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Edit proposal using AI"""
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.post(
            f"{KICKSTART_API}/api/proposals/{proposal_id}/edit",
            json=edit_data
        )
        return r.json() if r.status_code == 200 else {"error": r.text}