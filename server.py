# server.py
import os
import traceback
from typing import Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 🔹 Import your exported Agent Builder workflow code
# Replace 'agent_workflow' and 'run_workflow'
# with the actual module + function name from your export
from agent_workflow import run_workflow

# Load .env if running locally
load_dotenv()

app = FastAPI()

# CORS so Freshdesk + n8n can call the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to specific domains in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHATKIT_WORKFLOW_ID = os.getenv("CHATKIT_WORKFLOW_ID")

# ---------- Pydantic models ----------

class SessionRequest(BaseModel):
    user_id: Optional[str] = None


class N8nChatRequest(BaseModel):
    sessionId: Optional[str] = None
    message: str


# ---------- Health check ----------

@app.get("/")
async def root():
    return {
        "status": "ok",
        "has_api_key": bool(OPENAI_API_KEY),
        "has_workflow_id": bool(CHATKIT_WORKFLOW_ID),
    }


# ---------- ChatKit session endpoint (Freshdesk) ----------

@app.post("/api/chatkit/session")
async def create_chatkit_session(body: SessionRequest):
    """
    For Freshdesk ChatKit:
    Returns a client_secret for the given user.
    """
    if not OPENAI_API_KEY:
        raise HTTPException(500, "OPENAI_API_KEY is not set")
    if not CHATKIT_WORKFLOW_ID:
        raise HTTPException(500, "CHATKIT_WORKFLOW_ID is not set")

    user = body.user_id or "anonymous"

    try:
        resp = requests.post(
            "https://api.openai.com/v1/chatkit/sessions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
                "OpenAI-Beta": "chatkit_beta=v1",
            },
            json={
                "workflow": {"id": CHATKIT_WORKFLOW_ID},
                "user": user,
            },
            timeout=10,
        )

        if not resp.ok:
            raise HTTPException(
                status_code=500,
                detail=f"OpenAI ChatKit error {resp.status_code}: {resp.text}",
            )

        data = resp.json()
        client_secret = data.get("client_secret")
        if not client_secret:
            raise HTTPException(
                status_code=500,
                detail=f"Missing client_secret in ChatKit response: {data}",
            )

        return {"client_secret": client_secret}

    except HTTPException:
        raise
    except Exception as e:
        print("ERROR in /api/chatkit/session:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ---------- n8n chat endpoint (n8n chat widget) ----------

@app.post("/n8n/chat")
async def n8n_chat(req: N8nChatRequest):
    """
    For n8n chat widget:
    Calls the SAME Agent Builder workflow code that powers ChatKit
    (via the exported Agents SDK code).
    """
    if not OPENAI_API_KEY:
        raise HTTPException(500, "OPENAI_API_KEY is not set")

    try:
        # Use sessionId to keep multi-turn context, if your exported code supports it
        session_id = req.sessionId or "anonymous"

        # 🔹 This is the key part: call the exported workflow instead of a plain model
        reply_text = run_workflow(input_text=req.message, session_id=session_id)

        return {"reply": reply_text}

    except Exception as e:
        print("ERROR in /n8n/chat:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
