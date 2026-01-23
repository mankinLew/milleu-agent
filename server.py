# server.py
import os
import traceback
from typing import Optional, Any

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent_workflow import run_workflow, WorkflowInput  # uses your exported agent

load_dotenv()

app = FastAPI()

# Get health endpoint
@app.get("/healthz")
async def healthz():
    return {"ok": True}

# CORS so Freshdesk + n8n can call the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHATKIT_WORKFLOW_ID = os.getenv("CHATKIT_WORKFLOW_ID")


# ----------------------------
# Models
# ----------------------------
class SessionRequest(BaseModel):
    user_id: Optional[str] = None


class N8nChatRequest(BaseModel):
    sessionId: Optional[str] = None
    message: str


# ----------------------------
# Helpers
# ----------------------------
def extract_reply_text(result: Any) -> str:
    """
    Normalize the agent result into a single string for n8n.
    """
    if result is None:
        return ""

    # Most branches return {"message": "..."}
    if isinstance(result, dict):
        if isinstance(result.get("message"), str):
            return result["message"]
        # Sometimes guardrails returns {"safe_text": "..."} etc.
        if isinstance(result.get("safe_text"), str):
            return result["safe_text"]
        # Fallback: show something sensible
        return str(result)

    # If it's already a string
    if isinstance(result, str):
        return result

    return str(result)


# ----------------------------
# Health check
# ----------------------------
@app.get("/")
async def root():
    return {
        "status": "ok",
        "has_api_key": bool(OPENAI_API_KEY),
        "has_workflow_id": bool(CHATKIT_WORKFLOW_ID),
    }


# ----------------------------
# ChatKit session endpoint (Freshdesk ChatKit widget)
# ----------------------------
@app.post("/api/chatkit/session")
async def create_chatkit_session(body: SessionRequest):
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
            json={"workflow": {"id": CHATKIT_WORKFLOW_ID}, "user": user},
            timeout=20,
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


# ----------------------------
# n8n chat endpoint (n8n chat widget)
# ----------------------------
@app.post("/n8n/chat")
async def n8n_chat(req: N8nChatRequest):
    """
    n8n sends: { "sessionId": "...", "message": "..." }
    We run the SAME exported Agent Builder workflow and return: { "reply": "..." }
    """
    try:
        # Run the exported agent workflow (async)
        workflow_input = WorkflowInput(input_as_text=req.message)
        result = await run_workflow(workflow_input)

        reply_text = extract_reply_text(result)
        return {"reply": reply_text}

    except Exception as e:
        print("ERROR in /n8n/chat:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
