# server.py
import os
import traceback
from typing import Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# Load .env if running locally
load_dotenv()

# -------------------------------------------------------------------
# FastAPI app setup
# -------------------------------------------------------------------
app = FastAPI()

# CORS: allow your frontends (Freshdesk + n8n widget) to call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten this to your domains if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Environment / clients
# -------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHATKIT_WORKFLOW_ID = os.getenv("CHATKIT_WORKFLOW_ID")

# OpenAI client for normal chat (/n8n/chat)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


# -------------------------------------------------------------------
# Models
# -------------------------------------------------------------------
class SessionRequest(BaseModel):
    user_id: Optional[str] = None


class N8nChatRequest(BaseModel):
    sessionId: Optional[str] = None
    message: str


# -------------------------------------------------------------------
# Health check
# -------------------------------------------------------------------
@app.get("/")
async def root():
    """
    Simple health endpoint.
    """
    return {
        "status": "ok",
        "has_api_key": bool(OPENAI_API_KEY),
        "has_workflow_id": bool(CHATKIT_WORKFLOW_ID),
    }


# -------------------------------------------------------------------
# ChatKit session endpoint (for Freshdesk widget)
# -------------------------------------------------------------------
@app.post("/api/chatkit/session")
async def create_chatkit_session(body: SessionRequest):
    """
    Creates a ChatKit session and returns client_secret for the JS widget.
    Called by your Freshdesk page.
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
            # Bubble up full response for easier debugging
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
        # Already constructed with proper status/detail
        raise
    except Exception as e:
        print("ERROR in /api/chatkit/session:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------------
# n8n chat endpoint (for n8n chat widget)
# -------------------------------------------------------------------
@app.post("/n8n/chat")
async def n8n_chat(req: N8nChatRequest):
    """
    Simple text chat endpoint for n8n.
    n8n will POST { "sessionId": "...", "message": "..." } here,
    and we return { "reply": "..." }.
    """
    if not OPENAI_API_KEY:
        raise HTTPException(500, "OPENAI_API_KEY is not set")

    try:
        # You can use sessionId to implement conversation history if you like.
        # For now we just send a stateless request.
        response = openai_client.responses.create(
            model="gpt-4.1-mini",  # choose your model
            instructions="You are a helpful support assistant.",
            input=req.message,
        )

        # New Responses API: convenience property to get combined text
        reply_text = response.output_text

        return {"reply": reply_text}

    except Exception as e:
        print("ERROR in /n8n/chat:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
