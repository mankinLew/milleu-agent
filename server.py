# server.py
import os
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import requests

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WORKFLOW_ID = os.getenv("CHATKIT_WORKFLOW_ID")


@app.get("/")
async def root():
    return {
        "status": "ok",
        "has_api_key": bool(OPENAI_API_KEY),
        "has_workflow_id": bool(WORKFLOW_ID),
    }


class SessionRequest(BaseModel):
    user_id: str | None = None


@app.post("/api/chatkit/session")
async def create_chatkit_session(body: SessionRequest):
    if not OPENAI_API_KEY:
        raise HTTPException(500, "OPENAI_API_KEY is not set")
    if not WORKFLOW_ID:
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
                "workflow": {"id": WORKFLOW_ID},
                "user": user,
            },
            timeout=10,
        )

        if not resp.ok:
            # Bubble up OpenAI error body so you can see what's wrong
            raise HTTPException(
                status_code=500,
                detail=f"OpenAI error {resp.status_code}: {resp.text}",
            )

        data = resp.json()
        client_secret = data.get("client_secret")
        if not client_secret:
            raise HTTPException(
                status_code=500,
                detail=f"Missing client_secret in OpenAI response: {data}",
            )

        return {"client_secret": client_secret}

    except HTTPException:
        # Already handled above
        raise
    except Exception as e:
        print("ERROR in /api/chatkit/session:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
