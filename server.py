import os
import traceback
import openai as openai_pkg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok", "openai_version": openai_pkg.__version__}

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
WORKFLOW_ID = os.getenv("CHATKIT_WORKFLOW_ID")

class SessionRequest(BaseModel):
    user_id: str | None = None

@app.post("/api/chatkit/session")
async def create_chatkit_session(body: SessionRequest):
    if not WORKFLOW_ID:
        raise HTTPException(500, "CHATKIT_WORKFLOW_ID is not set")
    if not client.api_key:
        raise HTTPException(500, "OPENAI_API_KEY is not set")

    try:
        user = body.user_id or "anonymous"

        # ✅ IMPORTANT: Use top-level `.chatkit`, not `.beta.chatkit`
        session = client.chatkit.sessions.create(
            user=user,
            workflow={"id": WORKFLOW_ID},
        )

        return {"client_secret": session.client_secret}

    except Exception as e:
        print("ERROR in /api/chatkit/session:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
