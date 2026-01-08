# server.py
import os
import traceback
import openai  # <--- add this
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

print("OpenAI package version on server:", openai.__version__)

app = FastAPI()

# CORS so Freshdesk / browser can call it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple health check / root
@app.get("/")
async def root():
    return {"status": "ok"}

# OpenAI + ChatKit
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
WORKFLOW_ID = os.getenv("CHATKIT_WORKFLOW_ID")

class SessionRequest(BaseModel):
    user_id: str | None = None

@app.post("/api/chatkit/session")
async def create_chatkit_session(body: SessionRequest):
    if not WORKFLOW_ID:
        raise HTTPException(
            status_code=500,
            detail="CHATKIT_WORKFLOW_ID is not set in environment variables",
        )
    if not client.api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not set in environment variables",
        )

    try:
        user = body.user_id or "anonymous"

        session = client.chatkit.sessions.create({
            "workflow": {"id": WORKFLOW_ID},
            "user": user,
        })

        return {"client_secret": session.client_secret}

    except Exception as e:
        # Print full traceback to Render logs
        print("ERROR in /api/chatkit/session:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
