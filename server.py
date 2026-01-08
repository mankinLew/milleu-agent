from openai import OpenAI
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import openai as openai_pkg  # just for version/debug

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "status": "ok",
        "openai_version": openai_pkg.__version__,
    }

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
WORKFLOW_ID = os.getenv("CHATKIT_WORKFLOW_ID")

class SessionRequest(BaseModel):
    user_id: str | None = None

@app.post("/api/chatkit/session")
async def create_chatkit_session(body: SessionRequest):
    if not WORKFLOW_ID:
        raise HTTPException(500, "CHATKIT_WORKFLOW_ID is not set")

    try:
        user = body.user_id or "anonymous"

        # 🔴 OLD (wrong)
        # session = client.chatkit.sessions.create({
        #     "workflow": {"id": WORKFLOW_ID},
        #     "user": user,
        # })

        # ✅ NEW (correct)
        session = client.beta.chatkit.sessions.create({
            "workflow": {"id": WORKFLOW_ID},
            "user": user,
        })

        return {"client_secret": session.client_secret}
    except Exception as e:
        print("ERROR in /api/chatkit/session:", e)
        raise HTTPException(status_code=500, detail=str(e))
