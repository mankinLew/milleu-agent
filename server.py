# server.py
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

app = FastAPI()

# Allow your Freshdesk portal to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for production you can restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
WORKFLOW_ID = os.getenv("CHATKIT_WORKFLOW_ID")

class SessionRequest(BaseModel):
    user_id: str | None = None

@app.post("/api/chatkit/session")
async def create_chatkit_session(body: SessionRequest):
    user = body.user_id or "anonymous"

    session = client.chatkit.sessions.create({
        "workflow": {"id": WORKFLOW_ID},
        "user": user
    })

    return {"client_secret": session.client_secret}
