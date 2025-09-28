# backend/app.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from agent import Agent

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

agent = Agent()

class Q(BaseModel):
    question: str

@app.post("/api/ask")
async def ask(q: Q):
    return agent.answer(q.question)
