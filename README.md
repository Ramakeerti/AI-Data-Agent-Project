# AI-Data-Agent-Project

## What this project does
Interactive conversational interface that answers complex analytical business questions over messy SQL data.
- Final answer appears in natural language.
- Supporting tables and charts are returned for evidence.
- Shows an explainable plan and generated SQL (toggleable).

## Tech stack
- Frontend: React + Recharts
- Backend: FastAPI (Python)
- Analytics DB: DuckDB (file-based)
- Optional LLM: OpenAI (set `OPENAI_API_KEY` to enable)

## Quick start (local)
1. Install prerequisites:
   - Node.js (for frontend)
   - Python 3.10+ and pip
2. Backend setup:
   ```bash
   cd backend
   pip install -r requirements.txt
   python db_init.py
   uvicorn backend.app:app --reload --port 8000

