"""FastAPI backend for the ShopEase support desk.

Endpoints
  GET  /health                       status of the API, API key, FAISS indexes and log file
  POST /tickets/stream               run a complaint through the four agents, streamed as NDJSON
  GET  /tickets                      all rows of the ticket log
  GET  /stats                        totals for the dashboard
  POST /tickets/{ticket_id}/review   human approval of an escalated reply (UserProxyAgent)

`streamlit run app.py` starts this API automatically. To run it on its own:
  uvicorn backend.api:app --port 8000
"""
from __future__ import annotations

import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend import config, ticket_log
from backend.agents.human_review import approve_reply
from backend.team import stream_ticket

app = FastAPI(title="ShopEase Support Desk API", version="1.0")


class TicketIn(BaseModel):
    complaint: str = Field(min_length=1)
    customer_name: str = ""
    order_id: str = ""


class ReviewIn(BaseModel):
    response: str = ""
    reviewed_by: str = Field(min_length=1)


@app.get("/health")
def health() -> dict:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    key_ok = bool(config.OPENAI_API_KEY)
    return {
        "status": "ok" if key_ok else "degraded",
        "api_key_loaded": key_ok,
        "chat_model": config.CHAT_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "catalog_index_built": (config.INDEX_DIR / "catalog.index").exists(),
        "kb_index_built": (config.INDEX_DIR / "kb.index").exists(),
        "log_writable": os.access(config.LOG_DIR, os.W_OK),
        "team": "SelectorGroupChat",
        "max_messages": config.MAX_MESSAGES,
    }


@app.post("/tickets/stream")
async def create_ticket(body: TicketIn) -> StreamingResponse:
    """Streams one JSON event per line (NDJSON) so the UI can show each agent live."""

    async def events():
        async for ev in stream_ticket(body.customer_name, body.order_id, body.complaint):
            yield json.dumps(ev, ensure_ascii=False) + "\n"

    return StreamingResponse(
        events(), media_type="application/x-ndjson", headers={"X-Accel-Buffering": "no"}
    )


@app.get("/tickets")
def list_tickets() -> list[dict]:
    return ticket_log.read_log().to_dict(orient="records")


@app.get("/stats")
def stats() -> dict:
    df = ticket_log.read_log()
    total = len(df)
    avg = df["urgency"].map({"Low": 1, "Medium": 2, "High": 3}).mean() if total else 0
    return {
        "total": total,
        "resolved": int((df["status"] == "Resolved").sum()),
        "escalated": int((df["status"] == "Escalated").sum()),
        "avg_urgency": round(float(avg), 2) if total else 0,
        "by_category": df["category"].value_counts().to_dict(),
    }


@app.post("/tickets/{ticket_id}/review")
def review_ticket(ticket_id: str, body: ReviewIn) -> dict:
    """Human-in-the-loop approval. A plain `def` so FastAPI runs it in a worker thread,
    where approve_reply() can start its own AutoGen (UserProxyAgent) run."""
    rows = ticket_log.read_log()
    match = rows[rows["ticket_id"] == ticket_id]
    if match.empty:
        raise HTTPException(404, f"{ticket_id} not found")
    t = match.iloc[0]
    if t["status"] != "Escalated":
        raise HTTPException(400, "Only escalated tickets need human review")
    approved, via_user_proxy = approve_reply(
        ticket_id, t["complaint"], t["escalation_reason"], t["response"], body.response
    )
    try:
        ticket_log.update_review(ticket_id, approved, body.reviewed_by.strip())
    except PermissionError:
        raise HTTPException(409, "tickets.csv is locked: close it if it is open in Excel")
    return {"ticket_id": ticket_id, "reviewed_by": body.reviewed_by.strip(),
            "via_user_proxy": via_user_proxy}