"""Admin REST API + web UI: python -m app.admin.api"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import db as database
from app.answer_service import build_reply
from app.config import ADMIN_API_HOST, ADMIN_API_KEY, ADMIN_API_PORT, DATA_DIR
from app.knowledge_base import KB_CSV, WIKI_SITE_DIR
from app.rag import check_ollama_health
from app.retrieval import refresh_retriever

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="SmartTherm FAQ Admin", version="0.2.0")


class FaqIn(BaseModel):
    question: str = Field(min_length=5)
    answer: str = Field(min_length=10)
    tags: str = ""
    active: bool = True


class FaqOut(FaqIn):
    id: int
    created_at: str
    updated_at: str


class SearchIn(BaseModel):
    query: str = Field(min_length=2)


class DraftIn(BaseModel):
    question: str
    answer: str
    tags: str = ""


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if ADMIN_API_KEY and x_api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.on_event("startup")
def startup() -> None:
    database.init_db()
    refresh_retriever()


@app.get("/")
def admin_ui():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "Admin UI not found"}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if WIKI_SITE_DIR.exists():
    app.mount(
        "/wiki",
        StaticFiles(directory=WIKI_SITE_DIR, html=True),
        name="wiki",
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "ollama": check_ollama_health()}


@app.get("/dashboard")
def dashboard(_: None = Depends(require_api_key)) -> dict[str, Any]:
    return database.dashboard_stats()


@app.get("/faq", response_model=list[FaqOut])
def list_faq(_: None = Depends(require_api_key)) -> list[dict[str, Any]]:
    return database.list_faq(active_only=False)


@app.get("/faq/{faq_id}", response_model=FaqOut)
def get_faq(faq_id: int, _: None = Depends(require_api_key)) -> dict[str, Any]:
    row = database.get_faq(faq_id)
    if not row:
        raise HTTPException(404, "Not found")
    return row


@app.post("/faq", response_model=dict[str, int])
def create_faq(body: FaqIn, _: None = Depends(require_api_key)) -> dict[str, int]:
    fid = database.upsert_faq(None, body.question, body.answer, body.tags, body.active)
    refresh_retriever()
    return {"id": fid}


@app.put("/faq/{faq_id}", response_model=dict[str, int])
def update_faq(
    faq_id: int, body: FaqIn, _: None = Depends(require_api_key)
) -> dict[str, int]:
    if not database.get_faq(faq_id):
        raise HTTPException(404, "Not found")
    database.upsert_faq(faq_id, body.question, body.answer, body.tags, body.active)
    refresh_retriever()
    return {"id": faq_id}


@app.delete("/faq/{faq_id}")
def delete_faq(faq_id: int, _: None = Depends(require_api_key)) -> dict[str, str]:
    if not database.get_faq(faq_id):
        raise HTTPException(404, "Not found")
    database.delete_faq(faq_id, soft=True)
    refresh_retriever()
    return {"status": "deactivated"}


@app.post("/faq/import-csv")
def import_csv(
    replace: bool = False,
    _: None = Depends(require_api_key),
) -> dict[str, int]:
    path = KB_CSV
    if not path.exists():
        raise HTTPException(404, f"Missing {path}")
    df = pd.read_csv(path).fillna("")
    n = database.import_faq_from_rows(df.to_dict(orient="records"), replace=replace)
    refresh_retriever()
    return {"imported": n}


@app.post("/faq/export-csv")
def export_csv(_: None = Depends(require_api_key)) -> dict[str, Any]:
    """Save DB → data/faq_seed.csv (then run build_wiki_site to refresh wiki HTML)."""
    n = database.export_faq_to_csv(KB_CSV)
    return {"exported": n, "path": str(KB_CSV)}


@app.get("/stats/feedback")
def stats(_: None = Depends(require_api_key)) -> dict[str, Any]:
    return database.feedback_stats()


@app.post("/search")
def search_test(body: SearchIn, _: None = Depends(require_api_key)) -> dict[str, Any]:
    reply = build_reply(body.query)
    return {
        "mode": reply.mode.value,
        "faq_id": reply.faq_id,
        "similarity": reply.similarity,
        "used_rag": reply.used_rag,
        "text_preview": reply.text[:500],
        "matches": [
            {
                "faq_id": m.faq_id,
                "similarity": round(m.similarity, 4),
                "question": m.question[:200],
                "tags": m.tags,
            }
            for m in reply.matches
        ],
    }


@app.get("/escalations")
def list_escalations(
    status: str = "open", _: None = Depends(require_api_key)
) -> list[dict[str, Any]]:
    return database.list_escalations(status=status or None, limit=100)


@app.post("/escalations/{esc_id}/resolve")
def resolve_esc(esc_id: int, _: None = Depends(require_api_key)) -> dict[str, str]:
    if not database.get_escalation(esc_id):
        raise HTTPException(404)
    database.resolve_escalation(esc_id)
    return {"status": "resolved"}


@app.get("/drafts")
def list_drafts(_: None = Depends(require_api_key)) -> list[dict[str, Any]]:
    return database.list_faq_drafts()


@app.post("/drafts")
def create_draft(body: DraftIn, _: None = Depends(require_api_key)) -> dict[str, int]:
    did = database.create_faq_draft(body.question, body.answer, body.tags)
    return {"id": did}


@app.post("/drafts/{draft_id}/publish")
def publish_draft(draft_id: int, _: None = Depends(require_api_key)) -> dict[str, int]:
    try:
        fid = database.publish_faq_draft(draft_id)
    except ValueError:
        raise HTTPException(404, "Draft not found")
    refresh_retriever()
    return {"faq_id": fid}


@app.post("/index/rebuild")
def rebuild_index(_: None = Depends(require_api_key)) -> dict[str, int]:
    n = refresh_retriever()
    return {"indexed": n}


def main() -> None:
    database.init_db()
    uvicorn.run(app, host=ADMIN_API_HOST, port=ADMIN_API_PORT)


if __name__ == "__main__":
    main()
