"""Regression: bot must not return casual chat junk for common questions."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import db as database
from app.answer_service import build_reply
from app.retrieval import ResponseMode, get_retriever
from app.text_quality import CHAT_SLANG


def setup_module():
    database.init_db()
    df = pd.read_csv(ROOT / "data" / "faq_seed.csv").fillna("")
    database.import_faq_from_rows(df.to_dict(orient="records"), replace=True)
    get_retriever().reload_index()


def test_boiler_connection_not_chat_junk():
    reply = build_reply("как подключить котёл?")
    assert reply.mode == ResponseMode.AUTO
    assert not CHAT_SLANG.search(reply.text)
    assert "подключ" in reply.text.lower()
    assert "1)" in reply.text or "порядок" in reply.text.lower()
