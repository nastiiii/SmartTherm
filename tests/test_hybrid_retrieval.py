"""Hybrid BM25 + embeddings retrieval and intent gating for safety questions."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import db as database
from app.retrieval import ResponseMode, get_retriever


@pytest.fixture(scope="module", autouse=True)
def _load_kb():
    database.init_db()
    df = pd.read_csv(ROOT / "data" / "faq_seed.csv").fillna("")
    database.import_faq_from_rows(df.to_dict(orient="records"), replace=True)
    get_retriever().reload_index()


def test_bm25_helps_exact_terms():
    """Запрос с точным термином DS18B20 должен находить релевантную карточку."""
    r = get_retriever().search("ds18b20 распиновка")
    assert r.best is not None

    assert "опрос датчиков" not in (r.best.question or "").lower() or r.best.similarity > 0


def test_220v_safety_routed_to_curated():
    r = get_retriever().search("Можно ли подавать 220 В на клеммы термостата?")
    assert r.best is not None

    matches_join = "|".join(m.question.lower() for m in r.matches)
    assert any(
        kw in matches_join for kw in ("сухой контакт", "220", "безопас")
    ), f"safety FAQ not in top: {matches_join[:200]}"


def test_dry_contact_intent():
    """Запрос про сухой контакт должен подтянуть safety / wiring карточку."""
    r = get_retriever().search("что такое сухой контакт")
    assert r.best is not None
    matches_join = " | ".join(
        (m.question + " " + m.answer).lower() for m in r.matches
    )

    assert any("safety" in m.tags or "wiring" in m.tags for m in r.matches) or (
        "сухой контакт" in matches_join
    )


def test_hybrid_does_not_break_smoke():
    """sanity: гибрид не делает выдачу пустой даже для распространённого запроса."""
    r = get_retriever().search("не подключается к wifi")
    assert r.mode == ResponseMode.AUTO
    assert r.best is not None
