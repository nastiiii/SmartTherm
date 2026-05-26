import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import db as database
from app.retrieval import ResponseMode, get_retriever


def setup_module():
    database.init_db()
    df = pd.read_csv(ROOT / "data" / "faq_seed.csv").fillna("")
    database.import_faq_from_rows(df.to_dict(orient="records"), replace=True)
    get_retriever().reload_index()


def test_wifi_high_confidence():
    r = get_retriever().search("не подключается к wifi")
    assert r.best is not None
    assert r.mode == ResponseMode.AUTO
    assert r.best.similarity >= 0.7


def test_unknown_escalate_or_clarify():
    r = get_retriever().search("абракадабра xyz несуществующая проблема")
    assert r.mode in (ResponseMode.ESCALATE, ResponseMode.CLARIFY)
