"""FAQ logic filter rejects mismatched chat Q/A pairs."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.faq_logic import is_logically_consistent


def test_rejects_mismatched_chat_pair():
    q = "как подключить котёл к smarttherm?"
    a = "ааа, газ сикось-накось, хрен подлезешь за баками"
    assert not is_logically_consistent(q, a, "from_chat;wiring", qa_sim=0.1)


def test_accepts_curated_boiler_card():
    q = "Как подключить котёл к SmartTherm (общая схема)?"
    a = (
        "Общий порядок подключения: 1) Уточните модель котла. "
        "2) Отключите питание. 3) Подключите клеммы OT."
    )
    assert is_logically_consistent(q, a, "curated;wiring;opentherm")
