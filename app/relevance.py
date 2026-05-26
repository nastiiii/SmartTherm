"""Check whether a FAQ answer actually fits the user question (before AUTO / RAG)."""

from __future__ import annotations

import logging
import re

import httpx

from app.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    USE_RELEVANCE_JUDGE,
)
from app.query_normalize import token_overlap, token_set
from app.rag import ollama_enabled

logger = logging.getLogger(__name__)


_TOPIC_GATES: list[tuple[re.Pattern[str], tuple[str, ...]]] = [
    (
        re.compile(r"(?i)forecast|прогноз\s+погод|weather\s+api"),
        ("forecast", "прогноз", "openweather", "weather", "погодозавис", "home assistant"),
    ),
    (
        re.compile(r"(?i)погод.{0,20}(api|сервис|источник|forecast)"),
        ("forecast", "прогноз", "openweather", "weather", "mqtt", "home assistant"),
    ),
]

_JUDGE_SYSTEM = """Ты проверяешь, подходит ли карточка FAQ на вопрос пользователя SmartTherm.
Ответь ровно одним словом:
YES — карточка прямо отвечает на вопрос;
NO — другая тема, общий шаблон, про котёл/прошивку когда спрашивали другое, или ответ не по сути."""


def _topic_gate(query: str, faq_question: str, faq_answer: str) -> bool:
    blob = f"{faq_question} {faq_answer}".lower()
    for pat, required_any in _TOPIC_GATES:
        if pat.search(query):
            if not any(term in blob for term in required_any):
                return False
    return True


def heuristic_relevant(query: str, faq_question: str, faq_answer: str) -> bool:
    """Fast lexical check — no LLM."""
    if not _topic_gate(query, faq_question, faq_answer):
        return False

    q_ov = token_overlap(query, faq_question)
    a_ov = token_overlap(query, faq_answer)
    combined_ov = max(q_ov, a_ov * 0.85)


    if len(query.split()) <= 5:
        shared = token_set(query) & token_set(faq_question + " " + faq_answer)
        if shared and any(len(w) >= 6 for w in shared):
            return True
        if q_ov >= 0.10:
            return True
        if re.search(r"(?i)esp32|smarttherm|ds18b20", query) and re.search(
            r"(?i)esp32|smarttherm|ds18b20|датчик|прошивк", faq_question + faq_answer
        ):
            return True
        return False

    if combined_ov < 0.10:
        return False


    if "?" in query or re.search(
        r"(?i)^(как|что|где|можно|какой|подскаж|почему)\b", query
    ):
        if q_ov < 0.06 and a_ov < 0.05:
            return False

    return True


def _ollama_judge(query: str, faq_question: str, faq_answer: str) -> bool | None:
    if not ollama_enabled():
        return None

    faq_a = faq_answer[:1200] + ("…" if len(faq_answer) > 1200 else "")
    user = (
        f"Вопрос пользователя:\n{query}\n\n"
        f"Карточка FAQ — вопрос:\n{faq_question}\n\n"
        f"Карточка FAQ — ответ:\n{faq_a}\n\n"
        "Подходит ли эта карточка? YES или NO."
    )
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        "options": {"temperature": 0.0, "num_predict": 8},
    }
    try:
        with httpx.Client(timeout=min(OLLAMA_TIMEOUT, 45)) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            text = (r.json().get("message") or {}).get("content", "").strip().upper()
        if text.startswith("YES") or text == "Y":
            return True
        if text.startswith("NO") or text == "N":
            return False
        logger.info("Relevance judge unclear: %r", text[:40])
        return None
    except Exception as e:
        logger.warning("Relevance judge failed: %s", e)
        return None


def is_answer_relevant(
    query: str,
    faq_question: str,
    faq_answer: str,
    *,
    answer_text: str | None = None,
    match_similarity: float | None = None,
) -> bool:
    """
    True if we should send this FAQ to the user.
    Heuristic first; optional Ollama YES/NO when USE_RELEVANCE_JUDGE=1.
    """
    if not heuristic_relevant(query, faq_question, faq_answer):
        return False

    if answer_text and not heuristic_relevant(query, faq_question, answer_text):
        return False

    q_ov = token_overlap(query, faq_question)

    if match_similarity is not None and match_similarity >= 0.78 and q_ov >= 0.12:
        return True

    if USE_RELEVANCE_JUDGE:
        judged = _ollama_judge(query, faq_question, faq_answer)
        if judged is not None:
            return judged

    return True
