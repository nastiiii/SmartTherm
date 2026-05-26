"""Controlled RAG via local Ollama — answers only from retrieved FAQ context."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TEMPERATURE,
    OLLAMA_TIMEOUT,
)
from app.text_quality import is_good_faq_card, rag_faithful_to_source, rag_output_ok

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Ты помощник техподдержки SmartTherm (контроллер отопления).
Составь ответ пользователю СТРОГО на основе блока «Контекст» ниже.

Правила:
- Язык: русский. Допустимы только аббревиатуры Wi-Fi, MQTT, OpenTherm, HA, GPIO, USB, OT.
- Не используй английские слова и фразы (never, reset, step, please, check и т.д.).
- Не добавляй клеммы, версии прошивки, настройки и советы, которых нет в контексте.
- Формат: 4–6 коротких нумерованных пунктов или коротких абзацев, без вступления.
- Не выдумывай диагноз — только инструкции из контекста.
- Если в контексте нет ответа на вопрос — ответь ровно одной строкой: INSUFFICIENT_CONTEXT"""

CHAT_TIER_SYSTEM = """Ты помощник техподдержки SmartTherm.
В блоке «Контекст» — обсуждения из публичного Telegram-чата сообщества (не официальная документация).
Сформулируй понятный ответ на русском, опираясь на этот контекст и здравый смысл по теме отопления (OpenTherm, on/off, Wi-Fi, MQTT, Home Assistant, датчики).

Правила:
- Язык: русский. Можно использовать только аббревиатуры: Wi-Fi, MQTT, OpenTherm, HA, GPIO, USB, OT, ESP32, DS18B20.
- Не выдумывай конкретные клеммы, версии прошивки и пункты меню, которых нет в контексте.
- Если детали зависят от модели котла — попроси уточнить модель.
- 4–6 коротких пунктов или абзацев, без вступления."""

GENERAL_SYSTEM = """Ты помощник техподдержки SmartTherm — контроллера управления отопительным котлом (OpenTherm / on/off, Wi-Fi, MQTT, интеграция с Home Assistant).
Пользователь задал вопрос, на который в нашей базе знаний нет прямого ответа.

Ответь на русском, опираясь на общие знания об отоплении, протоколе OpenTherm, ESP32 и Home Assistant. Это вспомогательный ответ — позже его проверит человек-эксперт.

Правила:
- Язык: русский. Допустимы аббревиатуры Wi-Fi, MQTT, OpenTherm, HA, GPIO, USB, OT, ESP32, DS18B20, NTC.
- Не выдумывай конкретные клеммы SmartTherm, номера пунктов меню и точные версии прошивки.
- Если требуется модель котла или версия прошивки — попроси их у пользователя.
- 4–6 коротких пунктов; в конце явно отметь, что для точного решения стоит уточнить детали у эксперта.
- Если вопрос совсем не про SmartTherm / отопление / умный дом / датчики — ответь ровно: OUT_OF_SCOPE"""


def ollama_enabled() -> bool:
    return bool(OLLAMA_BASE_URL.strip())


def _build_user_prompt(query: str, chunks: list[dict[str, Any]]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        q = c.get("question", "")
        a = c.get("answer", "")
        if not is_good_faq_card(q, a, c.get("tags", "")):
            continue
        parts.append(f"[{i}] Вопрос: {q}\nОтвет: {a}")
    ctx = "\n\n".join(parts)
    return (
        f"Контекст:\n{ctx}\n\n"
        f"Вопрос пользователя: {query}\n\n"
        "Сформируй готовый ответ для Telegram, опираясь только на контекст."
    )


def generate_rag_answer(
    query: str,
    chunks: list[dict[str, Any]],
    *,
    source_answer: str | None = None,
) -> str | None:
    if not ollama_enabled() or not chunks:
        return None

    user_prompt = _build_user_prompt(query, chunks)
    if "[1]" not in user_prompt:
        return None

    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "options": {
            "temperature": OLLAMA_TEMPERATURE,
            "top_p": 0.9,
            "repeat_penalty": 1.12,
            "num_predict": OLLAMA_NUM_PREDICT,
            "num_ctx": OLLAMA_NUM_CTX,
        },
    }

    try:
        with httpx.Client(timeout=OLLAMA_TIMEOUT) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
        content = (data.get("message") or {}).get("content", "").strip()
        if not content or "INSUFFICIENT_CONTEXT" in content.upper():
            return None
        if not rag_output_ok(content):
            logger.info("RAG output rejected by language filter")
            return None
        if source_answer and not rag_faithful_to_source(content, source_answer):
            logger.info("RAG output rejected: low overlap with source FAQ")
            return None
        return content
    except Exception as e:
        logger.warning("Ollama RAG failed: %s", e)
        return None


def _call_ollama(system: str, user: str, *, num_predict: int | None = None) -> str | None:
    if not ollama_enabled():
        return None
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {
            "temperature": OLLAMA_TEMPERATURE,
            "top_p": 0.9,
            "repeat_penalty": 1.12,
            "num_predict": num_predict or OLLAMA_NUM_PREDICT,
            "num_ctx": OLLAMA_NUM_CTX,
        },
    }
    try:
        with httpx.Client(timeout=OLLAMA_TIMEOUT) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
        return (data.get("message") or {}).get("content", "").strip()
    except Exception as e:
        logger.warning("Ollama call failed: %s", e)
        return None


def _format_history(history: list[dict[str, Any]] | None) -> str:
    if not history:
        return ""
    lines = []
    for turn in history[-3:]:
        uq = (turn.get("user_query") or "").strip()
        ba = (turn.get("bot_answer") or "").strip()
        if not uq:
            continue
        ba_short = ba[:280] + ("…" if len(ba) > 280 else "")
        lines.append(f"Пользователь: {uq}\nБот: {ba_short}")
    if not lines:
        return ""
    return (
        "Предыдущий диалог (для понимания контекста, не цитируй его дословно):\n"
        + "\n---\n".join(lines)
        + "\n\n"
    )


def generate_chat_tier_answer(
    query: str,
    chunks: list[dict[str, Any]],
    *,
    history: list[dict[str, Any]] | None = None,
) -> str | None:
    """RAG ответ на основе нестрого курируемых фрагментов чата (Tier 2)."""
    if not chunks:
        return None
    parts = []
    for i, c in enumerate(chunks, 1):
        q = c.get("question", "")
        a = c.get("answer", "")
        parts.append(f"[{i}] Реплика из чата:\nВопрос: {q}\nОтвет: {a}")
    history_block = _format_history(history)
    user = (
        f"{history_block}"
        f"Контекст (фрагменты обсуждений в чате SmartTherm):\n"
        + "\n\n".join(parts)
        + f"\n\nВопрос пользователя: {query}\n\n"
        "Сформируй полезный ответ для Telegram."
    )
    content = _call_ollama(CHAT_TIER_SYSTEM, user)
    if not content or "INSUFFICIENT_CONTEXT" in content.upper():
        return None
    if not rag_output_ok(content):
        logger.info("Chat-tier RAG rejected by language filter")
        return None
    return content


def generate_general_answer(
    query: str, *, history: list[dict[str, Any]] | None = None
) -> str | None:
    """Общий ответ LLM по теме SmartTherm (Tier 3) — без жёсткого контекста."""
    history_block = _format_history(history)
    user = (
        f"{history_block}"
        f"Вопрос пользователя из чата техподдержки SmartTherm:\n{query}\n\n"
        "Дай полезный ответ, придерживаясь правил."
    )
    content = _call_ollama(GENERAL_SYSTEM, user, num_predict=OLLAMA_NUM_PREDICT)
    if not content:
        return None
    if "OUT_OF_SCOPE" in content.upper():
        return None
    if not rag_output_ok(content):
        logger.info("General LLM answer rejected by language filter")
        return None
    return content


def check_ollama_health() -> dict[str, Any]:
    if not ollama_enabled():
        return {"enabled": False, "ok": False, "model": OLLAMA_MODEL}
    try:
        base = OLLAMA_BASE_URL.rstrip("/")
        with httpx.Client(timeout=10) as client:
            r = client.get(f"{base}/api/tags")
            r.raise_for_status()
            models = [m.get("name") for m in r.json().get("models", [])]
        model_ready = any(
            (OLLAMA_MODEL.split(":")[0] in (m or "")) for m in models
        )
        return {
            "enabled": True,
            "ok": True,
            "models": models,
            "configured_model": OLLAMA_MODEL,
            "model_pulled": model_ready,
        }
    except Exception as e:
        return {
            "enabled": True,
            "ok": False,
            "configured_model": OLLAMA_MODEL,
            "error": str(e),
        }
