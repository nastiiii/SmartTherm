"""Hierarchical answer pipeline: FAQ → chat-RAG → general LLM → escalate."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from app.config import (
    CHAT_TIER_MIN_SIM,
    RAG_FALLBACK_TO_FAQ,
    RAG_TOP_CHUNKS,
    USE_CHAT_TIER,
    USE_GENERAL_LLM_FALLBACK,
    USE_RAG_FOR_AUTO,
)
from app.rag import (
    generate_chat_tier_answer,
    generate_general_answer,
    generate_rag_answer,
    ollama_enabled,
)
from app.relevance import is_answer_relevant
from app.retrieval import Match, ResponseMode, RetrievalResult, get_retriever
from app.text_quality import is_curated, is_good_faq_card, is_helpful_answer

SOFT_ANSWER_MIN_SIM = 0.52


_FOLLOWUP_HINTS = re.compile(
    r"(?i)\b(а\s+что|и\s+что|дальше|потом|ещё|еще|тогда|почему|зачем|"
    r"подробн|как\s+именно|пример|уточни|а\s+если|baxi|protherm|navien|"
    r"vaillant|bosch|buderus|ariston|zota|вайлант|бакси|протерм|"
    r"\d{2,4})\b"
)


def _merge_with_history(query: str, history: list[dict[str, Any]] | None) -> str:
    """If user sent a short follow-up — glue it with the last user query."""
    if not history:
        return query
    q = (query or "").strip()
    word_count = len(re.findall(r"\w+", q))
    if word_count == 0:
        return query
    is_short = word_count <= 4
    has_hint = bool(_FOLLOWUP_HINTS.search(q))
    if not (is_short or has_hint):
        return query
    prev_q = ""
    for turn in reversed(history):
        prev_q = (turn.get("user_query") or "").strip()
        if prev_q:
            break
    if not prev_q or prev_q == q:
        return query
    return f"{prev_q} — {q}"

CHAT_TIER_NOTE = (
    "\n\n_Ответ собран из обсуждений в чате сообщества SmartTherm — "
    "не официальная инструкция. При сомнениях нажмите «Позвать эксперта»._"
)
GENERAL_NOTE = (
    "\n\n_Это ответ языковой модели на основе общих знаний об отоплении и SmartTherm; "
    "в базе знаний точного ответа нет. Проверьте детали у эксперта._"
)


class AnswerTier(str, Enum):
    CURATED_FAQ = "curated_faq"
    CHAT_KB = "chat_kb"
    GENERAL_LLM = "general_llm"
    ESCALATED = "escalated"


@dataclass
class BotReply:
    mode: ResponseMode
    text: str
    faq_id: int | None
    similarity: float | None
    matches: list[Match]
    clarification_text: str | None = None
    used_rag: bool = False
    source_answer: str | None = None
    tier: AnswerTier = AnswerTier.CURATED_FAQ
    effective_query: str = ""
    history_used: bool = False


def _chunks_from_matches(
    matches: list[Match],
    *,
    limit: int | None = None,
    require_quality: bool = True,
) -> list[dict[str, Any]]:
    n = limit or RAG_TOP_CHUNKS
    chunks = []
    for m in matches[: n * 2]:
        if require_quality and not is_good_faq_card(m.question, m.answer, m.tags):
            continue
        chunks.append(
            {"question": m.question, "answer": m.answer, "tags": m.tags}
        )
        if len(chunks) >= n:
            break
    return chunks


def _polish_curated(query: str, matches: list[Match], fallback: str) -> tuple[str, bool]:
    if not ollama_enabled() or not matches:
        return fallback, False
    chunks = _chunks_from_matches(matches)
    if not chunks:
        return fallback, False
    rag_text = generate_rag_answer(query, chunks, source_answer=fallback)
    if rag_text:
        return rag_text, True
    return fallback, False


def _try_chat_tier(
    query: str,
    matches: list[Match],
    history: list[dict[str, Any]] | None = None,
) -> str | None:
    if not USE_CHAT_TIER or not ollama_enabled():
        return None
    pool = [m for m in matches if m.similarity >= CHAT_TIER_MIN_SIM]
    if not pool:
        return None
    chunks = _chunks_from_matches(pool, require_quality=False)
    if not chunks:
        return None
    return generate_chat_tier_answer(query, chunks, history=history)


def _try_general(
    query: str, history: list[dict[str, Any]] | None = None
) -> str | None:
    if not USE_GENERAL_LLM_FALLBACK or not ollama_enabled():
        return None
    return generate_general_answer(query, history=history)


def _escalate_reply(result: RetrievalResult, query: str) -> BotReply:
    text = (
        "Не уверен, что нашёл точный ответ — передаю вопрос эксперту.\n\n"
        "Если есть уточнения (модель котла, прошивка, датчик) — "
        "напишите следующим сообщением: добавлю к тому же обращению."
    )
    return BotReply(
        mode=ResponseMode.ESCALATE,
        text=text,
        faq_id=None,
        similarity=result.best.similarity if result.best else None,
        matches=result.matches,
        tier=AnswerTier.ESCALATED,
    )


def _curated_auto_reply(query: str, b: Match, result: RetrievalResult) -> BotReply:
    answer = b.answer
    used_rag = False
    if USE_RAG_FOR_AUTO and is_helpful_answer(b.answer, b.tags):
        polished, used_rag = _polish_curated(query, result.matches, answer)
        if used_rag:
            if is_answer_relevant(
                query,
                b.question,
                b.answer,
                answer_text=polished,
                match_similarity=b.similarity,
            ):
                answer = polished
            else:
                used_rag = False
                if not RAG_FALLBACK_TO_FAQ:
                    answer = b.answer
    return BotReply(
        mode=ResponseMode.AUTO,
        text=answer,
        faq_id=b.faq_id,
        similarity=b.similarity,
        matches=result.matches,
        used_rag=used_rag,
        source_answer=b.answer,
        tier=AnswerTier.CURATED_FAQ,
    )


def _chat_tier_reply(
    query: str,
    result: RetrievalResult,
    history: list[dict[str, Any]] | None = None,
) -> BotReply | None:
    text = _try_chat_tier(query, result.matches, history=history)
    if not text:
        return None
    best = result.best
    return BotReply(
        mode=ResponseMode.AUTO,
        text=text + CHAT_TIER_NOTE,
        faq_id=best.faq_id if best else None,
        similarity=best.similarity if best else None,
        matches=result.matches,
        used_rag=True,
        source_answer=best.answer if best else None,
        tier=AnswerTier.CHAT_KB,
    )


def _general_reply(
    query: str,
    result: RetrievalResult,
    history: list[dict[str, Any]] | None = None,
) -> BotReply | None:
    text = _try_general(query, history=history)
    if not text:
        return None
    return BotReply(
        mode=ResponseMode.AUTO,
        text=text + GENERAL_NOTE,
        faq_id=None,
        similarity=result.best.similarity if result.best else None,
        matches=result.matches,
        used_rag=True,
        source_answer=None,
        tier=AnswerTier.GENERAL_LLM,
    )


def build_reply(
    query: str,
    history: list[dict[str, Any]] | None = None,
) -> BotReply:
    """Build assistant reply.

    `history` — последние ходы диалога (oldest → newest), элементы вида
    ``{"user_query": ..., "bot_answer": ..., "tier": ...}``. Если запрос
    короткий или содержит маркеры follow-up («дальше», «а если», модель котла) —
    мы склеиваем его с предыдущим вопросом перед поиском.
    """
    effective_query = _merge_with_history(query, history)
    history_used = effective_query != query
    result: RetrievalResult = get_retriever().search(effective_query)

    def _finalize(reply: BotReply) -> BotReply:
        reply.effective_query = effective_query
        reply.history_used = history_used
        return reply


    if result.mode == ResponseMode.ESCALATE and result.best:
        helpful = next(
            (m for m in result.matches if is_helpful_answer(m.answer, m.tags)),
            None,
        )
        pick = helpful or result.best
        if (
            pick.similarity >= SOFT_ANSWER_MIN_SIM
            and is_good_faq_card(pick.question, pick.answer, pick.tags)
            and is_answer_relevant(effective_query, pick.question, pick.answer)
        ):
            result = RetrievalResult(
                mode=ResponseMode.AUTO, matches=result.matches, best=pick
            )

    if result.mode == ResponseMode.AUTO and result.best:
        b = result.best
        if not is_helpful_answer(b.answer, b.tags):
            helpful = next(
                (m for m in result.matches if is_helpful_answer(m.answer, m.tags)),
                None,
            )
            if helpful:
                b = helpful
                result = RetrievalResult(
                    mode=ResponseMode.AUTO, matches=result.matches, best=b
                )

        if is_curated(b.tags) and is_answer_relevant(
            effective_query, b.question, b.answer, match_similarity=b.similarity
        ):
            return _finalize(_curated_auto_reply(effective_query, b, result))


        chat_reply = _chat_tier_reply(effective_query, result, history=history)
        if chat_reply is not None:
            return _finalize(chat_reply)


        general = _general_reply(effective_query, result, history=history)
        if general is not None:
            return _finalize(general)

        return _finalize(_escalate_reply(result, effective_query))

    if result.mode == ResponseMode.CLARIFY:
        clarify = result.clarification_text or (
            "Выберите подходящий вариант кнопкой ниже."
        )
        return _finalize(
            BotReply(
                mode=ResponseMode.CLARIFY,
                text=clarify,
                faq_id=result.best.faq_id if result.best else None,
                similarity=result.best.similarity if result.best else None,
                matches=result.matches,
                tier=AnswerTier.CURATED_FAQ,
            )
        )


    chat_reply = _chat_tier_reply(effective_query, result, history=history)
    if chat_reply is not None:
        return _finalize(chat_reply)

    general = _general_reply(effective_query, result, history=history)
    if general is not None:
        return _finalize(general)

    return _finalize(_escalate_reply(result, effective_query))
