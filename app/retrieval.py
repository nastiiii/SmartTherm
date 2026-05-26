"""Embedding-based FAQ retrieval with three confidence modes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from app import db as database
from app.config import (
    BM25_WEIGHT,
    HIGH_TH,
    LEXICAL_AUTO_OVERLAP,
    MID_TH,
    MODEL_NAME,
    TOP_K,
    USE_EMBEDDING_CACHE,
    USE_HYBRID_RETRIEVAL,
)
from app.embeddings_cache import load_cache, save_cache
from app.query_normalize import normalize_query, token_overlap
from app.text_quality import is_curated, is_good_faq_card, is_helpful_answer


class ResponseMode(str, Enum):
    AUTO = "auto"
    CLARIFY = "clarify"
    ESCALATE = "escalate"


@dataclass
class Match:
    faq_id: int
    question: str
    answer: str
    tags: str
    similarity: float


@dataclass
class RetrievalResult:
    mode: ResponseMode
    matches: list[Match]
    best: Match | None
    clarification_text: str | None = None


_CURATED_INTENTS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"(?i)(как\s+)?подключ(ить|ение).{0,25}кот"),
        "подключить котёл",
    ),
    (
        re.compile(r"(?i)(home\s*assist|homeassistant|хоум\s*ассист|\bha\b)"),
        "home assistant",
    ),
    (re.compile(r"(?i)opentherm|open.?therm|\bot\b"), "opentherm"),
    (re.compile(r"(?i)не\s+подключ.{0,12}wi-?fi"), "wi-fi"),
    (re.compile(r"(?i)\bmqtt\b|mosquitto"), "mqtt"),
    (
        re.compile(r"(?i)(где|куда).{0,20}(став|стоит|повес|установ).{0,15}термостат"),
        "термостат",
    ),
    (
        re.compile(r"(?i)термостат.{0,25}(комнат|став|где|стоять)"),
        "термостат",
    ),
    (
        re.compile(
            r"(?i)(можно|опасно|разреш).{0,25}220.{0,10}(в|вольт)|"
            r"220.{0,15}(на\s+клемм|термостат|вход\s+котл|сухой\s+контакт)"
        ),
        "220",
    ),
    (
        re.compile(r"(?i)сухой\s+контакт"),
        "сухой контакт",
    ),
]


def _index_text(row: dict[str, Any]) -> str:
    """Embedding text: curated cards include answer lead-in for better recall."""
    q = str(row["question"])
    tags = (row.get("tags") or "").replace(";", " ")
    if is_curated(str(row.get("tags") or "")):
        lead = str(row["answer"])[:400]
        return f"{q} {lead} {tags}"
    return f"{q} {tags}"


_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)
_BM25_STOP = {
    "и", "в", "во", "на", "не", "что", "как", "у", "от", "к", "по", "с", "со",
    "это", "the", "of", "for", "to",
}


def _tokenize(text: str) -> list[str]:
    return [
        w for w in (m.group(0).lower() for m in _TOKEN_RE.finditer(text or ""))
        if w not in _BM25_STOP and len(w) >= 2
    ]


def _bm25_doc(row: dict[str, Any]) -> list[str]:
    q = str(row["question"])
    tags = (row.get("tags") or "").replace(";", " ")
    if is_curated(str(row.get("tags") or "")):
        ans = str(row["answer"])[:600]
        return _tokenize(f"{q} {ans} {tags}")
    return _tokenize(f"{q} {tags}")


class FaqRetriever:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self._emb: np.ndarray | None = None
        self._faq: list[dict[str, Any]] = []
        self._bm25 = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

    def _build_bm25(self) -> None:
        if not USE_HYBRID_RETRIEVAL or not self._faq:
            self._bm25 = None
            return
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            self._bm25 = None
            return
        corpus = [_bm25_doc(row) for row in self._faq]
        if not any(corpus):
            self._bm25 = None
            return
        self._bm25 = BM25Okapi(corpus)

    def reload_index(self) -> int:
        self._load_model()
        self._faq = database.list_faq(active_only=True)
        if not self._faq:
            self._emb = None
            self._bm25 = None
            return 0

        if USE_EMBEDDING_CACHE:
            cached_emb, cached_ids = load_cache(self._faq)
            if cached_emb is not None and cached_ids is not None:
                id_to_row = {int(f["id"]): f for f in self._faq}
                ordered = [id_to_row[i] for i in cached_ids if i in id_to_row]
                if len(ordered) == len(self._faq):
                    self._faq = ordered
                    self._emb = cached_emb
                    self._build_bm25()
                    return len(self._faq)

        texts = [_index_text(f) for f in self._faq]
        emb = self._model.encode(
            texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False
        )
        self._emb = np.asarray(emb, dtype="float32")
        if USE_EMBEDDING_CACHE:
            save_cache(self._faq, self._emb, [int(f["id"]) for f in self._faq])
        self._build_bm25()
        return len(self._faq)

    def _bm25_scores(self, query: str) -> np.ndarray | None:
        if self._bm25 is None or not self._faq:
            return None
        tokens = _tokenize(query)
        if not tokens:
            return None
        raw = np.asarray(self._bm25.get_scores(tokens), dtype="float32")
        if raw.size == 0:
            return None
        max_score = float(raw.max())
        if max_score <= 0:
            return np.zeros_like(raw)
        return raw / max_score

    def search(self, query: str, top_k: int = TOP_K) -> RetrievalResult:
        if not self._faq or self._emb is None:
            self.reload_index()
        if not self._faq or self._emb is None:
            return RetrievalResult(
                mode=ResponseMode.ESCALATE,
                matches=[],
                best=None,
                clarification_text=None,
            )

        query = normalize_query(query)
        intent_row = self._match_curated_intent(query)
        q_emb = self._model.encode([query], normalize_embeddings=True)
        q_emb = np.asarray(q_emb[0], dtype="float32")
        cos_sims = self._emb @ q_emb
        bm25 = self._bm25_scores(query)
        if bm25 is not None and bm25.shape == cos_sims.shape and BM25_WEIGHT > 0:
            sims = (1.0 - BM25_WEIGHT) * cos_sims + BM25_WEIGHT * bm25
        else:
            sims = cos_sims
        idx = np.argsort(-sims)[: max(top_k * 4, 12)]

        scored: list[Match] = []
        for i in idx:
            f = self._faq[int(i)]
            q = str(f["question"])
            a = str(f["answer"])
            tags = str(f.get("tags") or "")
            if not is_good_faq_card(q, a, tags):
                continue
            sim = float(sims[i])

            q_overlap = token_overlap(query, q)
            a_overlap = token_overlap(query, a)
            if is_curated(tags):
                sim += 0.14
            else:
                sim -= 0.12
            if q_overlap >= 0.25:
                sim += 0.10 * q_overlap
            if a_overlap >= 0.20:
                sim += 0.08 * a_overlap
            if not is_helpful_answer(a, tags):
                sim -= 0.25
            scored.append(
                Match(
                    faq_id=int(f["id"]),
                    question=q,
                    answer=a,
                    tags=tags,
                    similarity=sim,
                )
            )
        scored.sort(key=lambda m: -m.similarity)
        matches = scored[:top_k]

        best = _pick_best_match(scored, query) if scored else None
        if intent_row and (
            best is None
            or not is_curated(best.tags)
            or best.similarity < HIGH_TH
        ):
            ir = intent_row
            best = Match(
                faq_id=int(ir["id"]),
                question=str(ir["question"]),
                answer=str(ir["answer"]),
                tags=str(ir.get("tags") or ""),
                similarity=max(best.similarity if best else 0.0, HIGH_TH + 0.05),
            )
            if not any(m.faq_id == best.faq_id for m in matches):
                matches = [best] + matches
                matches = matches[:top_k]
        if not best:
            return RetrievalResult(
                mode=ResponseMode.ESCALATE, matches=[], best=None
            )

        overlap_best = token_overlap(query, best.question)
        force_auto = overlap_best >= LEXICAL_AUTO_OVERLAP and is_curated(best.tags)

        if best.similarity >= HIGH_TH or force_auto:
            mode = ResponseMode.AUTO
            clarify = None
        elif best.similarity >= MID_TH:
            mode = ResponseMode.CLARIFY
            clarify = build_clarification(matches)
        elif (
            overlap_best >= 0.32
            and is_good_faq_card(best.question, best.answer, best.tags)
            and is_curated(best.tags)
        ):
            mode = ResponseMode.AUTO
            clarify = None
        else:
            mode = ResponseMode.ESCALATE
            clarify = None

        return RetrievalResult(
            mode=mode,
            matches=matches,
            best=best,
            clarification_text=clarify,
        )


    def _match_curated_intent(self, query: str) -> dict[str, Any] | None:
        for pat, needle in _CURATED_INTENTS:
            if not pat.search(query):
                continue
            for row in self._faq:
                tags = str(row.get("tags") or "")
                if not is_curated(tags):
                    continue
                if needle in str(row["question"]).lower():
                    return row
        return None


def _pick_best_match(scored: list[Match], query: str) -> Match | None:
    """Prefer curated / instructional answers over chat one-liners."""
    if not scored:
        return None
    helpful = [m for m in scored if is_helpful_answer(m.answer, m.tags)]
    pool = helpful or scored
    curated = [m for m in pool if is_curated(m.tags)]
    if curated:
        return curated[0]

    for m in pool:
        if token_overlap(query, m.question) >= 0.35:
            return m
    return pool[0]


def build_clarification(matches: list[Match]) -> str:
    """Short clarifying question from top candidate topics."""
    if len(matches) < 2:
        q = matches[0].question[:80]
        return f"Уточните, пожалуйста: вы имеете в виду «{q}…»?"

    from app.text_quality import button_label_from_question

    labels = []
    for m in matches[:3]:
        lbl = button_label_from_question(m.question)
        if lbl not in labels:
            labels.append(lbl)

    if len(labels) >= 2:
        return (
            "Похоже, вопрос можно отнести к разным темам. "
            "Выберите подходящий вариант кнопкой ниже."
        )
    return "Уточните запрос или выберите вариант кнопкой ниже."


_retriever: FaqRetriever | None = None


def get_retriever() -> FaqRetriever:
    global _retriever
    if _retriever is None:
        _retriever = FaqRetriever()
        _retriever.reload_index()
    return _retriever


def refresh_retriever() -> int:
    r = get_retriever()
    return r.reload_index()
