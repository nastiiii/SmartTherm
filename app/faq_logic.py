"""Logical consistency checks for FAQ question–answer pairs."""

from __future__ import annotations

import re
from typing import Any

from app.query_normalize import token_overlap
from app.text_quality import is_curated, is_good_faq_card, is_helpful_answer


DOMAIN = re.compile(
    r"(?i)(smarttherm|смарттерм|кот[её]л|opentherm|open.?therm|датчик|"
    r"wifi|wi-fi|mqtt|прошив|термостат|гвс|отоплен|esp32|esp8266|"
    r"home\s*assistant|насос|реле|gpio|клемм|бойлер|baxi|контроллер|"
    r"esphome|температур|отоплен|модуляц|термистор|ds18b20|сухой контакт)"
)


SKIP_QUESTION = re.compile(
    r"(?i)(перепалка|благодарю за крайне|что хочу сказать|"
    r"обзор это активность|где отчеты выклываются|"
    r"^спасибо\b|^ок\.?$|^ага\.?$|wildberries|озон\b|гидропон|"
    r"купил на али панель|трава и близко)"
)
SKIP_ANSWER = re.compile(
    r"(?i)^(спасибо,|очень приятно получить|доброжелательный и главное|"
    r"тьфу-тьфу|парьтесь|лол\b)"
)


QA_SIM_THRESHOLD = 0.40

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        from app.config import MODEL_NAME

        _model = SentenceTransformer(MODEL_NAME)
    return _model


def qa_embedding_similarity(question: str, answer: str) -> float:
    model = _get_model()
    qe = model.encode(
        [f"вопрос: {question}"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    ae = model.encode(
        [f"ответ: {answer}"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    return float((qe * ae).sum())


def batch_qa_similarities(questions: list[str], answers: list[str]) -> list[float]:
    model = _get_model()
    qe = model.encode(
        [f"вопрос: {q}" for q in questions],
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    )
    ae = model.encode(
        [f"ответ: {a}" for a in answers],
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    )
    return [float((qe[i] * ae[i]).sum()) for i in range(len(questions))]


def is_domain_relevant(question: str, answer: str) -> bool:
    blob = f"{question} {answer}"
    return bool(DOMAIN.search(blob))


def is_support_question(question: str) -> bool:
    q = (question or "").strip()
    if len(q) < 15:
        return False
    if SKIP_QUESTION.search(q):
        return False
    return True


def is_logically_consistent(
    question: str,
    answer: str,
    tags: str = "",
    *,
    qa_sim: float | None = None,
) -> bool:
    """Question and answer belong together and fit support FAQ."""
    q, a = (question or "").strip(), (answer or "").strip()
    tags = tags or ""

    if not is_good_faq_card(q, a, tags):
        return False

    if is_curated(tags):
        return True

    if not is_support_question(q):
        return False
    if SKIP_ANSWER.search(a):
        return False
    if not is_domain_relevant(q, a):
        return False

    sim = qa_sim
    if sim is None:
        sim = qa_embedding_similarity(q, a)
    if sim < QA_SIM_THRESHOLD:
        return False


    if token_overlap(q, a) < 0.06 and sim < 0.48:
        return False

    return True


def filter_faq_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return logically valid FAQ rows and removal stats."""
    stats: dict[str, int] = {"kept": 0, "removed": 0}
    curated: list[dict[str, Any]] = []
    chat: list[dict[str, Any]] = []

    for row in records:
        tags = str(row.get("tags") or "")
        if is_curated(tags):
            if is_logically_consistent(
                str(row["question"]), str(row["answer"]), tags
            ):
                curated.append(row)
            else:
                stats["removed_curated"] = stats.get("removed_curated", 0) + 1
        else:
            chat.append(row)

    chat_ok: list[dict[str, Any]] = []
    if chat:
        sims = batch_qa_similarities(
            [str(r["question"]) for r in chat],
            [str(r["answer"]) for r in chat],
        )
        for row, sim in zip(chat, sims):
            tags = str(row.get("tags") or "")
            if is_logically_consistent(
                str(row["question"]),
                str(row["answer"]),
                tags,
                qa_sim=sim,
            ):
                chat_ok.append(row)
            else:
                reason = "low_sim" if sim < QA_SIM_THRESHOLD else "logic"
                stats[f"removed_{reason}"] = stats.get(f"removed_{reason}", 0) + 1

    kept = curated + chat_ok
    stats["kept"] = len(kept)
    stats["removed"] = len(records) - len(kept)
    return kept, stats
