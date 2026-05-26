"""Filter low-quality FAQ / RAG text."""

from __future__ import annotations

import re


JUNK_ANSWER = re.compile(
    r"(?i)(rockchip|cortex-a53|bogomips|cpu\)|описание из отзыва|"
    r"ozon|wildberries|entity_id:\s*sensor\.|/homeassistant/esphome/)"
)

CHAT_SLANG = re.compile(
    r"(?i)(хрен\b|говен|сикось|фиг пойм|блин\b|лол\b|тьфу|крюки не доходят)"
)
CHAT_REACTION = re.compile(
    r"(?i)^(ааа[,\s]|ну вот |не совсем\.|да там |кстати,|во[,.\s]|ага[,.\s])"
)
INSTRUCTIONAL = re.compile(
    r"(?i)(подключ|настрой|провер|клемм|инструкц|режим|отключите|включите|"
    r"выберите|уточните|важно:|шаг|схем|open.?therm|сухой контакт|\d\))"
)
JUNK_QUESTION = re.compile(
    r"(?i)^(доброе время суток|привет|спасибо|ок)\b"
)
CHAT_QUESTION = re.compile(
    r"(?i)(говен|хрен|пощелкай|розетк|насос и возможно)"
)

ALLOWED_LATIN = re.compile(
    r"(?i)\b(opentherm|mqtt|wifi|wi-fi|gpio|usb|ha|esphome|esp32|esp8266|"
    r"on/off|ch2|pid|dns|dhcp|ssid|ot|baxi)\b"
)
LATIN_WORD = re.compile(r"\b[a-z]{4,}\b", re.I)


def is_helpful_answer(answer: str, tags: str = "") -> bool:
    """Answer looks like support text, not a chat remark."""
    a = (answer or "").strip()
    if len(a) < 50:
        return False
    if CHAT_SLANG.search(a):

        if CHAT_REACTION.match(a) or re.match(r"(?i)^ну вот\b", a):
            return False
        if not re.search(
            r"(?i)(подключите|отключите|проверьте|настройте|выполните|"
            r"убедитесь|шаг\s*\d|\d\)\s)",
            a,
        ):
            return False
    if CHAT_REACTION.match(a) and not INSTRUCTIONAL.search(a):
        return False
    if "from_chat" in (tags or ""):
        if len(a) < 100 and not INSTRUCTIONAL.search(a):
            return False
        if not INSTRUCTIONAL.search(a):
            casual = re.search(
                r"(?i)\b(думаю|наверное|вроде|кажется|у меня|получишь|парьтесь)\b",
                a,
            )
            if casual:
                return False
    return True


def is_good_faq_card(question: str, answer: str, tags: str = "") -> bool:
    q, a = (question or "").strip(), (answer or "").strip()
    if len(q) < 12 or len(a) < 40:
        return False
    if JUNK_QUESTION.search(q) or JUNK_ANSWER.search(a) or CHAT_QUESTION.search(q):
        return False
    if len(a) > 1800 and JUNK_ANSWER.search(a[:400]):
        return False
    if not is_helpful_answer(a, tags):
        return False
    return True


def is_curated(tags: str) -> bool:
    return "from_chat" not in (tags or "")


def rag_output_ok(text: str) -> bool:
    """Reject RAG if it mixes English or looks broken."""
    if not text or len(text) < 20:
        return False
    if JUNK_ANSWER.search(text):
        return False

    lowered = text.lower()
    for m in LATIN_WORD.finditer(text):
        w = m.group(0).lower()
        if ALLOWED_LATIN.search(w):
            continue

        if w in (
            "never",
            "reset",
            "connect",
            "device",
            "step",
            "note",
            "please",
            "check",
            "ensure",
            "follow",
            "important",
        ):
            return False

    latin = len(LATIN_WORD.findall(text))
    cyrillic = len(re.findall(r"[а-яё]", lowered))
    if latin > 3 and latin > cyrillic // 20:
        return False
    return True


def _significant_tokens(text: str) -> set[str]:
    words = re.findall(r"[а-яёa-z0-9]{4,}", (text or "").lower())
    stop = {
        "этого",
        "котор",
        "можно",
        "нужно",
        "если",
        "тогда",
        "чтобы",
        "будет",
        "есть",
        "или",
        "также",
        "после",
        "перед",
        "когда",
        "где",
        "какой",
        "какая",
        "какие",
        "важно",
        "smarttherm",
    }
    return {w for w in words if w not in stop}


def source_overlap(rag_text: str, source: str) -> float:
    """Доля значимых токенов источника, встретившихся в RAG-ответе (0..1)."""
    src = _significant_tokens(source)
    out = _significant_tokens(rag_text)
    if not src or not out:
        return 0.0
    return len(src & out) / max(len(src), 1)


def rag_faithful_to_source(rag_text: str, source: str, min_overlap: float = 0.12) -> bool:
    """RAG must stay close to FAQ — reject obvious hallucinations."""
    return source_overlap(rag_text, source) >= min_overlap


def button_label_from_question(question: str, max_len: int = 56) -> str:
    q = re.sub(r"\s+", " ", (question or "").strip())
    if len(q) <= max_len:
        return q
    return q[: max_len - 1] + "…"
