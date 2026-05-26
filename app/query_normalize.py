"""Normalize user queries for better FAQ matching."""

from __future__ import annotations

import re

_REPLACEMENTS = (
    (r"(?i)homeassist", "home assistant"),
    (r"(?i)home\s*assistant", "home assistant"),
    (r"(?i)\bha\b", "home assistant"),
    (r"(?i)опентерм", "opentherm"),
    (r"(?i)вай\s*фай|wifi", "wi-fi"),
    (r"(?i)смарттерм|smarttherm", "smarttherm"),
)


def normalize_query(text: str) -> str:
    t = (text or "").strip()
    for pat, repl in _REPLACEMENTS:
        t = re.sub(pat, repl, t)
    t = re.sub(r"\s+", " ", t)
    return t


def token_set(text: str) -> set[str]:
    t = normalize_query(text).lower()
    t = re.sub(r"[^\w\sа-яё-]", " ", t, flags=re.I)
    words = {w for w in t.split() if len(w) > 2}
    stop = {
        "как",
        "что",
        "это",
        "или",
        "для",
        "при",
        "нет",
        "ли",
        "есть",
        "можно",
        "надо",
        "где",
        "почему",
        "когда",
        "the",
        "and",
    }
    return words - stop


def token_overlap(a: str, b: str) -> float:
    sa, sb = token_set(a), token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
