"""Text anonymization for chat exports and knowledge base."""

from __future__ import annotations

import re

PHONE_RE = re.compile(r"(\+?\d[\d\-\s()]{7,}\d)")
EMAIL_RE = re.compile(r"[\w.\-]+@[\w.\-]+\.\w+")
URL_RE = re.compile(r"https?://\S+")
TG_USER_RE = re.compile(r"@[\w\d_]{4,}")
CARD_RE = re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b")
PASSPORT_RE = re.compile(r"\b\d{2}\s?\d{2}\s?\d{6}\b")
IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
MAC_RE = re.compile(
    r"\b([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"
)

ADDR_RE = re.compile(
    r"(?i)\b(ул\.|улица|пр\.|проспект|д\.|дом|кв\.|квартира)\s+[\w\d\-.]+"
)


def anonymize(text: str, *, mask_urls: bool = True) -> str:
    if not text:
        return ""
    s = text
    s = EMAIL_RE.sub("<EMAIL>", s)
    s = PHONE_RE.sub("<PHONE>", s)
    s = CARD_RE.sub("<CARD>", s)
    s = PASSPORT_RE.sub("<ID>", s)
    s = IP_RE.sub("<IP>", s)
    s = MAC_RE.sub("<MAC>", s)
    s = ADDR_RE.sub("<ADDRESS>", s)
    s = TG_USER_RE.sub("<USER>", s)
    if mask_urls:
        s = URL_RE.sub("<URL>", s)
    return s.strip()
