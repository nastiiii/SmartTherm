"""Paths and constants for the shared FAQ / wiki knowledge base."""

from __future__ import annotations

from pathlib import Path

from app.config import DATA_DIR, ROOT


KB_CSV = DATA_DIR / "faq_seed.csv"
WIKI_SITE_DIR = ROOT / "wiki" / "site"
TARGET_CARD_COUNT = 650
