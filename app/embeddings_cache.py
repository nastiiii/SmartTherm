"""Persist FAQ embedding matrix to disk for faster restarts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from app.config import DATA_DIR, MODEL_NAME

CACHE_DIR = DATA_DIR / "cache"
META_FILE = CACHE_DIR / "embeddings_meta.json"
VEC_FILE = CACHE_DIR / "embeddings.npy"
IDS_FILE = CACHE_DIR / "faq_ids.json"


def _faq_fingerprint(faq_rows: list[dict]) -> str:
    payload = json.dumps(
        [
            (r["id"], r["question"], r.get("tags"), r.get("updated_at"))
            for r in faq_rows
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_cache(faq_rows: list[dict]) -> tuple[np.ndarray | None, list[int] | None]:
    if not META_FILE.exists() or not VEC_FILE.exists() or not IDS_FILE.exists():
        return None, None
    try:
        meta = json.loads(META_FILE.read_text(encoding="utf-8"))
        if meta.get("model") != MODEL_NAME:
            return None, None
        if meta.get("fingerprint") != _faq_fingerprint(faq_rows):
            return None, None
        emb = np.load(VEC_FILE)
        ids = json.loads(IDS_FILE.read_text(encoding="utf-8"))
        return emb, ids
    except Exception:
        return None, None


def save_cache(faq_rows: list[dict], emb: np.ndarray, ids: list[int]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(VEC_FILE, emb)
    IDS_FILE.write_text(json.dumps(ids), encoding="utf-8")
    META_FILE.write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "index": "question+tags-v2",
                "fingerprint": _faq_fingerprint(faq_rows),
                "count": len(ids),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
