
"""
Create FAQ draft candidates from qa_domain_good.csv (for expert review).
Does NOT auto-publish — writes data/faq_drafts_candidates.csv
"""

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.anonymize import anonymize

SRC = ROOT / "data" / "qa_domain_good.csv"
OUT = ROOT / "data" / "faq_drafts_candidates.csv"
MIN_Q, MIN_A = 25, 40
MAX_ROWS = 80


def clean(s: str) -> str:
    s = anonymize(str(s or ""))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main() -> None:
    df = pd.read_csv(SRC).fillna("")
    rows = []
    seen = set()
    for _, r in df.iterrows():
        q = clean(r.get("q_text", ""))
        a = clean(r.get("a_text", ""))
        if len(q) < MIN_Q or len(a) < MIN_A:
            continue
        key = q[:80].lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "question": q[:400],
                "answer": a[:2000],
                "tags": "draft;from_chat",
                "session_id": r.get("session_id", ""),
            }
        )
        if len(rows) >= MAX_ROWS:
            break
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"Wrote {len(out)} draft candidates to {OUT}")


if __name__ == "__main__":
    main()
