
"""Load data/faq_drafts_candidates.csv into faq_drafts table."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import db as database

SRC = ROOT / "data" / "faq_drafts_candidates.csv"


def main() -> None:
    if not SRC.exists():
        print("Run expand_faq_drafts.py first")
        sys.exit(1)
    database.init_db()
    df = pd.read_csv(SRC).fillna("")
    n = 0
    for _, r in df.iterrows():
        database.create_faq_draft(
            str(r["question"]),
            str(r["answer"]),
            str(r.get("tags") or "draft"),
        )
        n += 1
    print(f"Loaded {n} drafts into DB")


if __name__ == "__main__":
    main()
