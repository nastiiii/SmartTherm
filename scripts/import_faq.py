"""Import FAQ cards from data/faq_seed.csv into SQLite."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from app import db as database
from app.retrieval import get_retriever


def main() -> None:
    path = ROOT / "data" / "faq_seed.csv"
    df = pd.read_csv(path).fillna("")
    database.init_db()
    n = database.import_faq_from_rows(df.to_dict(orient="records"), replace=True)
    get_retriever().reload_index()
    print(f"Imported {n} FAQ cards from {path}")


if __name__ == "__main__":
    main()
