
"""Initialize DB, import FAQ, rebuild embedding index."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import db as database
from app.retrieval import get_retriever

if __name__ == "__main__":
    import pandas as pd

    database.init_db()
    df = pd.read_csv(ROOT / "data" / "faq_seed.csv").fillna("")
    n = database.import_faq_from_rows(df.to_dict(orient="records"), replace=True)
    indexed = get_retriever().reload_index()
    print(f"FAQ imported: {n}, indexed: {indexed}")
