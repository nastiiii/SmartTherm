"""Non-interactive retrieval demo for report (Fig. 1). Writes data/demo_retrieval_log.txt"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import db as database
from app.config import HIGH_TH, MID_TH
from app.retrieval import ResponseMode, get_retriever

DEMO_QUERIES = [
    "не подключается к wifi",
    "котел baxi opentherm горячая вода",
    "ds18b20 несколько датчиков на одной линии",
]

OUT = ROOT / "data" / "demo_retrieval_log.txt"


def main() -> None:
    database.init_db()
    import pandas as pd

    path = ROOT / "data" / "faq_seed.csv"
    df = pd.read_csv(path).fillna("")
    database.import_faq_from_rows(df.to_dict(orient="records"), replace=True)

    retriever = get_retriever()
    n = retriever.reload_index()

    lines = [f"Loaded FAQ: {n} items", f"Thresholds: HIGH={HIGH_TH}, MID={MID_TH}", ""]

    for user_q in DEMO_QUERIES:
        result = retriever.search(user_q)
        lines.append(f"USER> {user_q}")
        lines.append("\nTop matches:")
        for rank, m in enumerate(result.matches, 1):
            lines.append(f"{rank}) sim={m.similarity:.3f} | {m.question[:120]}")
        lines.append("\nSYSTEM DECISION:")
        if result.mode == ResponseMode.AUTO:
            lines.append("Mode: AUTO-ANSWER (high confidence)")
            lines.append(f"\nANSWER:\n{result.best.answer[:500]}...")
        elif result.mode == ResponseMode.CLARIFY:
            lines.append("Mode: CLARIFY (medium confidence)")
            lines.append(result.clarification_text or "")
        else:
            lines.append("Mode: ESCALATE (low confidence)")
            if result.best:
                lines.append(f"Closest tags: {result.best.tags}")
        lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", OUT)
    print(OUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
