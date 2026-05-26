
"""
Evaluate retrieval quality on data/eval_gold.csv.
Writes data/eval_report.md and data/eval_report.json for the coursework report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import db as database
from app.knowledge_base import KB_CSV
from app.retrieval import ResponseMode, get_retriever

GOLD = ROOT / "data" / "eval_gold.csv"
REPORT_MD = ROOT / "data" / "eval_report.md"
REPORT_JSON = ROOT / "data" / "eval_report.json"


def _tag_hit(expected: str, actual: str) -> bool:
    if not expected.strip():
        return True
    exp = {t.strip() for t in expected.split(";") if t.strip()}
    act = {t.strip() for t in (actual or "").split(";") if t.strip()}
    return bool(exp & act)


def _id_hit(expected_id: int, got_id: int | None) -> bool:
    if expected_id <= 0:
        return got_id is None or True
    return got_id == expected_id


def main() -> None:
    if not GOLD.exists():
        print("Run: python scripts/build_eval_gold.py")
        sys.exit(1)

    database.init_db()
    df_seed = pd.read_csv(KB_CSV).fillna("")
    database.import_faq_from_rows(df_seed.to_dict(orient="records"), replace=True)
    retriever = get_retriever()
    retriever.reload_index()

    gold = pd.read_csv(GOLD).fillna("")
    results = []
    id_ok = tag_ok = mode_ok = 0
    mrr_sum = 0.0
    n = len(gold)

    for _, row in gold.iterrows():
        q = str(row["query"])
        exp_id = int(row.get("expected_faq_id") or 0)
        exp_tags = str(row.get("expected_tags") or "")
        exp_mode = str(row.get("expected_mode") or "auto").lower()

        res = retriever.search(q)
        got_id = res.best.faq_id if res.best else None
        got_tags = res.best.tags if res.best else ""
        got_mode = res.mode.value

        id_match = _id_hit(exp_id, got_id) if exp_id > 0 else True
        tag_match = _tag_hit(exp_tags, got_tags) if exp_id <= 0 or not id_match else True
        if exp_id > 0:
            hit = id_match
        else:
            hit = got_mode == exp_mode
        if exp_id > 0 and id_match:
            id_ok += 1
        if exp_id > 0 and (id_match or tag_match):
            tag_ok += 1
        if got_mode == exp_mode:
            mode_ok += 1

        rr = 0.0
        if exp_id > 0:
            for rank, m in enumerate(res.matches, 1):
                if m.faq_id == exp_id:
                    rr = 1.0 / rank
                    break
        mrr_sum += rr

        results.append(
            {
                "query": q,
                "expected_faq_id": exp_id,
                "got_faq_id": got_id,
                "expected_mode": exp_mode,
                "got_mode": got_mode,
                "similarity": round(res.best.similarity, 4) if res.best else 0.0,
                "id_match": id_match,
                "mode_match": got_mode == exp_mode,
            }
        )

    id_queries = sum(1 for _, r in gold.iterrows() if int(r.get("expected_faq_id") or 0) > 0)
    metrics = {
        "n_queries": n,
        "n_with_expected_id": id_queries,
        "top1_id_accuracy": round(id_ok / id_queries, 4) if id_queries else 0.0,
        "top1_id_or_tag_accuracy": round(tag_ok / id_queries, 4) if id_queries else 0.0,
        "mode_accuracy": round(mode_ok / n, 4),
        "mrr": round(mrr_sum / id_queries, 4) if id_queries else 0.0,
        "faq_indexed": len(retriever._faq),
    }

    REPORT_JSON.write_text(
        json.dumps({"metrics": metrics, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fails = [r for r in results if not r["id_match"] and r["expected_faq_id"] > 0][:15]
    md = [
        "# Отчёт eval_retrieval",
        "",
        f"- Запросов: **{metrics['n_queries']}**",
        f"- С эталонной карточкой: **{metrics['n_with_expected_id']}**",
        f"- Top-1 accuracy (id): **{metrics['top1_id_accuracy']:.1%}**",
        f"- Top-1 id или тег: **{metrics['top1_id_or_tag_accuracy']:.1%}**",
        f"- MRR: **{metrics['mrr']:.3f}**",
        f"- Точность режима (auto/clarify/escalate): **{metrics['mode_accuracy']:.1%}**",
        f"- FAQ в индексе: **{metrics['faq_indexed']}**",
        "",
        "## Примеры ошибок Top-1",
        "",
    ]
    for f in fails:
        md.append(
            f"- «{f['query'][:70]}…» → id {f['got_faq_id']} "
            f"(ожид. {f['expected_faq_id']}), sim={f['similarity']}, mode={f['got_mode']}"
        )
    REPORT_MD.write_text("\n".join(md), encoding="utf-8")
    print(REPORT_MD.read_text(encoding="utf-8"))
    print(f"\nSaved {REPORT_JSON}")


if __name__ == "__main__":
    main()
