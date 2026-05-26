
"""Measure how often Tier-1 RAG output drifts from the source FAQ.

Для каждого запроса из ``data/eval_gold.csv``:
  1. Через ``FaqRetriever`` находим лучшую карточку.
  2. Просим Ollama сформулировать Tier-1 RAG ответ (по этой карточке).
  3. Считаем лексическое перекрытие со значимыми токенами карточки.
  4. Сравниваем поведение «до фильтра» и «после фильтра» (``rag_faithful_to_source``).

Дополнительно: метрика отдельно по группам ``curated`` и ``from_chat`` (если карточка
из чата — мы и не ожидаем дословного совпадения, но всё равно полезно увидеть цифру).

Выход:
  data/hallucination_report.md   — таблица + сводка
  data/hallucination_report.json — то же машиночитаемо, плюс полные тексты RAG-ответов
                                   (для ручного аудита).

Запуск:
    make eval-hallucinations
    # или
    python scripts/eval_hallucinations.py --limit 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import db as database
from app.knowledge_base import KB_CSV
from app.rag import check_ollama_health, generate_rag_answer, ollama_enabled
from app.retrieval import get_retriever
from app.text_quality import (
    is_curated,
    is_good_faq_card,
    rag_faithful_to_source,
    source_overlap,
)

GOLD = ROOT / "data" / "eval_gold.csv"
OUT_MD = ROOT / "data" / "hallucination_report.md"
OUT_JSON = ROOT / "data" / "hallucination_report.json"

THRESHOLD = 0.12


def _bucket(tags: str) -> str:
    return "curated" if is_curated(tags) else "from_chat"


def _summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rag_rows = [r for r in rows if r["rag_attempted"]]
    drifted = [r for r in rag_rows if not r["faithful"]]
    kept = [r for r in rows if r["kept"]]
    by_bucket: dict[str, dict[str, Any]] = {}
    for bucket in ("curated", "from_chat"):
        bucket_rag = [r for r in rag_rows if r["bucket"] == bucket]
        if not bucket_rag:
            continue
        bucket_drift = [r for r in bucket_rag if not r["faithful"]]
        by_bucket[bucket] = {
            "rag_attempts": len(bucket_rag),
            "drifts_pre_filter": len(bucket_drift),
            "hallucination_rate": round(len(bucket_drift) / len(bucket_rag), 3),
            "mean_overlap": round(mean(r["overlap"] for r in bucket_rag), 3),
        }
    return {
        "total_queries": len(rows),
        "rag_attempts": len(rag_rows),
        "rag_drifts_pre_filter": len(drifted),
        "rag_rejected_by_filter_after": len(rag_rows) - len(kept),
        "hallucination_rate_pre_filter": (
            round(len(drifted) / len(rag_rows), 3) if rag_rows else 0.0
        ),
        "kept_after_filter_rate": (
            round(len(kept) / len(rag_rows), 3) if rag_rows else 0.0
        ),
        "mean_overlap": (
            round(mean(r["overlap"] for r in rag_rows), 3) if rag_rows else 0.0
        ),
        "threshold_overlap": THRESHOLD,
        "by_bucket": by_bucket,
    }


def _write_md(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    md = [
        "# Hallucination report — Tier-1 RAG vs source FAQ",
        "",
        f"Порог фильтра `rag_faithful_to_source` = **{THRESHOLD}**. "
        "Запрос → лучшая карточка → ответ Qwen2.5 → перекрытие со значимыми токенами карточки.",
        "",
        "## Сводка",
        "",
        f"- Запросов всего: **{summary['total_queries']}**",
        f"- Попыток Tier-1 RAG: **{summary['rag_attempts']}**",
        f"- Доля «уплываний» **до фильтра** (hallucination_rate): "
        f"**{summary['hallucination_rate_pre_filter']:.1%}**",
        f"- Доля RAG-ответов, прошедших фильтр (kept_rate): "
        f"**{summary['kept_after_filter_rate']:.1%}**",
        f"- Среднее перекрытие: **{summary['mean_overlap']:.3f}**",
        "",
    ]
    if summary["by_bucket"]:
        md += [
            "### По типу карточки",
            "",
            "| Тип | RAG attempts | drifts | hallucination_rate | mean overlap |",
            "|---|---:|---:|---:|---:|",
        ]
        for bucket, b in summary["by_bucket"].items():
            md.append(
                f"| {bucket} | {b['rag_attempts']} | {b['drifts_pre_filter']} | "
                f"{b['hallucination_rate']:.1%} | {b['mean_overlap']:.3f} |"
            )
        md.append("")
    md += [
        "## По запросам",
        "",
        "| Query | FAQ | bucket | overlap | faithful | kept | note |",
        "|---|---:|---|---:|:---:|:---:|---|",
    ]
    for r in rows:
        q = (r["query"] or "")[:60].replace("|", "\\|")
        md.append(
            f"| {q} | {r['best_faq_id']} | {r['bucket']} | "
            f"{r['overlap']:.3f} | {'YES' if r['faithful'] else 'NO'} | "
            f"{'YES' if r['kept'] else 'NO'} | {r['note'] or ''} |"
        )
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Обработать только первые N запросов (для быстрого smoke-test)",
    )
    args = parser.parse_args()

    if not GOLD.exists():
        print("Run: python scripts/build_eval_gold.py")
        sys.exit(1)
    if not ollama_enabled():
        print("Ollama is not configured (OLLAMA_BASE_URL is empty).")
        sys.exit(1)

    health = check_ollama_health()
    if not health.get("ok"):
        print("Ollama is not reachable:", health.get("error"))
        sys.exit(1)
    if not health.get("model_pulled"):
        print(
            "Warning: model not pulled. Run `make ollama-pull` (or `ollama pull"
            f" {health.get('configured_model')}`)."
        )

    database.init_db()
    df_seed = pd.read_csv(KB_CSV).fillna("")
    database.import_faq_from_rows(df_seed.to_dict(orient="records"), replace=True)
    retriever = get_retriever()
    retriever.reload_index()

    gold = pd.read_csv(GOLD).fillna("")
    if args.limit > 0:
        gold = gold.head(args.limit)

    rows: list[dict[str, Any]] = []
    started = time.time()

    for _, g in gold.iterrows():
        query = str(g["query"]).strip()
        if not query:
            continue
        result = retriever.search(query)
        best = result.best
        row: dict[str, Any] = {
            "query": query,
            "best_faq_id": best.faq_id if best else None,
            "bucket": _bucket(best.tags) if best else "—",
            "overlap": 0.0,
            "faithful": False,
            "kept": False,
            "rag_attempted": False,
            "rag_text": "",
            "source_answer": best.answer if best else "",
            "note": "",
        }
        if not best:
            row["note"] = "no candidate"
            rows.append(row)
            continue
        if not is_good_faq_card(best.question, best.answer, best.tags):
            row["note"] = "low-quality candidate"
            rows.append(row)
            continue

        chunks = [{"question": best.question, "answer": best.answer, "tags": best.tags}]
        rag = generate_rag_answer(query, chunks, source_answer=None)
        row["rag_attempted"] = True
        if not rag:
            row["note"] = "INSUFFICIENT_CONTEXT or filter"
            rows.append(row)
            continue
        ov = source_overlap(rag, best.answer)
        row["overlap"] = round(ov, 3)
        row["faithful"] = ov >= THRESHOLD
        row["kept"] = row["faithful"] and rag_faithful_to_source(rag, best.answer)
        row["rag_text"] = rag
        rows.append(row)

    elapsed = time.time() - started
    summary = _summarise(rows)
    summary["elapsed_sec"] = round(elapsed, 1)

    OUT_JSON.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_md(rows, summary)
    print(f"Wrote {OUT_MD} and {OUT_JSON}")
    for k, v in summary.items():
        if k == "by_bucket":
            continue
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
