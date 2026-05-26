
"""Проверка RAG через Ollama — отчёт data/rag_test_report.md (для курсовой / ablation)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.answer_service import build_reply
from app.config import OLLAMA_MODEL
from app.rag import check_ollama_health, generate_rag_answer
from app.retrieval import get_retriever

SAMPLE_QUERIES = [
    "не подключается к wifi",
    "как подключить котёл к smarttherm",
    "openherm не работает что проверить",
    "как связать с home assistant через mqtt",
    "какая версия прошивки",
    "два контроллера мешают друг другу",
]


def main() -> None:
    health = check_ollama_health()
    lines = [
        "# Тест RAG (Ollama)",
        "",
        f"- Модель: `{OLLAMA_MODEL}`",
        f"- Ollama: `{health}`",
        "",
    ]

    if not health.get("ok"):
        lines.append(
            "**Ollama недоступна.** Запустите: `docker compose up -d ollama` "
            "и `make ollama-pull`."
        )
        out = ROOT / "data" / "rag_test_report.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        print(out)
        sys.exit(1)

    retriever = get_retriever()
    retriever.reload_index()

    for q in SAMPLE_QUERIES:
        result = retriever.search(q)
        lines.append(f"## «{q}»")
        lines.append("")
        if not result.best:
            lines.append("_Нет совпадений в базе._")
            lines.append("")
            continue

        chunks = [
            {
                "question": m.question,
                "answer": m.answer,
                "tags": m.tags,
            }
            for m in result.matches[:3]
        ]
        faq = result.best.answer
        rag = generate_rag_answer(q, chunks, source_answer=faq)
        reply = build_reply(q)

        lines.append(f"- Режим: **{reply.mode.value}**, FAQ id={reply.faq_id}, RAG в боте={reply.used_rag}")
        lines.append("")
        lines.append("**Исходный FAQ (фрагмент):**")
        lines.append("")
        lines.append(faq[:600] + ("…" if len(faq) > 600 else ""))
        lines.append("")
        if rag:
            lines.append("**Ответ Ollama RAG:**")
            lines.append("")
            lines.append(rag)
        else:
            lines.append("_RAG не сгенерирован (фильтр или INSUFFICIENT_CONTEXT) — в боте будет текст FAQ._")
        lines.append("")

    out = ROOT / "data" / "rag_test_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
