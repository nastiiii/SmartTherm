
"""
Build expanded knowledge base: curated cards + quality-filtered chat Q&A.
Output: data/faq_seed.csv (target ~500–800 cards).
"""

from __future__ import annotations

import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.anonymize import anonymize
from app.knowledge_base import KB_CSV, TARGET_CARD_COUNT
from app.faq_logic import QA_SIM_THRESHOLD, batch_qa_similarities, is_logically_consistent
from app.text_quality import is_good_faq_card

SEED = KB_CSV
QA_SRC = ROOT / "data" / "qa_domain_good.csv"
CLUSTERS = ROOT / "data" / "top_question_clusters.csv"
OUT = KB_CSV

MIN_Q, MIN_A = 20, 70
MAX_ANSWER_LEN = 2200
DEDUP_SIM = 0.80

POS = re.compile(
    r"(?i)\b(smarttherm|смарттерм|контроллер|кот[её]л|opentherm|open.?therm|"
    r"датчик|ds18b20|wifi|wi-fi|прошивк|mqtt|gpio|клемм|подключ|отоплен|гвс|"
    r"насос|реле|термостат|1-wire|esp32|esp8266|baxi|navien|ebus|home.?assistant)\b"
)
NEG = re.compile(
    r"(?i)\b(ozon|озон|wildberries|доставк|перезаказ|трек|упаковк|"
    r"цена|стоимост|маркетплейс|контрафакт)\b"
)
BAD_ANS = re.compile(
    r"(?i)\b(там выше|в закрепе|ищите в|зачем-зачем|лол|ха-ха)\b"
)
BAD_Q = re.compile(r"(?i)^(ага|ок|спасибо|понятно|да|нет)\.?$")


def clean(text: str) -> str:
    t = anonymize(str(text or ""))
    t = re.sub(r"\s+", " ", t).strip()
    return t


def infer_tags(q: str, a: str) -> str:
    blob = f"{q} {a}".lower()
    tags = []
    rules = [
        ("opentherm", r"opentherm|open.?therm|\bot\b"),
        ("wifi", r"wifi|wi-fi|ssid|роутер"),
        ("mqtt", r"mqtt|home.?assistant|\bha\b"),
        ("sensors", r"датчик|ds18b20|1-wire|ntc|термистор"),
        ("firmware", r"прошивк|firmware|версия|build|обновлен"),
        ("baxi", r"baxi|бакси"),
        ("wiring", r"клемм|подключ|провод|gpio|пин"),
        ("power", r"питани|блок питания|5v|usb"),
        ("settings", r"настройк|уставк|пид|pid"),
        ("hot_water", r"гвс|горяч|бойлер|ch2"),
        ("pressure", r"давлен"),
        ("safety", r"220|сухой контакт|безопас"),
        ("hardware", r"smarttherm32|esp32|esp8266|din"),
        ("diagnostics", r"не работает|ошибк|диагност|зависа"),
    ]
    for tag, pat in rules:
        if re.search(pat, blob, re.I):
            tags.append(tag)
    if not tags:
        tags.append("general")
    return ";".join(dict.fromkeys(tags))


def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower()[:200], b.lower()[:200]).ratio()


def normalize_question(q: str) -> str:
    q = q.lower().strip()
    q = re.sub(r"[^\w\sа-яё]", " ", q, flags=re.I)
    q = re.sub(r"\s+", " ", q)
    return q


def format_answer(a: str) -> str:
    a = clean(a)
    if len(a) > MAX_ANSWER_LEN:
        a = a[: MAX_ANSWER_LEN - 3] + "..."
    return a


def load_preserved() -> list[dict]:
    """Hand-edited cards kept across rebuilds. Chat-derived rows are rebuilt from QA each run."""
    if not SEED.exists():
        return []
    df = pd.read_csv(SEED).fillna("")
    rows = []
    for _, r in df.iterrows():
        q = clean(r["question"])
        a = format_answer(r["answer"])
        tags = str(r.get("tags") or "")
        if "from_chat" in tags:
            continue
        if not is_good_faq_card(q, a, tags):
            continue
        if len(q) < 10 or len(a) < 20:
            continue
        if "curated" not in tags:
            tags = (tags + ";curated").strip(";")
        rows.append(
            {
                "id": int(r["id"]),
                "question": q,
                "answer": a,
                "tags": tags,
                "source": "curated",
            }
        )
    return rows


def _qa_candidates(df: pd.DataFrame) -> list[dict]:
    raw: list[dict] = []
    for _, r in df.iterrows():
        q = clean(r.get("q_text", ""))
        a = format_answer(r.get("a_text", ""))
        if len(q) < MIN_Q or len(a) < MIN_A:
            continue
        if BAD_Q.match(q) or BAD_ANS.search(a) or NEG.search(q + " " + a):
            continue
        if not POS.search(q + " " + a):
            continue
        tags = infer_tags(q, a) + ";from_chat"
        if not is_good_faq_card(q, a, tags):
            continue
        raw.append(
            {
                "question": q[:450],
                "answer": a,
                "tags": tags,
                "source": "qa_domain_good",
                "_score": len(a) + len(q),
            }
        )
    if not raw:
        return []
    sims = batch_qa_similarities(
        [c["question"] for c in raw],
        [c["answer"] for c in raw],
    )
    cands = []
    for c, sim in zip(raw, sims):
        if sim < QA_SIM_THRESHOLD:
            continue
        if not is_logically_consistent(
            c["question"], c["answer"], c["tags"], qa_sim=sim
        ):
            continue
        c["_score"] += int(sim * 100)
        cands.append(c)
    cands.sort(key=lambda x: -x["_score"])
    return cands


def pick_from_qa(
    existing_questions: list[str], target_new: int
) -> list[dict]:
    df = pd.read_csv(QA_SRC).fillna("")
    out = []
    seen_norm = {normalize_question(q) for q in existing_questions}

    for cand in _qa_candidates(df):
        if len(out) >= target_new:
            break
        q = cand["question"]
        a = cand["answer"]
        nq = normalize_question(q)
        if nq in seen_norm:
            continue
        if any(similar(q, eq) > DEDUP_SIM for eq in existing_questions):
            continue
        seen_norm.add(nq)
        existing_questions.append(q)
        row = {k: v for k, v in cand.items() if k != "_score"}
        out.append(row)
    return out


def pick_from_clusters(existing_questions: list[str], start_id: int) -> list[dict]:
    if not CLUSTERS.exists():
        return []
    df = pd.read_csv(CLUSTERS).fillna("")
    qa = pd.read_csv(QA_SRC).fillna("")
    out = []
    next_id = start_id
    for _, c in df.iterrows():
        rep = clean(c.get("rep_question", ""))
        if len(rep) < MIN_Q or any(similar(rep, eq) > 0.75 for eq in existing_questions):
            continue
        kw = rep.lower().split()[:6]
        best_a = ""
        for _, r in qa.iterrows():
            q = clean(r.get("q_text", ""))
            a = format_answer(r.get("a_text", ""))
            if len(a) < MIN_A:
                continue
            if any(w in q.lower() or w in a.lower() for w in kw if len(w) > 4):
                if len(a) > len(best_a):
                    best_a = a
        if not best_a:
            continue
        existing_questions.append(rep)
        out.append(
            {
                "id": next_id,
                "question": rep[:450],
                "answer": best_a,
                "tags": infer_tags(rep, best_a) + ";cluster",
                "source": "cluster",
            }
        )
        next_id += 1
    return out


def main() -> None:
    preserved = load_preserved()
    questions = [r["question"] for r in preserved]
    target_new = max(0, TARGET_CARD_COUNT - len(preserved))
    new_qa = pick_from_qa(questions, target_new)

    rows = list(preserved)
    next_id = max((r["id"] for r in preserved), default=0) + 1
    for r in new_qa:
        r["id"] = next_id
        next_id += 1
        rows.append(r)

    for cr in pick_from_clusters(questions, next_id):
        if not any(similar(cr["question"], r["question"]) > 0.8 for r in rows):
            rows.append(cr)
            next_id = cr["id"] + 1


    for i, row in enumerate(rows, 1):
        row["id"] = i

    df = pd.DataFrame(rows)[["id", "question", "answer", "tags"]]
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} knowledge-base cards to {OUT} (target {TARGET_CARD_COUNT})")
    print(f"  hand-curated kept: {sum(1 for r in rows if r.get('source') == 'curated')}")
    print(f"  from chat QA: {sum(1 for r in rows if r.get('source')=='qa_domain_good')}")
    print(f"  from clusters: {sum(1 for r in rows if r.get('source')=='cluster')}")


if __name__ == "__main__":
    main()
