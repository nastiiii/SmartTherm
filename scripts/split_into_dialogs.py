import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

SRC_JSON = "data/result.json"
OUT_CSV  = "data/dialog_messages.csv"
OUT_SUM  = "data/dialog_summary.csv"

GAP_MINUTES = 45

def parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)

def normalize_text(x) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        parts = []
        for el in x:
            if isinstance(el, str):
                parts.append(el)
            elif isinstance(el, dict):
                t = el.get("text", "")
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
    return ""

def get_topic_id(m: dict):

    for k in ["topic_id", "thread_id", "message_thread_id"]:
        if k in m:
            return m.get(k)

    return None

def main():
    p = Path(SRC_JSON)
    data = json.loads(p.read_text(encoding="utf-8"))
    msgs = data.get("messages", [])

    rows = []
    for m in msgs:
        if m.get("type") != "message":
            continue
        if m.get("action"):
            continue
        text = normalize_text(m.get("text", ""))
        if not text or not str(m.get("date","")):
            continue
        rows.append({
            "id": m.get("id"),
            "date": m.get("date"),
            "dt": parse_dt(m["date"]),
            "from": m.get("from"),
            "from_id": m.get("from_id"),
            "text": text.strip(),
            "reply_to": m.get("reply_to_message_id"),
            "topic_id": get_topic_id(m),
        })

    df = pd.DataFrame(rows).sort_values("dt").reset_index(drop=True)
    print("Messages kept:", len(df))


    has_topics = df["topic_id"].notna().any()
    if has_topics:

        df["session_id"] = -1
        sid = 0
        gap = timedelta(minutes=GAP_MINUTES)

        for topic, g in df.groupby("topic_id", dropna=False):
            g = g.sort_values("dt")
            prev = None
            for idx, r in g.iterrows():
                if prev is None or (r["dt"] - prev) > gap:
                    sid += 1
                df.loc[idx, "session_id"] = sid
                prev = r["dt"]
        mode = "topic+time"
    else:

        gap = timedelta(minutes=GAP_MINUTES)
        sid = 0
        session = []
        prev = None
        session_ids = []
        for dt in df["dt"]:
            if prev is None or (dt - prev) > gap:
                sid += 1
            session_ids.append(sid)
            prev = dt
        df["session_id"] = session_ids
        mode = "time_only"

    print("Segmentation mode:", mode)
    print("Sessions:", df["session_id"].nunique())


    def first_lines(s):
        s = s.head(6).tolist()
        return " | ".join([x.replace("\n"," ")[:120] for x in s])

    summ = (
        df.groupby("session_id")
        .agg(
            start=("dt","min"),
            end=("dt","max"),
            n_messages=("id","count"),
            participants=("from", lambda x: len(set([str(i) for i in x if pd.notna(i)]))),
            preview=("text", first_lines),
        )
        .reset_index()
        .sort_values(["n_messages","start"], ascending=[False, True])
    )

    df.drop(columns=["dt"]).to_csv(OUT_CSV, index=False, encoding="utf-8")
    summ.to_csv(OUT_SUM, index=False, encoding="utf-8")

    print("Saved:", OUT_CSV)
    print("Saved:", OUT_SUM)
    print("\nTop 5 sessions by size:")
    print(summ.head(5)[["session_id","n_messages","start","preview"]].to_string(index=False))

if __name__ == "__main__":
    main()