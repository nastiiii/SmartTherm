import pandas as pd
import re
from datetime import datetime, timedelta

SRC = "data/dialog_messages.csv"
EXPERTS = "data/experts.txt"
OUT = "data/qa_from_sessions.csv"

QMARK = re.compile(r"\?")
QWORDS = re.compile(r"(?i)\b(как|почему|зачем|что|куда|можно ли|подскажите|не работает|ошибка|помогите)\b")
BADQ = re.compile(r"(?i)\b(спасибо|ок|ага|понятно|привет|добрый)\b")

def is_question(t: str) -> bool:
    t = (t or "").strip()
    if len(t) < 15: 
        return False
    if BADQ.search(t):
        return False
    return bool(QMARK.search(t) or QWORDS.search(t))

def parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)

def main():
    df = pd.read_csv(SRC)
    df["text"] = df["text"].astype(str)
    df["dt"] = df["date"].apply(parse_dt)

    experts = set()
    with open(EXPERTS, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                experts.add(name)

    rows = []
    window = timedelta(minutes=30)

    for sid, g in df.groupby("session_id"):
        g = g.sort_values("dt").reset_index(drop=True)


        for i in range(len(g)):
            author = str(g.loc[i, "from"])
            text = g.loc[i, "text"]

            if author in experts:
                continue
            if not is_question(text):
                continue

            q_dt = g.loc[i, "dt"]
            q_text = text


            a_text = None
            a_author = None
            for j in range(i+1, len(g)):
                dtj = g.loc[j, "dt"]
                if dtj - q_dt > window:
                    break
                author_j = str(g.loc[j, "from"])
                text_j = g.loc[j, "text"]

                if author_j in experts and len(text_j.strip()) >= 20:
                    a_text = text_j
                    a_author = author_j
                    break

            if a_text:
                rows.append({
                    "session_id": sid,
                    "q_date": g.loc[i, "date"],
                    "q_from": author,
                    "q_text": q_text,
                    "a_date": g.loc[j, "date"],
                    "a_from": a_author,
                    "a_text": a_text
                })

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print("Saved:", OUT)
    print("Pairs:", len(out))
    print("Example:")
    if len(out) > 0:
        print(out.head(5)[["q_text","a_text"]].to_string(index=False))

if __name__ == "__main__":
    main()