import pandas as pd
import re

SRC="data/qa_domain.csv"
DST="data/qa_domain_good.csv"

BAD_ANS = re.compile(r"(?i)\b(там выше|в инструкции|ищите|ссылк|зачем-зачем|ага|ок|понятно)\b")
def good(a: str) -> bool:
    a = (a or "").strip()
    if len(a) < 60:
        return False
    if BAD_ANS.search(a):
        return False
    return True

df = pd.read_csv(SRC)
df = df[df["a_text"].astype(str).map(good)].copy()
df.to_csv(DST, index=False, encoding="utf-8")
print("Saved:", DST)
print("Good pairs:", len(df))
print(df.head(5)[["q_text","a_text"]].to_string(index=False))