import pandas as pd
from collections import Counter

SRC = "data/dialog_messages.csv"

df = pd.read_csv(SRC)
df["text"] = df["text"].astype(str)


msg_cnt = Counter(df["from"].fillna("unknown"))
avg_len = df.groupby("from")["text"].apply(lambda s: s.str.len().mean()).to_dict()

rows = []
for name, c in msg_cnt.most_common():
    if name == "unknown":
        continue
    rows.append((c, avg_len.get(name, 0), name))

rows.sort(reverse=True, key=lambda x: (x[0], x[1]))

print("Top likely experts (count, avg_len, name):")
for c, al, name in rows[:40]:
    print(f"{c:5d}  {al:6.1f}  {name}")