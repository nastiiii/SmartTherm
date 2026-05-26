import pandas as pd
import re

SRC = "data/messages_extracted.csv"
DST = "data/questions_only.csv"


QMARK = re.compile(r"\?")
QWORDS = re.compile(r"(?i)\b(как|почему|зачем|что|куда|откуда|когда|можно ли|подскажите|не работает|ошибка|помогите)\b")


BAD = re.compile(r"(?i)\b(спасибо|пожалуйста|ок|ага|понятно|ясно|сделал|получилось|не получилось|добрый|привет|ха-ха|лол)\b")

def is_question(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 15:
        return False
    if BAD.search(t):
        return False

    if QMARK.search(t) or QWORDS.search(t):
        return True
    return False

def main():
    df = pd.read_csv(SRC)
    df["text_clean"] = df["text_clean"].astype(str)
    df = df[df["text_clean"].map(is_question)].copy()


    df = df[df["text_clean"].str.len() <= 400]

    df.to_csv(DST, index=False, encoding="utf-8")
    print("Saved:", DST)
    print("Questions:", len(df))
    print("Examples:")
    print(df["text_clean"].head(10).to_string(index=False))

if __name__ == "__main__":
    main()