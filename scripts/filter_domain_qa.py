import pandas as pd
import re

SRC = "data/qa_from_sessions.csv"
DST = "data/qa_domain.csv"

POS = re.compile(
    r"(?i)\b("
    r"smarttherm|смарттерм|контроллер|кот[её]л|отоплен|гвс|радиатор|"
    r"датчик|сенсор|температур|t1|t2|ntc|ds18b20|1-wire|onewire|"
    r"wifi|wi-fi|роутер|сеть|ssid|парол|"
    r"opentherm|open therm|ot\b|"
    r"esp32|esp|прошивк|firmware|update|обновлен|верс(ия|ии)|build|"
    r"пин|gpio|клемм|подключ|схем|питани|5v|12v|220|"
    r"модуляц|насос|клапан|реле|mqtt|алиса|home assistant|knx"
    r")\b"
)

NEG = re.compile(
    r"(?i)\b("
    r"ozon|озон|wildberries|wb\b|вайлдберриз|"
    r"заказ|доставк|курьер|возврат|перезаказ|этикетк|трек|упаковк|"
    r"продавц|склад|оплат|банк"
    r")\b"
)

def main():
    df = pd.read_csv(SRC)
    qt = df["q_text"].astype(str)
    at = df["a_text"].astype(str)

    pos = qt.str.contains(POS) | at.str.contains(POS)
    neg = qt.str.contains(NEG) | at.str.contains(NEG)

    out = df[pos & (~neg)].copy()
    out.to_csv(DST, index=False, encoding="utf-8")

    print("Total pairs:", len(df))
    print("Domain pairs:", len(out))
    print("\nExamples:")
    print(out.head(10)[["q_text","a_text"]].to_string(index=False))

if __name__ == "__main__":
    main()