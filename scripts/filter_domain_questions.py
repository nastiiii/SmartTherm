import pandas as pd
import re

SRC = "data/questions_only.csv"
DST = "data/domain_questions.csv"


POS = re.compile(
    r"(?i)\b("
    r"smarttherm|смарттерм|контроллер|кот[её]л|горелк|отоплен|гвс|радиатор|"
    r"датчик|сенсор|температур|t1|t2|ntc|ds18b20|1-wire|onewire|"
    r"wifi|wi-fi|роутер|точк[аи] доступ|сеть|ssid|парол|"
    r"opentherm|open therm|ot\b|"
    r"esp32|esp|прошивк|firmware|update|обновлен|верс(ия|ии)|build|"
    r"пин|gpio|клемм|подключ|схем|питани|5v|12v|220|"
    r"модуляц|модуляция|насос|клапан|реле"
    r")\b"
)


NEG = re.compile(
    r"(?i)\b("
    r"ozon|озон|wildberries|wb\b|вайлдберриз|"
    r"заказ|доставк|курьер|возврат|перезаказ|"
    r"этикетк|трек|упаковк|продавц|склад|"
    r"москва|питер|адрес|"
    r"цена|стоимост|оплат"
    r")\b"
)

def main():
    df = pd.read_csv(SRC)
    t = df["text_clean"].astype(str)

    df["pos"] = t.str.contains(POS)
    df["neg"] = t.str.contains(NEG)

    out = df[df["pos"] & (~df["neg"])].copy()
    out.to_csv(DST, index=False, encoding="utf-8")

    print("Total questions:", len(df))
    print("Domain questions:", len(out))
    print("\nExamples (domain):")
    print(out["text_clean"].head(15).to_string(index=False))

if __name__ == "__main__":
    main()