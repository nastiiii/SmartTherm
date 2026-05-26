
"""Build gold evaluation set from curated FAQ + paraphrases + edge cases."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.knowledge_base import KB_CSV

OUT = ROOT / "data" / "eval_gold.csv"


PARAPHRASES: list[tuple[str, int, str, str]] = [
    ("не подключается к wifi", 11, "wifi", "auto"),
    ("wi-fi не работает smarttherm", 11, "wifi", "auto"),
    ("как подключить котёл", 40, "installation", "auto"),
    ("как подключить котел к smarttherm", 40, "installation", "auto"),
    ("схема подключения opentherm", 40, "opentherm", "auto"),
    ("котел baxi opentherm горячая вода пропала", 21, "baxi;hot_water", "auto"),
    ("baxi гвс не греет после smarttherm", 21, "baxi", "auto"),
    ("opentherm не работает нет обмена", 20, "opentherm", "auto"),
    ("не горит значок ot на котле", 20, "opentherm", "auto"),
    ("ds18b20 несколько датчиков на одной линии", 10, "sensors;1wire", "auto"),
    ("можно ли несколько ds18b20", 10, "sensors", "auto"),
    ("есть ли интеграция с home assistant", 41, "homeassistant", "auto"),
    ("как связать smarttherm с ha через mqtt", 41, "mqtt;homeassistant", "auto"),
    ("какая версия прошивки установлена", 4, "firmware", "auto"),
    ("где посмотреть версию прошивки", 4, "firmware", "auto"),
    ("можно ли другой датчик температуры", 5, "sensors", "auto"),
    ("два контроллера на котле мешают", 2, "compatibility", "auto"),
    ("сухой контакт 220 в на термостат", 14, "safety", "auto"),
    ("блок питания usb зависает", 16, "power", "auto"),
    ("smarttherm не работает с чего начать", 7, "diagnostics", "auto"),
    ("on/off или opentherm что выбрать", 19, "opentherm", "auto"),
    ("подключить провода opentherm", 25, "opentherm;wiring", "auto"),
    ("mqtt mosquitto настройка smarttherm", 41, "mqtt", "auto"),
    ("термостат в комнате или у котла", 8, "sensors", "auto"),
    ("что приложить эксперту для помощи", 18, "support", "auto"),
    ("проблемы после обновления прошивки", 13, "firmware", "auto"),
    ("чем отличается smarttherm32", 9, "hardware", "auto"),
    ("реле на smarttherm можно", 6, "wiring", "auto"),
    ("котел поддерживает opentherm как проверить", 12, "opentherm", "auto"),
    ("датчик давления smarttherm32", 24, "pressure", "auto"),
    ("абракадабра xyz несуществующий вопрос", 0, "", "escalate"),
    ("когда лучше сажать картофель", 0, "", "escalate"),
    ("цена доставки ozon", 0, "", "escalate"),
]


def main() -> None:
    df = pd.read_csv(KB_CSV).fillna("")
    curated = df[~df.tags.str.contains("from_chat", na=False)]

    rows: list[dict] = []
    seen: set[str] = set()

    def add(q: str, fid: int, tags: str, mode: str, source: str) -> None:
        key = q.lower().strip()
        if key in seen:
            return
        seen.add(key)
        rows.append(
            {
                "query": q,
                "expected_faq_id": fid,
                "expected_tags": tags,
                "expected_mode": mode,
                "source": source,
            }
        )

    for q, fid, tags, mode in PARAPHRASES:
        add(q, fid, tags, mode, "paraphrase")

    for _, r in curated.iterrows():
        q = str(r["question"]).strip()
        add(q, int(r["id"]), str(r.get("tags") or ""), "auto", "curated_question")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "query",
                "expected_faq_id",
                "expected_tags",
                "expected_mode",
                "source",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} eval queries → {OUT}")


if __name__ == "__main__":
    main()
