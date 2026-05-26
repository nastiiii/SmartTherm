
"""Keep only logically consistent FAQ pairs; preserve hand-curated cards."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.faq_logic import filter_faq_records
from app.knowledge_base import KB_CSV

OUT = KB_CSV

EXTRA_CURATED = [
    (
        "Как подключить котёл к SmartTherm (общая схема)?",
        "Общий порядок подключения:\n\n"
        "1) Уточните модель котла и тип управления: OpenTherm (цифровая шина) "
        "или on/off (сухой контакт / комнатный термостат).\n"
        "2) Отключите питание котла. Подключите клеммы OT или клеммы комнатного "
        "термостата по инструкции котла и SmartTherm (витая пара, без 220 В на OT).\n"
        "3) В настройках котла включите режим внешнего термостата / OpenTherm.\n"
        "4) В SmartTherm выберите тот же режим (OT или on/off), укажите модель котла.\n"
        "5) Включите питание, проверьте обмен (температура, статус OT, нет ошибок).\n\n"
        "Важно: на время настройки оставьте один активный контроллер на шине. "
        "Для разбора пришлите модель котла и фото клемм.",
        "wiring;opentherm;installation;curated",
    ),
    (
        "Есть ли прямая интеграция SmartTherm с Home Assistant?",
        "Прямой «официальной» интеграции HA в виде одного плагина обычно нет.\n\n"
        "Типовые варианты:\n"
        "1) MQTT — если прошивка публикует топики, подключите MQTT-интеграцию в Home Assistant.\n"
        "2) ESPHome — у части пользователей связка SmartTherm ↔ HA через ESPHome "
        "(зависит от прошивки и конфигурации).\n"
        "3) Альтернативные прошивки — расширенные сценарии через MQTT/ESPHome.\n\n"
        "Для настройки нужны: адрес MQTT-брокера, учётные данные, список топиков "
        "или конфиг ESPHome. Уточните версию прошивки SmartTherm.",
        "homeassistant;mqtt;integration;curated",
    ),
]


def main() -> None:
    df = pd.read_csv(OUT).fillna("")
    records = df.to_dict(orient="records")
    print(f"Auditing {len(records)} FAQ cards (logic + Q↔A similarity)…")

    kept, stats = filter_faq_records(records)

    existing_q = {str(x["question"]).lower()[:60] for x in kept}
    next_id = max((int(x["id"]) for x in kept), default=0) + 1
    for q, a, tags in EXTRA_CURATED:
        if q.lower()[:60] in existing_q:
            continue
        kept.append({"id": next_id, "question": q, "answer": a, "tags": tags})
        next_id += 1

    for i, row in enumerate(kept, 1):
        row["id"] = i

    out = pd.DataFrame(kept)[["id", "question", "answer", "tags"]]
    out.to_csv(OUT, index=False, encoding="utf-8")

    print(f"FAQ: {len(df)} -> {len(out)} (removed {len(df) - len(out)})")
    for k, v in sorted(stats.items()):
        if k not in ("kept", "removed") and v:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
