"""Relevance gate before AUTO answers."""

from app.relevance import heuristic_relevant, is_answer_relevant


def test_forecast_query_rejects_unrelated_esp32_faq():
    query = "Подскажите, какой forecast погоды можно использовать?"
    faq_q = "Как связаны датчики температуры и температура котла?"
    faq_a = (
        "Датчики температуры в помещениях. Котел сообщает температуру по OpenTherm. "
        "Опрос датчиков раз в 1-2 минуты."
    )
    assert not heuristic_relevant(query, faq_q, faq_a)


def test_wifi_query_accepts_wifi_faq():
    query = "не подключается к wifi"
    faq_q = "Не подключается к Wi-Fi — что проверить?"
    faq_a = "Проверьте сеть 2.4 ГГц, WPA2, DHCP на роутере."
    assert heuristic_relevant(query, faq_q, faq_a)


def test_thermostat_placement_short_query():
    query = "Где ставить термостат?"
    faq_q = "Где должен стоять термостат/датчик: у котла или в комнате?"
    faq_a = "Комнатный термостат должен измерять температуру в помещении, а не рядом с котлом."
    assert heuristic_relevant(query, faq_q, faq_a)


def test_boiler_connect_curated():
    query = "как подключить котёл"
    faq_q = "Как подключить котёл к SmartTherm (общая схема)?"
    faq_a = "Отключите питание. Подключите OpenTherm или on/off по инструкции."
    assert heuristic_relevant(query, faq_q, faq_a)


def test_is_answer_relevant_heuristic_only(monkeypatch):
    import app.relevance as rel

    monkeypatch.setattr(rel, "USE_RELEVANCE_JUDGE", False)
    query = "не подключается к wifi"
    faq_q = "Wi-Fi не подключается"
    faq_a = "Сеть 2.4 ГГц WPA2."
    assert rel.is_answer_relevant(query, faq_q, faq_a)
