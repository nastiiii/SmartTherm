"""RAG output filters."""

from app.text_quality import rag_faithful_to_source, rag_output_ok, source_overlap


def test_rag_output_ok_rejects_english_leak():
    bad = "Please check the device and reset never step."
    assert not rag_output_ok(bad)


def test_rag_output_ok_accepts_russian():
    good = (
        "1) Проверьте Wi-Fi: сеть 2,4 ГГц, WPA2.\n"
        "2) Убедитесь, что DHCP включён на роутере.\n"
        "3) Перезагрузите SmartTherm."
    )
    assert rag_output_ok(good)


def test_rag_faithful_to_source():
    source = (
        "1) Отключите питание котла. 2) Подключите клеммы OpenTherm. "
        "3) Включите режим OT в настройках котла."
    )
    faithful = (
        "1) Отключите питание котла.\n"
        "2) Подключите клеммы OpenTherm по инструкции.\n"
        "3) Включите режим OT в настройках."
    )
    hallucination = "Подключите 220 В к клеммам OT и обновите прошивку до версии 9.9."
    assert rag_faithful_to_source(faithful, source)
    assert not rag_faithful_to_source(hallucination, source)


def test_source_overlap_is_a_proportion():
    """Faithful answer должен иметь высокое перекрытие, галлюцинация — низкое."""
    source = (
        "Проверьте сеть 2.4 ГГц, поддерживаемый протокол шифрования WPA2-PSK, "
        "правильность SSID и пароля, уровень сигнала роутера. Перезагрузите устройство."
    )
    faithful = (
        "1) Используйте сеть 2.4 ГГц.\n"
        "2) Уточните SSID и пароль, шифрование WPA2-PSK.\n"
        "3) Проверьте уровень сигнала от роутера.\n"
        "4) Перезагрузите устройство."
    )
    hallucination = (
        "Откройте крышку котла, замените керамический предохранитель и обновите "
        "прошивку через USB на версию 12.4."
    )
    assert source_overlap(faithful, source) >= 0.5
    assert source_overlap(hallucination, source) < 0.12

    assert source_overlap("", source) == 0.0
    assert source_overlap(faithful, "") == 0.0
