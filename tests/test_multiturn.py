"""Multi-turn dialog history: short follow-up merges with previous query."""

from app.answer_service import _merge_with_history


def test_merge_short_followup_with_previous_query():
    history = [
        {
            "user_query": "Не подключается к Wi-Fi",
            "bot_answer": "Проверьте 2.4 ГГц и WPA2.",
            "tier": "curated_faq",
        }
    ]
    result = _merge_with_history("а если ESP32?", history)
    assert "Не подключается к Wi-Fi" in result
    assert "ESP32" in result


def test_followup_boiler_model_is_merged():
    history = [
        {
            "user_query": "Почему пропала горячая вода?",
            "bot_answer": "Проверьте режим ГВС и CH2.",
            "tier": "chat_kb",
        }
    ]
    result = _merge_with_history("Baxi Eco Four", history)
    assert "пропала горячая вода" in result.lower()
    assert "Baxi" in result


def test_long_independent_query_not_merged():
    history = [
        {
            "user_query": "Не подключается к Wi-Fi",
            "bot_answer": "Проверьте 2.4 ГГц",
            "tier": "curated_faq",
        }
    ]
    new_q = "Как настроить интеграцию SmartTherm с Home Assistant через MQTT?"
    result = _merge_with_history(new_q, history)
    assert result == new_q


def test_merge_without_history():
    assert _merge_with_history("короткий вопрос", None) == "короткий вопрос"
    assert _merge_with_history("короткий вопрос", []) == "короткий вопрос"


def test_db_dialog_history_roundtrip(tmp_path, monkeypatch):
    import app.db as database
    from app.config import ROOT

    db_path = tmp_path / "test.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    database.init_db()
    chat_id = 999
    for i in range(8):
        database.push_dialog_turn(
            chat_id, 1, f"q{i}", f"a{i}", tier="curated_faq", faq_id=i
        )
    hist = database.get_dialog_history(chat_id, limit=3)
    assert len(hist) == 3

    assert [h["user_query"] for h in hist] == ["q5", "q6", "q7"]
    database.clear_dialog_history(chat_id)
    assert database.get_dialog_history(chat_id) == []
