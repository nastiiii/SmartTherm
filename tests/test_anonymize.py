import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.anonymize import anonymize


def test_phone_and_email():
    s = "Пишите на test@mail.ru или +7 999 123-45-67"
    out = anonymize(s)
    assert "<EMAIL>" in out
    assert "<PHONE>" in out


def test_telegram_user():
    out = anonymize("Спросите у @someuser123")
    assert "<USER>" in out
