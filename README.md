# SmartTherm — цифровой ассистент техподдержки

Telegram-бот для техподдержки контроллера котла SmartTherm. Отвечает на вопросы пользователей через четырёхуровневую систему: курируемая база FAQ, ответы по истории чата, общий LLM с дисклеймером и эскалация к оператору.

Полный текст курсовой работы: [`docs/Coursework_SmartTherm_2026.pdf`](docs/Coursework_SmartTherm_2026.pdf).

## Архитектура

```
Telegram → handlers → answer_service
                          │
                          ├─ Tier 1: FAQ (гибридный поиск BM25 + embeddings) → SQLite
                          ├─ Tier 2: ответы по истории чата + LLM-judge
                          ├─ Tier 3: общий ответ Ollama (qwen3:14b) + дисклеймер
                          └─ Tier 4: эскалация → OPERATOR_CHAT_ID

Admin API + статическая wiki ← faq_seed.csv
                                          │
                  Внешний Ollama-сервер ◄──┘
                  http://smarttherm.ru:11434
```

Компоненты:

- `app/bot/` — Telegram-бот (python-telegram-bot).
- `app/admin/` — FastAPI-админка и wiki.
- `app/retrieval.py`, `app/rag.py`, `app/relevance.py` — поиск, RAG, LLM-judge.
- `data/faq_seed.csv` — единый источник базы знаний.

## Конфигурация Ollama

Ollama развёрнута на отдельном сервере с GPU (`OLLAMA_BASE_URL=http://smarttherm.ru:11434`). Бот ходит туда по HTTP, локально Ollama поднимать не нужно. Модель по умолчанию — `qwen3:14b`; на сервере также доступны `qwen3.5:9b`, `gemma2:stable`, `llama3.2` и др.

Для локальной разработки без внешнего сервера можно поднять Ollama в Docker:

```bash
docker compose --profile local-ollama up -d ollama
docker compose exec ollama ollama pull qwen3:14b
# и в .env: OLLAMA_BASE_URL=http://localhost:11434
```

## Запуск локально

```bash
make install                      # зависимости
cp .env.example .env              # TELEGRAM_BOT_TOKEN, OPERATOR_CHAT_ID, ADMIN_API_KEY
make faq-audit                    # очистка FAQ + индекс + wiki
make bot                          # запустить бота
make admin                        # админка: http://localhost:8088/
```

## Деплой на сервер

```bash
ssh user@host
git clone <repo> smarttherm && cd smarttherm
cp .env.example .env              # заполнить токены и ADMIN_API_PORT
docker compose up -d --build      # bot + admin (Ollama берётся внешняя)
docker compose logs -f bot
```

Порт админки задаётся в `.env` (`ADMIN_API_PORT`, по умолчанию 8088).

## Тесты и оценка

```bash
make test                         # unit-тесты
make eval                         # метрики retrieval → data/eval_report.md
make eval-hallucinations          # метрика галлюцинаций
```
