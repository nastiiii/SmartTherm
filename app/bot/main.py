"""Run Telegram bot: python -m app.bot.main"""

import logging

from telegram.ext import Application, CallbackQueryHandler, CommandHandler, MessageHandler, filters

from app import db as database
from app.bot.handlers import (
    admin_command,
    callback_handler,
    cancel_command,
    handle_message,
    start_command,
)
from app.config import OLLAMA_MODEL, TELEGRAM_BOT_TOKEN, TELEGRAM_PROXY, USE_RAG_FOR_AUTO
from app.rag import check_ollama_health
from app.retrieval import get_retriever

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN in environment or .env")

    database.init_db()
    n = get_retriever().reload_index()
    logger.info("FAQ index loaded: %d cards", n)

    ollama = check_ollama_health()
    if USE_RAG_FOR_AUTO:
        if ollama.get("ok") and ollama.get("model_pulled"):
            logger.info("Ollama RAG ready: model=%s", OLLAMA_MODEL)
        else:
            logger.warning(
                "Ollama RAG enabled but not ready (%s). "
                "Answers will use FAQ text until model is pulled. See docs/SERVER_AND_OLLAMA.md",
                ollama,
            )

    builder = Application.builder().token(TELEGRAM_BOT_TOKEN)
    if TELEGRAM_PROXY:
        logger.info("Using proxy for Telegram: %s", TELEGRAM_PROXY)
        builder = builder.proxy(TELEGRAM_PROXY).get_updates_proxy(TELEGRAM_PROXY)
    app = builder.build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", start_command))
    app.add_handler(CommandHandler("cancel", cancel_command))
    app.add_handler(CommandHandler("admin", admin_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(callback_handler))

    logger.info("Bot polling started")
    app.run_polling(allowed_updates=["message", "callback_query"])


if __name__ == "__main__":
    main()
