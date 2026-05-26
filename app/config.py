import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


load_dotenv(ROOT / ".env")


MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
HIGH_TH = float(os.getenv("RETRIEVAL_HIGH_TH", "0.52"))
MID_TH = float(os.getenv("RETRIEVAL_MID_TH", "0.42"))

LEXICAL_AUTO_OVERLAP = float(os.getenv("RETRIEVAL_LEXICAL_AUTO", "0.38"))
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "3"))
USE_EMBEDDING_CACHE = os.getenv("USE_EMBEDDING_CACHE", "1") == "1"


USE_HYBRID_RETRIEVAL = os.getenv("USE_HYBRID_RETRIEVAL", "1") == "1"
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.30"))


DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'data' / 'assistant.db'}")


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
OPERATOR_CHAT_ID = os.getenv("OPERATOR_CHAT_ID", "")
ALLOWED_CHAT_IDS = os.getenv("ALLOWED_CHAT_IDS", "")
ADMIN_USER_IDS = os.getenv("ADMIN_USER_IDS", "")

TELEGRAM_PROXY = os.getenv("TELEGRAM_PROXY", "") or os.getenv("HTTPS_PROXY", "")


ADMIN_API_HOST = os.getenv("ADMIN_API_HOST", "0.0.0.0")
ADMIN_API_PORT = int(os.getenv("ADMIN_API_PORT", "8080"))
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.05"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "700"))
RAG_TOP_CHUNKS = int(os.getenv("RAG_TOP_CHUNKS", "3"))
_ollama_configured = bool(OLLAMA_BASE_URL.strip())


def _rag_flag(name: str, default_if_unset: str) -> bool:
    """If env var unset: use default_if_unset when Ollama URL is configured."""
    val = os.getenv(name)
    if val is None:
        return _ollama_configured and default_if_unset == "1"
    return val.strip() == "1"


USE_RAG_FOR_AUTO = _rag_flag("USE_RAG_FOR_AUTO", "1")
USE_RAG_FOR_CLARIFY = _rag_flag("USE_RAG_FOR_CLARIFY", "0")
USE_RAG_FOR_ESCALATE = _rag_flag("USE_RAG_FOR_ESCALATE", "0")

RAG_FALLBACK_TO_FAQ = os.getenv("RAG_FALLBACK_TO_FAQ", "1") == "1"

USE_RELEVANCE_JUDGE = _rag_flag("USE_RELEVANCE_JUDGE", "1")

RAG_SHOW_FOOTER = os.getenv("RAG_SHOW_FOOTER", "0") == "1"


USE_CHAT_TIER = _rag_flag("USE_CHAT_TIER", "1")

USE_GENERAL_LLM_FALLBACK = _rag_flag("USE_GENERAL_LLM_FALLBACK", "1")

CHAT_TIER_MIN_SIM = float(os.getenv("CHAT_TIER_MIN_SIM", "0.45"))


LOG_ALL_MESSAGES = os.getenv("LOG_ALL_MESSAGES", "1") == "1"
ESCALATE_ON_NOT_HELPFUL = os.getenv("ESCALATE_ON_NOT_HELPFUL", "1") == "1"
