import os

from dotenv import load_dotenv

load_dotenv()

GIGACHAT_TOKEN = os.getenv("GIGACHAT_TOKEN")
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_TOKEN")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_THINKING = os.getenv("DEEPSEEK_THINKING", "disabled").strip().lower()
if DEEPSEEK_THINKING not in {"enabled", "disabled"}:
    DEEPSEEK_THINKING = "disabled"
DEEPSEEK_MAX_TOKENS = int(os.getenv("DEEPSEEK_MAX_TOKENS", "3500"))
DEEPSEEK_TIMEOUT_SECONDS = int(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "45"))
DEEPSEEK_MAX_RETRIES = int(os.getenv("DEEPSEEK_MAX_RETRIES", "1"))

LLM_PROVIDERS = ("gigachat", "deepseek")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gigachat").strip().lower()
if LLM_PROVIDER not in LLM_PROVIDERS:
    LLM_PROVIDER = "gigachat"
LLM_USE_LANGGRAPH = os.getenv("LLM_USE_LANGGRAPH", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

MAX_TRANSCRIPT_CHARS = int(os.getenv("MAX_TRANSCRIPT_CHARS", "100000"))
LLM_LONG_TRANSCRIPT_CHARS = int(os.getenv("LLM_LONG_TRANSCRIPT_CHARS", "30000"))
LLM_TRANSCRIPT_CONTEXT_CHARS = int(os.getenv("LLM_TRANSCRIPT_CONTEXT_CHARS", "24000"))
LLM_PARALLEL_LONG_TRANSCRIPTS = os.getenv("LLM_PARALLEL_LONG_TRANSCRIPTS", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LLM_PARALLEL_LLM_MERGE = os.getenv("LLM_PARALLEL_LLM_MERGE", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LLM_CHUNK_CHARS = int(os.getenv("LLM_CHUNK_CHARS", "12000"))
LLM_CHUNK_OVERLAP_CHARS = int(os.getenv("LLM_CHUNK_OVERLAP_CHARS", "700"))
LLM_CHUNK_MAX_WORKERS = int(os.getenv("LLM_CHUNK_MAX_WORKERS", "4"))
