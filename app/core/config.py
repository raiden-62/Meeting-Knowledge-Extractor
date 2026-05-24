import os

from dotenv import load_dotenv

load_dotenv()

GIGACHAT_TOKEN = os.getenv("GIGACHAT_TOKEN")
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_TOKEN")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

LLM_PROVIDERS = ("gigachat", "deepseek")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gigachat").strip().lower()
if LLM_PROVIDER not in LLM_PROVIDERS:
    LLM_PROVIDER = "gigachat"

MAX_TRANSCRIPT_CHARS = int(os.getenv("MAX_TRANSCRIPT_CHARS", "100000"))
LLM_LONG_TRANSCRIPT_CHARS = int(os.getenv("LLM_LONG_TRANSCRIPT_CHARS", "30000"))
LLM_TRANSCRIPT_CONTEXT_CHARS = int(os.getenv("LLM_TRANSCRIPT_CONTEXT_CHARS", "24000"))
