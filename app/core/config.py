import os

from dotenv import load_dotenv

load_dotenv()

GIGACHAT_TOKEN = os.getenv("GIGACHAT_TOKEN")
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat")
MAX_TRANSCRIPT_CHARS = 20_000
