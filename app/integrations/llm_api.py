import time

from gigachat import GigaChat

from app.core.config import GIGACHAT_MODEL, GIGACHAT_TOKEN
from app.core.logger import logger

_gigachat_client: GigaChat | None = None


def get_gigachat_client() -> GigaChat:
    global _gigachat_client

    if not GIGACHAT_TOKEN:
        raise RuntimeError("GIGACHAT_TOKEN is not configured")

    if _gigachat_client is None:
        _gigachat_client = GigaChat(
            credentials=GIGACHAT_TOKEN,
            verify_ssl_certs=False,
            scope="GIGACHAT_API_PERS",
            model=GIGACHAT_MODEL,
        )

    return _gigachat_client


def gigachat_request(
    request: str,
    max_retries: int = 2,
    backoff_seconds: float = 0.5,
) -> dict[str, str]:
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = get_gigachat_client().chat(request)
            content = response.choices[0].message.content
            return {"answer": content, "model_name": GIGACHAT_MODEL}
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                logger.warning(
                    "GigaChat request failed: %s. Retrying (%s/%s)",
                    exc,
                    attempt,
                    max_retries,
                )
                time.sleep(backoff_seconds * attempt)

    raise RuntimeError(f"GigaChat request failed: {last_error}")
