import time

from app.services.llm_service import extract_output
from app.core.logger import logger


def process_meeting(
    transcript: str,
    provider: str | None = None,
    memory_context: str | None = None,
):
    start_time = time.time()
    logger.info("Started transcript processing")

    if provider is None and memory_context is None:
        extracted = extract_output(transcript)
    else:
        extracted = extract_output(
            transcript,
            provider=provider,
            memory_context=memory_context,
        )

    elapsed = time.time() - start_time
    metrics = extracted.setdefault("metrics", {})
    if isinstance(metrics, dict):
        metrics["response_time_seconds"] = round(elapsed, 3)

    logger.info(f"Processed meeting in {elapsed:.2f} seconds")

    return extracted
