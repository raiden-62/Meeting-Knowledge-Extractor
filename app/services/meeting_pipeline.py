import time

from app.services.llm_service import extract_output
from app.core.logger import logger


def process_meeting(transcript: str):
    start_time = time.time()
    logger.info("Started transcript processing")

    extracted = extract_output(transcript)

    elapsed = time.time() - start_time
    metrics = extracted.setdefault("metrics", {})
    if isinstance(metrics, dict):
        metrics["response_time_seconds"] = round(elapsed, 3)

    logger.info(f"Processed meeting in {elapsed:.2f} seconds")

    return extracted
