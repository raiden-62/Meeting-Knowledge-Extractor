import time

from app.integrations.ml_adapter import extract_entities
from app.services.llm_service import extract_output
from app.core.logger import logger


def process_meeting(transcript: str):
    start_time = time.time()

    logger.info("Started transcript processing")

    # notimplemented = "123"
    # raw_data = extract_entities(notimplemented)

    extracted = extract_output(transcript)

    elapsed = time.time() - start_time

    logger.info(f"Processed meeting in {elapsed:.2f} seconds")

    return extracted