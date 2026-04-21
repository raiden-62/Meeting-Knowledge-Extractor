from app.integrations.ml_adapter import extract_entities
from app.services.llm_service import format_output


def process_meeting(transcript: str):
    # Step 1: ML extraction (teammate logic later)
    raw_data = extract_entities(transcript)

    # Step 2: LLM formatting (you)
    formatted = format_output(raw_data)

    return formatted