from app.integrations.ml_adapter import extract_entities
from app.services.llm_service import extract_output


def process_meeting(transcript: str):
    # notimplemented = "123"
    # raw_data = extract_entities(notimplemented)

    extracted = extract_output(transcript)

    return extracted