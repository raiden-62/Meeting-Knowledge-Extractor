import time

from app.services.meeting_pipeline import process_meeting

def run_lite_pipeline(transcript: str):
    start = time.time()

    result = process_meeting(transcript)

    elapsed = time.time() - start

    return result, elapsed