import time

from sqlalchemy.orm import Session

from app.db import models
from app.services.agents import ProjectMemoryAgent, TaskLifecycleAgent
from app.services.meeting_pipeline import process_meeting


def run_extraction(
    db: Session,
    transcript: models.Transcript,
    provider: str | None = None,
) -> models.ExtractionRun:
    start = time.time()
    memory_agent = ProjectMemoryAgent()
    lifecycle_agent = TaskLifecycleAgent()
    memory = memory_agent.build(db, transcript)
    memory_context = memory_agent.render(memory)
    raw_output = process_meeting(
        transcript.content,
        provider=provider,
        memory_context=memory_context,
    )
    elapsed = time.time() - start

    run = models.ExtractionRun(
        transcript_id=transcript.id,
        provider=raw_output.get("source", "gigachat"),
        model_name=raw_output.get("model_name") or raw_output.get("source", "gigachat"),
        status="completed",
        response_time_seconds=elapsed,
        raw_response=raw_output,
    )
    db.add(run)
    db.flush()

    lifecycle_agent.apply(
        db,
        transcript,
        run,
        raw_output,
        infer_updates=False,
    )
    memory_agent.update_summary(db, transcript, raw_output)
    run.raw_response = raw_output

    return run
