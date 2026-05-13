import time

from sqlalchemy.orm import Session

from app.db import models
from app.services.meeting_pipeline import process_meeting


def get_or_create_person(db: Session, project_id: int, name: str) -> models.Person:
    person = (
        db.query(models.Person)
        .filter(models.Person.project_id == project_id, models.Person.name == name)
        .first()
    )
    if person:
        return person

    person = models.Person(project_id=project_id, name=name)
    db.add(person)
    db.flush()
    return person


def run_extraction(db: Session, transcript: models.Transcript) -> models.ExtractionRun:
    start = time.time()
    raw_output = process_meeting(transcript.content)
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

    decisions = raw_output.get("decisions", []) if isinstance(raw_output, dict) else []
    for decision in decisions:
        if not str(decision).strip():
            continue
        db.add(
            models.Decision(
                project_id=transcript.project_id,
                source_run_id=run.id,
                description=str(decision).strip(),
            )
        )

    tasks = raw_output.get("tasks", []) if isinstance(raw_output, dict) else []
    if not isinstance(tasks, list) or not tasks:
        people_map = raw_output.get("people", {}) if isinstance(raw_output, dict) else {}
        tasks = []
        if isinstance(people_map, dict):
            for person_name, items in people_map.items():
                for task in items or []:
                    tasks.append({"description": task, "assignee": person_name})

    for item in tasks:
        if isinstance(item, dict):
            description = str(item.get("description") or item.get("task") or "").strip()
            assignee = str(item.get("assignee") or "").strip()
            status = str(item.get("status") or "todo").strip() or "todo"
            priority = str(item.get("priority") or "medium").strip() or "medium"
        else:
            description = str(item).strip()
            assignee = ""
            status = "todo"
            priority = "medium"

        if not description:
            continue

        person_id = None
        if assignee:
            person_id = get_or_create_person(db, transcript.project_id, assignee).id

        db.add(
            models.Task(
                project_id=transcript.project_id,
                person_id=person_id,
                source_run_id=run.id,
                description=description,
                status=status,
                priority=priority,
            )
        )

    return run
