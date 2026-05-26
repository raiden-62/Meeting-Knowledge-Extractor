from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db import models
from app.db.database import get_db
from app.schemas.schemas import (
    DecisionCreate,
    DecisionRead,
    PersonCreate,
    PersonRead,
    PersonUpdate,
    ProjectCreate,
    ProjectRead,
    RunRead,
    TaskCreate,
    TaskRead,
    TaskUpdate,
    TranscriptRead,
)
from app.services.extraction_service import run_extraction
from app.services.llm_service import LLMProviderError
from app.services.project_validation import (
    clean_required_text,
    read_transcript_upload,
    validate_priority,
    validate_provider,
    validate_status,
    validate_transcript_content,
    parse_submitted_meeting_date,
)

router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.get("", response_model=list[ProjectRead])
def list_projects(db: Session = Depends(get_db)):
    return db.query(models.Project).order_by(models.Project.created_at.desc()).all()


@router.post("", response_model=ProjectRead)
def create_project(payload: ProjectCreate, db: Session = Depends(get_db)):
    project = models.Project(name=payload.name, description=payload.description)
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


@router.get("/{project_id}", response_model=ProjectRead)
def get_project(project_id: int, db: Session = Depends(get_db)):
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.post("/{project_id}/transcripts", response_model=TranscriptRead)
def add_transcript(
    project_id: int,
    transcript_text: str | None = Form(None),
    meeting_date: str | None = Form(None),
    file: UploadFile | None = File(None),
    db: Session = Depends(get_db),
):
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    content = None
    filename = None

    if file is not None:
        content = read_transcript_upload(file)
        filename = file.filename

    if transcript_text:
        content = transcript_text

    if content is None:
        raise HTTPException(status_code=400, detail="Transcript content is required")

    content = validate_transcript_content(content)
    parsed_meeting_date = parse_submitted_meeting_date(meeting_date, content)

    transcript = models.Transcript(
        project_id=project_id,
        content=content,
        source_filename=filename,
        meeting_date=parsed_meeting_date,
    )
    db.add(transcript)
    db.commit()
    db.refresh(transcript)
    return transcript


@router.get("/{project_id}/transcripts", response_model=list[TranscriptRead])
def list_transcripts(project_id: int, db: Session = Depends(get_db)):
    return (
        db.query(models.Transcript)
        .filter(models.Transcript.project_id == project_id)
        .order_by(models.Transcript.created_at.desc())
        .all()
    )


@router.post("/{project_id}/extract", response_model=RunRead)
def extract_for_transcript(
    project_id: int,
    transcript_id: int = Form(...),
    provider: str | None = Form(None),
    db: Session = Depends(get_db),
):
    transcript = (
        db.query(models.Transcript)
        .filter(
            models.Transcript.id == transcript_id,
            models.Transcript.project_id == project_id,
        )
        .first()
    )
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found")

    try:
        run = run_extraction(db, transcript, provider=validate_provider(provider))
    except LLMProviderError as exc:
        raise HTTPException(
            status_code=502,
            detail={"provider": exc.provider, "reason": exc.reason},
        ) from exc
    db.commit()
    db.refresh(run)
    return run


@router.get("/{project_id}/people", response_model=list[PersonRead])
def list_people(project_id: int, db: Session = Depends(get_db)):
    return (
        db.query(models.Person)
        .filter(models.Person.project_id == project_id)
        .order_by(models.Person.name.asc())
        .all()
    )


@router.post("/{project_id}/people", response_model=PersonRead)
def create_person(project_id: int, payload: PersonCreate, db: Session = Depends(get_db)):
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    person = models.Person(project_id=project_id, name=payload.name, role=payload.role)
    db.add(person)
    db.commit()
    db.refresh(person)
    return person


@router.patch("/{project_id}/people/{person_id}", response_model=PersonRead)
def update_person(
    project_id: int,
    person_id: int,
    payload: PersonUpdate,
    db: Session = Depends(get_db),
):
    person = (
        db.query(models.Person)
        .filter(models.Person.id == person_id, models.Person.project_id == project_id)
        .first()
    )
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    if payload.name is not None:
        person.name = payload.name
    if payload.role is not None:
        person.role = payload.role

    db.commit()
    db.refresh(person)
    return person


@router.get("/{project_id}/tasks", response_model=list[TaskRead])
def list_tasks(project_id: int, db: Session = Depends(get_db)):
    return (
        db.query(models.Task)
        .filter(models.Task.project_id == project_id)
        .order_by(models.Task.created_at.desc())
        .all()
    )


@router.post("/{project_id}/tasks", response_model=TaskRead)
def create_task(project_id: int, payload: TaskCreate, db: Session = Depends(get_db)):
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if payload.person_id:
        person = (
            db.query(models.Person)
            .filter(
                models.Person.id == payload.person_id,
                models.Person.project_id == project_id,
            )
            .first()
        )
        if not person:
            raise HTTPException(status_code=400, detail="Assignee not found")

    task = models.Task(
        project_id=project_id,
        person_id=payload.person_id,
        description=clean_required_text(payload.description, "Task description"),
        status=validate_status(payload.status),
        priority=validate_priority(payload.priority),
        due_date=payload.due_date,
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


@router.patch("/{project_id}/tasks/{task_id}", response_model=TaskRead)
def update_task(
    project_id: int,
    task_id: int,
    payload: TaskUpdate,
    db: Session = Depends(get_db),
):
    task = (
        db.query(models.Task)
        .filter(models.Task.id == task_id, models.Task.project_id == project_id)
        .first()
    )
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    update_data = payload.model_dump(exclude_unset=True)

    if "description" in update_data:
        task.description = clean_required_text(update_data["description"], "Task description")
    if "status" in update_data:
        task.status = validate_status(update_data["status"])
    if "priority" in update_data:
        task.priority = validate_priority(update_data["priority"])
    if "due_date" in update_data:
        task.due_date = update_data["due_date"]
    if "person_id" in update_data:
        if update_data["person_id"]:
            person = (
                db.query(models.Person)
                .filter(
                    models.Person.id == update_data["person_id"],
                    models.Person.project_id == project_id,
                )
                .first()
            )
            if not person:
                raise HTTPException(status_code=400, detail="Assignee not found")
        task.person_id = update_data["person_id"]

    db.commit()
    db.refresh(task)
    return task


@router.delete("/{project_id}/tasks/{task_id}")
def delete_task(project_id: int, task_id: int, db: Session = Depends(get_db)):
    task = (
        db.query(models.Task)
        .filter(models.Task.id == task_id, models.Task.project_id == project_id)
        .first()
    )
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    db.delete(task)
    db.commit()
    return {"status": "deleted", "id": task_id}


@router.get("/{project_id}/decisions", response_model=list[DecisionRead])
def list_decisions(project_id: int, db: Session = Depends(get_db)):
    return (
        db.query(models.Decision)
        .filter(models.Decision.project_id == project_id)
        .order_by(models.Decision.created_at.desc())
        .all()
    )


@router.post("/{project_id}/decisions", response_model=DecisionRead)
def create_decision(
    project_id: int,
    payload: DecisionCreate,
    db: Session = Depends(get_db),
):
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    decision = models.Decision(project_id=project_id, description=payload.description)
    db.add(decision)
    db.commit()
    db.refresh(decision)
    return decision


@router.get("/{project_id}/runs", response_model=list[RunRead])
def list_runs(project_id: int, db: Session = Depends(get_db)):
    return (
        db.query(models.ExtractionRun)
        .join(models.Transcript)
        .filter(models.Transcript.project_id == project_id)
        .order_by(models.ExtractionRun.created_at.desc())
        .all()
    )
