from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.core.config import MAX_TRANSCRIPT_CHARS
from app.db import models
from app.db.database import get_db
from app.services.extraction_service import run_extraction

router = APIRouter(tags=["ui"])

templates = Jinja2Templates(directory="app/frontend/templates")
templates.env.globals["max_transcript_chars"] = MAX_TRANSCRIPT_CHARS


def parse_due_date(value: str | None):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid due_date") from exc


def validate_transcript_content(content: str) -> str:
    cleaned = content.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Transcript content is required")
    if len(cleaned) > MAX_TRANSCRIPT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Transcript must be no longer than {MAX_TRANSCRIPT_CHARS} chars",
        )
    return cleaned


@router.get("/", response_class=HTMLResponse)
def home():
    return RedirectResponse(url="/projects", status_code=302)


@router.get("/projects", response_class=HTMLResponse)
def projects_page(request: Request, db: Session = Depends(get_db)):
    projects = db.query(models.Project).order_by(models.Project.created_at.desc()).all()
    return templates.TemplateResponse(
        request,
        "projects.html",
        {"projects": projects},
    )


@router.post("/projects")
def create_project(
    name: str = Form(...),
    description: str | None = Form(None),
    db: Session = Depends(get_db),
):
    project = models.Project(name=name, description=description)
    db.add(project)
    db.commit()
    db.refresh(project)
    return RedirectResponse(url=f"/projects/{project.id}", status_code=303)


@router.get("/projects/{project_id}", response_class=HTMLResponse)
def project_detail(
    request: Request,
    project_id: int,
    status: str | None = None,
    person_id: str | None = None,
    priority: str | None = None,
    db: Session = Depends(get_db),
):
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    transcripts = (
        db.query(models.Transcript)
        .filter(models.Transcript.project_id == project_id)
        .order_by(models.Transcript.created_at.desc())
        .all()
    )
    runs = (
        db.query(models.ExtractionRun)
        .join(models.Transcript)
        .filter(models.Transcript.project_id == project_id)
        .order_by(models.ExtractionRun.created_at.desc())
        .all()
    )
    people = (
        db.query(models.Person)
        .filter(models.Person.project_id == project_id)
        .order_by(models.Person.name.asc())
        .all()
    )
    all_tasks = (
        db.query(models.Task)
        .filter(models.Task.project_id == project_id)
        .order_by(models.Task.created_at.desc())
        .all()
    )
    tasks_query = db.query(models.Task).filter(models.Task.project_id == project_id)
    selected_person_id = int(person_id) if person_id else None
    if status:
        tasks_query = tasks_query.filter(models.Task.status == status)
    if selected_person_id:
        tasks_query = tasks_query.filter(models.Task.person_id == selected_person_id)
    if priority:
        tasks_query = tasks_query.filter(models.Task.priority == priority)
    tasks = tasks_query.order_by(models.Task.created_at.desc()).all()
    decisions = (
        db.query(models.Decision)
        .filter(models.Decision.project_id == project_id)
        .order_by(models.Decision.created_at.desc())
        .all()
    )

    latest_run = runs[0] if runs else None
    latest_output = latest_run.raw_response if latest_run and latest_run.raw_response else {}
    task_stats = {
        "total": len(all_tasks),
        "todo": sum(1 for task in all_tasks if task.status == "todo"),
        "in_progress": sum(1 for task in all_tasks if task.status == "in_progress"),
        "done": sum(1 for task in all_tasks if task.status == "done"),
    }

    return templates.TemplateResponse(
        request,
        "project_detail.html",
        {
            "project": project,
            "transcripts": transcripts,
            "runs": runs,
            "people": people,
            "tasks": tasks,
            "all_tasks": all_tasks,
            "decisions": decisions,
            "latest_output": latest_output,
            "task_stats": task_stats,
            "filters": {
                "status": status or "",
                "person_id": selected_person_id or "",
                "priority": priority or "",
            },
        },
    )


@router.post("/projects/{project_id}/transcripts")
def upload_transcript(
    project_id: int,
    transcript_text: str | None = Form(None),
    file: UploadFile | None = File(None),
    db: Session = Depends(get_db),
):
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    content = None
    filename = None

    if file is not None:
        content = file.file.read().decode("utf-8", errors="ignore")
        filename = file.filename

    if transcript_text:
        content = transcript_text

    if content is None:
        raise HTTPException(status_code=400, detail="Transcript content is required")

    content = validate_transcript_content(content)

    transcript = models.Transcript(
        project_id=project_id,
        content=content,
        source_filename=filename,
    )
    db.add(transcript)
    db.commit()

    return RedirectResponse(url=f"/projects/{project_id}", status_code=303)


@router.post("/projects/{project_id}/extract")
def extract_transcript(
    project_id: int,
    transcript_id: int = Form(...),
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

    run_extraction(db, transcript)
    db.commit()

    return RedirectResponse(url=f"/projects/{project_id}", status_code=303)


@router.post("/projects/{project_id}/people")
def add_person(
    project_id: int,
    name: str = Form(...),
    role: str | None = Form(None),
    db: Session = Depends(get_db),
):
    person = models.Person(project_id=project_id, name=name, role=role)
    db.add(person)
    db.commit()
    return RedirectResponse(url=f"/projects/{project_id}", status_code=303)


@router.post("/projects/{project_id}/people/{person_id}/edit")
def edit_person(
    project_id: int,
    person_id: int,
    name: str = Form(...),
    role: str | None = Form(None),
    db: Session = Depends(get_db),
):
    person = (
        db.query(models.Person)
        .filter(models.Person.id == person_id, models.Person.project_id == project_id)
        .first()
    )
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    person.name = name
    person.role = role
    db.commit()
    return RedirectResponse(url=f"/projects/{project_id}", status_code=303)


@router.post("/projects/{project_id}/tasks")
def add_task(
    project_id: int,
    description: str = Form(...),
    person_id: str | None = Form(None),
    status: str = Form("todo"),
    priority: str = Form("medium"),
    due_date: str | None = Form(None),
    db: Session = Depends(get_db),
):
    parsed_person_id = int(person_id) if person_id else None
    parsed_due_date = parse_due_date(due_date)

    if parsed_person_id:
        person = (
            db.query(models.Person)
            .filter(models.Person.id == parsed_person_id, models.Person.project_id == project_id)
            .first()
        )
        if not person:
            raise HTTPException(status_code=400, detail="Assignee not found")

    task = models.Task(
        project_id=project_id,
        person_id=parsed_person_id,
        description=description,
        status=status,
        priority=priority,
        due_date=parsed_due_date,
    )
    db.add(task)
    db.commit()
    return RedirectResponse(url=f"/projects/{project_id}", status_code=303)


@router.post("/projects/{project_id}/tasks/{task_id}/edit")
def edit_task(
    project_id: int,
    task_id: int,
    description: str = Form(...),
    person_id: str | None = Form(None),
    status: str = Form("todo"),
    priority: str = Form("medium"),
    due_date: str | None = Form(None),
    db: Session = Depends(get_db),
):
    task = (
        db.query(models.Task)
        .filter(models.Task.id == task_id, models.Task.project_id == project_id)
        .first()
    )
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    parsed_person_id = int(person_id) if person_id else None
    parsed_due_date = parse_due_date(due_date)

    if parsed_person_id:
        person = (
            db.query(models.Person)
            .filter(models.Person.id == parsed_person_id, models.Person.project_id == project_id)
            .first()
        )
        if not person:
            raise HTTPException(status_code=400, detail="Assignee not found")

    task.description = description
    task.person_id = parsed_person_id
    task.status = status
    task.priority = priority
    task.due_date = parsed_due_date

    db.commit()
    return RedirectResponse(url=f"/projects/{project_id}", status_code=303)


@router.post("/projects/{project_id}/decisions")
def add_decision(
    project_id: int,
    description: str = Form(...),
    db: Session = Depends(get_db),
):
    decision = models.Decision(project_id=project_id, description=description)
    db.add(decision)
    db.commit()
    return RedirectResponse(url=f"/projects/{project_id}", status_code=303)
