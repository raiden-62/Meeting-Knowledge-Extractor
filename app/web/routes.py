from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.core.config import LLM_PROVIDER, LLM_PROVIDERS, MAX_TRANSCRIPT_CHARS
from app.core.time import app_now
from app.db import models
from app.db.database import get_db
from app.services.agents import TaskLifecycleAgent
from app.services.extraction_service import run_extraction
from app.services.llm_service import LLMProviderError
from app.services.project_service import delete_project
from app.services.project_validation import (
    clean_optional_text,
    clean_required_text,
    parse_due_date,
    parse_optional_int,
    parse_submitted_meeting_date,
    read_transcript_upload,
    validate_priority,
    validate_provider,
    validate_status,
    validate_transcript_content,
)

router = APIRouter(tags=["ui"])

templates = Jinja2Templates(directory="app/frontend/templates")
templates.env.globals["max_transcript_chars"] = MAX_TRANSCRIPT_CHARS
templates.env.globals["llm_provider"] = LLM_PROVIDER
templates.env.globals["llm_providers"] = LLM_PROVIDERS

STATUS_LABELS = {"todo": "сделать", "in_progress": "в работе", "done": "готово"}
PRIORITY_LABELS = {"low": "низкая", "medium": "средняя", "high": "высокая"}


def accepted_task_from_review(
    db: Session,
    project_id: int,
    source_run_id: int | None,
    description: str,
    person_id: int | None,
    status: str,
    priority: str,
    due_date,
) -> models.Task:
    assignee_name = ""
    if person_id:
        person = (
            db.query(models.Person)
            .filter(models.Person.id == person_id, models.Person.project_id == project_id)
            .first()
        )
        if not person:
            raise HTTPException(status_code=400, detail="Ответственный не найден")
        assignee_name = person.name

    existing_tasks = (
        db.query(models.Task)
        .filter(models.Task.project_id == project_id)
        .order_by(models.Task.updated_at.desc())
        .all()
    )
    lifecycle = TaskLifecycleAgent()
    existing = lifecycle.find_matching_task(existing_tasks, description, assignee_name)
    if existing:
        lifecycle._merge_task(existing, person_id, status, priority, due_date)
        return existing

    task = models.Task(
        project_id=project_id,
        person_id=person_id,
        source_run_id=source_run_id,
        description=description,
        status=status,
        priority=priority,
        due_date=due_date,
    )
    db.add(task)
    return task


def find_project_matches(project_id: int, query: str | None, db: Session) -> dict[str, list[dict]]:
    cleaned = (query or "").strip()
    if len(cleaned) < 2:
        return {"query": cleaned, "tasks": [], "decisions": [], "transcripts": [], "people": []}

    lowered = cleaned.casefold()

    def hit(value: str | None) -> bool:
        return lowered in (value or "").casefold()

    tasks = [
        {
            "id": task.id,
            "title": task.description,
            "meta": (
                f"{STATUS_LABELS.get(task.status, task.status)} · "
                f"{PRIORITY_LABELS.get(task.priority, task.priority)}"
            ),
        }
        for task in db.query(models.Task).filter(models.Task.project_id == project_id).all()
        if hit(task.description) or hit(task.status) or hit(task.priority) or hit(task.assignee.name if task.assignee else "")
    ]
    decisions = [
        {"id": decision.id, "title": decision.description, "meta": decision.created_at.strftime("%Y-%m-%d")}
        for decision in db.query(models.Decision).filter(models.Decision.project_id == project_id).all()
        if hit(decision.description)
    ]
    transcripts = [
        {
            "id": transcript.id,
            "title": transcript.source_filename or f"Transcript #{transcript.id}",
            "meta": transcript.created_at.strftime("%Y-%m-%d %H:%M"),
            "snippet": _snippet(transcript.content, lowered),
        }
        for transcript in db.query(models.Transcript).filter(models.Transcript.project_id == project_id).all()
        if hit(transcript.content) or hit(transcript.source_filename)
    ]
    people = [
        {"id": person.id, "title": person.name, "meta": person.role or "Без роли"}
        for person in db.query(models.Person).filter(models.Person.project_id == project_id).all()
        if hit(person.name) or hit(person.role)
    ]
    return {
        "query": cleaned,
        "tasks": tasks[:20],
        "decisions": decisions[:20],
        "transcripts": transcripts[:10],
        "people": people[:20],
    }


def _snippet(text: str, lowered_query: str, radius: int = 90) -> str:
    lowered = text.casefold()
    index = lowered.find(lowered_query)
    if index == -1:
        return text[: radius * 2].strip()
    start = max(index - radius, 0)
    end = min(index + len(lowered_query) + radius, len(text))
    prefix = "..." if start else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end].strip()}{suffix}"


def build_project_report(project: models.Project, db: Session, markdown: bool = True) -> str:
    tasks = (
        db.query(models.Task)
        .filter(models.Task.project_id == project.id)
        .order_by(models.Task.status.asc(), models.Task.created_at.desc())
        .all()
    )
    decisions = (
        db.query(models.Decision)
        .filter(models.Decision.project_id == project.id)
        .order_by(models.Decision.created_at.desc())
        .all()
    )
    runs = (
        db.query(models.ExtractionRun)
        .join(models.Transcript)
        .filter(models.Transcript.project_id == project.id)
        .order_by(models.ExtractionRun.created_at.desc())
        .all()
    )
    latest_output = runs[0].raw_response if runs and runs[0].raw_response else {}
    risks = latest_output.get("accepted_risks") or latest_output.get("risks") or []
    summary = latest_output.get("summary") or "Сводка пока не сформирована."

    if markdown:
        lines = [
            f"# {project.name}",
            "",
            project.description or "Без описания",
            "",
            "## Сводка",
            summary,
            "",
            "## Задачи",
        ]
        lines.extend(
            (
                f"- [{STATUS_LABELS.get(task.status, task.status)}] {task.description} "
                f"({task.assignee.name if task.assignee else 'без ответственного'}, "
                f"{PRIORITY_LABELS.get(task.priority, task.priority)})"
            )
            for task in tasks
        )
        if not tasks:
            lines.append("- Нет задач")
        lines.extend(["", "## Решения"])
        lines.extend(f"- {decision.description}" for decision in decisions)
        if not decisions:
            lines.append("- Нет решений")
        lines.extend(["", "## Риски"])
        lines.extend(f"- {risk}" for risk in risks)
        if not risks:
            lines.append("- Нет рисков")
        return "\n".join(lines).strip() + "\n"

    lines = [
        project.name,
        project.description or "Без описания",
        "",
        "Сводка:",
        summary,
        "",
        "Задачи:",
    ]
    lines.extend(
        (
            f"- {STATUS_LABELS.get(task.status, task.status)}: {task.description} "
            f"({task.assignee.name if task.assignee else 'без ответственного'}, "
            f"{PRIORITY_LABELS.get(task.priority, task.priority)})"
        )
        for task in tasks
    )
    lines.extend(["", "Решения:"])
    lines.extend(f"- {decision.description}" for decision in decisions)
    lines.extend(["", "Риски:"])
    lines.extend(f"- {risk}" for risk in risks)
    return "\n".join(lines).strip() + "\n"


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
    project = models.Project(
        name=clean_required_text(name, "Project name"),
        description=clean_optional_text(description),
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return RedirectResponse(url=f"/projects/{project.id}", status_code=303)


@router.post("/projects/{project_id}/edit")
def edit_project(
    project_id: int,
    name: str = Form(...),
    description: str | None = Form(None),
    db: Session = Depends(get_db),
):
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project.name = clean_required_text(name, "Project name")
    project.description = clean_optional_text(description)
    db.commit()
    return RedirectResponse(url=f"/projects/{project_id}", status_code=303)


@router.post("/projects/{project_id}/delete")
def remove_project(project_id: int, db: Session = Depends(get_db)):
    if not delete_project(db, project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    return RedirectResponse(url="/projects", status_code=303)


@router.get("/projects/{project_id}", response_class=HTMLResponse)
def project_detail(
    request: Request,
    project_id: int,
    status: str | None = None,
    person_id: str | None = None,
    priority: str | None = None,
    q: str | None = None,
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
    pending_task_suggestions = (
        db.query(models.TaskSuggestion)
        .filter(
            models.TaskSuggestion.project_id == project_id,
            models.TaskSuggestion.review_status == "pending",
        )
        .order_by(models.TaskSuggestion.created_at.desc())
        .all()
    )
    tasks_query = db.query(models.Task).filter(models.Task.project_id == project_id)
    selected_person_id = parse_optional_int(person_id, "person_id")
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
            "pending_task_suggestions": pending_task_suggestions,
            "decisions": decisions,
            "latest_output": latest_output,
            "latest_run": latest_run,
            "search": find_project_matches(project_id, q, db),
            "task_stats": task_stats,
            "filters": {
                "status": status or "",
                "person_id": selected_person_id or "",
                "priority": priority or "",
                "q": q or "",
            },
            "today_date": app_now().date().isoformat(),
        },
    )


@router.post("/projects/{project_id}/transcripts")
def upload_transcript(
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

    return RedirectResponse(url=f"/projects/{project_id}", status_code=303)


@router.post("/projects/{project_id}/extract")
def extract_transcript(
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
        run_extraction(db, transcript, provider=validate_provider(provider))
    except LLMProviderError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"{exc.provider}: {exc.reason}",
        ) from exc
    db.commit()

    return RedirectResponse(url=f"/projects/{project_id}", status_code=303)


@router.get("/projects/{project_id}/export.{extension}")
def export_project_report(
    project_id: int,
    extension: str,
    db: Session = Depends(get_db),
):
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if extension not in {"md", "txt"}:
        raise HTTPException(status_code=404, detail="Export format not found")

    markdown = extension == "md"
    content = build_project_report(project, db, markdown=markdown)
    media_type = "text/markdown; charset=utf-8" if markdown else "text/plain; charset=utf-8"
    filename = quote(f"{project.name}-meeting-report.{extension}")
    return PlainTextResponse(
        content,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"},
    )


@router.post("/projects/{project_id}/people")
def add_person(
    project_id: int,
    name: str = Form(...),
    role: str | None = Form(None),
    db: Session = Depends(get_db),
):
    person = models.Person(
        project_id=project_id,
        name=clean_required_text(name, "Person name"),
        role=clean_optional_text(role),
    )
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

    person.name = clean_required_text(name, "Person name")
    person.role = clean_optional_text(role)
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
    parsed_person_id = parse_optional_int(person_id, "person_id")
    parsed_due_date = parse_due_date(due_date)
    cleaned_description = clean_required_text(description, "Task description")
    cleaned_status = validate_status(status)
    cleaned_priority = validate_priority(priority)

    if parsed_person_id:
        person = (
            db.query(models.Person)
            .filter(models.Person.id == parsed_person_id, models.Person.project_id == project_id)
            .first()
        )
        if not person:
            raise HTTPException(status_code=400, detail="Ответственный не найден")

    task = models.Task(
        project_id=project_id,
        person_id=parsed_person_id,
        description=cleaned_description,
        status=cleaned_status,
        priority=cleaned_priority,
        due_date=parsed_due_date,
    )
    db.add(task)
    db.commit()
    return RedirectResponse(url=f"/projects/{project_id}", status_code=303)


@router.post("/projects/{project_id}/task-suggestions/{suggestion_id}/accept")
def accept_task_suggestion(
    project_id: int,
    suggestion_id: int,
    description: str = Form(...),
    person_id: str | None = Form(None),
    status: str = Form("todo"),
    priority: str = Form("medium"),
    due_date: str | None = Form(None),
    db: Session = Depends(get_db),
):
    suggestion = (
        db.query(models.TaskSuggestion)
        .filter(
            models.TaskSuggestion.id == suggestion_id,
            models.TaskSuggestion.project_id == project_id,
            models.TaskSuggestion.review_status == "pending",
        )
        .first()
    )
    if not suggestion:
        raise HTTPException(status_code=404, detail="Task suggestion not found")

    cleaned_description = description.strip()
    if not cleaned_description:
        raise HTTPException(status_code=400, detail="Task description is required")

    parsed_due_date = parse_due_date(due_date)
    parsed_person_id = parse_optional_int(person_id, "person_id")
    cleaned_status = validate_status(status)
    cleaned_priority = validate_priority(priority)
    accepted_task_from_review(
        db,
        project_id,
        suggestion.source_run_id,
        cleaned_description,
        parsed_person_id,
        cleaned_status,
        cleaned_priority,
        parsed_due_date,
    )
    suggestion.description = cleaned_description
    suggestion.assignee_name = (
        db.query(models.Person.name)
        .filter(models.Person.id == parsed_person_id, models.Person.project_id == project_id)
        .scalar()
        if parsed_person_id
        else None
    )
    suggestion.status = cleaned_status
    suggestion.priority = cleaned_priority
    suggestion.due_date = parsed_due_date
    suggestion.review_status = "accepted"
    suggestion.reviewed_at = app_now()

    db.commit()
    return RedirectResponse(url=f"/projects/{project_id}", status_code=303)


@router.post("/projects/{project_id}/task-suggestions/{suggestion_id}/reject")
def reject_task_suggestion(
    project_id: int,
    suggestion_id: int,
    db: Session = Depends(get_db),
):
    suggestion = (
        db.query(models.TaskSuggestion)
        .filter(
            models.TaskSuggestion.id == suggestion_id,
            models.TaskSuggestion.project_id == project_id,
            models.TaskSuggestion.review_status == "pending",
        )
        .first()
    )
    if not suggestion:
        raise HTTPException(status_code=404, detail="Task suggestion not found")

    suggestion.review_status = "rejected"
    suggestion.reviewed_at = app_now()
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

    parsed_person_id = parse_optional_int(person_id, "person_id")
    parsed_due_date = parse_due_date(due_date)
    cleaned_description = clean_required_text(description, "Task description")
    cleaned_status = validate_status(status)
    cleaned_priority = validate_priority(priority)

    if parsed_person_id:
        person = (
            db.query(models.Person)
            .filter(models.Person.id == parsed_person_id, models.Person.project_id == project_id)
            .first()
        )
        if not person:
            raise HTTPException(status_code=400, detail="Ответственный не найден")

    task.description = cleaned_description
    task.person_id = parsed_person_id
    task.status = cleaned_status
    task.priority = cleaned_priority
    task.due_date = parsed_due_date

    db.commit()
    return RedirectResponse(url=f"/projects/{project_id}", status_code=303)


@router.post("/projects/{project_id}/tasks/{task_id}/delete")
def delete_task(
    project_id: int,
    task_id: int,
    db: Session = Depends(get_db),
):
    task = (
        db.query(models.Task)
        .filter(models.Task.id == task_id, models.Task.project_id == project_id)
        .first()
    )
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    db.delete(task)
    db.commit()
    return RedirectResponse(url=f"/projects/{project_id}", status_code=303)


@router.post("/projects/{project_id}/decisions")
def add_decision(
    project_id: int,
    description: str = Form(...),
    db: Session = Depends(get_db),
):
    decision = models.Decision(
        project_id=project_id,
        description=clean_required_text(description, "Decision description"),
    )
    db.add(decision)
    db.commit()
    return RedirectResponse(url=f"/projects/{project_id}", status_code=303)
