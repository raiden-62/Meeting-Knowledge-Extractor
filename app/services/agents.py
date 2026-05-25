import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any

from sqlalchemy.orm import Session

from app.db import models

DONE_MARKERS = (
    "готов",
    "готова",
    "готово",
    "готовы",
    "сделал",
    "сделала",
    "сделали",
    "выполнил",
    "выполнила",
    "выполнили",
    "закрыл",
    "закрыла",
    "закрыли",
    "завершил",
    "завершила",
    "завершили",
    "исправил",
    "исправила",
    "исправили",
    "done",
    "completed",
    "finished",
    "fixed",
    "resolved",
)

IN_PROGRESS_MARKERS = (
    "в работе",
    "занимается",
    "начал",
    "начала",
    "начали",
    "делает",
    "работает над",
    "in progress",
    "started",
    "working on",
)

NEGATIVE_DONE_MARKERS = (
    "не готов",
    "не готова",
    "не готово",
    "не сделали",
    "не сделано",
    "не выполн",
    "не закры",
    "not done",
    "not ready",
    "not completed",
)

STOPWORDS = {
    "and",
    "for",
    "from",
    "the",
    "will",
    "with",
    "все",
    "для",
    "или",
    "как",
    "над",
    "она",
    "они",
    "под",
    "при",
    "про",
    "что",
    "это",
    "будет",
    "готов",
    "готова",
    "готово",
    "делает",
    "задача",
    "начал",
    "начала",
    "сделал",
    "сделала",
    "сделали",
    "проверить",
    "подготовить",
    "обновить",
    "исправить",
    "сделать",
    "создать",
}

PRIORITY_RANK = {"low": 0, "medium": 1, "high": 2}
STATUS_RANK = {"todo": 0, "in_progress": 1, "done": 2}
ALLOWED_STATUSES = {"todo", "in_progress", "done"}
MONTHS_RU = {
    "января": 1,
    "январь": 1,
    "февраля": 2,
    "февраль": 2,
    "марта": 3,
    "март": 3,
    "апреля": 4,
    "апрель": 4,
    "мая": 5,
    "май": 5,
    "июня": 6,
    "июнь": 6,
    "июля": 7,
    "июль": 7,
    "августа": 8,
    "август": 8,
    "сентября": 9,
    "сентябрь": 9,
    "октября": 10,
    "октябрь": 10,
    "ноября": 11,
    "ноябрь": 11,
    "декабря": 12,
    "декабрь": 12,
}


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(" \t\r\n-•*")


def _normalize_text(value: Any) -> str:
    return re.sub(r"[^0-9a-zа-яё]+", " ", _clean_text(value).casefold()).replace("ё", "е").strip()


def _stem_token(token: str) -> str:
    for suffix in (
        "иями",
        "ями",
        "ами",
        "его",
        "ому",
        "ыми",
        "ими",
        "ую",
        "юю",
        "ая",
        "яя",
        "ое",
        "ее",
        "ые",
        "ие",
        "ый",
        "ий",
        "ой",
        "ам",
        "ям",
        "ах",
        "ях",
        "ов",
        "ев",
        "ей",
        "ия",
        "ию",
        "ью",
        "а",
        "я",
        "у",
        "ю",
        "ы",
        "и",
        "е",
        "о",
    ):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _tokens(value: Any) -> set[str]:
    tokens = set()
    for token in _normalize_text(value).split():
        if len(token) < 3 or token in STOPWORDS:
            continue
        tokens.add(_stem_token(token))
    return tokens


def _text_similarity(left: str, right: str) -> float:
    left_norm = _normalize_text(left)
    right_norm = _normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0

    sequence_score = SequenceMatcher(None, left_norm, right_norm).ratio()
    left_tokens = _tokens(left_norm)
    right_tokens = _tokens(right_norm)
    if not left_tokens or not right_tokens:
        return sequence_score

    overlap = len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))
    return max(sequence_score, overlap)


def _task_assignee_name(task: models.Task) -> str:
    if hasattr(task, "assignee_name"):
        return str(getattr(task, "assignee_name") or "")
    return task.assignee.name if task.assignee else ""


def _safe_date(value: Any):
    if not value:
        return None
    if hasattr(value, "isoformat") and not isinstance(value, str):
        return value
    text = _clean_text(value)
    for date_format in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, date_format).date()
        except ValueError:
            pass
    match = re.search(r"\b([0-2]?\d|3[01])[./](0?[1-9]|1[0-2])\b", text)
    if match:
        try:
            return datetime(datetime.utcnow().year, int(match.group(2)), int(match.group(1))).date()
        except ValueError:
            return None
    month_names = "|".join(MONTHS_RU)
    month_match = re.search(
        rf"\b([0-2]?\d|3[01])\s+({month_names})(?:\s+(20\d{{2}}))?\b",
        text.casefold().replace("ё", "е"),
    )
    if month_match:
        try:
            return datetime(
                int(month_match.group(3) or datetime.utcnow().year),
                MONTHS_RU[month_match.group(2)],
                int(month_match.group(1)),
            ).date()
        except ValueError:
            return None
    return None


def _should_update_status(current: str, incoming: str) -> bool:
    return STATUS_RANK.get(incoming, 0) > STATUS_RANK.get(current, 0)


def _split_segments(text: str) -> list[str]:
    return [
        _clean_text(segment)
        for segment in re.split(r"(?<!\d)[.!?]+(?!\d)|[\n;]+", text or "")
        if _clean_text(segment)
    ]


class ProjectMemoryAgent:
    def ensure_table(self, db: Session) -> None:
        models.ProjectMemory.__table__.create(bind=db.get_bind(), checkfirst=True)

    def build(self, db: Session, transcript: models.Transcript) -> dict[str, Any]:
        self.ensure_table(db)
        project = db.query(models.Project).filter(models.Project.id == transcript.project_id).first()
        all_tasks = (
            db.query(models.Task)
            .filter(models.Task.project_id == transcript.project_id)
            .order_by(models.Task.updated_at.desc())
            .limit(120)
            .all()
        )
        tasks = self._select_relevant_tasks(project, transcript, all_tasks)
        decisions = (
            db.query(models.Decision)
            .filter(models.Decision.project_id == transcript.project_id)
            .order_by(models.Decision.created_at.desc())
            .limit(12)
            .all()
        )
        people = (
            db.query(models.Person)
            .filter(models.Person.project_id == transcript.project_id)
            .order_by(models.Person.name.asc())
            .all()
        )
        memory_record = (
            db.query(models.ProjectMemory)
            .filter(models.ProjectMemory.project_id == transcript.project_id)
            .first()
        )

        return {
            "project": project,
            "current_transcript": transcript,
            "tasks": tasks,
            "decisions": decisions,
            "people": people,
            "memory_summary": memory_record.summary if memory_record else "",
        }

    def _select_relevant_tasks(
        self,
        project: models.Project | None,
        transcript: models.Transcript,
        tasks: list[models.Task],
        limit: int = 25,
    ) -> list[models.Task]:
        if not tasks:
            return []

        context = " ".join(
            [
                transcript.content,
                transcript.source_filename or "",
                project.name if project else "",
                project.description if project and project.description else "",
            ]
        )
        context_tokens = _tokens(context)

        ranked: list[tuple[int, int, models.Task]] = []
        for index, task in enumerate(tasks):
            task_tokens = _tokens(task.description)
            overlap = len(task_tokens & context_tokens)
            assignee = _task_assignee_name(task)
            score = overlap * 4
            if assignee and _normalize_text(assignee) in _normalize_text(context):
                score += 4
            if task.status != "done":
                score += 3
            if score > 0:
                ranked.append((score, index, task))

        relevant = [task for _, _, task in sorted(ranked, key=lambda item: (-item[0], item[1]))[:limit]]
        if len(relevant) < min(limit, 10):
            seen = {task.id for task in relevant}
            for task in tasks:
                if task.id in seen:
                    continue
                relevant.append(task)
                seen.add(task.id)
                if len(relevant) >= min(limit, 10):
                    break
        return relevant[:limit]

    def render(self, memory: dict[str, Any], max_chars: int = 6000) -> str:
        project = memory.get("project")
        current_transcript = memory.get("current_transcript")
        lines = [
            f"Проект: {project.name if project else 'без названия'}",
            f"Описание: {project.description if project and project.description else 'нет'}",
            (
                "Текущая стенограмма: "
                f"#{current_transcript.id}, файл: "
                f"{current_transcript.source_filename or 'вставленный текст'}"
                if current_transcript
                else "Текущая стенограмма: нет данных"
            ),
        ]

        memory_summary = _clean_text(memory.get("memory_summary") or "")
        if memory_summary:
            lines.append("Сжатая память проекта:")
            lines.append(memory_summary[:1800])

        people = memory.get("people") or []
        if people:
            lines.append("Участники:")
            for person in people[:25]:
                role = f", роль: {person.role}" if person.role else ""
                lines.append(f"- {person.name}{role}")

        tasks = memory.get("tasks") or []
        open_tasks = [task for task in tasks if task.status != "done"]
        done_tasks = [task for task in tasks if task.status == "done"]
        if open_tasks:
            lines.append("Открытые задачи:")
            for task in open_tasks[:35]:
                assignee = _task_assignee_name(task) or "без ответственного"
                due_date = f", срок: {task.due_date.isoformat()}" if task.due_date else ""
                lines.append(
                    f"- #{task.id} [{task.status}, {task.priority}{due_date}] "
                    f"{assignee}: {task.description}"
                )
        if done_tasks:
            lines.append("Недавно закрытые задачи:")
            for task in done_tasks[:15]:
                assignee = _task_assignee_name(task) or "без ответственного"
                lines.append(f"- #{task.id} [done] {assignee}: {task.description}")

        decisions = memory.get("decisions") or []
        if decisions:
            lines.append("Последние решения:")
            for decision in decisions[:12]:
                lines.append(f"- {decision.description}")

        result = "\n".join(lines)
        return result[:max_chars]

    def update_summary(
        self,
        db: Session,
        transcript: models.Transcript,
        raw_output: dict[str, Any],
        max_chars: int = 3000,
    ) -> None:
        self.ensure_table(db)
        project = db.query(models.Project).filter(models.Project.id == transcript.project_id).first()
        tasks = (
            db.query(models.Task)
            .filter(models.Task.project_id == transcript.project_id)
            .order_by(models.Task.updated_at.desc())
            .limit(30)
            .all()
        )
        decisions = (
            db.query(models.Decision)
            .filter(models.Decision.project_id == transcript.project_id)
            .order_by(models.Decision.created_at.desc())
            .limit(8)
            .all()
        )

        lines = [
            f"Проект: {project.name if project else transcript.project_id}",
            f"Последняя сводка: {_clean_text(raw_output.get('summary') or 'нет')}",
        ]
        open_tasks = [task for task in tasks if task.status != "done"][:18]
        if open_tasks:
            lines.append("Ключевые открытые задачи:")
            for task in open_tasks:
                assignee = _task_assignee_name(task) or "без ответственного"
                lines.append(f"- #{task.id} [{task.status}, {task.priority}] {assignee}: {task.description}")
        if decisions:
            lines.append("Последние решения:")
            for decision in decisions:
                lines.append(f"- {decision.description}")

        summary = "\n".join(lines)[:max_chars]
        memory = (
            db.query(models.ProjectMemory)
            .filter(models.ProjectMemory.project_id == transcript.project_id)
            .first()
        )
        if memory is None:
            memory = models.ProjectMemory(project_id=transcript.project_id, summary=summary)
            db.add(memory)
        else:
            memory.summary = summary


class TaskLifecycleAgent:
    def apply(
        self,
        db: Session,
        transcript: models.Transcript,
        run: models.ExtractionRun,
        raw_output: dict[str, Any],
        extra_updates: list[dict[str, str | int | None]] | None = None,
        infer_updates: bool = False,
    ) -> dict[str, int]:
        stats = {
            "created_tasks": 0,
            "updated_tasks": 0,
            "created_decisions": 0,
            "deduplicated_tasks": 0,
        }

        self._add_decisions(db, transcript, run, raw_output, stats)

        existing_tasks = (
            db.query(models.Task)
            .filter(models.Task.project_id == transcript.project_id)
            .order_by(models.Task.updated_at.desc())
            .all()
        )
        updates = list(raw_output.get("task_updates") or [])
        if extra_updates:
            updates.extend(extra_updates)
        if infer_updates:
            updates.extend(self.infer_task_updates(transcript.content, existing_tasks))
        self._apply_task_updates(transcript, existing_tasks, updates, stats)

        tasks = self._task_payloads_from_output(raw_output)
        for item in tasks:
            description = _clean_text(item.get("description") or item.get("task"))
            if not description:
                continue

            assignee = _clean_text(item.get("assignee") or "")
            person_id = None
            if assignee:
                person_id = get_or_create_person(db, transcript.project_id, assignee).id

            status = _clean_text(item.get("status") or "todo").lower()
            if status not in ALLOWED_STATUSES:
                status = "todo"
            priority = _clean_text(item.get("priority") or "medium").lower()
            if priority not in PRIORITY_RANK:
                priority = "medium"
            due_date = _safe_date(
                item.get("due_date")
                or item.get("deadline")
                or item.get("due")
                or item.get("date")
                or item.get("task_date")
                or item.get("deadline_date")
                or item.get("dueDate")
            )

            existing = self.find_matching_task(existing_tasks, description, assignee)
            if existing:
                self._merge_task(existing, person_id, status, priority, due_date)
                stats["deduplicated_tasks"] += 1
                continue

            task = models.Task(
                project_id=transcript.project_id,
                person_id=person_id,
                source_run_id=run.id,
                description=description,
                status=status,
                priority=priority,
                due_date=due_date,
            )
            db.add(task)
            db.flush()
            existing_tasks.append(task)
            stats["created_tasks"] += 1

        notes = raw_output.setdefault("agent_notes", [])
        if not isinstance(notes, list):
            notes = []
            raw_output["agent_notes"] = notes
        notes.append(
            "LifecycleAgent: "
            f"создано задач {stats['created_tasks']}, "
            f"обновлено задач {stats['updated_tasks']}, "
            f"дубликатов предотвращено {stats['deduplicated_tasks']}."
        )
        raw_output["lifecycle"] = stats
        run.raw_response = raw_output
        return stats

    def infer_task_updates(
        self,
        transcript: str,
        existing_tasks: list[models.Task],
    ) -> list[dict[str, str | int | None]]:
        updates: list[dict[str, str | int | None]] = []
        seen: set[tuple[int, str]] = set()

        for segment in _split_segments(transcript):
            status = self._status_from_segment(segment)
            if not status:
                continue

            for task in existing_tasks:
                if task.status == status:
                    continue
                assignee = _task_assignee_name(task)
                score = _text_similarity(segment, task.description)
                assignee_seen = assignee and _normalize_text(assignee) in _normalize_text(segment)
                threshold = 0.48 if assignee_seen else 0.62
                if score < threshold:
                    continue

                key = (task.id, status)
                if key in seen:
                    continue
                seen.add(key)
                updates.append(
                    {
                        "task_id": task.id,
                        "description": task.description,
                        "assignee": assignee or None,
                        "status": status,
                        "reason": f"Найдено в стенограмме: {segment}",
                    }
                )

        return updates

    def find_matching_task(
        self,
        tasks: list[models.Task],
        description: str,
        assignee: str | None = None,
    ) -> models.Task | None:
        best_task = None
        best_score = 0.0
        normalized_assignee = _normalize_text(assignee)

        for task in tasks:
            score = _text_similarity(description, task.description)
            task_assignee = _normalize_text(_task_assignee_name(task))
            if normalized_assignee:
                if task_assignee and task_assignee == normalized_assignee:
                    score += 0.1
                elif task_assignee and task_assignee != normalized_assignee:
                    score -= 0.2
            if score > best_score:
                best_task = task
                best_score = score

        threshold = 0.62 if normalized_assignee else 0.72
        return best_task if best_task is not None and best_score >= threshold else None

    def _add_decisions(
        self,
        db: Session,
        transcript: models.Transcript,
        run: models.ExtractionRun,
        raw_output: dict[str, Any],
        stats: dict[str, int],
    ) -> None:
        existing = {
            _normalize_text(decision.description)
            for decision in db.query(models.Decision)
            .filter(models.Decision.project_id == transcript.project_id)
            .all()
        }
        for decision in raw_output.get("decisions", []) or []:
            description = _clean_text(decision)
            key = _normalize_text(description)
            if not description or key in existing:
                continue
            db.add(
                models.Decision(
                    project_id=transcript.project_id,
                    source_run_id=run.id,
                    description=description,
                )
            )
            existing.add(key)
            stats["created_decisions"] += 1

    def _apply_task_updates(
        self,
        transcript: models.Transcript,
        existing_tasks: list[models.Task],
        updates: list[dict[str, Any]],
        stats: dict[str, int],
    ) -> None:
        tasks_by_id = {task.id: task for task in existing_tasks}
        for update in updates:
            if not isinstance(update, dict):
                continue
            status = _clean_text(update.get("status") or "done").lower()
            if status not in ALLOWED_STATUSES:
                status = "done"

            task = None
            task_id = update.get("task_id") or update.get("id")
            if task_id is not None:
                try:
                    task = tasks_by_id.get(int(task_id))
                except (TypeError, ValueError):
                    task = None
            if task is None:
                task = self.find_matching_task(
                    existing_tasks,
                    _clean_text(update.get("description") or update.get("task")),
                    _clean_text(update.get("assignee") or ""),
                )
            if (
                task is None
                or task.project_id != transcript.project_id
                or not _should_update_status(task.status, status)
            ):
                due_date = _safe_date(
                    update.get("due_date")
                    or update.get("deadline")
                    or update.get("due")
                    or update.get("date")
                    or update.get("task_date")
                    or update.get("deadline_date")
                    or update.get("dueDate")
                )
                if task is not None and due_date and task.project_id == transcript.project_id:
                    task.due_date = due_date
                continue

            task.status = status
            due_date = _safe_date(
                update.get("due_date")
                or update.get("deadline")
                or update.get("due")
                or update.get("date")
                or update.get("task_date")
                or update.get("deadline_date")
                or update.get("dueDate")
            )
            if due_date:
                task.due_date = due_date
            stats["updated_tasks"] += 1

    def _task_payloads_from_output(self, raw_output: dict[str, Any]) -> list[dict[str, Any]]:
        tasks = raw_output.get("tasks", []) if isinstance(raw_output, dict) else []
        if isinstance(tasks, list) and tasks:
            return [item if isinstance(item, dict) else {"description": item} for item in tasks]

        people_map = raw_output.get("people", {}) if isinstance(raw_output, dict) else {}
        result: list[dict[str, Any]] = []
        if isinstance(people_map, dict):
            for person_name, items in people_map.items():
                for task in items or []:
                    result.append({"description": task, "assignee": person_name})
        return result

    def _merge_task(
        self,
        task: models.Task,
        person_id: int | None,
        status: str,
        priority: str,
        due_date,
    ) -> None:
        if person_id and task.person_id != person_id:
            task.person_id = person_id
        if _should_update_status(task.status, status):
            task.status = status
        if PRIORITY_RANK.get(priority, 1) > PRIORITY_RANK.get(task.priority, 1):
            task.priority = priority
        if due_date:
            task.due_date = due_date

    def _status_from_segment(self, segment: str) -> str | None:
        lowered = _normalize_text(segment)
        if any(marker in lowered for marker in NEGATIVE_DONE_MARKERS):
            return None
        if any(marker in lowered for marker in DONE_MARKERS):
            return "done"
        if any(marker in lowered for marker in IN_PROGRESS_MARKERS):
            return "in_progress"
        return None


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
