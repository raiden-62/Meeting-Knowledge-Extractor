from datetime import datetime
from typing import Any

from fastapi import HTTPException, UploadFile

from app.core.config import LLM_PROVIDERS, MAX_TRANSCRIPT_CHARS
from app.services.transcript_dates import parse_meeting_date, resolve_meeting_date

ALLOWED_STATUSES = {"todo", "in_progress", "done"}
ALLOWED_PRIORITIES = {"low", "medium", "high"}


def clean_optional_text(value: str | None) -> str | None:
    cleaned = (value or "").strip()
    return cleaned or None


def clean_required_text(value: str, field_name: str) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail=f"{field_name} is required")
    return cleaned


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


def validate_transcript_file(file: UploadFile) -> None:
    filename = (file.filename or "").lower()
    if not filename.endswith((".txt", ".md")):
        raise HTTPException(status_code=400, detail="Only .txt and .md transcripts are supported")


def has_transcript_file(file: UploadFile | None) -> bool:
    return bool(file and (file.filename or "").strip())


def read_transcript_upload(file: UploadFile) -> str:
    validate_transcript_file(file)
    data = file.file.read()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="Transcript file must be valid UTF-8") from exc


def validate_provider(provider: str | None) -> str | None:
    if provider is None or not provider.strip():
        return None
    cleaned = provider.strip().lower()
    if cleaned not in LLM_PROVIDERS:
        allowed = ", ".join(LLM_PROVIDERS)
        raise HTTPException(status_code=400, detail=f"Provider must be one of: {allowed}")
    return cleaned


def parse_submitted_meeting_date(value: str | None, content: str):
    if value and value.strip():
        parsed = parse_meeting_date(value)
        if not parsed:
            raise HTTPException(status_code=400, detail="Invalid meeting_date")
        return parsed
    return resolve_meeting_date(None, content)


def parse_due_date(value: str | None):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid due_date") from exc


def parse_optional_int(value: Any, field_name: str) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid {field_name}") from exc


def validate_status(value: str | None) -> str:
    cleaned = (value or "todo").strip().lower()
    if cleaned not in ALLOWED_STATUSES:
        allowed = ", ".join(sorted(ALLOWED_STATUSES))
        raise HTTPException(status_code=400, detail=f"Status must be one of: {allowed}")
    return cleaned


def validate_priority(value: str | None) -> str:
    cleaned = (value or "medium").strip().lower()
    if cleaned not in ALLOWED_PRIORITIES:
        allowed = ", ".join(sorted(ALLOWED_PRIORITIES))
        raise HTTPException(status_code=400, detail=f"Priority must be one of: {allowed}")
    return cleaned
