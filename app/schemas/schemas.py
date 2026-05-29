from datetime import date, datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.config import LLM_PROVIDERS, MAX_TRANSCRIPT_CHARS
from app.services.project_validation import ALLOWED_PRIORITIES, ALLOWED_STATUSES


class AnalyzeRequest(BaseModel):
    transcript: str = Field(..., min_length=1, max_length=MAX_TRANSCRIPT_CHARS)
    provider: Optional[str] = None

    @field_validator("transcript")
    @classmethod
    def transcript_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Transcript content is required")
        return cleaned

    @field_validator("provider")
    @classmethod
    def provider_must_be_supported(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        cleaned = value.strip().lower()
        if cleaned not in LLM_PROVIDERS:
            allowed = ", ".join(LLM_PROVIDERS)
            raise ValueError(f"Provider must be one of: {allowed}")
        return cleaned


class ExtractedTask(BaseModel):
    description: str
    assignee: Optional[str] = None
    status: str = "todo"
    priority: str = "medium"
    due_date: Optional[str] = None


class ExtractionMetrics(BaseModel):
    transcript_chars: int = 0
    llm_transcript_chars: int = 0
    decisions_count: int = 0
    tasks_count: int = 0
    people_count: int = 0
    risks_count: int = 0
    task_updates_count: int = 0
    parallel_chunks_count: int = 0
    parallel_workers: int = 0
    response_time_seconds: Optional[float] = None


class ExtractedTaskUpdate(BaseModel):
    task_id: Optional[int] = None
    description: str = ""
    assignee: Optional[str] = None
    status: str = "done"
    due_date: Optional[str] = None
    reason: Optional[str] = None


class AnalyzeResponse(BaseModel):
    summary: str = ""
    decisions: List[str] = Field(default_factory=list)
    tasks: List[ExtractedTask] = Field(default_factory=list)
    task_updates: List[ExtractedTaskUpdate] = Field(default_factory=list)
    people: Dict[str, List[str]] = Field(default_factory=dict)
    risks: List[str] = Field(default_factory=list)
    metrics: ExtractionMetrics = Field(default_factory=ExtractionMetrics)
    agent_notes: List[str] = Field(default_factory=list)
    source: str = "fallback"
    model_name: Optional[str] = None


class ChatRequest(BaseModel):
    message: str


class ProjectCreate(BaseModel):
    name: str = Field(..., max_length=200)
    description: Optional[str] = None

    @field_validator("name")
    @classmethod
    def name_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Project name is required")
        return cleaned

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return value.strip() or None


class ProjectRead(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TranscriptRead(BaseModel):
    id: int
    project_id: int
    source_filename: Optional[str] = None
    meeting_date: Optional[date] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class RunRead(BaseModel):
    id: int
    transcript_id: int
    provider: str
    model_name: Optional[str] = None
    status: str
    response_time_seconds: Optional[float] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PersonCreate(BaseModel):
    name: str = Field(..., max_length=200)
    role: Optional[str] = None

    @field_validator("name")
    @classmethod
    def person_name_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Person name is required")
        return cleaned

    @field_validator("role")
    @classmethod
    def normalize_role(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return value.strip() or None


class PersonUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=200)
    role: Optional[str] = None

    @field_validator("name")
    @classmethod
    def updated_person_name_must_not_be_blank(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Person name is required")
        return cleaned

    @field_validator("role")
    @classmethod
    def normalize_updated_role(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return value.strip() or None


class PersonRead(BaseModel):
    id: int
    project_id: int
    name: str
    role: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TaskCreate(BaseModel):
    description: str
    person_id: Optional[int] = None
    status: Optional[str] = "todo"
    priority: Optional[str] = "medium"
    due_date: Optional[date] = None

    @field_validator("description")
    @classmethod
    def task_description_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Task description is required")
        return cleaned

    @field_validator("status")
    @classmethod
    def task_status_must_be_supported(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        cleaned = value.strip().lower()
        if cleaned not in ALLOWED_STATUSES:
            allowed = ", ".join(sorted(ALLOWED_STATUSES))
            raise ValueError(f"Status must be one of: {allowed}")
        return cleaned

    @field_validator("priority")
    @classmethod
    def task_priority_must_be_supported(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        cleaned = value.strip().lower()
        if cleaned not in ALLOWED_PRIORITIES:
            allowed = ", ".join(sorted(ALLOWED_PRIORITIES))
            raise ValueError(f"Priority must be one of: {allowed}")
        return cleaned


class TaskUpdate(BaseModel):
    description: Optional[str] = None
    person_id: Optional[int] = None
    status: Optional[str] = None
    priority: Optional[str] = None
    due_date: Optional[date] = None

    @field_validator("description")
    @classmethod
    def updated_task_description_must_not_be_blank(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Task description is required")
        return cleaned

    @field_validator("status")
    @classmethod
    def updated_task_status_must_be_supported(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip().lower()
        if cleaned not in ALLOWED_STATUSES:
            allowed = ", ".join(sorted(ALLOWED_STATUSES))
            raise ValueError(f"Status must be one of: {allowed}")
        return cleaned

    @field_validator("priority")
    @classmethod
    def updated_task_priority_must_be_supported(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip().lower()
        if cleaned not in ALLOWED_PRIORITIES:
            allowed = ", ".join(sorted(ALLOWED_PRIORITIES))
            raise ValueError(f"Priority must be one of: {allowed}")
        return cleaned


class TaskRead(BaseModel):
    id: int
    project_id: int
    person_id: Optional[int] = None
    description: str
    status: str
    priority: str
    due_date: Optional[date] = None
    meeting_date: Optional[date] = None
    last_updated_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DecisionCreate(BaseModel):
    description: str

    @field_validator("description")
    @classmethod
    def decision_description_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Decision description is required")
        return cleaned


class DecisionRead(BaseModel):
    id: int
    project_id: int
    description: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
