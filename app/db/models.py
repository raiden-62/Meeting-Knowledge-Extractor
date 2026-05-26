from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    transcripts: Mapped[list["Transcript"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
    )
    people: Mapped[list["Person"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
    )
    tasks: Mapped[list["Task"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
    )
    task_suggestions: Mapped[list["TaskSuggestion"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
    )
    decisions: Mapped[list["Decision"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
    )
    memory: Mapped["ProjectMemory | None"] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
    )


class ProjectMemory(Base):
    __tablename__ = "project_memory"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), unique=True, nullable=False)
    summary: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    project: Mapped[Project] = relationship(back_populates="memory")


class Transcript(Base):
    __tablename__ = "transcripts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    project: Mapped[Project] = relationship(back_populates="transcripts")
    runs: Mapped[list["ExtractionRun"]] = relationship(
        back_populates="transcript",
        cascade="all, delete-orphan",
    )


class ExtractionRun(Base):
    __tablename__ = "extraction_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    transcript_id: Mapped[int] = mapped_column(
        ForeignKey("transcripts.id"),
        nullable=False,
    )
    provider: Mapped[str] = mapped_column(String(100), nullable=False)
    model_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    status: Mapped[str] = mapped_column(String(40), default="completed")
    response_time_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw_response: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    transcript: Mapped[Transcript] = relationship(back_populates="runs")
    tasks: Mapped[list["Task"]] = relationship(
        back_populates="source_run",
        cascade="all, delete-orphan",
    )
    task_suggestions: Mapped[list["TaskSuggestion"]] = relationship(
        back_populates="source_run",
        cascade="all, delete-orphan",
    )
    decisions: Mapped[list["Decision"]] = relationship(
        back_populates="source_run",
        cascade="all, delete-orphan",
    )


class Person(Base):
    __tablename__ = "people"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    role: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    project: Mapped[Project] = relationship(back_populates="people")
    tasks: Mapped[list["Task"]] = relationship(back_populates="assignee")


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False)
    person_id: Mapped[int | None] = mapped_column(ForeignKey("people.id"))
    source_run_id: Mapped[int | None] = mapped_column(ForeignKey("extraction_runs.id"))
    description: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(40), default="todo")
    priority: Mapped[str] = mapped_column(String(40), default="medium")
    due_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    project: Mapped[Project] = relationship(back_populates="tasks")
    assignee: Mapped[Person | None] = relationship(back_populates="tasks")
    source_run: Mapped[ExtractionRun | None] = relationship(back_populates="tasks")

    @property
    def meeting_date(self) -> datetime:
        if self.source_run and self.source_run.transcript:
            return self.source_run.transcript.created_at
        return self.created_at

    @property
    def last_updated_at(self) -> datetime:
        return self.updated_at


class TaskSuggestion(Base):
    __tablename__ = "task_suggestions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False)
    source_run_id: Mapped[int | None] = mapped_column(ForeignKey("extraction_runs.id"))
    description: Mapped[str] = mapped_column(Text, nullable=False)
    assignee_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    status: Mapped[str] = mapped_column(String(40), default="todo")
    priority: Mapped[str] = mapped_column(String(40), default="medium")
    due_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    confidence_level: Mapped[str] = mapped_column(String(20), default="low")
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_flags: Mapped[list | None] = mapped_column(JSON, nullable=True)
    confidence_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    review_status: Mapped[str] = mapped_column(String(20), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    project: Mapped[Project] = relationship(back_populates="task_suggestions")
    source_run: Mapped[ExtractionRun | None] = relationship(back_populates="task_suggestions")


class Decision(Base):
    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False)
    source_run_id: Mapped[int | None] = mapped_column(ForeignKey("extraction_runs.id"))
    description: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    project: Mapped[Project] = relationship(back_populates="decisions")
    source_run: Mapped[ExtractionRun | None] = relationship(back_populates="decisions")
