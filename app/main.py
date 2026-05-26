from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sqlalchemy import inspect, text

from app.api.routes.extract import router as analyze_router
from app.api.routes.mcp import router as mcp_router
from app.api.routes.projects import router as projects_router
from app.db import models
from app.db.database import Base, engine
from app.web.routes import router as ui_router


def init_db() -> None:
    _ = models
    Base.metadata.create_all(bind=engine)
    ensure_schema()


def ensure_schema() -> None:
    inspector = inspect(engine)
    if "transcripts" not in inspector.get_table_names():
        return

    transcript_columns = {column["name"] for column in inspector.get_columns("transcripts")}
    if "meeting_date" in transcript_columns:
        return

    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE transcripts ADD COLUMN meeting_date DATE"))
        if engine.dialect.name == "sqlite":
            backfill_sql = "UPDATE transcripts SET meeting_date = DATE(created_at) WHERE meeting_date IS NULL"
        else:
            backfill_sql = "UPDATE transcripts SET meeting_date = CAST(created_at AS DATE) WHERE meeting_date IS NULL"
        connection.execute(text(backfill_sql))


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Meeting Knowledge Extractor",
    description=(
        "Business-ready service for extracting meeting summaries, decisions, "
        "tasks, responsible people and risks from transcripts."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(analyze_router)
app.include_router(mcp_router)
app.include_router(projects_router)
app.include_router(ui_router)

app.mount("/static", StaticFiles(directory="app/frontend/static"), name="static")

init_db()


@app.get("/health")
def health():
    return {"status": "ok"}
