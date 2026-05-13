from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes.extract import router as analyze_router
from app.api.routes.mcp import router as mcp_router
from app.api.routes.projects import router as projects_router
from app.db import models
from app.db.database import Base, engine
from app.web.routes import router as ui_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ = models
    Base.metadata.create_all(bind=engine)
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


@app.get("/health")
def health():
    return {"status": "ok"}
