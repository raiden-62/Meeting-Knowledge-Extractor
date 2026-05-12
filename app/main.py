from fastapi import FastAPI
from app.api.routes.extract import router as analyze_router
from app.api.routes.mcp import router as mcp_router
from fastapi.responses import FileResponse

app = FastAPI(title="Meeting Knowledge Extractor")

app.include_router(analyze_router)
app.include_router(mcp_router)

@app.get("/")
def frontend():
    return FileResponse("app/frontend/index.html")

@app.get("/health")
def health():
    return {"status": "ok"}