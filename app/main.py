from fastapi import FastAPI
from app.api.routes.extract import router as analyze_router
from app.api.routes.mcp import router as mcp_router

app = FastAPI(title="Meeting Knowledge Extractor")

app.include_router(analyze_router)
app.include_router(mcp_router)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/health")
def health():
    return {"status": "ok"}