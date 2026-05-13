from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from app.integrations.mcp_tool import TOOL_SCHEMA, execute_tool

router = APIRouter(
    prefix="/mcp",
    tags=["mcp"]
)


@router.get("/tool")
def get_tool_schema():
    return TOOL_SCHEMA


@router.post("/execute")
def execute_mcp_tool(arguments: dict[str, Any]):
    try:
        return execute_tool(arguments)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
