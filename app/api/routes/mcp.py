from fastapi import APIRouter

from app.integrations.mcp_tool import TOOL_SCHEMA

router = APIRouter(
    prefix="/mcp",
    tags=["mcp"]
)


@router.get("/tool")
def get_tool_schema():
    return TOOL_SCHEMA