from fastapi import APIRouter
from app.schemas.meeting import AnalyzeRequest, AnalyzeResponse
from app.services.meeting_pipeline import process_meeting

router = APIRouter(prefix="/analyze", tags=["analyze"])


@router.post("", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    return process_meeting(request.transcript)