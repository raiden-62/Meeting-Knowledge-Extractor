from fastapi import APIRouter, HTTPException
from app.schemas.schemas import AnalyzeRequest, AnalyzeResponse
from app.services.meeting_pipeline import process_meeting
from app.services.llm_service import LLMProviderError

router = APIRouter(prefix="/analyze", tags=["analyze"])

@router.post("", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    try:
        return process_meeting(request.transcript, provider=request.provider)
    except LLMProviderError as exc:
        raise HTTPException(
            status_code=502,
            detail={"provider": exc.provider, "reason": exc.reason},
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
