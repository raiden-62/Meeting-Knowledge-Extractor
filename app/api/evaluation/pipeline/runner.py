import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from app.integrations import llm_api
from app.services import llm_service
from app.services.meeting_pipeline import process_meeting


@dataclass(frozen=True)
class PipelineRunOptions:
    provider: str | None = None
    model: str | None = None


@dataclass(frozen=True)
class PipelineRunResult:
    response: dict
    response_time_seconds: float
    provider: str | None
    requested_model: str | None
    actual_model: str | None


@contextmanager
def _temporary_model_override(provider: str | None, model: str | None) -> Iterator[None]:
    if not model:
        yield
        return

    selected_provider = (provider or "").strip().lower()
    if selected_provider not in {"gigachat", "deepseek"}:
        yield
        return

    attr = "GIGACHAT_MODEL" if selected_provider == "gigachat" else "DEEPSEEK_MODEL"
    old_api_model = getattr(llm_api, attr)
    old_service_model = getattr(llm_service, attr)
    old_gigachat_client = getattr(llm_api, "_gigachat_client", None)

    setattr(llm_api, attr, model)
    setattr(llm_service, attr, model)
    if selected_provider == "gigachat":
        llm_api._gigachat_client = None

    try:
        yield
    finally:
        setattr(llm_api, attr, old_api_model)
        setattr(llm_service, attr, old_service_model)
        if selected_provider == "gigachat":
            llm_api._gigachat_client = old_gigachat_client


def run_pipeline(transcript: str, options: PipelineRunOptions | None = None) -> PipelineRunResult:
    options = options or PipelineRunOptions()

    start = time.time()
    with _temporary_model_override(options.provider, options.model):
        response = process_meeting(transcript, provider=options.provider)
    elapsed = time.time() - start

    return PipelineRunResult(
        response=response,
        response_time_seconds=elapsed,
        provider=response.get("source") or options.provider,
        requested_model=options.model,
        actual_model=response.get("model_name"),
    )
