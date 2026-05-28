from app.api.evaluation.pipeline.runner import PipelineRunOptions, run_pipeline


def run_lite_pipeline(
    transcript: str,
    provider: str | None = None,
    model: str | None = None,
):
    result = run_pipeline(
        transcript,
        PipelineRunOptions(provider=provider, model=model),
    )

    return result.response, result.response_time_seconds
