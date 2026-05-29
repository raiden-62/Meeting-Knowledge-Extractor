from app.api.evaluation.pipeline.config import EvaluationConfig
from app.api.evaluation.pipeline.dataset_loader import TranscriptCase
from app.api.evaluation.pipeline.deterministic_scorer import score_expected_vs_actual
from app.api.evaluation.pipeline.expected_loader import load_expected
from app.api.evaluation.pipeline.judge import default_evaluation, evaluate_response
from app.api.evaluation.pipeline.runner import PipelineRunOptions, run_pipeline


def skipped_evaluation() -> dict:
    evaluation = default_evaluation()
    evaluation["judge_skipped"] = True
    return evaluation


def failed_evaluation(error: Exception) -> dict:
    evaluation = default_evaluation()
    evaluation["error"] = str(error)
    return evaluation


def evaluate_case(case: TranscriptCase, config: EvaluationConfig) -> dict:
    expected = load_expected(config.expected_dir, case.file_name)
    if expected is None and config.require_expected:
        raise FileNotFoundError(f"Expected labels not found for {case.file_name}")

    run_result = run_pipeline(
        case.transcript,
        PipelineRunOptions(provider=config.provider, model=config.extractor_model),
    )

    deterministic = None
    if config.deterministic_scoring_enabled and expected is not None:
        deterministic = score_expected_vs_actual(expected.labels, run_result.response)

    if config.judge_enabled:
        evaluation = evaluate_response(
            case.transcript,
            run_result.response,
            provider=config.judge_provider,
            model=config.judge_model,
        )
    else:
        evaluation = skipped_evaluation()

    return {
        "file": case.file_name,
        "path": str(case.path),
        "response_time_seconds": run_result.response_time_seconds,
        "lite_response": run_result.response,
        "expected": {
            "path": str(expected.path),
            "source_transcript": expected.payload.get("source_transcript"),
            "label_model": expected.payload.get("label_model"),
        } if expected else None,
        "deterministic": deterministic,
        "evaluation": evaluation,
        "run": {
            "provider": run_result.provider,
            "requested_model": run_result.requested_model,
            "actual_model": run_result.actual_model,
            "judge_provider": config.judge_provider if config.judge_enabled else None,
            "judge_model": config.judge_model if config.judge_enabled else None,
        },
    }
