import argparse
from pathlib import Path

from app.api.evaluation.pipeline.config import (
    DEFAULT_DATASET_DIR,
    DEFAULT_EXPECTED_DIR,
    DEFAULT_REPORTS_DIR,
    EvaluationConfig,
    normalize_provider,
)
from app.api.evaluation.pipeline.dataset_loader import load_transcripts
from app.api.evaluation.pipeline.evaluator import evaluate_case, failed_evaluation
from app.api.evaluation.pipeline.report_generator import (
    generate_summary_report,
    save_detailed_report,
)
from app.core.config import LLM_PROVIDERS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run transcript extraction evaluation.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--expected-dir", type=Path, default=DEFAULT_EXPECTED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--pattern", default="*.txt", help="Dataset glob pattern.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--provider", choices=LLM_PROVIDERS, default=None)
    parser.add_argument("--extractor-model", default=None, help="Override extractor model for this evaluation run.")
    parser.add_argument(
        "--model",
        dest="legacy_model",
        default=None,
        help="Deprecated alias for --extractor-model.",
    )
    parser.add_argument("--skip-judge", action="store_true", help="Run extraction only and skip LLM judge.")
    parser.add_argument(
        "--judge-provider",
        choices=LLM_PROVIDERS,
        default=None,
        help="Judge provider. Defaults to the extractor provider.",
    )
    parser.add_argument("--judge-model", default=None, help="Override judge model for this evaluation run.")
    parser.add_argument("--skip-deterministic", action="store_true", help="Skip expected JSON scoring.")
    parser.add_argument(
        "--require-expected",
        action="store_true",
        help="Fail a transcript when its expected JSON file is missing.",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="Exit with code 1 if average deterministic score is below this value.",
    )
    parser.add_argument(
        "--no-transcript-in-report",
        action="store_true",
        help="Do not store full transcript text in detailed reports.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first transcript error.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> EvaluationConfig:
    provider = normalize_provider(args.provider)
    judge_provider = normalize_provider(args.judge_provider or provider)
    extractor_model = args.extractor_model or args.legacy_model
    return EvaluationConfig(
        dataset_dir=args.dataset_dir,
        expected_dir=args.expected_dir,
        output_dir=args.output_dir,
        dataset_pattern=args.pattern,
        limit=args.limit,
        provider=provider,
        extractor_model=extractor_model,
        judge_enabled=not args.skip_judge,
        judge_provider=judge_provider,
        judge_model=args.judge_model,
        deterministic_scoring_enabled=not args.skip_deterministic,
        require_expected=args.require_expected,
        fail_under=args.fail_under,
        include_transcript_in_report=not args.no_transcript_in_report,
        continue_on_error=not args.fail_fast,
    )


def run_evaluation(config: EvaluationConfig) -> list[dict]:
    transcripts = load_transcripts(
        config.dataset_dir,
        pattern=config.dataset_pattern,
        limit=config.limit,
    )
    all_results: list[dict] = []

    print("Starting evaluation")
    print(
        f"Dataset: {config.dataset_dir} ({config.dataset_pattern}), "
        f"expected={config.expected_dir}, "
        f"files={len(transcripts)}, provider={config.provider}, "
        f"extractor_model={config.extractor_model or 'env-default'}, "
        f"deterministic={'on' if config.deterministic_scoring_enabled else 'off'}, "
        f"judge={'on' if config.judge_enabled else 'off'}, "
        f"judge_provider={config.judge_provider if config.judge_enabled else 'none'}, "
        f"judge_model={config.judge_model or 'env-default' if config.judge_enabled else 'none'}"
    )

    for case in transcripts:
        print(f"Processing {case.file_name}")

        try:
            detailed_result = evaluate_case(case, config)
        except Exception as exc:
            if not config.continue_on_error:
                raise
            detailed_result = {
                "file": case.file_name,
                "path": str(case.path),
                "response_time_seconds": 0,
                "lite_response": {},
                "expected": None,
                "deterministic": None,
                "evaluation": failed_evaluation(exc),
                "run": {
                    "provider": config.provider,
                    "requested_model": config.extractor_model,
                    "actual_model": None,
                    "judge_provider": config.judge_provider if config.judge_enabled else None,
                    "judge_model": config.judge_model if config.judge_enabled else None,
                },
            }

        all_results.append(detailed_result)
        save_detailed_report(
            file_name=case.file_name,
            transcript=case.transcript,
            lite_response=detailed_result["lite_response"],
            evaluation=detailed_result["evaluation"],
            response_time=detailed_result["response_time_seconds"],
            output_dir=config.output_dir,
            run_metadata=detailed_result["run"],
            include_transcript=config.include_transcript_in_report,
        )

    generate_summary_report(
        all_results,
        output_dir=config.output_dir,
        run_config=config.to_report_dict(),
    )
    return all_results


def main() -> None:
    parser = build_parser()
    config = config_from_args(parser.parse_args())
    results = run_evaluation(config)
    if config.fail_under is not None:
        scores = [
            result["deterministic"]["overall_score"]
            for result in results
            if isinstance(result.get("deterministic"), dict)
        ]
        if scores:
            average = sum(scores) / len(scores)
            if average < config.fail_under:
                raise SystemExit(
                    f"Average deterministic score {average:.4f} is below {config.fail_under:.4f}"
                )
    print("Evaluation complete")


if __name__ == "__main__":
    main()
