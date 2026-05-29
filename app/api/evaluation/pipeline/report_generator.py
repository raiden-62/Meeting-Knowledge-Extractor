import json
from pathlib import Path


def _average(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def save_detailed_report(
    file_name: str,
    transcript: str,
    lite_response: dict,
    evaluation: dict,
    response_time: float,
    output_dir: Path | None = None,
    run_metadata: dict | None = None,
    include_transcript: bool = True,
):
    report = {
        "file": file_name,
        "response_time_seconds": response_time,
        "lite_response": lite_response,
        "evaluation": evaluation,
        "run": run_metadata or {},
    }
    if include_transcript:
        report["transcript"] = transcript

    current_dir = Path(__file__).parent
    reports_dir = (output_dir or current_dir.parent / "reports") / "detailed"

    reports_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path = reports_dir / f"{file_name}.json"

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2
        )


def generate_summary_report(
    results: list[dict],
    output_dir: Path | None = None,
    run_config: dict | None = None,
):
    total_files = len(results)
    current_dir = Path(__file__).parent
    reports_dir = output_dir or current_dir.parent / "reports"
    reports_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if total_files == 0:
        summary = {
            "total_files": 0,
            "config": run_config or {},
            "results": [],
        }
        summary_path = reports_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary

    avg_response_time = round(
        sum(r["response_time_seconds"] for r in results) / total_files,
        2,
    )
    judge_results = [r.get("evaluation", {}) for r in results if isinstance(r.get("evaluation"), dict)]
    deterministic_results = [
        r.get("deterministic")
        for r in results
        if isinstance(r.get("deterministic"), dict)
    ]
    deterministic_scores = [
        float(item["overall_score"])
        for item in deterministic_results
        if item.get("overall_score") is not None
    ]
    category_summary: dict[str, dict] = {}
    for category in ("tasks", "task_updates", "decisions", "risks", "people"):
        values = [
            item.get("categories", {}).get(category, {})
            for item in deterministic_results
        ]
        category_summary[category] = {
            "average_precision": _average([float(value["precision"]) for value in values if value.get("precision") is not None]),
            "average_recall": _average([float(value["recall"]) for value in values if value.get("recall") is not None]),
            "average_f1": _average([float(value["f1"]) for value in values if value.get("f1") is not None]),
            "total_expected": sum(int(value.get("expected", 0)) for value in values),
            "total_actual": sum(int(value.get("actual", 0)) for value in values),
            "total_matched": sum(int(value.get("matched", 0)) for value in values),
            "total_missed": sum(int(value.get("missed", 0)) for value in values),
            "total_extra": sum(int(value.get("extra", 0)) for value in values),
        }

    summary = {
        "total_files": total_files,
        "config": run_config or {},
        "average_response_time": avg_response_time,
        "deterministic": {
            "scored_files": len(deterministic_results),
            "average_overall_score": _average(deterministic_scores),
            "categories": category_summary,
        },
        "judge": {
            "average_clarity_rating": _average(
                [float(item["clarity_rating"]) for item in judge_results if item.get("clarity_rating") is not None]
            ),
            "average_overall_score": _average(
                [float(item["overall_score"]) for item in judge_results if item.get("overall_score") is not None]
            ),
            "total_missed_tasks": sum(int(item.get("missed_tasks", 0)) for item in judge_results),
            "total_missed_task_updates": sum(int(item.get("missed_task_updates", 0)) for item in judge_results),
            "total_missed_decisions": sum(int(item.get("missed_decisions", 0)) for item in judge_results),
            "total_missed_risks": sum(int(item.get("missed_risks", 0)) for item in judge_results),
            "total_assignee_errors": sum(int(item.get("assignee_errors", 0)) for item in judge_results),
            "total_status_errors": sum(int(item.get("status_errors", 0)) for item in judge_results),
            "total_priority_errors": sum(int(item.get("priority_errors", 0)) for item in judge_results),
            "total_due_date_errors": sum(int(item.get("due_date_errors", 0)) for item in judge_results),
            "total_hallucinated_items": sum(int(item.get("hallucinated_items", 0)) for item in judge_results),
        },
    }

    summary_path = reports_dir / "summary.json"

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2
        )
    return summary
