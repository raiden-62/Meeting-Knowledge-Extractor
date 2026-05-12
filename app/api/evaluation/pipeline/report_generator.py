import json
from pathlib import Path


def save_detailed_report(
    file_name: str,
    transcript: str,
    lite_response: dict,
    evaluation: dict,
    response_time: float
):
    report = {
        "file": file_name,
        "response_time_seconds": response_time,
        "transcript": transcript,
        "lite_response": lite_response,
        "evaluation": evaluation
    }

    current_dir = Path(__file__).parent
    reports_dir = current_dir.parent / "reports" / "detailed"

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


def generate_summary_report(results: list[dict]):
    total_files = len(results)

    avg_response_time = sum(
        r["response_time_seconds"]
        for r in results
    ) / total_files
    avg_response_time = round(avg_response_time, 2)

    avg_clarity = sum(
        r["evaluation"]["clarity_rating"]
        for r in results
    ) / total_files

    avg_score = sum(
        r["evaluation"]["overall_score"]
        for r in results
    ) / total_files

    total_missed_tasks = sum(
        r["evaluation"]["missed_tasks"]
        for r in results
    )

    total_missed_decisions = sum(
        r["evaluation"]["missed_decisions"]
        for r in results
    )

    total_misattributed = sum(
        r["evaluation"]["misattributed_tasks"]
        for r in results
    )

    summary = {
        "total_files": total_files,
        "average_response_time": avg_response_time,
        "average_clarity_rating": avg_clarity,
        "average_overall_score": avg_score,
        "total_missed_tasks": total_missed_tasks,
        "total_missed_decisions": total_missed_decisions,
        "total_misattributed_tasks": total_misattributed
    }

    current_dir = Path(__file__).parent

    reports_dir = current_dir.parent / "reports"

    reports_dir.mkdir(
        parents=True,
        exist_ok=True
    )

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