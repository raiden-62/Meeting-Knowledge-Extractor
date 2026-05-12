import json
from pathlib import Path

from app.api.evaluation.pipeline.lite_runner import run_lite_pipeline
from app.api.evaluation.pipeline.judge import evaluate_response
from app.api.evaluation.pipeline.report_generator import (
    save_detailed_report,
    generate_summary_report
)

CURRENT_DIR = Path(__file__).parent
DATASET_DIR = CURRENT_DIR.parent / "dataset"


def load_transcripts():
    transcripts = []

    for file_path in DATASET_DIR.glob("*.txt"):
        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:
            transcripts.append(
                {
                    "file_name": file_path.stem,
                    "transcript": f.read()
                }
            )

    return transcripts


def main():
    transcripts = load_transcripts()

    all_results = []

    print("Starting evaluation")

    for item in transcripts:
        file_name = item["file_name"]
        transcript = item["transcript"]

        print(f"Processing {file_name}")

        lite_response, response_time = run_lite_pipeline(
            transcript
        )

        evaluation = evaluate_response(
            transcript,
            lite_response
        )

        detailed_result = {
            "file": file_name,
            "response_time_seconds": response_time,
            "lite_response": lite_response,
            "evaluation": evaluation
        }

        all_results.append(detailed_result)

        save_detailed_report(
            file_name=file_name,
            transcript=transcript,
            lite_response=lite_response,
            evaluation=evaluation,
            response_time=response_time
        )


    generate_summary_report(all_results)

    print("Evaluation complete")


if __name__ == "__main__":
    main()