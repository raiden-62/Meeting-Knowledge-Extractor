import json

from app.api.evaluation.pipeline.report_generator import generate_summary_report


def test_summary_aggregates_field_accuracy(tmp_path):
    results = [
        {
            "response_time_seconds": 1.0,
            "deterministic": {
                "overall_score": 0.75,
                "categories": {
                    "tasks": {
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1": 1.0,
                        "expected": 2,
                        "actual": 2,
                        "matched": 2,
                        "missed": 0,
                        "extra": 0,
                        "field_totals": {"due_date": 2},
                        "field_matches": {"due_date": 1},
                    }
                },
            },
            "evaluation": {},
        }
    ]

    summary = generate_summary_report(results, output_dir=tmp_path)

    assert summary["deterministic"]["categories"]["tasks"]["fields"]["due_date"] == {
        "matched": 1,
        "total": 2,
        "accuracy": 0.5,
    }
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8")) == summary
