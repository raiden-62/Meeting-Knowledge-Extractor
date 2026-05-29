import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExpectedLabels:
    file_name: str
    path: Path
    payload: dict[str, Any]
    labels: dict[str, Any]


def expected_path_for(expected_dir: Path, file_name: str) -> Path:
    return expected_dir / f"{file_name}.expected.json"


def load_expected(expected_dir: Path, file_name: str) -> ExpectedLabels | None:
    path = expected_path_for(expected_dir, file_name)
    if not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload.get("labels") if isinstance(payload, dict) else None
    if not isinstance(labels, dict):
        labels = payload if isinstance(payload, dict) else {}

    return ExpectedLabels(
        file_name=file_name,
        path=path,
        payload=payload,
        labels=labels,
    )
