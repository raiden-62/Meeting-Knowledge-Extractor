from dataclasses import asdict, dataclass
from pathlib import Path

from app.core.config import LLM_PROVIDER, LLM_PROVIDERS


CURRENT_DIR = Path(__file__).parent
DEFAULT_DATASET_DIR = CURRENT_DIR.parent / "dataset"
DEFAULT_EXPECTED_DIR = CURRENT_DIR.parent / "expected"
DEFAULT_REPORTS_DIR = CURRENT_DIR.parent / "reports"


@dataclass(frozen=True)
class EvaluationConfig:
    dataset_dir: Path = DEFAULT_DATASET_DIR
    expected_dir: Path = DEFAULT_EXPECTED_DIR
    output_dir: Path = DEFAULT_REPORTS_DIR
    dataset_pattern: str = "*.txt"
    limit: int | None = None
    provider: str = LLM_PROVIDER
    model: str | None = None
    judge_enabled: bool = True
    deterministic_scoring_enabled: bool = True
    require_expected: bool = False
    fail_under: float | None = None
    include_transcript_in_report: bool = True
    continue_on_error: bool = True

    def to_report_dict(self) -> dict:
        data = asdict(self)
        data["dataset_dir"] = str(self.dataset_dir)
        data["expected_dir"] = str(self.expected_dir)
        data["output_dir"] = str(self.output_dir)
        return data


def normalize_provider(provider: str | None) -> str:
    selected = (provider or LLM_PROVIDER).strip().lower()
    if selected not in LLM_PROVIDERS:
        allowed = ", ".join(LLM_PROVIDERS)
        raise ValueError(f"Provider must be one of: {allowed}")
    return selected
