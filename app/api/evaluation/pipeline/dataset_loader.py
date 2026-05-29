from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TranscriptCase:
    file_name: str
    path: Path
    transcript: str


def load_transcripts(
    dataset_dir: Path,
    pattern: str = "*.txt",
    limit: int | None = None,
) -> list[TranscriptCase]:
    paths = sorted(dataset_dir.glob(pattern))
    if limit is not None:
        paths = paths[: max(limit, 0)]

    transcripts: list[TranscriptCase] = []
    for file_path in paths:
        if not file_path.is_file():
            continue
        transcripts.append(
            TranscriptCase(
                file_name=file_path.stem,
                path=file_path,
                transcript=file_path.read_text(encoding="utf-8"),
            )
        )

    return transcripts
