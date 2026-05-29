import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from app.api.evaluation.pipeline.config import DEFAULT_DATASET_DIR, DEFAULT_EXPECTED_DIR
from app.api.evaluation.pipeline.dataset_loader import TranscriptCase, load_transcripts
from app.api.evaluation.pipeline.silver_truth_prompt import SILVER_TRUTH_PROMPT
from app.core.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MAX_RETRIES,
    DEEPSEEK_TIMEOUT_SECONDS,
)
from app.services.llm_service import parse_json_response

DEFAULT_SILVER_MODEL = os.getenv("SILVER_TRUTH_MODEL", "deepseek-v4-pro")
DEFAULT_MAX_TOKENS = int(os.getenv("SILVER_TRUTH_MAX_TOKENS", "8000"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate persistent silver truth JSON files.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--expected-dir", type=Path, default=DEFAULT_EXPECTED_DIR)
    parser.add_argument("--pattern", default="*.txt", help="Dataset glob pattern.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default=DEFAULT_SILVER_MODEL)
    parser.add_argument("--thinking", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def _deepseek_silver_request(
    prompt: str,
    model: str,
    thinking: str,
    max_tokens: int,
    temperature: float,
) -> tuple[dict[str, Any], str]:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")

    url = f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "thinking": {"type": thinking},
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    last_error: Exception | None = None
    attempts = max(1, DEEPSEEK_MAX_RETRIES)
    with requests.Session() as session:
        for attempt in range(1, attempts + 1):
            try:
                response = session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=300  #DEEPSEEK_TIMEOUT_SECONDS,
                )
                if response.status_code >= 400:
                    raise RuntimeError(
                        f"DeepSeek returned {response.status_code}: {response.text[:500]}"
                    )

                data = response.json()
                content = data["choices"][0]["message"]["content"]
                parsed = parse_json_response(content)
                if not isinstance(parsed, dict):
                    raise RuntimeError("DeepSeek response was not valid JSON")
                return parsed, data.get("model") or model
            except Exception as exc:
                last_error = exc
                if attempt < attempts:
                    sleep_seconds = 0.75 * attempt
                    print(
                        f"  request failed: {exc}; retrying in {sleep_seconds:.1f}s "
                        f"({attempt}/{attempts})"
                    )
                    time.sleep(sleep_seconds)

    raise RuntimeError(f"DeepSeek silver truth request failed: {last_error}")


def _expected_path(expected_dir: Path, case: TranscriptCase) -> Path:
    return expected_dir / f"{case.file_name}.expected.json"


def _build_expected_payload(
    case: TranscriptCase,
    labels: dict[str, Any],
    model: str,
    requested_model: str,
    thinking: str,
    elapsed: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_transcript": case.path.name,
        "label_model": model,
        "requested_label_model": requested_model,
        "thinking": thinking,
        "label_created_at": datetime.now(timezone.utc).isoformat(),
        "generation_time_seconds": round(elapsed, 3),
        "labels": labels,
    }


def generate_for_case(
    case: TranscriptCase,
    expected_dir: Path,
    model: str,
    thinking: str,
    max_tokens: int,
    temperature: float,
    overwrite: bool,
) -> str:
    output_path = _expected_path(expected_dir, case)
    if output_path.exists() and not overwrite:
        print(f"  skipped existing: {output_path}")
        return "skipped"

    prompt = (
        SILVER_TRUTH_PROMPT
        .replace("{file_name}", case.path.name)
        .replace("{transcript}", case.transcript)
    )
    start = time.time()
    labels, actual_model = _deepseek_silver_request(
        prompt=prompt,
        model=model,
        thinking=thinking,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    elapsed = time.time() - start

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _build_expected_payload(
        case=case,
        labels=labels,
        model=actual_model,
        requested_model=model,
        thinking=thinking,
        elapsed=elapsed,
    )
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  saved: {output_path} ({elapsed:.2f}s)")
    return "generated"


def main() -> None:
    args = build_parser().parse_args()
    cases = load_transcripts(args.dataset_dir, pattern=args.pattern, limit=args.limit)
    args.expected_dir.mkdir(parents=True, exist_ok=True)

    print("Starting silver truth generation")
    print(f"Dataset: {args.dataset_dir} ({args.pattern}), files={len(cases)}")
    print(
        f"Expected dir: {args.expected_dir}, model={args.model}, "
        f"thinking={args.thinking}, overwrite={args.overwrite}"
    )

    total_start = time.time()
    stats = {"generated": 0, "skipped": 0, "failed": 0}
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.path.name} ({len(case.transcript)} chars)")
        try:
            status = generate_for_case(
                case=case,
                expected_dir=args.expected_dir,
                model=args.model,
                thinking=args.thinking,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                overwrite=args.overwrite,
            )
            stats[status] += 1
        except Exception as exc:
            stats["failed"] += 1
            print(f"  failed: {exc}")
            if args.fail_fast:
                raise

    total_elapsed = time.time() - total_start
    print(
        "Silver truth generation complete: "
        f"generated={stats['generated']}, skipped={stats['skipped']}, "
        f"failed={stats['failed']}, total_time={total_elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
