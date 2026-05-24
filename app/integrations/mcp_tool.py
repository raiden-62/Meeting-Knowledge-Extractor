import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.core.config import LLM_PROVIDERS, MAX_TRANSCRIPT_CHARS
from app.schemas.schemas import AnalyzeRequest
from app.services.meeting_pipeline import process_meeting

TOOL_SCHEMA = {
    "name": "extract_meeting_knowledge",
    "description": "Extract summary, decisions, tasks, responsible people and risks from a meeting transcript",
    "input_schema": {
        "type": "object",
        "properties": {
            "transcript": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_TRANSCRIPT_CHARS,
                "description": "Meeting transcript text up to 20,000 characters",
            },
            "provider": {
                "type": "string",
                "enum": list(LLM_PROVIDERS),
                "description": "Optional LLM provider: gigachat or deepseek",
            }
        },
        "required": ["transcript"],
    },
}


def execute_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    request = AnalyzeRequest.model_validate(arguments)
    return process_meeting(request.transcript, provider=request.provider)


def _load_payload(stdin_text: str, transcript_arg: str | None) -> dict[str, Any]:
    if transcript_arg:
        return {"transcript": transcript_arg}

    stdin_text = stdin_text.strip()
    if not stdin_text:
        raise ValueError("Provide JSON on stdin or pass --transcript")

    try:
        payload = json.loads(stdin_text)
    except json.JSONDecodeError:
        payload = {"transcript": stdin_text}

    if not isinstance(payload, dict):
        raise ValueError("MCP payload must be a JSON object")

    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Meeting Knowledge Extractor MCP tool")
    parser.add_argument("--transcript", help="Transcript text. If omitted, stdin is used.")
    parser.add_argument("--provider", choices=LLM_PROVIDERS, help="LLM provider to use.")
    args = parser.parse_args(argv)

    try:
        payload = _load_payload(sys.stdin.read(), args.transcript)
        if args.provider:
            payload["provider"] = args.provider
        result = execute_tool(payload)
    except (ValidationError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    except Exception as exc:
        print(json.dumps({"error": f"Tool execution failed: {exc}"}, ensure_ascii=False), file=sys.stderr)
        return 2

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
