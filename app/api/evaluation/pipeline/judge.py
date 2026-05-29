import json
import time

from app.api.evaluation.pipeline.config import normalize_provider
from app.api.evaluation.pipeline.judge_prompt import JUDGE_PROMPT
from app.api.evaluation.pipeline.runner import temporary_model_override
from app.core.logger import logger
from app.integrations.llm_api import deepseek_request, gigachat_request


def judge_request(
    request: str,
    provider: str | None,
    model: str | None = None,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
) -> str:
    selected_provider = normalize_provider(provider)
    print(
        "Sending request to "
        f"{selected_provider} judge"
        f"{f' ({model})' if model else ''}"
    )

    last_error: Exception | None = None
    with temporary_model_override(selected_provider, model):
        for attempt in range(1, max_retries + 1):
            try:
                if selected_provider == "gigachat":
                    return gigachat_request(request, max_retries=1)["answer"]
                if selected_provider == "deepseek":
                    return deepseek_request(request, max_retries=1)["answer"]
            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    logger.warning(
                        "%s judge request failed: %s. Retrying (%s/%s)",
                        selected_provider,
                        exc,
                        attempt,
                        max_retries,
                    )
                    time.sleep(backoff_seconds * attempt)

    raise RuntimeError(f"{selected_provider} judge request failed: {last_error}")


def extract_json_block(text: str) -> str:
    if not text:
        return ""

    cleaned = text.strip()

    if cleaned.startswith("```"):
        newline_index = cleaned.find("\n")
        if newline_index != -1:
            cleaned = cleaned[newline_index + 1:]
        end_fence = cleaned.rfind("```")
        if end_fence != -1:
            cleaned = cleaned[:end_fence]

    cleaned = cleaned.replace("вЂў", "")

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start:end + 1]

    return cleaned.strip()


def parse_json_response(text: str) -> dict | None:
    if not text:
        return None

    for candidate in (text, extract_json_block(text)):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return None


def default_evaluation() -> dict:
    return {
        "missed_tasks": 0,
        "missed_task_updates": 0,
        "missed_decisions": 0,
        "missed_risks": 0,
        "assignee_errors": 0,
        "status_errors": 0,
        "priority_errors": 0,
        "due_date_errors": 0,
        "hallucinated_items": 0,
        "clarity_rating": 0,
        "overall_score": 0,
        "comments": "",
    }


def evaluate_response(
    transcript: str,
    lite_response: dict,
    provider: str | None = None,
    model: str | None = None,
) -> dict:
    prompt = JUDGE_PROMPT.format(
        transcript=transcript,
        response=json.dumps(
            lite_response,
            ensure_ascii=False,
            indent=2,
        ),
    )

    selected_provider = normalize_provider(provider)
    raw_response = judge_request(prompt, provider=selected_provider, model=model)
    parsed = parse_json_response(raw_response)

    if not isinstance(parsed, dict):
        logger.error("Judge response is not valid JSON. Using defaults.")
        return default_evaluation()

    required_keys = {
        "missed_tasks",
        "missed_task_updates",
        "missed_decisions",
        "missed_risks",
        "assignee_errors",
        "status_errors",
        "priority_errors",
        "due_date_errors",
        "hallucinated_items",
        "clarity_rating",
        "overall_score",
        "comments",
    }

    if not required_keys.issubset(parsed.keys()):
        logger.error("Judge response JSON schema mismatch. Using defaults.")
        return default_evaluation()

    parsed["judge_provider"] = selected_provider
    parsed["judge_model"] = model
    return parsed
