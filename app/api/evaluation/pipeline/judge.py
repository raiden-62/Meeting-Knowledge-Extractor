import json
import time
from fastapi import HTTPException
from gigachat import GigaChat
from app.core.config import GIGACHAT_TOKEN
from app.core.logger import logger

from app.api.evaluation.pipeline.judge_prompt import JUDGE_PROMPT

gigachat_judge: GigaChat | None = None


def get_gigachat_judge() -> GigaChat:
    global gigachat_judge

    if not GIGACHAT_TOKEN:
        raise RuntimeError("GIGACHAT_TOKEN is not configured")

    if gigachat_judge is None:
        gigachat_judge = GigaChat(
            credentials=GIGACHAT_TOKEN,
            verify_ssl_certs=False,
            scope="GIGACHAT_API_PERS",
            # model="GigaChat-Pro" uses too many tokens out of 50k
        )

    return gigachat_judge

def gigachat_request(
    request: str,
    max_retries: int = 3,
    backoff_seconds: float = 1.0
):
    print("Sending request to GigaChat judge")

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = get_gigachat_judge().chat(request)

            content = response.choices[0].message.content

            return content
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                logger.warning(
                    "GigaChat judge request failed: %s. Retrying (%s/%s)",
                    e,
                    attempt,
                    max_retries
                )
                time.sleep(backoff_seconds * attempt)
            else:
                raise HTTPException(status_code=500, detail=str(e))

    raise HTTPException(status_code=500, detail=str(last_error))


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

    cleaned = cleaned.replace("•", "")

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
        "missed_decisions": 0,
        "misattributed_tasks": 0,
        "clarity_rating": 0,
        "overall_score": 0
    }

def evaluate_response(
    transcript: str,
    lite_response: dict
):
    prompt = JUDGE_PROMPT.format(
        transcript=transcript,
        response=json.dumps(
            lite_response,
            ensure_ascii=False,
            indent=2
        )
    )

    raw_response = gigachat_request(prompt)
    parsed = parse_json_response(raw_response)

    if not isinstance(parsed, dict):
        logger.error("Judge response is not valid JSON. Using defaults.")
        return default_evaluation()

    required_keys = {
        "missed_tasks",
        "missed_decisions",
        "misattributed_tasks",
        "clarity_rating",
        "overall_score"
    }

    if not required_keys.issubset(parsed.keys()):
        logger.error("Judge response JSON schema mismatch. Using defaults.")
        return default_evaluation()

    return parsed





