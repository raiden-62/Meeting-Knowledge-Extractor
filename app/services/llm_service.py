import requests
import json

from app.core.logger import logger
from app.integrations.llm_api import gigachat_request


def clean_json_response(text: str) -> str:
    text = text.strip()

    # Find the first code block start marker
    start_marker = None
    if text.startswith("```json"):
        start_marker = "```json"
    elif text.startswith("```"):
        start_marker = "```"

    if start_marker:
        # Remove the starting marker
        text = text[len(start_marker):].lstrip()

        # Find the end of the JSON block (second ```)
        end_marker_index = text.find("```")
        if end_marker_index != -1:
            text = text[:end_marker_index]  # Cut at the closing ```

    # Remove stray bullet characters
    text = text.replace("•", "")

    return text.strip()

def extract_output(raw_data: str) -> dict:
    prompt = """
    Проанализируй этот текст и извлеки задачи, решения и людей ответственных за задачи.
    Верни свой ответ ТОЛЬКО в виде json такого вида:
    {
      "decisions": [],
      "people": {
        "name1": ["task1", "task2"],
        "name2": ["task3", "task4"]
      }
    }

    Текст:
    """
    prompt += raw_data


    try:
        logger.info("Sending request to GigaChat")

        result = gigachat_request(prompt)

        logger.info("Received response from GigaChat")

        answer = result.get("answer")
        clean_answer = clean_json_response(answer)

        return json.loads(clean_answer)

    except requests.exceptions.Timeout:
        logger.error("GigaChat request timed out")

    except requests.exceptions.ConnectionError:
        logger.error("Could not connect to GigaChat")

    except requests.exceptions.HTTPError as e:
        logger.error(f"GigaChat HTTP error: {e}")

    except json.JSONDecodeError:
        logger.error("Failed to parse LLM response as JSON")

    except Exception as e:
        logger.error(f"Unexpected error: {e}")

    return {
        "decisions": [],
        "tasks": [],
        "responsible_people": []
    }