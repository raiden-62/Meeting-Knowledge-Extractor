import requests
import json

from app.core.logger import logger
from app.integrations.llm_api import gigachat_request


def clean_llama_json(llama_json):
    result = llama_json.strip()

    if result.startswith('```json'):
        result = result[7:]  # Remove ```json
    elif result.startswith('```'):
        result = result[3:]  # Remove ```

    if result.endswith('```'):
        result = result[:-3]  # Remove trailing ```

    return result

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
        #response = clean_llama_json(response)

        return json.loads(answer)

    except requests.exceptions.Timeout:
        logger.error("Ollama request timed out")

    except requests.exceptions.ConnectionError:
        logger.error("Could not connect to Ollama")

    except requests.exceptions.HTTPError as e:
        logger.error(f"Ollama HTTP error: {e}")

    except json.JSONDecodeError:
        logger.error("Failed to parse LLM response as JSON")

    except Exception as e:
        logger.error(f"Unexpected error: {e}")

    return {
        "decisions": [],
        "tasks": [],
        "responsible_people": []
    }