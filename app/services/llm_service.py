import requests
import json

from app.core.logger import logger
from app.core.config import OLLAMA_URL, OLLAMA_MODEL


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
    prompt = f"""
    Return ONLY valid JSON in this schema:

    {{
      "decisions": ["string"],
      "tasks": [
        {{
          "task": "string",
          "assignee": "string|null"
        }}
      ],
      "responsible_people": ["string"]
    }}

    Input:
    {raw_data}
    """

    try:
        logger.info("Sending request to Ollama")

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )

        response.raise_for_status()


        result = response.json()["response"]

        logger.info("Received response from Ollama")

        result = clean_llama_json(result)

        return json.loads(result)

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