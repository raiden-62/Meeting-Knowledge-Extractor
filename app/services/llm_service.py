import requests
import json


OLLAMA_URL = "http://localhost:11434/api/generate"


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

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "qwen2.5:7b",
            "prompt": prompt,
            "stream": False
        }
    )


    result = response.json()["response"]
    result = clean_llama_json(result)

    return json.loads(result)