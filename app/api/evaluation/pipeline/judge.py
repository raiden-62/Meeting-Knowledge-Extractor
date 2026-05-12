import json
from fastapi import HTTPException
from gigachat import GigaChat
from app.core.config import GIGACHAT_TOKEN

from app.api.evaluation.pipeline.judge_prompt import JUDGE_PROMPT

gigachat_judge = GigaChat(
    credentials=GIGACHAT_TOKEN,
    verify_ssl_certs = False,
    scope="GIGACHAT_API_PERS",
    #model="GigaChat-Pro"
)

def gigachat_request(request: str):
    print("Sending request to GigaChat Pro judge")
    try:
        response = gigachat_judge.chat(request)

        content = response.choices[0].message.content

        return content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    clean_response = clean_json_response(raw_response)
    print("---------Response thats supposed to be clean json-----------")
    print(clean_response)
    print("------------------------------------------------------------")
    return json.loads(clean_response)





