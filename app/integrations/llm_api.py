from fastapi import HTTPException
from gigachat import GigaChat
from app.schemas.schemas import ChatRequest
from app.core.config import GIGACHAT_TOKEN
from app.core.logger import logger

gigachat = GigaChat(
    credentials=GIGACHAT_TOKEN,
    verify_ssl_certs = False,
    scope="GIGACHAT_API_PERS"
)

def gigachat_request(request: str):
    try:
        response = gigachat.chat(request)

        content = response.choices[0].message.content

        return {"answer": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))