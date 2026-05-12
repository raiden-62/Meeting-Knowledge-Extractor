from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class AnalyzeRequest(BaseModel):
    transcript: str = Field(..., max_length=20000)

class Task(BaseModel):
    task: str
    assignee: Optional[str] = None

class AnalyzeResponse(BaseModel):
    decisions: List[str]
    people: Dict[str, List[str]]

class ChatRequest(BaseModel):
    message: str