from pydantic import BaseModel, Field
from typing import List, Optional

class AnalyzeRequest(BaseModel):
    transcript: str = Field(..., max_length=20000)

class Task(BaseModel):
    task: str
    assignee: Optional[str] = None

class AnalyzeResponse(BaseModel):
    decisions: List[str]
    tasks: List[Task]
    responsible_people: List[str]