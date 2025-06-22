# schemas.py
from pydantic import BaseModel
from typing import Optional

class AgentResponse(BaseModel):
    answer: str
    tool_used: str
    confidence: float
