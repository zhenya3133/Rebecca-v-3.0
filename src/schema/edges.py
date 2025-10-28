from datetime import datetime
from typing import Dict, Literal, Optional

from pydantic import BaseModel


class Edge(BaseModel):
    id: str
    source: str
    target: str
    etype: str
    created_at: datetime
    updated_at: datetime
    confidence: float = 0.7
    attrs: Dict[str, str] = {}
