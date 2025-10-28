from typing import List, Literal

from pydantic import BaseModel

from .nodes import NodeBase


class ContextPack(BaseModel):
    version: Literal["v1"] = "v1"
    query: str
    nodes: List[NodeBase]
    policies_applied: List[str]
    rationale: List[str]
    warnings: List[str] = []
    token_budget: int = 0
