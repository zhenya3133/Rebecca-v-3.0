from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


Privacy = Literal["public", "team", "owner"]


class EvidenceRef(BaseModel):
    kind: Literal["event", "artifact"]
    id: str
    confidence: float = 0.7


class NodeBase(BaseModel):
    id: str
    ntype: str
    created_at: datetime
    updated_at: datetime
    owner: str
    privacy: Privacy = "owner"
    version: str = "1.0.0"
    confidence: float = 0.7
    embedding: Optional[List[float]] = None
    attrs: Dict[str, str] = {}


class Event(NodeBase):
    ntype: Literal["Event"] = "Event"
    t_start: datetime
    t_end: Optional[datetime] = None
    actors: List[str] = []
    channel: Optional[str] = None
    raw_ref: Optional[str] = None


class Fact(NodeBase):
    ntype: Literal["Fact"] = "Fact"
    subject: str
    predicate: str
    object: str
    evidence: List[EvidenceRef] = []


class Procedure(NodeBase):
    ntype: Literal["Procedure"] = "Procedure"
    preconds: List[str] = []
    steps: List[str] = []
    postconds: List[str] = []
    metrics: Dict[str, float] = {}
