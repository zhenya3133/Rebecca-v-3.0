"""
Rebecca Meta-Agent Package

Мета-агент для интеллектуального управления агентной экосистемой Rebecca Platform.
Обеспечивает координацию между специализированными агентами для решения сложных задач.
"""

from .meta_agent import RebeccaMetaAgent
from .meta_agent import (
    TaskPlan, 
    AgentAssignment, 
    PlaybookStep, 
    ResourceAllocation,
    MetaAgentConfig
)

__version__ = "1.0.0"
__all__ = [
    "RebeccaMetaAgent",
    "TaskPlan",
    "AgentAssignment", 
    "PlaybookStep",
    "ResourceAllocation",
    "MetaAgentConfig"
]