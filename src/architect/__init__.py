"""Architect agent scaffold for Rebecca-Platform.

Exports placeholders inspired by CAMEL-AI OWL's graph-based planning and
NirDiamant's production-oriented agent pipelines. Concrete logic will be wired
after orchestrator integration.
"""

from .architectural_planner import ArchitecturalPlanner
from .diagram_generator import DiagramGenerator
from .standards_manager import StandardsManager

__all__ = [
    "ArchitecturalPlanner",
    "DiagramGenerator",
    "StandardsManager",
]
