"""Educator agent scaffold for Rebecca-Platform."""

from .knowledge_updater import KnowledgeUpdater
from .curriculum_manager import CurriculumManager
from .semantic_mapper import SemanticMapper
from .educator_main import EducatorAgent

__all__ = [
    "KnowledgeUpdater",
    "CurriculumManager",
    "SemanticMapper",
    "EducatorAgent",
]
