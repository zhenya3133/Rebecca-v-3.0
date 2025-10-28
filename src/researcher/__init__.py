"""Researcher agent scaffold for Rebecca-Platform."""

from .trend_scanner import TrendScanner
from .data_importer import DataImporter
from .source_manager import SourceManager
from .researcher_main import ResearcherAgent

__all__ = [
    "TrendScanner",
    "DataImporter",
    "SourceManager",
    "ResearcherAgent",
]
