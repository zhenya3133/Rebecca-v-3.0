"""
Integration тесты для KAG системы Rebecca-Platform
"""

__version__ = "1.0.0"
__author__ = "KAG Testing Framework"

# Экспорты основных тестовых классов
from .test_semantic_memory_integration import SemanticMemoryIntegrationTestSuite
from .test_episodic_memory_integration import EpisodicMemoryIntegrationTestSuite  
from .test_full_memory_stack_integration import TestFullMemoryStackIntegration

__all__ = [
    "SemanticMemoryIntegrationTestSuite",
    "EpisodicMemoryIntegrationTestSuite", 
    "TestFullMemoryStackIntegration"
]