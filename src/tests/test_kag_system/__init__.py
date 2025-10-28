"""
KAG Testing Framework для Rebecca-Platform

Этот модуль предоставляет комплексную систему тестирования для компонентов KAG:
- KAGGraph - граф знаний
- ConceptExtractor - извлечение концептов  
- ContextEngine - контекстуальная интеграция
- Memory Manager - 6 слоев памяти (semantic, episodic, procedural, vault, security, vector_store)

Структура тестирования:
- Unit тесты: тестирование отдельных компонентов
- Integration тесты: тестирование взаимодействия между компонентами
- Performance тесты: тестирование производительности и масштабируемости
- E2E тесты: тестирование полного workflow системы
- Knowledge Quality тесты: тестирование качества и целостности знаний
"""

# Фикстуры и тестовые данные
from .fixtures.conftest import *
from .test_data.cognitive_biases import *

# Unit тесты
from .unit.test_kag_graph import *
from .unit.test_concept_extractor import *
from .unit.test_context_engine import *
from .unit.test_kag_graph_extended import *

# Integration тесты
from .integration.test_semantic_memory_integration import *
from .integration.test_episodic_memory_integration import *
from .integration.test_full_memory_stack_integration import *

# Knowledge Quality тесты (заглушки для будущего развития)
from .knowledge_quality import *

# Performance тесты (заглушки для будущего развития)  
from .performance import *

# E2E тесты (заглушки для будущего развития)
from .e2e import *

__version__ = "1.0.0"
__author__ = "Rebecca-Platform Team"
__all__ = [
    # Основные компоненты
    "base_config",
    "strict_config", 
    "memory_manager_mock",
    "kag_graph_mock",
    "concept_extractor_mock",
    "context_engine_mock",
    "cognitive_bias_data",
    "kag_test_data",
    
    # Тестовые данные
    "COGNITIVE_BIASES_DATASET",
    "COMPREHENSIVE_BIAS_SCENARIOS", 
    "KNOWLEDGE_QUALITY_DATASETS",
    
    # Integration тесты
    "SemanticMemoryIntegrationTestSuite",
    "EpisodicMemoryIntegrationTestSuite",
    "TestFullMemoryStackIntegration"
]