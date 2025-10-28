"""Полноценный Memory Manager для Rebecca-Platform.

Предоставляет комплексную систему управления памятью с 6 слоями,
интеграцией с векторными хранилищами и отслеживанием изменений архитектуры.
"""

# Основные компоненты контекста памяти
from .memory_context import (
    MemoryContext,
    MemoryEntry,
    CORE,
    EPISODIC,
    SEMANTIC,
    PROCEDURAL,
    VAULT,
    SECURITY,
    MEMORY_LAYER_TYPES
)

# Векторное хранилище
from .vector_store_client import VectorStoreClient, VectorStoreConfig, VectorItem

# Полноценный MemoryManager
from .memory_manager import (
    MemoryManager,
    AdaptiveBlueprintTracker,
    MemoryLayerFactory,
    MemoryCache,
    LayerStats,
    CacheEntry,
    create_memory_manager,
    quick_memory_test
)

# Расширенный Blueprint Tracker
from .adaptive_blueprint import (
    BlueprintVersion,
    ResourceLink,
    ChangeAnalysis
)

# Слои памяти
from .core_memory import CoreMemory
from .episodic_memory import EpisodicMemory
from .semantic_memory import SemanticMemory
from .procedural_memory import ProceduralMemory
from .vault_memory import VaultMemory
from .security_memory import SecurityMemory

# Утилиты
import logging

def setup_logger(name: str) -> logging.Logger:
    """Создает и настраивает логгер."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Создаем консольный обработчик
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Добавляем обработчик к логгеру
    if not logger.handlers:
        logger.addHandler(handler)
    
    return logger

# Устаревшие интерфейсы (для обратной совместимости)
from .memory_manager_interface import (
    IMemoryManager,
    MemoryLayer,
    MemoryItem,
    MemoryFilter,
    LayerFactory,
    PerformanceOptimizer
)

# Фасад для обратной совместимости
from .memory_manager_main import MemoryManager as MemoryManagerFacade

# Документы
from .document_ingest import DocumentIngestor

__all__ = [
    # Контекст памяти
    "MemoryContext",
    "MemoryEntry", 
    "CORE",
    "EPISODIC",
    "SEMANTIC",
    "PROCEDURAL",
    "VAULT",
    "SECURITY",
    "MEMORY_LAYER_TYPES",
    
    # Векторное хранилище
    "VectorStoreClient",
    "VectorStoreConfig",
    "VectorItem",
    
    # Основной MemoryManager
    "MemoryManager",
    "AdaptiveBlueprintTracker",
    "MemoryLayerFactory", 
    "MemoryCache",
    "LayerStats",
    "CacheEntry",
    "create_memory_manager",
    "quick_memory_test",
    
    # Blueprint Tracker
    "BlueprintVersion",
    "ResourceLink",
    "ChangeAnalysis",
    
    # Слои памяти
    "CoreMemory",
    "EpisodicMemory", 
    "SemanticMemory",
    "ProceduralMemory",
    "VaultMemory",
    "SecurityMemory",
    
    # Устаревшие интерфейсы (для обратной совместимости)
    "IMemoryManager",
    "MemoryLayer",
    "MemoryItem", 
    "MemoryFilter",
    "LayerFactory",
    "PerformanceOptimizer",
    
    # Фасады
    "MemoryManagerFacade",
    "DocumentIngestor",
    
    # Утилиты
    "setup_logger",
]

# Версия пакета
__version__ = "2.0.0"

# Метаинформация
__author__ = "Rebecca Platform Team"
__description__ = "Полноценный Memory Manager с 6 слоями памяти и AdaptiveBlueprintTracker"
