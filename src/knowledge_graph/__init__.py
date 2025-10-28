"""
Модуль Knowledge Graph для Rebecca Platform
Реализует систему управления концептуальными графами знаний.

Ключевые компоненты:
- KAGGraph: Основной класс для работы с графом знаний
- Concept: Структура узла-концепта с метаданными и отношениями
- Relationship: Структура отношения между концептами
- QueryEngine: Система запросов к графу
- GraphTraversal: Алгоритмы обхода графа
"""

from .kag_graph import (
    KAGGraph,
    Concept, 
    Relationship,
    RelationshipType,
    QueryResult
)

from .query_engine import (
    KAGQueryEngine,
    QueryType,
    QueryContext,
    SearchFilters
)

from .graph_traversal import (
    GraphTraversal,
    TraversalType,
    SearchResult
)

# =============================================================================
# Contextual Knowledge Integration (Новая функциональность)
# =============================================================================

from .context_engine import (
    ContextEngine,
    ContextRequest,
    ContextResult,
    KnowledgeUnit,
    ConceptNode,
    ReasoningRelation,
    KnowledgeDomain,
    ConceptType,
    ReasoningHop,
    TemporalState,
    TemporalMetadata,
    DynamicContextBuilder,
    ContextAwareRetriever,
    MultiHopReasoningEngine,
    TemporalValidationEngine,
    CrossDomainLinkingEngine,
    create_context_engine,
    ContextAwareBaseAgent,
    integrate_context_awareness
)

from .agent_integration import (
    ContextAwareAgent,
    ContextAwareAgentFactory,
    create_context_aware_ecosystem,
    get_agent_context_summary
)

from .psychology_examples import (
    PsychologyContextExamples,
    PsychologyKnowledgeBase,
    PsychologyTaskTemplates,
    demonstrate_psychology_context_integration,
    demonstrate_knowledge_base_integration
)

__all__ = [
    # Основные классы
    'KAGGraph',
    'Concept', 
    'Relationship',
    'QueryResult',
    
    # Enum типы
    'RelationshipType',
    'QueryType',
    'TraversalType',
    
    # Движки и системы
    'KAGQueryEngine',
    'GraphTraversal',
    
    # Вспомогательные классы
    'QueryContext',
    'SearchFilters',
    'SearchResult'
]

# Настройка логгера модуля
import logging
logger = logging.getLogger(__name__)
logger.info("Knowledge Graph модуль загружен с поддержкой Contextual Knowledge Integration")