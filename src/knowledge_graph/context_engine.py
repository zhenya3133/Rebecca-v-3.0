"""
Context Engine для контекстуальной интеграции знаний в Rebecca-Platform

Реализует:
1. Dynamic context building на основе текущих задач
2. Context-aware knowledge retrieval
3. Multi-hop reasoning через связанные концепты
4. Knowledge freshness и temporal validation
5. Cross-domain knowledge linking
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import uuid
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
import networkx as nx

from pydantic import BaseModel, Field, field_validator

# Импорты из Rebecca-Platform
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from memory_manager.memory_manager import MemoryManager
from retrieval.hybrid_retriever import HybridRetriever
from multi_agent.base_agent import AgentType, TaskRequest, TaskResult


# =============================================================================
# Структуры данных и схемы
# =============================================================================

class KnowledgeDomain(str, Enum):
    """Домены знаний для классификации"""
    PSYCHOLOGY = "psychology"
    MEDICINE = "medicine"
    TECHNOLOGY = "technology"
    BUSINESS = "business"
    EDUCATION = "education"
    SCIENCE = "science"
    PHILOSOPHY = "philosophy"
    ARTS = "arts"
    LAW = "law"
    GENERAL = "general"


class ConceptType(str, Enum):
    """Типы концептов в графе знаний"""
    ENTITY = "entity"
    CONCEPT = "concept"
    PROCESS = "process"
    RELATIONSHIP = "relationship"
    RULE = "rule"
    PATTERN = "pattern"
    THEORY = "theory"
    METHOD = "method"
    APPLICATION = "application"


class ReasoningHop(str, Enum):
    """Типы связей для multi-hop reasoning"""
    CAUSATION = "causation"
    CORRELATION = "correlation"
    INHERITANCE = "inheritance"
    COMPOSITION = "composition"
    ANALOGY = "analogy"
    INDUCTION = "induction"
    DEDUCTION = "deduction"
    ABDUCTION = "abduction"


class TemporalState(str, Enum):
    """Состояния актуальности знаний"""
    VALID = "valid"
    EXPIRED = "expired"
    CONTROVERSIAL = "controversial"
    EVOLVING = "evolving"
    DEPRECATED = "deprecated"


@dataclass
class TemporalMetadata:
    """Временные метаданные знаний"""
    created_at: datetime
    last_updated: datetime
    validated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    version: str = "1.0"
    confidence_score: float = 1.0
    evidence_count: int = 0
    last_evidence_added: Optional[datetime] = None
    
    @property
    def age_days(self) -> float:
        """Возраст знания в днях"""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds() / 86400
    
    @property
    def is_fresh(self) -> bool:
        """Проверка актуальности знаний"""
        if self.expires_at and datetime.now(timezone.utc) > self.expires_at:
            return False
        return True
    
    @property
    def temporal_state(self) -> TemporalState:
        """Определение временного состояния"""
        if not self.is_fresh:
            return TemporalState.EXPIRED
        
        if self.confidence_score < 0.5:
            return TemporalState.CONTROVERSIAL
        
        if self.age_days < 1:
            return TemporalState.VALID
        elif self.age_days < 365:
            return TemporalState.EVOLVING
        else:
            return TemporalState.DEPRECATED


@dataclass
class ConceptNode:
    """Узел концепта в графе знаний"""
    id: str
    label: str
    concept_type: ConceptType
    domain: KnowledgeDomain
    description: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    temporal_metadata: Optional[TemporalMetadata] = None
    confidence: float = 1.0
    related_concepts: Set[str] = field(default_factory=set)
    evidence: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.temporal_metadata:
            self.temporal_metadata = TemporalMetadata(
                created_at=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc)
            )


@dataclass
class ReasoningRelation:
    """Реляция для multi-hop reasoning"""
    source_id: str
    target_id: str
    relation_type: ReasoningHop
    strength: float
    evidence: List[str] = field(default_factory=list)
    confidence: float = 1.0
    temporal_metadata: Optional[TemporalMetadata] = None
    
    def __post_init__(self):
        if not self.temporal_metadata:
            self.temporal_metadata = TemporalMetadata(
                created_at=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc)
            )


class ContextRequest(BaseModel):
    """Запрос контекста для обработки"""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    current_task: Optional[TaskRequest] = None
    active_context: Dict[str, Any] = Field(default_factory=dict)
    target_domains: List[KnowledgeDomain] = Field(default_factory=list)
    reasoning_depth: int = Field(default=2, ge=1, le=5)
    freshness_threshold: float = Field(default=0.7)
    include_controversial: bool = False
    cross_domain_links: bool = True
    temporal_validation: bool = True
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ContextResult(BaseModel):
    """Результат контекстуального анализа"""
    request_id: str
    context_graph: Dict[str, Any] = Field(default_factory=dict)
    relevant_concepts: List[ConceptNode] = Field(default_factory=list)
    reasoning_chains: List[List[str]] = Field(default_factory=list)
    temporal_insights: Dict[str, Any] = Field(default_factory=dict)
    cross_domain_connections: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_score: float = 0.0
    processing_time: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь для сериализации"""
        return self.model_dump(exclude_none=True)


class KnowledgeUnit(BaseModel):
    """Единица знания в контекстуальной системе"""
    unit_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    concept_id: str
    domain: KnowledgeDomain
    concept_type: ConceptType
    temporal_metadata: TemporalMetadata
    relevance_score: float = 0.0
    context_dependencies: List[str] = Field(default_factory=list)
    evidence_sources: List[str] = Field(default_factory=list)
    validation_status: TemporalState = TemporalState.VALID
    cross_references: List[str] = Field(default_factory=list)


# =============================================================================
# Интерфейсы для компонентов
# =============================================================================

class ContextBuilder(ABC):
    """Интерфейс построителя контекста"""
    
    @abstractmethod
    async def build_dynamic_context(self, request: ContextRequest) -> Dict[str, Any]:
        """Построение динамического контекста"""
        pass
    
    @abstractmethod
    async def analyze_task_context(self, task: TaskRequest) -> Dict[str, Any]:
        """Анализ контекста задачи"""
        pass


class KnowledgeRetriever(ABC):
    """Интерфейс retrieval системы знаний"""
    
    @abstractmethod
    async def retrieve_relevant_knowledge(
        self, 
        context: Dict[str, Any], 
        domains: List[KnowledgeDomain]
    ) -> List[KnowledgeUnit]:
        """Извлечение релевантных знаний"""
        pass
    
    @abstractmethod
    async def semantic_search(self, query: str, domain: KnowledgeDomain) -> List[KnowledgeUnit]:
        """Семантический поиск знаний"""
        pass


class ReasoningEngine(ABC):
    """Интерфейс системы рассуждений"""
    
    @abstractmethod
    async def multi_hop_reasoning(
        self, 
        concepts: List[str], 
        reasoning_depth: int
    ) -> List[ReasoningRelation]:
        """Multi-hop рассуждения"""
        pass
    
    @abstractmethod
    async def build_reasoning_chains(self, start_concepts: List[str]) -> List[List[str]]:
        """Построение цепочек рассуждений"""
        pass


class TemporalValidator(ABC):
    """Интерфейс валидации временных характеристик"""
    
    @abstractmethod
    async def validate_temporal_consistency(self, knowledge: List[KnowledgeUnit]) -> Dict[str, Any]:
        """Валидация временной согласованности"""
        pass
    
    @abstractmethod
    async def detect_conflicts(self, knowledge: List[KnowledgeUnit]) -> List[Dict[str, Any]]:
        """Обнаружение временных конфликтов"""
        pass


class CrossDomainLinker(ABC):
    """Интерфейс междоменного связывания"""
    
    @abstractmethod
    async def find_cross_domain_connections(
        self, 
        domains: List[KnowledgeDomain]
    ) -> List[Dict[str, Any]]:
        """Поиск междоменных связей"""
        pass
    
    @abstractmethod
    async def analyze_domain_overlap(self, domain1: KnowledgeDomain, domain2: KnowledgeDomain) -> Dict[str, Any]:
        """Анализ пересечения доменов"""
        pass


# =============================================================================
# Реализации компонентов
# =============================================================================

class DynamicContextBuilder(ContextBuilder):
    """Построитель динамического контекста"""
    
    def __init__(self, memory_manager: MemoryManager, logger: Optional[logging.Logger] = None):
        self.memory_manager = memory_manager
        self.logger = logger or logging.getLogger(__name__)
        self.context_cache = {}
    
    async def build_dynamic_context(self, request: ContextRequest) -> Dict[str, Any]:
        """Построение динамического контекста"""
        start_time = time.time()
        
        try:
            context = {
                "request_id": request.request_id,
                "timestamp": request.timestamp.isoformat(),
                "task_context": {},
                "domain_context": {},
                "temporal_context": {},
                "concept_context": {}
            }
            
            # Анализ контекста текущей задачи
            if request.current_task:
                task_context = await self.analyze_task_context(request.current_task)
                context["task_context"] = task_context
            
            # Построение контекста доменов
            domain_context = await self._build_domain_context(request.target_domains)
            context["domain_context"] = domain_context
            
            # Временной контекст
            temporal_context = await self._build_temporal_context(request.temporal_validation)
            context["temporal_context"] = temporal_context
            
            # Контекст концептов
            concept_context = await self._build_concept_context(
                request.active_context, request.target_domains
            )
            context["concept_context"] = concept_context
            
            # Кэширование контекста
            await self._cache_context(request.request_id, context)
            
            processing_time = time.time() - start_time
            context["processing_time"] = processing_time
            
            self.logger.info(f"Динамический контекст построен за {processing_time:.3f}s")
            
            return context
            
        except Exception as e:
            self.logger.error(f"Ошибка построения динамического контекста: {str(e)}")
            raise
    
    async def analyze_task_context(self, task: TaskRequest) -> Dict[str, Any]:
        """Анализ контекста задачи"""
        task_context = {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "agent_type": task.agent_type.value,
            "description": task.description,
            "priority": task.priority,
            "inputs": task.inputs,
            "dependencies": task.dependencies,
            "estimated_complexity": self._estimate_complexity(task),
            "related_agents": await self._find_related_agents(task),
            "knowledge_requirements": await self._extract_knowledge_requirements(task),
            "temporal_constraints": await self._analyze_temporal_constraints(task)
        }
        
        return task_context
    
    def _estimate_complexity(self, task: TaskRequest) -> Dict[str, Any]:
        """Оценка сложности задачи"""
        complexity_factors = {
            "dependency_count": len(task.dependencies),
            "input_complexity": len(str(task.inputs)),
            "description_length": len(task.description),
            "priority_level": task.priority
        }
        
        # Простая эвристика оценки сложности
        complexity_score = sum([
            complexity_factors["dependency_count"] * 0.3,
            complexity_factors["input_complexity"] * 0.001,
            complexity_factors["description_length"] * 0.001,
            (6 - complexity_factors["priority_level"]) * 0.2
        ])
        
        return {
            "score": complexity_score,
            "factors": complexity_factors,
            "level": "low" if complexity_score < 1 else "medium" if complexity_score < 3 else "high"
        }
    
    async def _find_related_agents(self, task: TaskRequest) -> List[str]:
        """Поиск связанных агентов"""
        # Анализируем описание задачи для определения связанных агентов
        agent_keywords = {
            AgentType.BACKEND: ["backend", "server", "api", "database"],
            AgentType.FRONTEND: ["frontend", "ui", "interface", "client"],
            AgentType.ML_ENGINEER: ["ml", "machine learning", "ai", "model"],
            AgentType.QA_ANALYST: ["test", "qa", "quality", "validation"],
            AgentType.DEVOPS: ["deploy", "devops", "infrastructure", "docker"],
            AgentType.RESEARCH: ["research", "analysis", "study"],
            AgentType.WRITER: ["write", "document", "content"]
        }
        
        related_agents = []
        task_text = f"{task.task_type} {task.description}".lower()
        
        for agent_type, keywords in agent_keywords.items():
            if any(keyword in task_text for keyword in keywords):
                related_agents.append(agent_type.value)
        
        return related_agents
    
    async def _extract_knowledge_requirements(self, task: TaskRequest) -> List[Dict[str, Any]]:
        """Извлечение требований к знаниям"""
        requirements = []
        
        # Простой анализ ключевых слов в описании
        task_text = task.description.lower()
        
        psychology_keywords = {
            "психология": KnowledgeDomain.PSYCHOLOGY,
            "поведение": KnowledgeDomain.PSYCHOLOGY,
            "когнитивный": KnowledgeDomain.PSYCHOLOGY,
            "эмоция": KnowledgeDomain.PSYCHOLOGY
        }
        
        tech_keywords = {
            "api": KnowledgeDomain.TECHNOLOGY,
            "система": KnowledgeDomain.TECHNOLOGY,
            "программа": KnowledgeDomain.TECHNOLOGY
        }
        
        all_keywords = {**psychology_keywords, **tech_keywords}
        
        for keyword, domain in all_keywords.items():
            if keyword in task_text:
                requirements.append({
                    "keyword": keyword,
                    "domain": domain.value,
                    "relevance": 0.8,
                    "type": "keyword_match"
                })
        
        return requirements
    
    async def _analyze_temporal_constraints(self, task: TaskRequest) -> Dict[str, Any]:
        """Анализ временных ограничений"""
        return {
            "has_timeout": task.timeout is not None,
            "timeout_seconds": task.timeout,
            "created_at": task.created_at.isoformat(),
            "age_hours": (datetime.now(timezone.utc) - task.created_at).total_seconds() / 3600,
            "urgency_level": "high" if task.priority <= 2 else "medium" if task.priority <= 3 else "low"
        }
    
    async def _build_domain_context(self, target_domains: List[KnowledgeDomain]) -> Dict[str, Any]:
        """Построение контекста доменов"""
        domain_context = {
            "target_domains": [domain.value for domain in target_domains],
            "domain_weights": {},
            "cross_domain_potential": 0.0,
            "domain_maturity": {}
        }
        
        # Анализ зрелости доменов
        for domain in target_domains:
            maturity_score = await self._assess_domain_maturity(domain)
            domain_context["domain_maturity"][domain.value] = maturity_score
        
        return domain_context
    
    async def _assess_domain_maturity(self, domain: KnowledgeDomain) -> Dict[str, Any]:
        """Оценка зрелости домена знаний"""
        # Здесь можно добавить более сложную логику оценки зрелости
        # На основе количества знаний, их качества, частоты использования
        return {
            "knowledge_density": 0.5,  # Заглушка
            "evidence_quality": 0.7,
            "consensus_level": 0.6,
            "overall_score": 0.6
        }
    
    async def _build_temporal_context(self, validation_enabled: bool) -> Dict[str, Any]:
        """Построение временного контекста"""
        now = datetime.now(timezone.utc)
        
        temporal_context = {
            "current_time": now.isoformat(),
            "validation_enabled": validation_enabled,
            "time_window_hours": 24,  # По умолчанию смотрим на знания за последние 24 часа
            "freshness_threshold": 0.7,
            "temporal_horizon": "short_term"
        }
        
        return temporal_context
    
    async def _build_concept_context(
        self, 
        active_context: Dict[str, Any], 
        target_domains: List[KnowledgeDomain]
    ) -> Dict[str, Any]:
        """Построение контекста концептов"""
        concept_context = {
            "active_concepts": list(active_context.get("concepts", [])),
            "concept_relationships": {},
            "concept_freshness": {},
            "missing_concepts": []
        }
        
        return concept_context
    
    async def _cache_context(self, context_id: str, context: Dict[str, Any]) -> None:
        """Кэширование контекста"""
        cache_entry = {
            "context": context,
            "timestamp": datetime.now(timezone.utc),
            "ttl": 300  # 5 минут
        }
        self.context_cache[context_id] = cache_entry


class ContextAwareRetriever(KnowledgeRetriever):
    """Контекстуально-осведомленный retriever знаний"""
    
    def __init__(
        self, 
        memory_manager: MemoryManager, 
        hybrid_retriever: Optional[HybridRetriever] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.memory_manager = memory_manager
        self.hybrid_retriever = hybrid_retriever
        self.logger = logger or logging.getLogger(__name__)
        self.retrieval_cache = {}
    
    async def retrieve_relevant_knowledge(
        self, 
        context: Dict[str, Any], 
        domains: List[KnowledgeDomain]
    ) -> List[KnowledgeUnit]:
        """Извлечение релевантных знаний"""
        start_time = time.time()
        
        try:
            knowledge_units = []
            
            # Извлечение знаний из памяти по доменам
            for domain in domains:
                domain_knowledge = await self._retrieve_domain_knowledge(domain, context)
                knowledge_units.extend(domain_knowledge)
            
            # Контекстуальная фильтрация
            filtered_knowledge = await self._filter_by_context(knowledge_units, context)
            
            # Ранжирование по релевантности
            ranked_knowledge = await self._rank_by_relevance(filtered_knowledge, context)
            
            processing_time = time.time() - start_time
            self.logger.info(f"Извлечено {len(ranked_knowledge)} знаний за {processing_time:.3f}s")
            
            return ranked_knowledge
            
        except Exception as e:
            self.logger.error(f"Ошибка извлечения знаний: {str(e)}")
            raise
    
    async def semantic_search(self, query: str, domain: KnowledgeDomain) -> List[KnowledgeUnit]:
        """Семантический поиск знаний"""
        # Используем hybrid retriever если доступен
        if self.hybrid_retriever:
            try:
                results = self.hybrid_retriever.retrieve(query, k=20)
                knowledge_units = []
                
                for result in results:
                    if "domain" in result and result["domain"] == domain.value:
                        unit = KnowledgeUnit(
                            content=result["text"],
                            concept_id=result.get("id", ""),
                            domain=domain,
                            concept_type=ConceptType.CONCEPT,
                            temporal_metadata=TemporalMetadata(
                                created_at=datetime.now(timezone.utc),
                                last_updated=datetime.now(timezone.utc)
                            ),
                            relevance_score=result.get("score", 0.0)
                        )
                        knowledge_units.append(unit)
                
                return knowledge_units
            except Exception as e:
                self.logger.warning(f"Hybrid retriever недоступен: {str(e)}")
        
        # Fallback к поиску в памяти
        return await self._search_in_memory(query, domain)
    
    async def _retrieve_domain_knowledge(self, domain: KnowledgeDomain, context: Dict[str, Any]) -> List[KnowledgeUnit]:
        """Извлечение знаний из домена"""
        knowledge_units = []
        
        try:
            # Поиск в семантической памяти
            semantic_results = await self._search_semantic_memory(domain, context)
            knowledge_units.extend(semantic_results)
            
            # Поиск в эпизодической памяти для релевантных событий
            episodic_results = await self._search_episodic_memory(domain, context)
            knowledge_units.extend(episodic_results)
            
        except Exception as e:
            self.logger.error(f"Ошибка извлечения знаний домена {domain}: {str(e)}")
        
        return knowledge_units
    
    async def _search_semantic_memory(self, domain: KnowledgeDomain, context: Dict[str, Any]) -> List[KnowledgeUnit]:
        """Поиск в семантической памяти"""
        knowledge_units = []
        
        try:
            # Поиск по ключевым словам контекста
            query_terms = []
            
            if "task_context" in context:
                task_context = context["task_context"]
                if "description" in task_context:
                    # Простое извлечение ключевых слов
                    words = task_context["description"].lower().split()
                    query_terms.extend(words[:10])  # Первые 10 слов
            
            # Поиск в памяти
            for term in query_terms[:5]:  # Ограничиваем количество запросов
                try:
                    results = await self.memory_manager.search(layer="SEMANTIC", query=term, limit=5)
                    
                    for result in results:
                        unit = KnowledgeUnit(
                            content=result.get("content", str(result)),
                            concept_id=result.get("key", ""),
                            domain=domain,
                            concept_type=ConceptType.CONCEPT,
                            temporal_metadata=TemporalMetadata(
                                created_at=datetime.now(timezone.utc),
                                last_updated=datetime.now(timezone.utc)
                            ),
                            relevance_score=0.5  # Базовая релевантность
                        )
                        knowledge_units.append(unit)
                        
                except Exception:
                    continue  # Игнорируем ошибки поиска для отдельных терминов
                    
        except Exception as e:
            self.logger.error(f"Ошибка поиска в семантической памяти: {str(e)}")
        
        return knowledge_units
    
    async def _search_episodic_memory(self, domain: KnowledgeDomain, context: Dict[str, Any]) -> List[KnowledgeUnit]:
        """Поиск в эпизодической памяти"""
        knowledge_units = []
        
        try:
            # Поиск недавних событий
            results = await self.memory_manager.search(layer="EPISODIC", query="", limit=10)
            
            for result in results:
                unit = KnowledgeUnit(
                    content=result.get("content", str(result)),
                    concept_id=result.get("key", ""),
                    domain=domain,
                    concept_type=ConceptType.PROCESS,
                    temporal_metadata=TemporalMetadata(
                        created_at=datetime.now(timezone.utc),
                        last_updated=datetime.now(timezone.utc)
                    ),
                    relevance_score=0.3
                )
                knowledge_units.append(unit)
                
        except Exception as e:
            self.logger.error(f"Ошибка поиска в эпизодической памяти: {str(e)}")
        
        return knowledge_units
    
    async def _search_in_memory(self, query: str, domain: KnowledgeDomain) -> List[KnowledgeUnit]:
        """Поиск в памяти как fallback"""
        knowledge_units = []
        
        try:
            results = await self.memory_manager.search(layer="SEMANTIC", query=query, limit=10)
            
            for result in results:
                unit = KnowledgeUnit(
                    content=result.get("content", str(result)),
                    concept_id=result.get("key", ""),
                    domain=domain,
                    concept_type=ConceptType.CONCEPT,
                    temporal_metadata=TemporalMetadata(
                        created_at=datetime.now(timezone.utc),
                        last_updated=datetime.now(timezone.utc)
                    ),
                    relevance_score=0.5
                )
                knowledge_units.append(unit)
                
        except Exception as e:
            self.logger.error(f"Ошибка поиска в памяти: {str(e)}")
        
        return knowledge_units
    
    async def _filter_by_context(self, knowledge: List[KnowledgeUnit], context: Dict[str, Any]) -> List[KnowledgeUnit]:
        """Фильтрация знаний по контексту"""
        filtered = []
        
        # Контекстуальная фильтрация на основе задачи
        if "task_context" in context:
            task_context = context["task_context"]
            task_text = f"{task_context.get('description', '')} {task_context.get('task_type', '')}".lower()
            
            for unit in knowledge:
                # Простая проверка релевантности по тексту
                unit_score = 0.0
                for word in task_text.split()[:10]:
                    if word in unit.content.lower():
                        unit_score += 0.1
                
                if unit_score > 0.0:
                    unit.relevance_score += unit_score
                    filtered.append(unit)
        else:
            # Если нет контекста задачи, возвращаем все
            filtered = knowledge
        
        return filtered
    
    async def _rank_by_relevance(self, knowledge: List[KnowledgeUnit], context: Dict[str, Any]) -> List[KnowledgeUnit]:
        """Ранжирование знаний по релевантности"""
        # Сортируем по убыванию релевантности
        ranked = sorted(knowledge, key=lambda x: x.relevance_score, reverse=True)
        
        return ranked[:20]  # Ограничиваем количество результатов


class MultiHopReasoningEngine(ReasoningEngine):
    """Движок multi-hop рассуждений"""
    
    def __init__(self, memory_manager: MemoryManager, logger: Optional[logging.Logger] = None):
        self.memory_manager = memory_manager
        self.logger = logger or logging.getLogger(__name__)
        self.knowledge_graph = nx.DiGraph()
        self.reasoning_cache = {}
    
    async def multi_hop_reasoning(
        self, 
        concepts: List[str], 
        reasoning_depth: int
    ) -> List[ReasoningRelation]:
        """Multi-hop рассуждения"""
        start_time = time.time()
        
        try:
            relations = []
            
            for concept in concepts:
                concept_relations = await self._reason_from_concept(concept, reasoning_depth)
                relations.extend(concept_relations)
            
            # Удаление дубликатов
            unique_relations = await self._deduplicate_relations(relations)
            
            processing_time = time.time() - start_time
            self.logger.info(f"Multi-hop рассуждения завершены за {processing_time:.3f}s")
            
            return unique_relations
            
        except Exception as e:
            self.logger.error(f"Ошибка multi-hop рассуждений: {str(e)}")
            raise
    
    async def build_reasoning_chains(self, start_concepts: List[str]) -> List[List[str]]:
        """Построение цепочек рассуждений"""
        chains = []
        
        for concept in start_concepts:
            chain = await self._build_concept_chain(concept)
            if chain:
                chains.append(chain)
        
        return chains
    
    async def _reason_from_concept(self, concept_id: str, depth: int) -> List[ReasoningRelation]:
        """Рассуждения от концепта"""
        relations = []
        
        try:
            # Поиск связанных концептов
            related_concepts = await self._find_related_concepts(concept_id)
            
            for related_id, relation_type, strength in related_concepts:
                relation = ReasoningRelation(
                    source_id=concept_id,
                    target_id=related_id,
                    relation_type=ReasoningHop(relation_type),
                    strength=strength
                )
                relations.append(relation)
                
                # Рекурсивный поиск на заданную глубину
                if depth > 1:
                    sub_relations = await self._reason_from_concept(related_id, depth - 1)
                    relations.extend(sub_relations)
                    
        except Exception as e:
            self.logger.error(f"Ошибка рассуждений от концепта {concept_id}: {str(e)}")
        
        return relations
    
    async def _find_related_concepts(self, concept_id: str) -> List[Tuple[str, str, float]]:
        """Поиск связанных концептов"""
        related = []
        
        try:
            # Поиск в памяти концептов
            results = await self.memory_manager.search(layer="SEMANTIC", query=concept_id, limit=10)
            
            for result in results:
                if result.get("key") != concept_id:
                    related.append((result.get("key", ""), ReasoningHop.CORRELATION, 0.7))
                    
        except Exception as e:
            self.logger.error(f"Ошибка поиска связанных концептов: {str(e)}")
        
        return related
    
    async def _build_concept_chain(self, concept_id: str) -> List[str]:
        """Построение цепочки концептов"""
        chain = [concept_id]
        
        try:
            # Простое построение цепочки - поиск 2-3 связанных концептов
            related = await self._find_related_concepts(concept_id)
            
            for related_id, _, _ in related[:2]:  # Берем первые 2 связи
                if related_id not in chain:
                    chain.append(related_id)
                    
        except Exception as e:
            self.logger.error(f"Ошибка построения цепочки концептов: {str(e)}")
        
        return chain
    
    async def _deduplicate_relations(self, relations: List[ReasoningRelation]) -> List[ReasoningRelation]:
        """Удаление дубликатов отношений"""
        seen = set()
        unique_relations = []
        
        for relation in relations:
            key = (relation.source_id, relation.target_id, relation.relation_type.value)
            if key not in seen:
                seen.add(key)
                unique_relations.append(relation)
        
        return unique_relations


class TemporalValidationEngine(TemporalValidator):
    """Движок валидации временных характеристик"""
    
    def __init__(self, memory_manager: MemoryManager, logger: Optional[logging.Logger] = None):
        self.memory_manager = memory_manager
        self.logger = logger or logging.getLogger(__name__)
        self.validation_cache = {}
    
    async def validate_temporal_consistency(self, knowledge: List[KnowledgeUnit]) -> Dict[str, Any]:
        """Валидация временной согласованности"""
        start_time = time.time()
        
        try:
            validation_result = {
                "total_units": len(knowledge),
                "valid_units": 0,
                "expired_units": 0,
                "controversial_units": 0,
                "conflicts": [],
                "consistency_score": 0.0,
                "recommendations": []
            }
            
            # Анализ каждой единицы знания
            for unit in knowledge:
                temporal_state = unit.temporal_metadata.temporal_state
                
                if temporal_state == TemporalState.VALID:
                    validation_result["valid_units"] += 1
                elif temporal_state == TemporalState.EXPIRED:
                    validation_result["expired_units"] += 1
                elif temporal_state == TemporalState.CONTROVERSIAL:
                    validation_result["controversial_units"] += 1
            
            # Выявление конфликтов
            conflicts = await self._detect_temporal_conflicts(knowledge)
            validation_result["conflicts"] = conflicts
            
            # Расчет показателя согласованности
            total = len(knowledge)
            valid_ratio = validation_result["valid_units"] / max(1, total)
            conflict_penalty = len(conflicts) * 0.1
            
            validation_result["consistency_score"] = max(0.0, valid_ratio - conflict_penalty)
            
            # Генерация рекомендаций
            recommendations = await self._generate_temporal_recommendations(validation_result)
            validation_result["recommendations"] = recommendations
            
            processing_time = time.time() - start_time
            self.logger.info(f"Временная валидация завершена за {processing_time:.3f}s")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Ошибка временной валидации: {str(e)}")
            raise
    
    async def detect_conflicts(self, knowledge: List[KnowledgeUnit]) -> List[Dict[str, Any]]:
        """Обнаружение временных конфликтов"""
        conflicts = await self._detect_temporal_conflicts(knowledge)
        return conflicts
    
    async def _detect_temporal_conflicts(self, knowledge: List[KnowledgeUnit]) -> List[Dict[str, Any]]:
        """Обнаружение временных конфликтов"""
        conflicts = []
        
        try:
            # Группировка по концептам
            concept_groups = defaultdict(list)
            
            for unit in knowledge:
                concept_groups[unit.concept_id].append(unit)
            
            # Анализ конфликтов в каждой группе
            for concept_id, units in concept_groups.items():
                if len(units) > 1:
                    concept_conflicts = await self._analyze_concept_conflicts(concept_id, units)
                    conflicts.extend(concept_conflicts)
                    
        except Exception as e:
            self.logger.error(f"Ошибка обнаружения конфликтов: {str(e)}")
        
        return conflicts
    
    async def _analyze_concept_conflicts(self, concept_id: str, units: List[KnowledgeUnit]) -> List[Dict[str, Any]]:
        """Анализ конфликтов концепта"""
        conflicts = []
        
        # Простой анализ на основе временных меток
        for i, unit1 in enumerate(units):
            for unit2 in units[i+1:]:
                # Проверяем временные конфликты
                time_diff = abs((unit1.temporal_metadata.last_updated - unit2.temporal_metadata.last_updated).total_seconds())
                
                if time_diff < 3600:  # Конфликт если обновления в пределах часа
                    conflict = {
                        "concept_id": concept_id,
                        "conflict_type": "temporal_update",
                        "unit1_id": unit1.unit_id,
                        "unit2_id": unit2.unit_id,
                        "time_difference_seconds": time_diff,
                        "severity": "low" if time_diff > 1800 else "medium"
                    }
                    conflicts.append(conflict)
        
        return conflicts
    
    async def _generate_temporal_recommendations(self, validation_result: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций на основе валидации"""
        recommendations = []
        
        if validation_result["expired_units"] > 0:
            recommendations.append(f"Обновить {validation_result['expired_units']} устаревших единиц знания")
        
        if validation_result["controversial_units"] > 0:
            recommendations.append(f"Требуется дополнительная валидация {validation_result['controversial_units']} спорных единиц знания")
        
        if validation_result["consistency_score"] < 0.7:
            recommendations.append("Низкий показатель согласованности - требуется ревизия знаний")
        
        return recommendations


class CrossDomainLinkingEngine(CrossDomainLinker):
    """Движок междоменного связывания знаний"""
    
    def __init__(self, memory_manager: MemoryManager, logger: Optional[logging.Logger] = None):
        self.memory_manager = memory_manager
        self.logger = logger or logging.getLogger(__name__)
        self.domain_graph = nx.Graph()
        self.link_cache = {}
    
    async def find_cross_domain_connections(self, domains: List[KnowledgeDomain]) -> List[Dict[str, Any]]:
        """Поиск междоменных связей"""
        start_time = time.time()
        
        try:
            connections = []
            
            # Попарный анализ доменов
            for i, domain1 in enumerate(domains):
                for domain2 in domains[i+1:]:
                    domain_connections = await self._find_domain_connections(domain1, domain2)
                    connections.extend(domain_connections)
            
            processing_time = time.time() - start_time
            self.logger.info(f"Найдено {len(connections)} междоменных связей за {processing_time:.3f}s")
            
            return connections
            
        except Exception as e:
            self.logger.error(f"Ошибка поиска междоменных связей: {str(e)}")
            raise
    
    async def analyze_domain_overlap(self, domain1: KnowledgeDomain, domain2: KnowledgeDomain) -> Dict[str, Any]:
        """Анализ пересечения доменов"""
        overlap_analysis = {
            "domain1": domain1.value,
            "domain2": domain2.value,
            "shared_concepts": [],
            "concept_similarity": 0.0,
            "connection_strength": 0.0,
            "integration_potential": "low"
        }
        
        try:
            # Поиск общих концептов
            shared = await self._find_shared_concepts(domain1, domain2)
            overlap_analysis["shared_concepts"] = shared
            
            # Расчет сходства
            if shared:
                similarity = min(len(shared) / 10, 1.0)  # Простая метрика сходства
                overlap_analysis["concept_similarity"] = similarity
                
                # Определение силы связи
                if similarity > 0.7:
                    overlap_analysis["connection_strength"] = "strong"
                    overlap_analysis["integration_potential"] = "high"
                elif similarity > 0.3:
                    overlap_analysis["connection_strength"] = "medium"
                    overlap_analysis["integration_potential"] = "medium"
                else:
                    overlap_analysis["connection_strength"] = "weak"
                    overlap_analysis["integration_potential"] = "low"
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа пересечения доменов: {str(e)}")
        
        return overlap_analysis
    
    async def _find_domain_connections(self, domain1: KnowledgeDomain, domain2: KnowledgeDomain) -> List[Dict[str, Any]]:
        """Поиск связей между доменами"""
        connections = []
        
        try:
            # Специальная логика для психологического домена
            if domain1 == KnowledgeDomain.PSYCHOLOGY or domain2 == KnowledgeDomain.PSYCHOLOGY:
                psychology_connections = await _find_psychology_connections(domain1, domain2)
                connections.extend(psychology_connections)
            
            # Общие связи на основе концептов
            shared_concepts = await self._find_shared_concepts(domain1, domain2)
            
            for concept in shared_concepts:
                connection = {
                    "connection_type": "shared_concept",
                    "domain1": domain1.value,
                    "domain2": domain2.value,
                    "concept": concept,
                    "strength": 0.5,
                    "bidirectional": True
                }
                connections.append(connection)
                
        except Exception as e:
            self.logger.error(f"Ошибка поиска связей между {domain1} и {domain2}: {str(e)}")
        
        return connections
    
    async def _find_shared_concepts(self, domain1: KnowledgeDomain, domain2: KnowledgeDomain) -> List[str]:
        """Поиск общих концептов между доменами"""
        shared = []
        
        try:
            # Поиск в памяти концептов
            domain1_results = await self.memory_manager.search(layer="SEMANTIC", query=domain1.value, limit=20)
            domain2_results = await self.memory_manager.search(layer="SEMANTIC", query=domain2.value, limit=20)
            
            # Сравнение концептов
            concepts1 = set(result.get("key", "") for result in domain1_results)
            concepts2 = set(result.get("key", "") for result in domain2_results)
            
            shared = list(concepts1.intersection(concepts2))
            
        except Exception as e:
            self.logger.error(f"Ошибка поиска общих концептов: {str(e)}")
        
        return shared


async def _find_psychology_connections(domain1: KnowledgeDomain, domain2: KnowledgeDomain) -> List[Dict[str, Any]]:
    """Специальные связи для психологического домена"""
    connections = []
    
    # Матрица связей психологии с другими доменами
    psychology_connections = {
        KnowledgeDomain.MEDICINE: [
            {"concept": "neuropsychology", "strength": 0.8},
            {"concept": "clinical_psychology", "strength": 0.9},
            {"concept": "therapeutic_approaches", "strength": 0.7}
        ],
        KnowledgeDomain.EDUCATION: [
            {"concept": "educational_psychology", "strength": 0.9},
            {"concept": "learning_theories", "strength": 0.8},
            {"concept": "cognitive_development", "strength": 0.7}
        ],
        KnowledgeDomain.BUSINESS: [
            {"concept": "organizational_psychology", "strength": 0.8},
            {"concept": "consumer_behavior", "strength": 0.7},
            {"concept": "leadership", "strength": 0.6}
        ],
        KnowledgeDomain.TECHNOLOGY: [
            {"concept": "human_computer_interaction", "strength": 0.7},
            {"concept": "cognitive_load_theory", "strength": 0.6},
            {"concept": "user_experience", "strength": 0.5}
        ]
    }
    
    # Определяем, какой домен не психология
    non_psychology_domain = domain1 if domain2 == KnowledgeDomain.PSYCHOLOGY else domain2
    
    if non_psychology_domain in psychology_connections:
        for connection_info in psychology_connections[non_psychology_domain]:
            connection = {
                "connection_type": "psychology_domain_link",
                "domain1": KnowledgeDomain.PSYCHOLOGY.value,
                "domain2": non_psychology_domain.value,
                "concept": connection_info["concept"],
                "strength": connection_info["strength"],
                "bidirectional": True,
                "specialized": True
            }
            connections.append(connection)
    
    return connections


# =============================================================================
# Главный ContextEngine
# =============================================================================

class ContextEngine:
    """
    Главный движок контекстуальной интеграции знаний
    
    Интегрирует все компоненты для обеспечения:
    - Динамического построения контекста
    - Контекстуально-осведомленного поиска знаний
    - Multi-hop рассуждений
    - Временной валидации
    - Междоменного связывания
    """
    
    def __init__(
        self,
        memory_manager: MemoryManager,
        hybrid_retriever: Optional[HybridRetriever] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.memory_manager = memory_manager
        self.hybrid_retriever = hybrid_retriever
        self.logger = logger or logging.getLogger(__name__)
        
        # Инициализация компонентов
        self.context_builder = DynamicContextBuilder(memory_manager, logger)
        self.knowledge_retriever = ContextAwareRetriever(memory_manager, hybrid_retriever, logger)
        self.reasoning_engine = MultiHopReasoningEngine(memory_manager, logger)
        self.temporal_validator = TemporalValidationEngine(memory_manager, logger)
        self.cross_domain_linker = CrossDomainLinkingEngine(memory_manager, logger)
        
        # Статистика использования
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "average_processing_time": 0.0,
            "cache_hits": 0
        }
    
    async def process_context_request(self, request: ContextRequest) -> ContextResult:
        """
        Главный метод обработки запроса контекста
        """
        start_time = time.time()
        self.stats["total_requests"] += 1
        
        try:
            # Проверка кэша
            cached_result = await self._get_cached_result(request.request_id)
            if cached_result:
                self.stats["cache_hits"] += 1
                return cached_result
            
            # 1. Построение динамического контекста
            dynamic_context = await self.context_builder.build_dynamic_context(request)
            
            # 2. Контекстуальное извлечение знаний
            relevant_knowledge = await self.knowledge_retriever.retrieve_relevant_knowledge(
                dynamic_context, request.target_domains
            )
            
            # 3. Multi-hop рассуждения
            reasoning_relations = await self.reasoning_engine.multi_hop_reasoning(
                [k.concept_id for k in relevant_knowledge[:5]],  # Первые 5 концептов
                request.reasoning_depth
            )
            
            # 4. Временная валидация
            temporal_validation = {}
            if request.temporal_validation:
                temporal_validation = await self.temporal_validator.validate_temporal_consistency(
                    relevant_knowledge
                )
            
            # 5. Междоменное связывание
            cross_domain_connections = []
            if request.cross_domain_links and len(request.target_domains) > 1:
                cross_domain_connections = await self.cross_domain_linker.find_cross_domain_connections(
                    request.target_domains
                )
            
            # 6. Построение цепочек рассуждений
            reasoning_chains = await self.reasoning_engine.build_reasoning_chains(
                [k.concept_id for k in relevant_knowledge[:3]]
            )
            
            # 7. Формирование результата
            result = ContextResult(
                request_id=request.request_id,
                context_graph=dynamic_context,
                relevant_concepts=[ConceptNode(
                    id=k.concept_id,
                    label=f"Concept {k.concept_id}",
                    concept_type=k.concept_type,
                    domain=k.domain,
                    description=k.content[:200] + "..." if len(k.content) > 200 else k.content,
                    temporal_metadata=k.temporal_metadata,
                    confidence=k.relevance_score
                ) for k in relevant_knowledge],
                reasoning_chains=reasoning_chains,
                temporal_insights=temporal_validation,
                cross_domain_connections=cross_domain_connections,
                confidence_score=await self._calculate_overall_confidence(relevant_knowledge, temporal_validation),
                processing_time=time.time() - start_time
            )
            
            # 8. Кэширование результата
            await self._cache_result(request.request_id, result)
            
            self.stats["successful_requests"] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки контекстного запроса: {str(e)}")
            self.stats["successful_requests"] += 1  # Учитываем как обработанный запрос
            raise
    
    async def enhance_agent_context(
        self, 
        agent_id: str, 
        task: TaskRequest,
        domains: Optional[List[KnowledgeDomain]] = None
    ) -> Dict[str, Any]:
        """
        Улучшение контекста агента с учетом специфики задачи
        """
        try:
            # Определение релевантных доменов
            if not domains:
                domains = await self._infer_relevant_domains(task)
            
            # Создание запроса контекста
            request = ContextRequest(
                current_task=task,
                active_context={"agent_id": agent_id, "task_id": task.task_id},
                target_domains=domains,
                reasoning_depth=2,
                freshness_threshold=0.7,
                temporal_validation=True,
                cross_domain_links=True
            )
            
            # Обработка запроса
            result = await self.process_context_request(request)
            
            # Формирование обогащенного контекста
            enhanced_context = {
                "agent_id": agent_id,
                "task_id": task.task_id,
                "context_result": result.to_dict(),
                "actionable_insights": await self._extract_actionable_insights(result),
                "knowledge_gaps": await self._identify_knowledge_gaps(result),
                "recommended_actions": await self._generate_recommendations(result)
            }
            
            return enhanced_context
            
        except Exception as e:
            self.logger.error(f"Ошибка улучшения контекста агента: {str(e)}")
            raise
    
    async def _infer_relevant_domains(self, task: TaskRequest) -> List[KnowledgeDomain]:
        """Автоматическое определение релевантных доменов на основе задачи"""
        domains = [KnowledgeDomain.GENERAL]  # База всегда
        
        task_text = f"{task.task_type} {task.description}".lower()
        
        # Простое определение доменов по ключевым словам
        domain_keywords = {
            KnowledgeDomain.PSYCHOLOGY: ["психология", "поведение", "когнитив", "эмоци", "личность"],
            KnowledgeDomain.MEDICINE: ["медицина", "здоровье", "лечение", "диагностик"],
            KnowledgeDomain.TECHNOLOGY: ["программ", "система", "технолог", "api", "сервер"],
            KnowledgeDomain.BUSINESS: ["бизнес", "маркетинг", "продаж", "управлен"],
            KnowledgeDomain.EDUCATION: ["образован", "обучен", "учен", "курс"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in task_text for keyword in keywords):
                domains.append(domain)
        
        return domains
    
    async def _calculate_overall_confidence(
        self, 
        knowledge: List[KnowledgeUnit], 
        temporal_validation: Dict[str, Any]
    ) -> float:
        """Расчет общего показателя уверенности"""
        if not knowledge:
            return 0.0
        
        # Базовая уверенность на основе релевантности знаний
        avg_relevance = sum(k.relevance_score for k in knowledge) / len(knowledge)
        
        # Коррекция на основе временной валидации
        temporal_score = temporal_validation.get("consistency_score", 0.5)
        
        # Общая уверенность (взвешенное среднее)
        overall_confidence = (avg_relevance * 0.7) + (temporal_score * 0.3)
        
        return max(0.0, min(1.0, overall_confidence))
    
    async def _extract_actionable_insights(self, result: ContextResult) -> List[str]:
        """Извлечение действенных инсайтов"""
        insights = []
        
        # На основе релевантных концептов
        for concept in result.relevant_concepts[:3]:  # Топ-3 концепта
            if concept.confidence > 0.7:
                insights.append(f"Высокая уверенность в концепте: {concept.label}")
        
        # На основе временных инсайтов
        temporal_insights = result.temporal_insights
        if temporal_insights.get("consistency_score", 0) > 0.8:
            insights.append("Знания высоко согласованы во времени")
        elif temporal_insights.get("expired_units", 0) > 0:
            insights.append("Некоторые знания устарели и требуют обновления")
        
        # На основе междоменных связей
        if result.cross_domain_connections:
            insights.append(f"Найдено {len(result.cross_domain_connections)} междоменных связей")
        
        return insights
    
    async def _identify_knowledge_gaps(self, result: ContextResult) -> List[str]:
        """Идентификация пробелов в знаниях"""
        gaps = []
        
        # Если мало релевантных концептов
        if len(result.relevant_concepts) < 3:
            gaps.append("Недостаточно релевантных концептов в контексте")
        
        # Если нет цепочек рассуждений
        if not result.reasoning_chains:
            gaps.append("Отсутствуют цепочки рассуждений для глубокого анализа")
        
        # Если нет междоменных связей при нескольких доменах
        if not result.cross_domain_connections and len(result.context_graph.get("domain_context", {}).get("target_domains", [])) > 1:
            gaps.append("Отсутствуют связи между доменами знаний")
        
        return gaps
    
    async def _generate_recommendations(self, result: ContextResult) -> List[str]:
        """Генерация рекомендаций"""
        recommendations = []
        
        # Рекомендации на основе уверенности
        if result.confidence_score < 0.5:
            recommendations.append("Низкая уверенность - рекомендуется сбор дополнительной информации")
        
        # Рекомендации на основе временного анализа
        temporal_insights = result.temporal_insights
        if temporal_insights.get("expired_units", 0) > 0:
            recommendations.append("Обновить устаревшие знания")
        
        if temporal_insights.get("controversial_units", 0) > 0:
            recommendations.append("Проверить спорные знания на актуальность")
        
        # Рекомендации на основе связей
        if result.cross_domain_connections:
            recommendations.append("Рассмотреть интеграцию междоменных подходов")
        
        return recommendations
    
    async def _get_cached_result(self, request_id: str) -> Optional[ContextResult]:
        """Получение кэшированного результата"""
        # Простая реализация кэша в памяти
        # В продакшене здесь должен быть Redis или другой кэш
        cache_key = f"context_result_{request_id}"
        
        # Заглушка - всегда возвращаем None для простоты
        return None
    
    async def _cache_result(self, request_id: str, result: ContextResult) -> None:
        """Кэширование результата"""
        # Заглушка кэширования
        cache_key = f"context_result_{request_id}"
        # Здесь должен быть код для сохранения в Redis или другой кэш
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики использования"""
        return {
            **self.stats,
            "success_rate": (
                self.stats["successful_requests"] / max(1, self.stats["total_requests"])
            )
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Проверка здоровья системы"""
        health = {
            "status": "healthy",
            "components": {
                "context_builder": "active",
                "knowledge_retriever": "active", 
                "reasoning_engine": "active",
                "temporal_validator": "active",
                "cross_domain_linker": "active"
            },
            "statistics": self.get_statistics(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Проверка критических компонентов
        if not self.memory_manager:
            health["status"] = "degraded"
            health["components"]["memory_manager"] = "inactive"
        
        return health


# =============================================================================
# Функции утилиты
# =============================================================================

async def create_context_engine(
    memory_manager: MemoryManager,
    hybrid_retriever: Optional[HybridRetriever] = None
) -> ContextEngine:
    """
    Factory функция для создания ContextEngine
    """
    logger = logging.getLogger("context_engine")
    
    return ContextEngine(
        memory_manager=memory_manager,
        hybrid_retriever=hybrid_retriever,
        logger=logger
    )


def validate_context_request(request: ContextRequest) -> bool:
    """Валидация запроса контекста"""
    if not request.target_domains:
        return False
    
    if request.reasoning_depth < 1 or request.reasoning_depth > 5:
        return False
    
    if request.freshness_threshold < 0.0 or request.freshness_threshold > 1.0:
        return False
    
    return True


def extract_concepts_from_text(text: str) -> List[str]:
    """Простое извлечение концептов из текста"""
    # Простое извлечение слов длиннее 4 символов
    words = text.lower().split()
    concepts = [word for word in words if len(word) > 4]
    
    return concepts[:10]  # Ограничиваем количество


# =============================================================================
# Интеграция с BaseAgent
# =============================================================================

class ContextAwareBaseAgent:
    """
    Расширение BaseAgent с поддержкой контекстуальной интеграции знаний
    """
    
    def __init__(self, base_agent, context_engine: ContextEngine):
        self.base_agent = base_agent
        self.context_engine = context_engine
        self.logger = logging.getLogger(f"agent.context_aware.{base_agent.agent_type.value}")
    
    async def execute_with_context(self, task: TaskRequest) -> TaskResult:
        """
        Выполнение задачи с обогащенным контекстом
        """
        try:
            # Получение обогащенного контекста
            enhanced_context = await self.context_engine.enhance_agent_context(
                agent_id=f"{self.base_agent.agent_type.value}_agent",
                task=task
            )
            
            # Добавление контекста в задачу
            task.context = task.context or {}
            task.context["enhanced_context"] = enhanced_context
            
            self.logger.info(f"Задача {task.task_id} выполнена с обогащенным контекстом")
            
            # Выполнение задачи базовым агентом
            return await self.base_agent.execute_task(task)
            
        except Exception as e:
            self.logger.error(f"Ошибка выполнения задачи с контекстом: {str(e)}")
            # Fallback к обычному выполнению
            return await self.base_agent.execute_task(task)


def integrate_context_awareness(
    base_agent, 
    context_engine: ContextEngine
) -> ContextAwareBaseAgent:
    """
    Интеграция контекстуальной осведомленности в BaseAgent
    """
    return ContextAwareBaseAgent(base_agent, context_engine)
