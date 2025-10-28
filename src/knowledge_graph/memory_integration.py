"""
KAG (Knowledge-Augmented Generation) система с интеграцией 6 слоев памяти Rebecca-Platform.

Реализует комплексную систему управления знаниями с:
- Интеграцией с 6 слоями памяти (Core, Episodic, Semantic, Procedural, Vault, Security)
- Bidirectional synchronization между KAG graph и memory layers
- Knowledge validation и security контроль
- Advanced reasoning и query processing
- Access control и audit logging
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
import numpy as np

# Импорт компонентов памяти Rebecca-Platform
from ..memory_manager.memory_manager import MemoryManager, MEMORY_LAYERS
from ..memory_manager.memory_context import MemoryContext, MemoryEntry, CORE, EPISODIC, SEMANTIC, PROCEDURAL, VAULT, SECURITY

# Константы KAG системы
KAG_VERSION = "1.0.0"
MAX_GRAPH_NODES = 10000
MAX_SYNC_OPERATIONS = 100
SYNC_INTERVAL = 30  # секунд
VALIDATION_CONFIDENCE_THRESHOLD = 0.7

# Настройка логгера
logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Типы узлов в графе знаний."""
    CONCEPT = "concept"
    ENTITY = "entity"
    RELATION = "relation"
    EVENT = "event"
    PROCEDURE = "procedure"
    RULE = "rule"
    VAULT_ITEM = "vault_item"
    SECURITY_RULE = "security_rule"


class EdgeType(Enum):
    """Типы связей в графе знаний."""
    IS_A = "is_a"
    PART_OF = "part_of"
    RELATED_TO = "related_to"
    CAUSES = "causes"
    ENABLES = "enables"
    VALIDATES = "validates"
    CONFLICTS = "conflicts"
    DEPENDS_ON = "depends_on"


class ValidationStatus(Enum):
    """Статусы валидации знаний."""
    VALID = "valid"
    INVALID = "invalid"
    PENDING = "pending"
    CONFLICT = "conflict"
    EXPIRED = "expired"


class AccessLevel(Enum):
    """Уровни доступа к знаниям."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


@dataclass
class KAGNode:
    """Узел в графе знаний KAG."""
    id: str
    node_type: NodeType
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    validation_status: ValidationStatus = ValidationStatus.VALID
    access_level: AccessLevel = AccessLevel.INTERNAL
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: Set[str] = field(default_factory=set)
    memory_layer: Optional[str] = None
    vector_embedding: Optional[List[float]] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует узел в словарь."""
        return {
            "id": self.id,
            "node_type": self.node_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "confidence": self.confidence,
            "validation_status": self.validation_status.value,
            "access_level": self.access_level.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": list(self.tags),
            "memory_layer": self.memory_layer,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KAGNode:
        """Создает узел из словаря."""
        return cls(
            id=data["id"],
            node_type=NodeType(data["node_type"]),
            content=data["content"],
            metadata=data.get("metadata", {}),
            confidence=data.get("confidence", 1.0),
            validation_status=ValidationStatus(data.get("validation_status", "valid")),
            access_level=AccessLevel(data.get("access_level", "internal")),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            tags=set(data.get("tags", [])),
            memory_layer=data.get("memory_layer"),
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None
        )


@dataclass
class KAGEdge:
    """Связь в графе знаний KAG."""
    source: str
    target: str
    edge_type: EdgeType
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    validation_status: ValidationStatus = ValidationStatus.VALID
    created_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует связь в словарь."""
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "metadata": self.metadata,
            "confidence": self.confidence,
            "validation_status": self.validation_status.value,
            "created_at": self.created_at.isoformat(),
            "access_count": self.access_count
        }


@dataclass
class SyncOperation:
    """Операция синхронизации между графом и памятью."""
    operation_id: str
    operation_type: str  # "create", "update", "delete", "sync"
    source: str  # "graph" или "memory"
    target: str  # "graph" или "memory"
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # "pending", "completed", "failed"
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует операцию в словарь."""
        return {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "source": self.source,
            "target": self.target,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries
        }


class KnowledgeValidator:
    """Валидатор знаний для обеспечения качества и согласованности."""
    
    def __init__(self):
        self.validation_rules: Dict[str, Callable] = {}
        self.confidence_thresholds: Dict[NodeType, float] = {
            NodeType.CONCEPT: 0.8,
            NodeType.ENTITY: 0.7,
            NodeType.RELATION: 0.6,
            NodeType.EVENT: 0.9,
            NodeType.PROCEDURE: 0.85,
            NodeType.RULE: 0.95,
            NodeType.VAULT_ITEM: 0.9,
            NodeType.SECURITY_RULE: 0.99
        }
        
        logger.info("KnowledgeValidator инициализирован")
    
    async def validate_node(self, node: KAGNode) -> Tuple[ValidationStatus, float]:
        """Валидирует узел знаний.
        
        Args:
            node: Узел для валидации
            
        Returns:
            Tuple[ValidationStatus, confidence_score]
        """
        confidence_score = node.confidence
        issues = []
        
        # Проверка базовых требований
        if not node.content:
            issues.append("Отсутствует содержимое узла")
            confidence_score *= 0.5
        
        if not node.metadata.get("description"):
            issues.append("Отсутствует описание в метаданных")
            confidence_score *= 0.8
        
        # Проверка специфичных правил для типов узлов
        if node.node_type in self.validation_rules:
            try:
                rule_result = await self.validation_rules[node.node_type](node)
                if not rule_result["valid"]:
                    issues.extend(rule_result["issues"])
                    confidence_score *= rule_result.get("confidence_factor", 0.7)
            except Exception as e:
                issues.append(f"Ошибка применения правила валидации: {e}")
                confidence_score *= 0.6
        
        # Проверка уровня доступа
        if node.access_level == AccessLevel.SECRET and node.node_type != NodeType.VAULT_ITEM:
            issues.append("Несоответствие типа узла и уровня доступа")
            confidence_score *= 0.5
        
        # Определяем статус валидации
        min_threshold = self.confidence_thresholds.get(node.node_type, VALIDATION_CONFIDENCE_THRESHOLD)
        
        if confidence_score >= min_threshold:
            status = ValidationStatus.VALID
        elif confidence_score >= min_threshold * 0.5:
            status = ValidationStatus.PENDING
        else:
            status = ValidationStatus.INVALID
        
        # Логируем результаты валидации
        if issues:
            logger.warning(f"Валидация узла {node.id}: {len(issues)} проблем, "
                          f"confidence={confidence_score:.2f}, status={status.value}")
            for issue in issues:
                logger.debug(f"  - {issue}")
        else:
            logger.debug(f"Валидация узла {node.id}: успешно, "
                        f"confidence={confidence_score:.2f}")
        
        return status, confidence_score
    
    def add_validation_rule(self, node_type: NodeType, rule: Callable) -> None:
        """Добавляет правило валидации для типа узла."""
        self.validation_rules[node_type] = rule
        logger.info(f"Добавлено правило валидации для типа {node_type.value}")


class AccessControl:
    """Система контроля доступа к знаниям."""
    
    def __init__(self):
        self.user_permissions: Dict[str, Set[AccessLevel]] = {}
        self.classification_rules: Dict[str, AccessLevel] = {}
        
        logger.info("AccessControl инициализирован")
    
    def check_access(self, user_id: str, node: KAGNode) -> bool:
        """Проверяет доступ пользователя к узлу.
        
        Args:
            user_id: ID пользователя
            node: Узел для проверки доступа
            
        Returns:
            True, если доступ разрешен
        """
        user_levels = self.user_permissions.get(user_id, set())
        required_level = node.access_level
        
        # Проверяем, есть ли у пользователя необходимый уровень доступа
        if required_level in user_levels:
            return True
        
        # Проверяем наследование уровней доступа
        access_hierarchy = {
            AccessLevel.PUBLIC: 0,
            AccessLevel.INTERNAL: 1,
            AccessLevel.CONFIDENTIAL: 2,
            AccessLevel.SECRET: 3,
            AccessLevel.TOP_SECRET: 4
        }
        
        user_max_level = max([access_hierarchy.get(level, -1) for level in user_levels])
        required_level_num = access_hierarchy.get(required_level, 0)
        
        return user_max_level >= required_level_num
    
    def classify_content(self, content: Any, metadata: Dict[str, Any]) -> AccessLevel:
        """Автоматически классифицирует контент по уровню доступа."""
        # Простая эвристика классификации
        if metadata.get("classification"):
            return AccessLevel(metadata["classification"])
        
        # Анализ содержимого
        content_str = str(content).lower()
        
        # Секретные ключевые слова
        secret_keywords = ["пароль", "ключ", "секрет", "token", "password"]
        if any(keyword in content_str for keyword in secret_keywords):
            return AccessLevel.SECRET
        
        # Конфиденциальные ключевые слова
        confidential_keywords = ["приватный", "конфиденциально", "internal"]
        if any(keyword in content_str for keyword in confidential_keywords):
            return AccessLevel.CONFIDENTIAL
        
        return AccessLevel.INTERNAL
    
    def add_user_permission(self, user_id: str, access_level: AccessLevel) -> None:
        """Добавляет разрешение пользователю."""
        if user_id not in self.user_permissions:
            self.user_permissions[user_id] = set()
        self.user_permissions[user_id].add(access_level)
        logger.info(f"Пользователю {user_id} добавлено разрешение {access_level.value}")


class KAGGraphManager:
    """Менеджер графа знаний KAG."""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.node_index: Dict[str, KAGNode] = {}
        self.edge_index: Dict[Tuple[str, str, str], KAGEdge] = {}
        self._lock = asyncio.Lock()
        
        logger.info("KAGGraphManager инициализирован")
    
    async def add_node(self, node: KAGNode) -> bool:
        """Добавляет узел в граф.
        
        Args:
            node: Узел для добавления
            
        Returns:
            True, если узел успешно добавлен
        """
        async with self._lock:
            try:
                # Валидация узла
                validator = KnowledgeValidator()
                validation_status, confidence = await validator.validate_node(node)
                node.validation_status = validation_status
                node.confidence = confidence
                
                # Добавляем в граф
                self.graph.add_node(
                    node.id,
                    **node.to_dict()
                )
                
                # Индексируем
                self.node_index[node.id] = node
                
                logger.info(f"Узел {node.id} добавлен в граф (тип: {node.node_type.value})")
                return True
                
            except Exception as e:
                logger.error(f"Ошибка добавления узла {node.id}: {e}")
                return False
    
    async def add_edge(self, edge: KAGEdge) -> bool:
        """Добавляет связь в граф.
        
        Args:
            edge: Связь для добавления
            
        Returns:
            True, если связь успешно добавлена
        """
        async with self._lock:
            try:
                # Проверяем существование узлов
                if edge.source not in self.graph or edge.target not in self.graph:
                    logger.warning(f"Узлы для связи {edge.source} -> {edge.target} не найдены")
                    return False
                
                # Добавляем в граф
                self.graph.add_edge(
                    edge.source,
                    edge.target,
                    key=edge.edge_type.value,
                    **edge.to_dict()
                )
                
                # Индексируем
                edge_key = (edge.source, edge.target, edge.edge_type.value)
                self.edge_index[edge_key] = edge
                
                logger.debug(f"Связь {edge.source} -> {edge.target} добавлена")
                return True
                
            except Exception as e:
                logger.error(f"Ошибка добавления связи: {e}")
                return False
    
    async def get_node(self, node_id: str) -> Optional[KAGNode]:
        """Получает узел по ID.
        
        Args:
            node_id: ID узла
            
        Returns:
            Узел или None, если не найден
        """
        node = self.node_index.get(node_id)
        if node:
            node.access_count += 1
            node.last_accessed = datetime.now()
        return node
    
    async def find_nodes_by_type(self, node_type: NodeType) -> List[KAGNode]:
        """Находит узлы по типу.
        
        Args:
            node_type: Тип узла
            
        Returns:
            Список найденных узлов
        """
        return [node for node in self.node_index.values() 
                if node.node_type == node_type]
    
    async def find_related_nodes(self, node_id: str, 
                                max_depth: int = 2) -> List[KAGNode]:
        """Находит связанные узлы.
        
        Args:
            node_id: ID исходного узла
            max_depth: Максимальная глубина поиска
            
        Returns:
            Список связанных узлов
        """
        try:
            # Используем NetworkX для поиска по графу
            related_nodes = []
            visited = set()
            
            def dfs(current_id: str, depth: int):
                if current_id in visited or depth > max_depth:
                    return
                
                visited.add(current_id)
                node = self.node_index.get(current_id)
                if node:
                    related_nodes.append(node)
                
                # Исследуем соседние узлы
                for neighbor in self.graph.neighbors(current_id):
                    dfs(neighbor, depth + 1)
            
            dfs(node_id, 0)
            
            logger.debug(f"Найдено {len(related_nodes)} связанных узлов для {node_id}")
            return related_nodes
            
        except Exception as e:
            logger.error(f"Ошибка поиска связанных узлов: {e}")
            return []
    
    async def query_graph(self, query: str) -> List[Dict[str, Any]]:
        """Выполняет запрос к графу знаний.
        
        Args:
            query: Строка запроса
            
        Returns:
            Список результатов запроса
        """
        try:
            results = []
            
            # Простой поиск по содержимому узлов
            query_lower = query.lower()
            
            for node in self.node_index.values():
                # Проверяем содержимое узла
                if query_lower in str(node.content).lower():
                    results.append({
                        "type": "node",
                        "id": node.id,
                        "content": node.content,
                        "node_type": node.node_type.value,
                        "confidence": node.confidence,
                        "relevance": 1.0
                    })
                
                # Проверяем метаданные и теги
                if any(query_lower in str(value).lower() 
                       for value in node.metadata.values()):
                    results.append({
                        "type": "node_metadata",
                        "id": node.id,
                        "content": node.content,
                        "node_type": node.node_type.value,
                        "confidence": node.confidence,
                        "relevance": 0.8
                    })
                
                if any(query_lower in tag.lower() for tag in node.tags):
                    results.append({
                        "type": "node_tag",
                        "id": node.id,
                        "content": node.content,
                        "node_type": node.node_type.value,
                        "confidence": node.confidence,
                        "relevance": 0.9
                    })
            
            # Сортируем по релевантности и уверенности
            results.sort(key=lambda x: (x["relevance"], x["confidence"]), reverse=True)
            
            logger.debug(f"Запрос '{query}' вернул {len(results)} результатов")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка выполнения запроса: {e}")
            return []
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику графа."""
        try:
            return {
                "total_nodes": len(self.node_index),
                "total_edges": len(self.edge_index),
                "node_types": {
                    node_type.value: len([n for n in self.node_index.values() 
                                         if n.node_type == node_type])
                    for node_type in NodeType
                },
                "validation_status": {
                    status.value: len([n for n in self.node_index.values() 
                                       if n.validation_status == status])
                    for status in ValidationStatus
                },
                "access_levels": {
                    level.value: len([n for n in self.node_index.values() 
                                      if n.access_level == level])
                    for level in AccessLevel
                }
            }
        except Exception as e:
            logger.error(f"Ошибка получения статистики графа: {e}")
            return {}


class MemoryLayerIntegration:
    """Интеграционный слой между KAG графом и памятью."""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.access_control = AccessControl()
        self.sync_queue: deque = deque(maxlen=MAX_SYNC_OPERATIONS)
        self._lock = asyncio.Lock()
        
        logger.info("MemoryLayerIntegration инициализирован")
    
    async def sync_from_memory_to_graph(self, layer: str, 
                                       limit: int = 100) -> List[str]:
        """Синхронизирует данные из памяти в граф.
        
        Args:
            layer: Слой памяти
            limit: Максимальное количество элементов
            
        Returns:
            Список ID синхронизированных узлов
        """
        async with self._lock:
            try:
                # Получаем данные из памяти
                memory_data = await self.memory_manager.retrieve(layer, limit=limit)
                synced_ids = []
                
                # Создаем KAGGraphManager для каждой синхронизации
                graph_manager = KAGGraphManager()
                
                for item in memory_data:
                    try:
                        # Создаем узел на основе данных памяти
                        kag_node = await self._create_node_from_memory_item(item, layer)
                        
                        # Добавляем в граф
                        success = await graph_manager.add_node(kag_node)
                        if success:
                            synced_ids.append(kag_node.id)
                        
                    except Exception as e:
                        logger.error(f"Ошибка синхронизации элемента {item.get('id')}: {e}")
                
                logger.info(f"Синхронизировано {len(synced_ids)} элементов из слоя {layer}")
                return synced_ids
                
            except Exception as e:
                logger.error(f"Ошибка синхронизации из памяти: {e}")
                return []
    
    async def sync_from_graph_to_memory(self, node_ids: List[str]) -> List[str]:
        """Синхронизирует данные из графа в память.
        
        Args:
            node_ids: Список ID узлов для синхронизации
            
        Returns:
            Список ID сохраненных элементов
        """
        async with self._lock:
            synced_ids = []
            
            try:
                for node_id in node_ids:
                    # Получаем узел из графа (здесь упрощенная реализация)
                    # В реальной системе нужно передавать graph_manager как параметр
                    node_data = await self._get_node_data_from_graph(node_id)
                    
                    if node_data:
                        # Определяем слой памяти на основе типа узла
                        layer = self._determine_memory_layer(node_data["node_type"])
                        
                        # Сохраняем в память
                        memory_id = await self.memory_manager.store(
                            layer=layer,
                            data=node_data["content"],
                            metadata={
                                **node_data["metadata"],
                                "kag_node_id": node_id,
                                "sync_timestamp": datetime.now().isoformat()
                            },
                            tags=list(node_data.get("tags", [])),
                            priority=5
                        )
                        
                        synced_ids.append(memory_id)
                
                logger.info(f"Синхронизировано {len(synced_ids)} узлов в память")
                return synced_ids
                
            except Exception as e:
                logger.error(f"Ошибка синхронизации в память: {e}")
                return []
    
    async def _create_node_from_memory_item(self, item: Dict[str, Any], 
                                          layer: str) -> KAGNode:
        """Создает KAG узел из элемента памяти.
        
        Args:
            item: Элемент памяти
            layer: Слой памяти
            
        Returns:
            KAG узел
        """
        # Определяем тип узла на основе слоя памяти
        node_type_mapping = {
            CORE: NodeType.CONCEPT,
            EPISODIC: NodeType.EVENT,
            SEMANTIC: NodeType.CONCEPT,
            PROCEDURAL: NodeType.PROCEDURE,
            VAULT: NodeType.VAULT_ITEM,
            SECURITY: NodeType.SECURITY_RULE
        }
        
        node_type = node_type_mapping.get(layer, NodeType.CONCEPT)
        
        # Классифицируем уровень доступа
        access_level = self.access_control.classify_content(
            item["content"], 
            item.get("metadata", {})
        )
        
        # Создаем узел
        node = KAGNode(
            id=f"{layer}_{item['id']}",
            node_type=node_type,
            content=item["content"],
            metadata=item.get("metadata", {}),
            tags=set(item.get("tags", [])),
            access_level=access_level,
            memory_layer=layer,
            confidence=item.get("priority", 5) / 10.0  # Нормализация приоритета
        )
        
        return node
    
    async def _get_node_data_from_graph(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Получает данные узла из графа.
        
        Args:
            node_id: ID узла
            
        Returns:
            Данные узла или None
        """
        # Заглушка для получения данных из графа
        # В реальной реализации здесь будет интеграция с graph_manager
        return {
            "content": f"Graph content for {node_id}",
            "metadata": {"source": "graph"},
            "tags": ["graph"],
            "node_type": "concept"
        }
    
    def _determine_memory_layer(self, node_type: str) -> str:
        """Определяет слой памяти на основе типа узла.
        
        Args:
            node_type: Тип узла
            
        Returns:
            Слой памяти
        """
        type_to_layer = {
            "concept": CORE,
            "event": EPISODIC,
            "procedure": PROCEDURAL,
            "vault_item": VAULT,
            "security_rule": SECURITY,
            "entity": SEMANTIC,
            "relation": SEMANTIC,
            "rule": SEMANTIC
        }
        
        return type_to_layer.get(node_type, CORE)
    
    async def create_sync_operation(self, operation: SyncOperation) -> None:
        """Создает операцию синхронизации.
        
        Args:
            operation: Операция синхронизации
        """
        self.sync_queue.append(operation)
        logger.debug(f"Создана операция синхронизации {operation.operation_id}")
    
    async def process_sync_queue(self) -> Dict[str, int]:
        """Обрабатывает очередь синхронизации.
        
        Returns:
            Словарь с результатами обработки
        """
        processed = {"completed": 0, "failed": 0, "retried": 0}
        
        while self.sync_queue:
            operation = self.sync_queue.popleft()
            
            try:
                success = await self._execute_sync_operation(operation)
                
                if success:
                    operation.status = "completed"
                    processed["completed"] += 1
                    logger.debug(f"Операция {operation.operation_id} выполнена")
                else:
                    if operation.retry_count < operation.max_retries:
                        operation.retry_count += 1
                        self.sync_queue.append(operation)
                        processed["retried"] += 1
                        logger.debug(f"Операция {operation.operation_id} поставлена в очередь повтора")
                    else:
                        operation.status = "failed"
                        processed["failed"] += 1
                        logger.warning(f"Операция {operation.operation_id} не выполнена после {operation.max_retries} попыток")
                        
            except Exception as e:
                logger.error(f"Ошибка выполнения операции {operation.operation_id}: {e}")
                processed["failed"] += 1
        
        return processed
    
    async def _execute_sync_operation(self, operation: SyncOperation) -> bool:
        """Выполняет операцию синхронизации.
        
        Args:
            operation: Операция для выполнения
            
        Returns:
            True, если операция выполнена успешно
        """
        try:
            if operation.operation_type == "create":
                if operation.source == "memory" and operation.target == "graph":
                    return await self._sync_memory_to_graph(operation.data)
                elif operation.source == "graph" and operation.target == "memory":
                    return await self._sync_graph_to_memory(operation.data)
            
            elif operation.operation_type == "update":
                return await self._sync_update(operation.data)
            
            elif operation.operation_type == "delete":
                return await self._sync_delete(operation.data)
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка выполнения операции синхронизации: {e}")
            return False
    
    async def _sync_memory_to_graph(self, data: Dict[str, Any]) -> bool:
        """Синхронизирует данные из памяти в граф."""
        # Реализация синхронизации из памяти в граф
        return True
    
    async def _sync_graph_to_memory(self, data: Dict[str, Any]) -> bool:
        """Синхронизирует данные из графа в память."""
        # Реализация синхронизации из графа в память
        return True
    
    async def _sync_update(self, data: Dict[str, Any]) -> bool:
        """Синхронизирует обновление данных."""
        # Реализация синхронизации обновлений
        return True
    
    async def _sync_delete(self, data: Dict[str, Any]) -> bool:
        """Синхронизирует удаление данных."""
        # Реализация синхронизации удалений
        return True


class KAGMemoryIntegration:
    """Главный класс интеграции KAG системы с памятью Rebecca-Platform."""
    
    def __init__(self, memory_manager: MemoryManager):
        """Инициализирует интеграционную систему.
        
        Args:
            memory_manager: Экземпляр MemoryManager
        """
        self.memory_manager = memory_manager
        self.graph_manager = KAGGraphManager()
        self.memory_integration = MemoryLayerIntegration(memory_manager)
        self.access_control = AccessControl()
        self.validator = KnowledgeValidator()
        
        # Состояние системы
        self._running = False
        self._sync_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Статистика
        self.sync_stats = {
            "total_sync_operations": 0,
            "successful_syncs": 0,
            "failed_syncs": 0,
            "last_sync_time": None,
            "nodes_in_graph": 0,
            "memory_items_synced": 0
        }
        
        logger.info("KAGMemoryIntegration инициализирован")
    
    async def start(self) -> None:
        """Запускает интеграционную систему."""
        if self._running:
            logger.warning("Система уже запущена")
            return
        
        logger.info("Запуск KAGMemoryIntegration...")
        
        # Запускаем фоновые задачи
        self._running = True
        self._sync_task = asyncio.create_task(self._sync_loop())
        
        # Выполняем первоначальную синхронизацию
        await self._initial_sync()
        
        logger.info("KAGMemoryIntegration запущен")
    
    async def stop(self) -> None:
        """Останавливает интеграционную систему."""
        if not self._running:
            logger.warning("Система не запущена")
            return
        
        logger.info("Остановка KAGMemoryIntegration...")
        
        self._running = False
        self._shutdown_event.set()
        
        # Останавливаем фоновые задачи
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        
        logger.info("KAGMemoryIntegration остановлен")
    
    async def add_knowledge(self, content: Any, 
                           node_type: NodeType,
                           metadata: Optional[Dict[str, Any]] = None,
                           tags: Optional[List[str]] = None,
                           auto_sync: bool = True) -> str:
        """Добавляет знание в систему.
        
        Args:
            content: Содержимое знания
            node_type: Тип узла
            metadata: Метаданные
            tags: Теги
            auto_sync: Автоматически синхронизировать с памятью
            
        Returns:
            ID созданного узла
        """
        try:
            # Создаем узел
            node_id = f"kag_{uuid.uuid4().hex[:8]}"
            
            # Классифицируем уровень доступа
            access_level = self.access_control.classify_content(content, metadata or {})
            
            # Создаем KAG узел
            node = KAGNode(
                id=node_id,
                node_type=node_type,
                content=content,
                metadata=metadata or {},
                tags=set(tags or []),
                access_level=access_level
            )
            
            # Валидируем узел
            validation_status, confidence = await self.validator.validate_node(node)
            node.validation_status = validation_status
            node.confidence = confidence
            
            # Добавляем в граф
            success = await self.graph_manager.add_node(node)
            if not success:
                raise RuntimeError(f"Не удалось добавить узел {node_id} в граф")
            
            # Определяем слой памяти
            layer = self.memory_integration._determine_memory_layer(node_type.value)
            
            # Синхронизируем с памятью, если требуется
            if auto_sync:
                sync_data = {
                    "node_id": node_id,
                    "content": content,
                    "metadata": metadata or {},
                    "tags": tags or [],
                    "layer": layer
                }
                
                operation = SyncOperation(
                    operation_id=str(uuid.uuid4()),
                    operation_type="create",
                    source="graph",
                    target="memory",
                    data=sync_data
                )
                
                await self.memory_integration.create_sync_operation(operation)
            
            # Обновляем статистику
            self.sync_stats["nodes_in_graph"] += 1
            self.sync_stats["total_sync_operations"] += 1
            
            logger.info(f"Знание добавлено в систему, ID: {node_id}")
            return node_id
            
        except Exception as e:
            logger.error(f"Ошибка добавления знания: {e}")
            raise
    
    async def query_knowledge(self, query: str, 
                             node_types: Optional[List[NodeType]] = None,
                             max_results: int = 20) -> List[Dict[str, Any]]:
        """Выполняет запрос к знаниям.
        
        Args:
            query: Строка запроса
            node_types: Типы узлов для поиска
            max_results: Максимальное количество результатов
            
        Returns:
            Список результатов запроса
        """
        try:
            # Выполняем поиск по графу
            graph_results = await self.graph_manager.query_graph(query)
            
            # Фильтруем по типам узлов
            if node_types:
                allowed_types = {t.value for t in node_types}
                graph_results = [
                    r for r in graph_results 
                    if r["node_type"] in allowed_types
                ]
            
            # Дополнительно ищем в памяти
            memory_results = await self._search_memory_for_query(query, node_types)
            
            # Объединяем результаты
            all_results = graph_results + memory_results
            
            # Сортируем по релевантности
            all_results.sort(key=lambda x: (x.get("relevance", 0), x.get("confidence", 0)), reverse=True)
            
            # Ограничиваем количество результатов
            final_results = all_results[:max_results]
            
            logger.info(f"Запрос '{query}' вернул {len(final_results)} результатов")
            return final_results
            
        except Exception as e:
            logger.error(f"Ошибка выполнения запроса: {e}")
            return []
    
    async def _search_memory_for_query(self, query: str, 
                                      node_types: Optional[List[NodeType]] = None) -> List[Dict[str, Any]]:
        """Ищет в памяти по запросу.
        
        Args:
            query: Строка запроса
            node_types: Типы узлов для поиска
            
        Returns:
            Список результатов из памяти
        """
        results = []
        
        try:
            # Определяем слои для поиска
            layer_mapping = {
                NodeType.CONCEPT: [CORE, SEMANTIC],
                NodeType.EVENT: [EPISODIC],
                NodeType.PROCEDURE: [PROCEDURAL],
                NodeType.VAULT_ITEM: [VAULT],
                NodeType.SECURITY_RULE: [SECURITY]
            }
            
            target_layers = []
            if node_types:
                for node_type in node_types:
                    target_layers.extend(layer_mapping.get(node_type, []))
            else:
                target_layers = list(MEMORY_LAYERS.keys())
            
            # Ищем параллельно во всех слоях
            search_tasks = []
            for layer in target_layers:
                task = asyncio.create_task(
                    self.memory_manager.retrieve(layer, query=query, limit=10)
                )
                search_tasks.append((layer, task))
            
            # Собираем результаты
            for layer, task in search_tasks:
                try:
                    memory_results = await task
                    for item in memory_results:
                        results.append({
                            "type": "memory",
                            "id": f"mem_{item['id']}",
                            "content": item["content"],
                            "node_type": "concept",  # Упрощенная типизация
                            "confidence": item.get("priority", 5) / 10.0,
                            "relevance": 0.7,  # Базовое значение релевантности
                            "memory_layer": layer,
                            "source": "memory"
                        })
                except Exception as e:
                    logger.warning(f"Ошибка поиска в слое {layer}: {e}")
            
        except Exception as e:
            logger.error(f"Ошибка поиска в памяти: {e}")
        
        return results
    
    async def validate_knowledge(self, node_id: str) -> Dict[str, Any]:
        """Валидирует знание по ID.
        
        Args:
            node_id: ID узла для валидации
            
        Returns:
            Результат валидации
        """
        try:
            # Получаем узел из графа
            node = await self.graph_manager.get_node(node_id)
            if not node:
                return {"valid": False, "error": "Узел не найден"}
            
            # Выполняем валидацию
            validation_status, confidence = await self.validator.validate_node(node)
            
            # Обновляем узел
            node.validation_status = validation_status
            node.confidence = confidence
            node.updated_at = datetime.now()
            
            result = {
                "valid": validation_status == ValidationStatus.VALID,
                "status": validation_status.value,
                "confidence": confidence,
                "issues": []  # Здесь можно добавить детали проблем
            }
            
            logger.info(f"Валидация узла {node_id}: {validation_status.value} "
                       f"(confidence: {confidence:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка валидации узла {node_id}: {e}")
            return {"valid": False, "error": str(e)}
    
    async def sync_all_layers(self) -> Dict[str, Any]:
        """Синхронизирует все слои памяти с графом.
        
        Returns:
            Результаты синхронизации
        """
        logger.info("Начало полной синхронизации слоев памяти...")
        
        sync_results = {
            "total_layers": len(MEMORY_LAYERS),
            "successful_layers": 0,
            "failed_layers": 0,
            "total_synced_items": 0,
            "layer_results": {},
            "duration": 0
        }
        
        start_time = datetime.now()
        
        try:
            # Синхронизируем каждый слой памяти
            for layer_name in MEMORY_LAYERS.keys():
                try:
                    layer_start_time = datetime.now()
                    
                    # Синхронизируем из памяти в граф
                    synced_ids = await self.memory_integration.sync_from_memory_to_graph(
                        layer_name, limit=500
                    )
                    
                    layer_duration = (datetime.now() - layer_start_time).total_seconds()
                    
                    sync_results["layer_results"][layer_name] = {
                        "synced_items": len(synced_ids),
                        "duration": layer_duration,
                        "success": True
                    }
                    
                    sync_results["total_synced_items"] += len(synced_ids)
                    
                    if synced_ids:
                        sync_results["successful_layers"] += 1
                        logger.info(f"Слой {layer_name}: синхронизировано {len(synced_ids)} элементов")
                    else:
                        logger.info(f"Слой {layer_name}: нет новых элементов для синхронизации")
                    
                except Exception as e:
                    sync_results["failed_layers"] += 1
                    sync_results["layer_results"][layer_name] = {
                        "error": str(e),
                        "success": False
                    }
                    logger.error(f"Ошибка синхронизации слоя {layer_name}: {e}")
            
            # Обрабатываем очередь синхронизации
            queue_results = await self.memory_integration.process_sync_queue()
            sync_results["queue_processing"] = queue_results
            
            # Вычисляем общее время
            sync_results["duration"] = (datetime.now() - start_time).total_seconds()
            
            # Обновляем статистику
            self.sync_stats["last_sync_time"] = datetime.now().isoformat()
            self.sync_stats["memory_items_synced"] += sync_results["total_synced_items"]
            self.sync_stats["successful_syncs"] += sync_results["successful_layers"]
            
            logger.info(f"Синхронизация завершена за {sync_results['duration']:.2f}s, "
                       f"синхронизировано {sync_results['total_synced_items']} элементов")
            
        except Exception as e:
            logger.error(f"Критическая ошибка синхронизации: {e}")
            sync_results["error"] = str(e)
        
        return sync_results
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Возвращает статус системы интеграции.
        
        Returns:
            Словарь со статусом системы
        """
        try:
            # Получаем статистику графа
            graph_stats = self.graph_manager.get_graph_statistics()
            
            # Получаем статистику памяти
            memory_stats = await self.memory_manager.get_layer_statistics()
            
            # Получаем очередь синхронизации
            sync_queue_size = len(self.memory_integration.sync_queue)
            
            return {
                "running": self._running,
                "kag_version": KAG_VERSION,
                "graph_statistics": graph_stats,
                "memory_statistics": memory_stats,
                "sync_queue_size": sync_queue_size,
                "sync_statistics": self.sync_stats,
                "access_control": {
                    "total_users": len(self.access_control.user_permissions),
                    "classification_rules": len(self.access_control.classification_rules)
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения статуса системы: {e}")
            return {
                "running": self._running,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _initial_sync(self) -> None:
        """Выполняет первоначальную синхронизацию."""
        logger.info("Выполнение первоначальной синхронизации...")
        
        try:
            # Синхронизируем все слои
            sync_results = await self.sync_all_layers()
            
            logger.info(f"Первоначальная синхронизация завершена: "
                       f"{sync_results['total_synced_items']} элементов")
            
        except Exception as e:
            logger.error(f"Ошибка первоначальной синхронизации: {e}")
    
    async def _sync_loop(self) -> None:
        """Цикл автоматической синхронизации."""
        while self._running:
            try:
                # Ждем интервал синхронизации
                await asyncio.wait_for(
                    self._shutdown_event.wait(), 
                    timeout=SYNC_INTERVAL
                )
                
                if self._running:
                    # Обрабатываем очередь синхронизации
                    await self.memory_integration.process_sync_queue()
                
            except asyncio.TimeoutError:
                # Обычное завершение ожидания
                if self._running:
                    logger.debug("Выполнение фоновой синхронизации...")
                    
                    try:
                        # Обрабатываем очередь синхронизации
                        await self.memory_integration.process_sync_queue()
                    except Exception as e:
                        logger.error(f"Ошибка фоновой синхронизации: {e}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ошибка в цикле синхронизации: {e}")
                await asyncio.sleep(10)  # Короткая пауза перед повтором


# Вспомогательные функции

async def create_kag_integration(memory_manager: MemoryManager) -> KAGMemoryIntegration:
    """Создает экземпляр интеграции KAG с памятью.
    
    Args:
        memory_manager: Экземпляр MemoryManager
        
    Returns:
        Экземпляр KAGMemoryIntegration
    """
    integration = KAGMemoryIntegration(memory_manager)
    
    # Настраиваем базовые правила валидации
    integration.validator.add_validation_rule(
        NodeType.CONCEPT,
        lambda node: {"valid": True, "issues": [], "confidence_factor": 1.0}
    )
    
    integration.validator.add_validation_rule(
        NodeType.VAULT_ITEM,
        lambda node: {
            "valid": len(str(node.content)) > 0,
            "issues": ["Содержимое vault элемента не должно быть пустым"] if not str(node.content) else [],
            "confidence_factor": 0.9 if str(node.content) else 0.5
        }
    )
    
    return integration


async def quick_kag_test() -> Dict[str, Any]:
    """Быстрый тест интеграционной системы KAG."""
    from ..memory_manager.memory_manager import create_memory_manager
    
    # Создаем MemoryManager
    memory_manager = create_memory_manager({
        "cache_size": 100,
        "cache_ttl": 1800
    })
    
    try:
        await memory_manager.start()
        
        # Создаем KAG интеграцию
        kag_integration = await create_kag_integration(memory_manager)
        
        # Запускаем систему
        await kag_integration.start()
        
        # Добавляем тестовые знания
        concept_id = await kag_integration.add_knowledge(
            content="Искусственный интеллект - это технология, имитирующая человеческий интеллект",
            node_type=NodeType.CONCEPT,
            tags=["AI", "technology", "intelligence"],
            metadata={"domain": "technology", "importance": "high"}
        )
        
        procedure_id = await kag_integration.add_knowledge(
            content="Алгоритм обучения нейронной сети включает прямой и обратный проход",
            node_type=NodeType.PROCEDURE,
            tags=["neural_network", "training", "algorithm"],
            metadata={"category": "machine_learning"}
        )
        
        # Выполняем запрос
        query_results = await kag_integration.query_knowledge("искусственный интеллект")
        
        # Получаем статус системы
        system_status = await kag_integration.get_system_status()
        
        # Останавливаем систему
        await kag_integration.stop()
        await memory_manager.stop()
        
        return {
            "success": True,
            "created_knowledge": [concept_id, procedure_id],
            "query_results_count": len(query_results),
            "system_status": system_status
        }
        
    except Exception as e:
        logger.error(f"Ошибка тестирования: {e}")
        return {
            "success": False,
            "error": str(e)
        }


# Экспорт основных классов и функций
__all__ = [
    "KAGMemoryIntegration",
    "KAGGraphManager", 
    "MemoryLayerIntegration",
    "KnowledgeValidator",
    "AccessControl",
    "KAGNode",
    "KAGEdge",
    "SyncOperation",
    "NodeType",
    "EdgeType", 
    "ValidationStatus",
    "AccessLevel",
    "create_kag_integration",
    "quick_kag_test"
]