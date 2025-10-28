"""
KAGGraph - основной класс для работы с концептуальным графом знаний.
Реализует архитектуру для хранения концептов, отношений и запросов к графу.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Callable
from enum import Enum
from collections import defaultdict, deque
import numpy as np
from pathlib import Path

# Импорты для интеграции с memory manager
import sys
sys.path.append('..')

try:
    from memory_manager.memory_manager_interface import MemoryManager, MemoryLayer, MemoryItem
    from memory_manager.memory_context import MemoryContext
except ImportError:
    # Fallback для случаев, когда memory manager недоступен
    MemoryManager = None
    MemoryLayer = None
    MemoryItem = None
    MemoryContext = None


class RelationshipType(Enum):
    """Типы отношений между концептами."""
    IS_A = "is_a"              # наследование (суперкласс)
    PART_OF = "part_of"        # часть целого
    RELATED_TO = "related_to"  # общая связь
    CAUSES = "causes"          # причинность
    INFLUENCES = "influences"  # влияние
    OPPOSITE_OF = "opposite_of" # противоположность
    SIMILAR_TO = "similar_to"  # похожесть
    LEADS_TO = "leads_to"      # приводит к
    CONTAINS = "contains"      # содержит
    USED_FOR = "used_for"      # используется для
    PRODUCES = "produces"      # производит


@dataclass
class Concept:
    """
    Структура узла-концепта в графе знаний.
    Содержит данные о концепте, его связях и метаданные.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    category: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 1.0
    source: str = "unknown"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Связанные концепты и отношения
    outgoing_relationships: Dict[str, 'Relationship'] = field(default_factory=dict)
    incoming_relationships: Dict[str, 'Relationship'] = field(default_factory=dict)
    
    # Кэш для вычислений
    _similarity_cache: Dict[str, float] = field(default_factory=dict)
    
    def add_relationship(self, relationship: 'Relationship') -> None:
        """Добавляет отношение к концепту."""
        if relationship.source_id == self.id:
            self.outgoing_relationships[relationship.id] = relationship
        elif relationship.target_id == self.id:
            self.incoming_relationships[relationship.id] = relationship
        else:
            raise ValueError("Отношение не связано с данным концептом")
    
    def remove_relationship(self, relationship_id: str) -> bool:
        """Удаляет отношение из концепта."""
        return (
            self.outgoing_relationships.pop(relationship_id, None) or
            self.incoming_relationships.pop(relationship_id, None)
        ) is not None
    
    def get_related_concepts(
        self, 
        relationship_types: Optional[Set[RelationshipType]] = None
    ) -> List['Concept']:
        """Получает связанные концепты."""
        related = []
        
        for rel in self.outgoing_relationships.values():
            if relationship_types is None or rel.relationship_type in relationship_types:
                related.append(rel.target)
        
        for rel in self.incoming_relationships.values():
            if relationship_types is None or rel.relationship_type in relationship_types:
                related.append(rel.source)
        
        return related
    
    def get_relationships(
        self, 
        relationship_types: Optional[Set[RelationshipType]] = None
    ) -> List['Relationship']:
        """Получает все отношения концепта."""
        all_relationships = list(self.outgoing_relationships.values())
        all_relationships.extend(self.incoming_relationships.values())
        
        if relationship_types:
            return [rel for rel in all_relationships 
                   if rel.relationship_type in relationship_types]
        
        return all_relationships
    
    def calculate_similarity(self, other: 'Concept') -> float:
        """Вычисляет схожесть с другим концептом."""
        # Проверяем кэш
        cache_key = f"{self.id}_{other.id}"
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]
        
        similarity = 0.0
        
        # Похожесть по категориям
        if self.category and other.category:
            if self.category == other.category:
                similarity += 0.4
            else:
                similarity += 0.1
        
        # Похожесть по тегам
        common_tags = set(self.tags) & set(other.tags)
        total_tags = set(self.tags) | set(other.tags)
        if total_tags:
            similarity += 0.3 * len(common_tags) / len(total_tags)
        
        # Похожесть по свойствам
        common_props = set(self.properties.keys()) & set(other.properties.keys())
        if common_props:
            prop_similarity = sum(
                1 for key in common_props
                if self.properties[key] == other.properties[key]
            ) / len(common_props)
            similarity += 0.2 * prop_similarity
        
        # Похожесть по структуре графа (общие соседи)
        self_neighbors = {n.id for n in self.get_related_concepts()}
        other_neighbors = {n.id for n in other.get_related_concepts()}
        common_neighbors = self_neighbors & other_neighbors
        total_neighbors = self_neighbors | other_neighbors
        if total_neighbors:
            similarity += 0.1 * len(common_neighbors) / len(total_neighbors)
        
        # Кэшируем результат
        self._similarity_cache[cache_key] = similarity
        other._similarity_cache[f"{other.id}_{self.id}"] = similarity
        
        return similarity
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует концепт в словарь."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'category': self.category,
            'metadata': self.metadata,
            'confidence_score': self.confidence_score,
            'source': self.source,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'tags': self.tags,
            'properties': self.properties,
            'relationships': [rel.to_dict() for rel in self.get_relationships()]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], concepts_cache: Optional[Dict[str, 'Concept']] = None) -> 'Concept':
        """Создает концепт из словаря."""
        concept = cls(
            id=data['id'],
            name=data['name'],
            description=data['description'],
            category=data['category'],
            metadata=data.get('metadata', {}),
            confidence_score=data.get('confidence_score', 1.0),
            source=data.get('source', 'unknown'),
            created_at=data.get('created_at', time.time()),
            updated_at=data.get('updated_at', time.time()),
            tags=data.get('tags', []),
            properties=data.get('properties', {})
        )
        
        # Сохраняем в кэше, если предоставлен
        if concepts_cache is not None:
            concepts_cache[concept.id] = concept
        
        return concept
    
    def update_timestamp(self) -> None:
        """Обновляет время последнего изменения."""
        self.updated_at = time.time()


@dataclass
class Relationship:
    """
    Структура отношения между концептами.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    relationship_type: RelationshipType = RelationshipType.RELATED_TO
    strength: float = 1.0  # Сила связи (0-1)
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    confidence_score: float = 1.0
    
    # Ссылки на объекты концептов (заполняются после загрузки)
    source: Optional[Concept] = None
    target: Optional[Concept] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует отношение в словарь."""
        return {
            'id': self.id,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'relationship_type': self.relationship_type.value,
            'strength': self.strength,
            'description': self.description,
            'metadata': self.metadata,
            'created_at': self.created_at,
            'confidence_score': self.confidence_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Relationship':
        """Создает отношение из словаря."""
        return cls(
            id=data['id'],
            source_id=data['source_id'],
            target_id=data['target_id'],
            relationship_type=RelationshipType(data['relationship_type']),
            strength=data.get('strength', 1.0),
            description=data.get('description', ''),
            metadata=data.get('metadata', {}),
            created_at=data.get('created_at', time.time()),
            confidence_score=data.get('confidence_score', 1.0)
        )


@dataclass
class QueryResult:
    """
    Результат запроса к графу знаний.
    """
    query: str
    results: List[Concept] = field(default_factory=list)
    paths: List[List[str]] = field(default_factory=list)  # Пути между концептами
    scores: List[float] = field(default_factory=list)
    execution_time: float = 0.0
    total_results: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует результат в словарь."""
        return {
            'query': self.query,
            'results': [concept.to_dict() for concept in self.results],
            'paths': self.paths,
            'scores': self.scores,
            'execution_time': self.execution_time,
            'total_results': self.total_results
        }


class KAGGraph:
    """
    Основной класс для работы с концептуальным графом знаний.
    Обеспечивает хранение концептов, отношений и выполнение запросов.
    """
    
    def __init__(
        self,
        memory_manager: Optional[MemoryManager] = None,
        persistence_path: Optional[str] = None
    ):
        """
        Инициализация графа знаний.
        
        Args:
            memory_manager: Интегрированный менеджер памяти
            persistence_path: Путь для сохранения на диск
        """
        self.concepts: Dict[str, Concept] = {}
        self.relationships: Dict[str, Relationship] = {}
        
        # Индексы для оптимизации поиска
        self.name_index: Dict[str, str] = {}  # name -> concept_id
        self.category_index: Dict[str, Set[str]] = defaultdict(set)  # category -> set(concept_ids)
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)  # tag -> set(concept_ids)
        self.type_index: Dict[RelationshipType, Set[str]] = defaultdict(set)  # rel_type -> set(relationship_ids)
        
        # Интеграция с memory manager
        self.memory_manager = memory_manager
        self.memory_context = None
        if memory_manager:
            self.memory_context = memory_manager.context
        
        # Настройки персистентности
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self.auto_save = True
        self.save_interval = 300  # секунд
        
        # Статистика
        self.stats = {
            'total_concepts': 0,
            'total_relationships': 0,
            'queries_executed': 0,
            'avg_query_time': 0.0,
            'last_save': time.time()
        }
        
        # Создаем директорию для сохранения
        if self.persistence_path:
            self.persistence_path.mkdir(parents=True, exist_ok=True)
    
    def add_concept(self, concept: Concept) -> str:
        """
        Добавляет концепт в граф.
        
        Args:
            concept: Концепт для добавления
            
        Returns:
            ID концепта
        """
        # Проверяем уникальность имени
        if concept.name in self.name_index:
            existing_id = self.name_index[concept.name]
            if existing_id != concept.id:
                print(f"Предупреждение: Концепт с именем '{concept.name}' уже существует")
        
        # Добавляем концепт
        self.concepts[concept.id] = concept
        self.name_index[concept.name] = concept.id
        
        # Обновляем индексы
        if concept.category:
            self.category_index[concept.category].add(concept.id)
        
        for tag in concept.tags:
            self.tag_index[tag].add(concept.id)
        
        # Сохраняем в памяти
        self._persist_concept(concept)
        
        # Обновляем статистику
        self.stats['total_concepts'] = len(self.concepts)
        
        return concept.id
    
    def remove_concept(self, concept_id: str) -> bool:
        """
        Удаляет концепт из графа.
        
        Args:
            concept_id: ID концепта для удаления
            
        Returns:
            True если успешно удален
        """
        if concept_id not in self.concepts:
            return False
        
        concept = self.concepts[concept_id]
        
        # Удаляем все связанные отношения
        related_relationships = concept.get_relationships()
        for rel in related_relationships:
            self.remove_relationship(rel.id)
        
        # Удаляем из индексов
        self.name_index.pop(concept.name, None)
        if concept.category:
            self.category_index[concept.category].discard(concept_id)
        
        for tag in concept.tags:
            self.tag_index[tag].discard(concept_id)
        
        # Удаляем из памяти
        self._remove_concept_from_memory(concept)
        
        # Удаляем концепт
        del self.concepts[concept_id]
        
        # Обновляем статистику
        self.stats['total_concepts'] = len(self.concepts)
        
        return True
    
    def add_relationship(self, relationship: Relationship) -> str:
        """
        Добавляет отношение в граф.
        
        Args:
            relationship: Отношение для добавления
            
        Returns:
            ID отношения
        """
        # Проверяем существование концептов
        if relationship.source_id not in self.concepts:
            raise ValueError(f"Исходный концепт {relationship.source_id} не найден")
        if relationship.target_id not in self.concepts:
            raise ValueError(f"Целевой концепт {relationship.target_id} не найден")
        
        # Добавляем отношение
        self.relationships[relationship.id] = relationship
        
        # Обновляем ссылки на концепты
        relationship.source = self.concepts[relationship.source_id]
        relationship.target = self.concepts[relationship.target_id]
        
        # Обновляем связи в концептах
        relationship.source.add_relationship(relationship)
        relationship.target.add_relationship(relationship)
        
        # Обновляем индексы
        self.type_index[relationship.relationship_type].add(relationship.id)
        
        # Сохраняем в памяти
        self._persist_relationship(relationship)
        
        # Обновляем статистику
        self.stats['total_relationships'] = len(self.relationships)
        
        return relationship.id
    
    def remove_relationship(self, relationship_id: str) -> bool:
        """
        Удаляет отношение из графа.
        
        Args:
            relationship_id: ID отношения для удаления
            
        Returns:
            True если успешно удалено
        """
        if relationship_id not in self.relationships:
            return False
        
        relationship = self.relationships[relationship_id]
        
        # Удаляем связи из концептов
        if relationship.source:
            relationship.source.remove_relationship(relationship_id)
        if relationship.target:
            relationship.target.remove_relationship(relationship_id)
        
        # Удаляем из индексов
        self.type_index[relationship.relationship_type].discard(relationship_id)
        
        # Удаляем из памяти
        self._remove_relationship_from_memory(relationship)
        
        # Удаляем отношение
        del self.relationships[relationship_id]
        
        # Обновляем статистику
        self.stats['total_relationships'] = len(self.relationships)
        
        return True
    
    def get_concept(self, concept_id: str) -> Optional[Concept]:
        """Получает концепт по ID."""
        return self.concepts.get(concept_id)
    
    def get_concept_by_name(self, name: str) -> Optional[Concept]:
        """Получает концепт по имени."""
        concept_id = self.name_index.get(name)
        if concept_id:
            return self.concepts.get(concept_id)
        return None
    
    def find_concepts(
        self,
        name: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        min_confidence: float = 0.0
    ) -> List[Concept]:
        """
        Находит концепты по критериям.
        
        Args:
            name: Частичное или полное имя
            category: Категория
            tags: Набор тегов
            min_confidence: Минимальная оценка уверенности
            
        Returns:
            Список найденных концептов
        """
        results = []
        
        for concept in self.concepts.values():
            # Фильтр по уверенности
            if concept.confidence_score < min_confidence:
                continue
            
            # Фильтр по имени
            if name and name.lower() not in concept.name.lower():
                continue
            
            # Фильтр по категории
            if category and concept.category != category:
                continue
            
            # Фильтр по тегам
            if tags and not tags.issubset(set(concept.tags)):
                continue
            
            results.append(concept)
        
        return results
    
    def get_connected_concepts(
        self,
        concept_id: str,
        max_depth: int = 1,
        relationship_types: Optional[Set[RelationshipType]] = None
    ) -> Dict[str, List[Concept]]:
        """
        Получает связанные концепты с указанием глубины связи.
        
        Args:
            concept_id: ID исходного концепта
            max_depth: Максимальная глубина поиска
            relationship_types: Типы отношений для поиска
            
        Returns:
            Словарь {глубина: [концепты]}
        """
        if concept_id not in self.concepts:
            return {}
        
        visited = set()
        result = defaultdict(list)
        queue = deque([(concept_id, 0)])
        visited.add(concept_id)
        
        while queue:
            current_id, depth = queue.popleft()
            
            if depth > max_depth:
                continue
            
            current_concept = self.concepts[current_id]
            
            # Получаем связанные концепты
            related = current_concept.get_related_concepts(relationship_types)
            for related_concept in related:
                if related_concept.id not in visited:
                    visited.add(related_concept.id)
                    result[depth + 1].append(related_concept)
                    queue.append((related_concept.id, depth + 1))
        
        return dict(result)
    
    def find_shortest_path(
        self,
        source_id: str,
        target_id: str,
        relationship_types: Optional[Set[RelationshipType]] = None
    ) -> Optional[List[str]]:
        """
        Находит кратчайший путь между концептами.
        
        Args:
            source_id: ID исходного концепта
            target_id: ID целевого концепта
            relationship_types: Типы отношений для поиска пути
            
        Returns:
            Список ID концептов в пути или None если путь не найден
        """
        if source_id not in self.concepts or target_id not in self.concepts:
            return None
        
        queue = deque([(source_id, [source_id])])
        visited = {source_id}
        
        while queue:
            current_id, path = queue.popleft()
            
            if current_id == target_id:
                return path
            
            current_concept = self.concepts[current_id]
            related = current_concept.get_related_concepts(relationship_types)
            
            for related_concept in related:
                if related_concept.id not in visited:
                    visited.add(related_concept.id)
                    new_path = path + [related_concept.id]
                    queue.append((related_concept.id, new_path))
        
        return None
    
    def calculate_graph_metrics(self) -> Dict[str, Any]:
        """
        Вычисляет метрики графа.
        
        Returns:
            Словарь с метриками графа
        """
        total_concepts = len(self.concepts)
        total_relationships = len(self.relationships)
        
        if total_concepts == 0:
            return {
                'total_concepts': 0,
                'total_relationships': 0,
                'density': 0.0,
                'avg_degree': 0.0,
                'clustering_coefficient': 0.0
            }
        
        # Плотность графа
        max_edges = total_concepts * (total_concepts - 1)
        density = total_relationships / max_edges if max_edges > 0 else 0.0
        
        # Средняя степень вершин
        degrees = [len(c.get_related_concepts()) for c in self.concepts.values()]
        avg_degree = sum(degrees) / len(degrees)
        
        # Коэффициент кластеризации
        clustering_coeffs = []
        for concept in self.concepts.values():
            neighbors = concept.get_related_concepts()
            if len(neighbors) < 2:
                clustering_coeffs.append(0.0)
                continue
            
            edges_between_neighbors = 0
            possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
            
            neighbor_ids = {n.id for n in neighbors}
            for neighbor in neighbors:
                neighbor_neighbors = {n.id for n in neighbor.get_related_concepts()}
                edges_between_neighbors += len(neighbor_ids & neighbor_neighbors)
            
            if possible_edges > 0:
                clustering_coeffs.append(edges_between_neighbors / possible_edges)
            else:
                clustering_coeffs.append(0.0)
        
        avg_clustering = sum(clustering_coeffs) / len(clustering_coeffs)
        
        return {
            'total_concepts': total_concepts,
            'total_relationships': total_relationships,
            'density': density,
            'avg_degree': avg_degree,
            'clustering_coefficient': avg_clustering,
            'categories': len(self.category_index),
            'unique_tags': len(self.tag_index),
            'relationship_types': len(self.type_index)
        }
    
    def export_graph(self, format_type: str = "json") -> Dict[str, Any]:
        """
        Экспортирует граф в различных форматах.
        
        Args:
            format_type: Тип формата ("json", "dict")
            
        Returns:
            Экспортированные данные
        """
        export_data = {
            'concepts': [concept.to_dict() for concept in self.concepts.values()],
            'relationships': [rel.to_dict() for rel in self.relationships.values()],
            'metadata': {
                'exported_at': time.time(),
                'total_concepts': len(self.concepts),
                'total_relationships': len(self.relationships),
                'version': '1.0'
            }
        }
        
        if format_type == "json":
            return export_data
        elif format_type == "dict":
            return export_data
        else:
            raise ValueError(f"Неподдерживаемый формат экспорта: {format_type}")
    
    def import_graph(self, data: Dict[str, Any]) -> bool:
        """
        Импортирует граф из данных.
        
        Args:
            data: Данные для импорта
            
        Returns:
            True если успешно импортирован
        """
        try:
            # Очищаем текущий граф
            self.concepts.clear()
            self.relationships.clear()
            self.name_index.clear()
            self.category_index.clear()
            self.tag_index.clear()
            self.type_index.clear()
            
            # Импортируем концепты
            concepts_cache = {}
            for concept_data in data.get('concepts', []):
                concept = Concept.from_dict(concept_data, concepts_cache)
                self.concepts[concept.id] = concept
                self.name_index[concept.name] = concept.id
                if concept.category:
                    self.category_index[concept.category].add(concept.id)
                for tag in concept.tags:
                    self.tag_index[tag].add(concept.id)
            
            # Импортируем отношения
            for rel_data in data.get('relationships', []):
                relationship = Relationship.from_dict(rel_data)
                if (relationship.source_id in self.concepts and 
                    relationship.target_id in self.concepts):
                    self.relationships[relationship.id] = relationship
                    relationship.source = self.concepts[relationship.source_id]
                    relationship.target = self.concepts[relationship.target_id]
                    relationship.source.add_relationship(relationship)
                    relationship.target.add_relationship(relationship)
                    self.type_index[relationship.relationship_type].add(relationship.id)
            
            # Обновляем статистику
            self.stats['total_concepts'] = len(self.concepts)
            self.stats['total_relationships'] = len(self.relationships)
            
            return True
            
        except Exception as e:
            print(f"Ошибка при импорте графа: {e}")
            return False
    
    # Методы интеграции с Memory Manager
    
    async def _persist_concept(self, concept: Concept) -> None:
        """Сохраняет концепт в память."""
        if not self.memory_manager:
            return
        
        try:
            memory_data = concept.to_dict()
            await self.memory_manager.store(
                layer=MemoryLayer.SEMANTIC,
                data=memory_data,
                metadata={
                    'type': 'kag_concept',
                    'concept_id': concept.id,
                    'name': concept.name,
                    'category': concept.category
                }
            )
        except Exception as e:
            print(f"Ошибка при сохранении концепта в память: {e}")
    
    async def _persist_relationship(self, relationship: Relationship) -> None:
        """Сохраняет отношение в память."""
        if not self.memory_manager:
            return
        
        try:
            memory_data = relationship.to_dict()
            await self.memory_manager.store(
                layer=MemoryLayer.SEMANTIC,
                data=memory_data,
                metadata={
                    'type': 'kag_relationship',
                    'relationship_id': relationship.id,
                    'source_id': relationship.source_id,
                    'target_id': relationship.target_id,
                    'relationship_type': relationship.relationship_type.value
                }
            )
        except Exception as e:
            print(f"Ошибка при сохранении отношения в память: {e}")
    
    async def _remove_concept_from_memory(self, concept: Concept) -> None:
        """Удаляет концепт из памяти."""
        if not self.memory_manager:
            return
        
        try:
            # Поиск и удаление из памяти
            results = await self.memory_manager.retrieve(
                layer=MemoryLayer.SEMANTIC,
                query={'type': 'kag_concept', 'concept_id': concept.id}
            )
            
            for item in results:
                await self.memory_manager.delete(
                    layer=MemoryLayer.SEMANTIC,
                    item_id=item.id
                )
        except Exception as e:
            print(f"Ошибка при удалении концепта из памяти: {e}")
    
    async def _remove_relationship_from_memory(self, relationship: Relationship) -> None:
        """Удаляет отношение из памяти."""
        if not self.memory_manager:
            return
        
        try:
            # Поиск и удаление из памяти
            results = await self.memory_manager.retrieve(
                layer=MemoryLayer.SEMANTIC,
                query={'type': 'kag_relationship', 'relationship_id': relationship.id}
            )
            
            for item in results:
                await self.memory_manager.delete(
                    layer=MemoryLayer.SEMANTIC,
                    item_id=item.id
                )
        except Exception as e:
            print(f"Ошибка при удалении отношения из памяти: {e}")
    
    async def load_from_memory(self) -> bool:
        """
        Загружает граф из памяти.
        
        Returns:
            True если успешно загружен
        """
        if not self.memory_manager:
            return False
        
        try:
            # Загружаем концепты
            concept_results = await self.memory_manager.retrieve(
                layer=MemoryLayer.SEMANTIC,
                query={'type': 'kag_concept'}
            )
            
            concepts_cache = {}
            for item in concept_results:
                concept_data = item.data
                concept = Concept.from_dict(concept_data, concepts_cache)
                self.concepts[concept.id] = concept
                self.name_index[concept.name] = concept.id
                if concept.category:
                    self.category_index[concept.category].add(concept.id)
                for tag in concept.tags:
                    self.tag_index[tag].add(concept.id)
            
            # Загружаем отношения
            relationship_results = await self.memory_manager.retrieve(
                layer=MemoryLayer.SEMANTIC,
                query={'type': 'kag_relationship'}
            )
            
            for item in relationship_results:
                relationship_data = item.data
                relationship = Relationship.from_dict(relationship_data)
                
                if (relationship.source_id in self.concepts and 
                    relationship.target_id in self.concepts):
                    self.relationships[relationship.id] = relationship
                    relationship.source = self.concepts[relationship.source_id]
                    relationship.target = self.concepts[relationship.target_id]
                    relationship.source.add_relationship(relationship)
                    relationship.target.add_relationship(relationship)
                    self.type_index[relationship.relationship_type].add(relationship.id)
            
            # Обновляем статистику
            self.stats['total_concepts'] = len(self.concepts)
            self.stats['total_relationships'] = len(self.relationships)
            
            return True
            
        except Exception as e:
            print(f"Ошибка при загрузке графа из памяти: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Получает статистику графа."""
        return self.stats.copy()
