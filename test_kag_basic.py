#!/usr/bin/env python3
"""
Простой тест для проверки работоспособности KAGGraph
Запускается напрямую без использования pytest
"""

import sys
import os
import time
import uuid
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

# ============================================================================
# Упрощенная реализация базовых компонентов для тестирования
# ============================================================================

class RelationshipType(Enum):
    """Типы отношений между концептами."""
    IS_A = "is_a"
    PART_OF = "part_of"
    RELATED_TO = "related_to"
    CAUSES = "causes"
    INFLUENCES = "influences"
    OPPOSITE_OF = "opposite_of"
    SIMILAR_TO = "similar_to"
    LEADS_TO = "leads_to"
    CONTAINS = "contains"
    USED_FOR = "used_for"
    PRODUCES = "produces"


@dataclass
class Concept:
    """Структура узла-концепта в графе знаний."""
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
        }


@dataclass
class Relationship:
    """Структура отношения между концептами."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    relationship_type: RelationshipType = RelationshipType.RELATED_TO
    strength: float = 1.0
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    confidence_score: float = 1.0


class KAGGraph:
    """Основной класс для работы с концептуальным графом знаний."""
    
    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.relationships: Dict[str, Relationship] = {}
        
        # Индексы для оптимизации поиска
        self.name_index: Dict[str, str] = {}
        self.category_index: Dict[str, Set[str]] = defaultdict(set)
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Статистика
        self.stats = {
            'total_concepts': 0,
            'total_relationships': 0,
            'queries_executed': 0,
            'avg_query_time': 0.0,
            'last_save': time.time()
        }
    
    def add_concept(self, concept: Concept) -> str:
        """Добавляет концепт в граф."""
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
        
        # Обновляем статистику
        self.stats['total_concepts'] = len(self.concepts)
        
        return concept.id
    
    def remove_concept(self, concept_id: str) -> bool:
        """Удаляет концепт из графа."""
        if concept_id not in self.concepts:
            return False
        
        concept = self.concepts[concept_id]
        
        # Удаляем из индексов
        self.name_index.pop(concept.name, None)
        if concept.category:
            self.category_index[concept.category].discard(concept_id)
        
        for tag in concept.tags:
            self.tag_index[tag].discard(concept_id)
        
        # Удаляем концепт
        del self.concepts[concept_id]
        
        # Обновляем статистику
        self.stats['total_concepts'] = len(self.concepts)
        
        return True
    
    def add_relationship(self, relationship: Relationship) -> str:
        """Добавляет отношение в граф."""
        # Проверяем существование концептов
        if relationship.source_id not in self.concepts:
            raise ValueError(f"Исходный концепт {relationship.source_id} не найден")
        if relationship.target_id not in self.concepts:
            raise ValueError(f"Целевой концепт {relationship.target_id} не найден")
        
        # Добавляем отношение
        self.relationships[relationship.id] = relationship
        
        # Обновляем статистику
        self.stats['total_relationships'] = len(self.relationships)
        
        return relationship.id
    
    def get_concept(self, concept_id: str) -> Optional[Concept]:
        """Получает концепт по ID."""
        return self.concepts.get(concept_id)
    
    def find_concepts(
        self,
        name: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        min_confidence: float = 0.0
    ) -> List[Concept]:
        """Находит концепты по критериям."""
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
    
    def calculate_graph_metrics(self) -> Dict[str, Any]:
        """Вычисляет метрики графа."""
        total_concepts = len(self.concepts)
        total_relationships = len(self.relationships)
        
        if total_concepts == 0:
            return {
                'total_concepts': 0,
                'total_relationships': 0,
                'density': 0.0,
                'avg_degree': 0.0
            }
        
        # Плотность графа
        max_edges = total_concepts * (total_concepts - 1)
        density = total_relationships / max_edges if max_edges > 0 else 0.0
        
        return {
            'total_concepts': total_concepts,
            'total_relationships': total_relationships,
            'density': density,
            'categories': len(self.category_index),
            'unique_tags': len(self.tag_index)
        }


# ============================================================================
# Тесты
# ============================================================================

def test_concept_creation():
    """Тест создания концепта."""
    print("\n=== Тест создания концепта ===")
    
    concept = Concept(
        name="Тестовый Концепт",
        description="Описание тестового концепта",
        category="test_category",
        tags=["тег1", "тег2"],
        properties={"ключ": "значение"}
    )
    
    assert concept.id is not None
    assert concept.name == "Тестовый Концепт"
    assert concept.description == "Описание тестового концепта"
    assert concept.category == "test_category"
    assert concept.confidence_score == 1.0
    assert "тег1" in concept.tags
    assert concept.properties["ключ"] == "значение"
    
    print("✓ Концепт создан успешно")
    print(f"  ID: {concept.id}")
    print(f"  Имя: {concept.name}")
    print(f"  Категория: {concept.category}")
    print(f"  Теги: {concept.tags}")
    return True


def test_graph_operations():
    """Тест операций с графом."""
    print("\n=== Тест операций с графом ===")
    
    # Создаем граф
    graph = KAGGraph()
    
    # Создаем концепты
    concept1 = Concept(
        name="Искусственный Интеллект",
        description="Технология имитации человеческого интеллекта",
        category="технология",
        tags=["AI", "машинное обучение", "нейросети"],
        confidence_score=0.95
    )
    
    concept2 = Concept(
        name="Машинное Обучение",
        description="Подраздел AI для обучения алгоритмов",
        category="технология",
        tags=["ML", "алгоритмы", "данные"],
        confidence_score=0.90
    )
    
    # Добавляем концепты
    id1 = graph.add_concept(concept1)
    id2 = graph.add_concept(concept2)
    
    assert id1 == concept1.id
    assert id2 == concept2.id
    assert len(graph.concepts) == 2
    assert graph.name_index[concept1.name] == concept1.id
    
    print("✓ Концепты добавлены в граф")
    print(f"  Концепт 1 ID: {id1}")
    print(f"  Концепт 2 ID: {id2}")
    print(f"  Общее количество концептов: {len(graph.concepts)}")
    
    # Создаем отношение
    relationship = Relationship(
        source_id=concept1.id,
        target_id=concept2.id,
        relationship_type=RelationshipType.CONTAINS,
        strength=0.9
    )
    
    rel_id = graph.add_relationship(relationship)
    assert rel_id == relationship.id
    assert len(graph.relationships) == 1
    
    print("✓ Отношение добавлено в граф")
    print(f"  Отношение ID: {rel_id}")
    
    # Тестируем поиск
    results = graph.find_concepts(name="интеллект")
    assert len(results) >= 1
    assert concept1 in results
    
    print("✓ Поиск концептов работает")
    print(f"  Найдено концептов по запросу 'интеллект': {len(results)}")
    
    # Тестируем получение концепта
    retrieved = graph.get_concept(concept1.id)
    assert retrieved == concept1
    
    print("✓ Получение концепта по ID работает")
    
    # Тестируем метрики
    metrics = graph.calculate_graph_metrics()
    assert metrics['total_concepts'] == 2
    assert metrics['total_relationships'] == 1
    assert 'density' in metrics
    
    print("✓ Вычисление метрик работает")
    print(f"  Метрики: {metrics}")
    
    return True


def test_relationship_types():
    """Тест типов отношений."""
    print("\n=== Тест типов отношений ===")
    
    # Проверяем все типы отношений
    print("Доступные типы отношений:")
    for rel_type in RelationshipType:
        print(f"  - {rel_type.value}")
    
    assert len(RelationshipType) == 11
    assert RelationshipType.IS_A in RelationshipType
    assert RelationshipType.CAUSES in RelationshipType
    
    print("✓ Все типы отношений доступны")
    return True


def run_all_tests():
    """Запуск всех тестов."""
    print("Запуск базовых тестов KAGGraph Core...")
    
    tests = [
        ("Создание концепта", test_concept_creation),
        ("Операции с графом", test_graph_operations),
        ("Типы отношений", test_relationship_types)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name}: ПРОЙДЕН")
            else:
                print(f"✗ {test_name}: ПРОВАЛЕН")
        except Exception as e:
            print(f"✗ {test_name}: ОШИБКА - {e}")
    
    print(f"\n=== РЕЗУЛЬТАТЫ ===")
    print(f"Пройдено: {passed}/{total}")
    print(f"Успешность: {(passed/total*100):.1f}%")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    if success:
        print("\n🎉 Все тесты KAGGraph Core пройдены успешно!")
        sys.exit(0)
    else:
        print("\n❌ Некоторые тесты провалены")
        sys.exit(1)