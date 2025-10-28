"""
Комплексные тесты для KAGGraph системы.
Тестирует все основные компоненты: концепты, отношения, запросы, персистентность.
"""

import pytest
import asyncio
import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Set
import json

# Импорты тестируемой системы
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from knowledge_graph import (
    KAGGraph, Concept, Relationship, RelationshipType, QueryResult,
    GraphTraversal, TraversalType, GraphPersistence, KAGQueryEngine,
    SearchFilters
)


class TestConcept:
    """Тесты для класса Concept."""
    
    def test_create_concept(self):
        """Тест создания концепта."""
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
    
    def test_concept_relationships(self):
        """Тест управления отношениями концепта."""
        concept1 = Concept(name="Концепт 1")
        concept2 = Concept(name="Концепт 2")
        
        relationship = Relationship(
            source_id=concept1.id,
            target_id=concept2.id,
            relationship_type=RelationshipType.RELATED_TO,
            strength=0.8
        )
        
        concept1.add_relationship(relationship)
        concept2.add_relationship(relationship)
        
        assert len(concept1.outgoing_relationships) == 1
        assert len(concept2.incoming_relationships) == 1
        assert concept1.outgoing_relationships[relationship.id] == relationship
        assert concept2.incoming_relationships[relationship.id] == relationship
        
        # Тест удаления отношения
        assert concept1.remove_relationship(relationship.id)
        assert len(concept1.outgoing_relationships) == 0
        assert len(concept2.incoming_relationships) == 0
    
    def test_concept_similarity(self):
        """Тест вычисления схожести концептов."""
        concept1 = Concept(
            name="Кот",
            category="животное",
            tags=["пушистый", "млекопитающее"],
            properties={"размер": "маленький"}
        )
        
        concept2 = Concept(
            name="Собака",
            category="животное",
            tags=["пушистый", "млекопитающее"],
            properties={"размер": "средний"}
        )
        
        concept3 = Concept(
            name="Автомобиль",
            category="техника",
            tags=["транспорт"],
            properties={"размер": "большой"}
        )
        
        similarity_12 = concept1.calculate_similarity(concept2)
        similarity_13 = concept1.calculate_similarity(concept3)
        
        assert similarity_12 > similarity_13
        assert 0.0 <= similarity_12 <= 1.0
        assert 0.0 <= similarity_13 <= 1.0
    
    def test_concept_to_dict(self):
        """Тест сериализации концепта."""
        concept = Concept(
            name="Тестовый Концепт",
            description="Описание",
            category="test"
        )
        
        concept_dict = concept.to_dict()
        
        assert isinstance(concept_dict, dict)
        assert concept_dict['name'] == "Тестовый Концепт"
        assert concept_dict['category'] == "test"
        assert 'relationships' in concept_dict
    
    def test_concept_from_dict(self):
        """Тест десериализации концепта."""
        concept_data = {
            'id': 'test-id',
            'name': 'Импортированный концепт',
            'description': 'Импортированное описание',
            'category': 'imported',
            'confidence_score': 0.9,
            'tags': ['imported'],
            'properties': {},
            'relationships': []
        }
        
        concept = Concept.from_dict(concept_data)
        
        assert concept.id == 'test-id'
        assert concept.name == 'Импортированный концепт'
        assert concept.confidence_score == 0.9


class TestRelationship:
    """Тесты для класса Relationship."""
    
    def test_create_relationship(self):
        """Тест создания отношения."""
        source_concept = Concept(name="Источник")
        target_concept = Concept(name="Цель")
        
        relationship = Relationship(
            source_id=source_concept.id,
            target_id=target_concept.id,
            relationship_type=RelationshipType.IS_A,
            strength=0.9,
            description="Тестовое отношение"
        )
        
        assert relationship.id is not None
        assert relationship.source_id == source_concept.id
        assert relationship.target_id == target_concept.id
        assert relationship.relationship_type == RelationshipType.IS_A
        assert relationship.strength == 0.9
        assert relationship.confidence_score == 1.0
    
    def test_relationship_serialization(self):
        """Тест сериализации отношения."""
        relationship = Relationship(
            source_id="source-1",
            target_id="target-1",
            relationship_type=RelationshipType.CAUSES,
            strength=0.7
        )
        
        relationship_dict = relationship.to_dict()
        
        assert isinstance(relationship_dict, dict)
        assert relationship_dict['source_id'] == "source-1"
        assert relationship_dict['relationship_type'] == "causes"
        assert relationship_dict['strength'] == 0.7
        
        # Тест десериализации
        restored_relationship = Relationship.from_dict(relationship_dict)
        assert restored_relationship.source_id == "source-1"
        assert restored_relationship.relationship_type == RelationshipType.CAUSES


class TestKAGGraph:
    """Тесты для основного класса KAGGraph."""
    
    def setup_method(self):
        """Настройка для каждого теста."""
        self.graph = KAGGraph()
        
        # Создаем тестовые концепты
        self.concept1 = Concept(
            name="Искусственный Интеллект",
            description="Технология имитации человеческого интеллекта",
            category="технология",
            tags=["AI", "машинное обучение", "нейросети"],
            confidence_score=0.95
        )
        
        self.concept2 = Concept(
            name="Машинное Обучение",
            description="Подраздел AI для обучения алгоритмов",
            category="технология",
            tags=["ML", "алгоритмы", "данные"],
            confidence_score=0.90
        )
        
        self.concept3 = Concept(
            name="Python",
            description="Язык программирования",
            category="язык программирования",
            tags=["программирование", "язык"],
            confidence_score=0.85
        )
    
    def test_add_concept(self):
        """Тест добавления концепта в граф."""
        concept_id = self.graph.add_concept(self.concept1)
        
        assert concept_id == self.concept1.id
        assert self.concept1.id in self.graph.concepts
        assert self.graph.concepts[self.concept1.id] == self.concept1
        assert self.graph.name_index[self.concept1.name] == self.concept1.id
    
    def test_remove_concept(self):
        """Тест удаления концепта из графа."""
        # Добавляем концепт с отношением
        self.graph.add_concept(self.concept1)
        self.graph.add_concept(self.concept2)
        
        relationship = Relationship(
            source_id=self.concept1.id,
            target_id=self.concept2.id,
            relationship_type=RelationshipType.RELATED_TO
        )
        
        self.graph.add_relationship(relationship)
        
        # Удаляем концепт
        result = self.graph.remove_concept(self.concept1.id)
        
        assert result is True
        assert self.concept1.id not in self.graph.concepts
        assert self.concept1.name not in self.graph.name_index
        assert self.concept2.id in self.graph.concepts  # Другой концепт не удален
        
        # Отношение должно быть удалено автоматически
        assert relationship.id not in self.graph.relationships
    
    def test_add_relationship(self):
        """Тест добавления отношения в граф."""
        self.graph.add_concept(self.concept1)
        self.graph.add_concept(self.concept2)
        
        relationship = Relationship(
            source_id=self.concept1.id,
            target_id=self.concept2.id,
            relationship_type=RelationshipType.LEADS_TO,
            strength=0.8
        )
        
        rel_id = self.graph.add_relationship(relationship)
        
        assert rel_id == relationship.id
        assert rel_id in self.graph.relationships
        assert self.graph.relationships[rel_id] == relationship
        assert relationship.source == self.concept1
        assert relationship.target == self.concept2
    
    def test_remove_relationship(self):
        """Тест удаления отношения из графа."""
        self.graph.add_concept(self.concept1)
        self.graph.add_concept(self.concept2)
        
        relationship = Relationship(
            source_id=self.concept1.id,
            target_id=self.concept2.id,
            relationship_type=RelationshipType.RELATED_TO
        )
        
        rel_id = self.graph.add_relationship(relationship)
        
        result = self.graph.remove_relationship(rel_id)
        
        assert result is True
        assert rel_id not in self.graph.relationships
        assert len(self.concept1.outgoing_relationships) == 0
        assert len(self.concept2.incoming_relationships) == 0
    
    def test_get_concept(self):
        """Тест получения концепта по ID."""
        self.graph.add_concept(self.concept1)
        
        retrieved = self.graph.get_concept(self.concept1.id)
        assert retrieved == self.concept1
        
        # Тест несуществующего концепта
        assert self.graph.get_concept("non-existent") is None
    
    def test_get_concept_by_name(self):
        """Тест получения концепта по имени."""
        self.graph.add_concept(self.concept1)
        
        retrieved = self.graph.get_concept_by_name("Искусственный Интеллект")
        assert retrieved == self.concept1
        
        # Тест несуществующего имени
        assert self.graph.get_concept_by_name("Несуществующий") is None
    
    def test_find_concepts(self):
        """Тест поиска концептов."""
        self.graph.add_concept(self.concept1)
        self.graph.add_concept(self.concept2)
        self.graph.add_concept(self.concept3)
        
        # Поиск по имени
        results = self.graph.find_concepts(name="интеллект")
        assert len(results) >= 1
        assert self.concept1 in results
        
        # Поиск по категории
        results = self.graph.find_concepts(category="технология")
        assert len(results) == 2
        assert self.concept1 in results
        assert self.concept2 in results
        
        # Поиск по тегам
        results = self.graph.find_concepts(tags={"AI"})
        assert len(results) == 1
        assert self.concept1 in results
    
    def test_get_connected_concepts(self):
        """Тест получения связанных концептов."""
        self.graph.add_concept(self.concept1)
        self.graph.add_concept(self.concept2)
        
        relationship = Relationship(
            source_id=self.concept1.id,
            target_id=self.concept2.id,
            relationship_type=RelationshipType.RELATED_TO
        )
        
        self.graph.add_relationship(relationship)
        
        connected = self.graph.get_connected_concepts(
            self.concept1.id,
            max_depth=1
        )
        
        assert 1 in connected
        assert self.concept2 in connected[1]
    
    def test_find_shortest_path(self):
        """Тест поиска кратчайшего пути."""
        # Создаем цепочку: A -> B -> C
        concept_a = Concept(name="A")
        concept_b = Concept(name="B") 
        concept_c = Concept(name="C")
        
        self.graph.add_concept(concept_a)
        self.graph.add_concept(concept_b)
        self.graph.add_concept(concept_c)
        
        rel1 = Relationship(
            source_id=concept_a.id,
            target_id=concept_b.id,
            relationship_type=RelationshipType.RELATED_TO
        )
        
        rel2 = Relationship(
            source_id=concept_b.id,
            target_id=concept_c.id,
            relationship_type=RelationshipType.RELATED_TO
        )
        
        self.graph.add_relationship(rel1)
        self.graph.add_relationship(rel2)
        
        # Ищем путь A -> C
        path = self.graph.find_shortest_path(concept_a.id, concept_c.id)
        
        assert path is not None
        assert len(path) == 3
        assert path[0] == concept_a.id
        assert path[1] == concept_b.id
        assert path[2] == concept_c.id
        
        # Тест недостижимого пути
        concept_d = Concept(name="D")
        self.graph.add_concept(concept_d)
        
        path = self.graph.find_shortest_path(concept_a.id, concept_d.id)
        assert path is None
    
    def test_calculate_graph_metrics(self):
        """Тест вычисления метрик графа."""
        # Добавляем концепты и отношения
        self.graph.add_concept(self.concept1)
        self.graph.add_concept(self.concept2)
        self.graph.add_concept(self.concept3)
        
        relationship1 = Relationship(
            source_id=self.concept1.id,
            target_id=self.concept2.id,
            relationship_type=RelationshipType.RELATED_TO
        )
        
        relationship2 = Relationship(
            source_id=self.concept2.id,
            target_id=self.concept3.id,
            relationship_type=RelationshipType.RELATED_TO
        )
        
        self.graph.add_relationship(relationship1)
        self.graph.add_relationship(relationship2)
        
        metrics = self.graph.calculate_graph_metrics()
        
        assert metrics['total_concepts'] == 3
        assert metrics['total_relationships'] == 2
        assert 'density' in metrics
        assert 'avg_degree' in metrics
        assert 'clustering_coefficient' in metrics
        assert 0.0 <= metrics['density'] <= 1.0
    
    def test_export_import_graph(self):
        """Тест экспорта и импорта графа."""
        # Создаем граф с данными
        self.graph.add_concept(self.concept1)
        self.graph.add_concept(self.concept2)
        
        relationship = Relationship(
            source_id=self.concept1.id,
            target_id=self.concept2.id,
            relationship_type=RelationshipType.RELATED_TO
        )
        
        self.graph.add_relationship(relationship)
        
        # Экспортируем
        export_data = self.graph.export_graph("json")
        
        assert 'concepts' in export_data
        assert 'relationships' in export_data
        assert 'metadata' in export_data
        assert len(export_data['concepts']) == 2
        assert len(export_data['relationships']) == 1
        
        # Создаем новый граф и импортируем
        new_graph = KAGGraph()
        success = new_graph.import_graph(export_data)
        
        assert success is True
        assert len(new_graph.concepts) == 2
        assert len(new_graph.relationships) == 1
        
        # Проверяем, что концепты импортированы корректно
        imported_concept = new_graph.get_concept(self.concept1.id)
        assert imported_concept is not None
        assert imported_concept.name == self.concept1.name


class TestGraphTraversal:
    """Тесты для алгоритмов обхода графа."""
    
    def setup_method(self):
        """Настройка для каждого теста."""
        self.graph = KAGGraph()
        self.traversal = GraphTraversal(self.graph)
        
        # Создаем тестовую структуру графа
        self.nodes = {}
        for i in range(6):
            concept = Concept(name=f"Узел_{i}")
            self.graph.add_concept(concept)
            self.nodes[i] = concept
        
        # Создаем связи: 0 -> 1 -> 2 -> 3 -> 4 -> 5
        for i in range(5):
            relationship = Relationship(
                source_id=self.nodes[i].id,
                target_id=self.nodes[i+1].id,
                relationship_type=RelationshipType.RELATED_TO,
                strength=1.0 - (i * 0.1)  # Убывающая сила связи
            )
            self.graph.add_relationship(relationship)
    
    def test_bfs_algorithm(self):
        """Тест алгоритма поиска в ширину."""
        result = self.traversal.find_path(
            source_id=self.nodes[0].id,
            target_id=self.nodes[3].id,
            algorithm=TraversalType.BFS
        )
        
        assert result is not None
        assert len(result.path) == 4
        assert result.path[0] == self.nodes[0].id
        assert result.path[3] == self.nodes[3].id
        assert result.cost > 0
        assert 0.0 <= result.confidence <= 1.0
    
    def test_dfs_algorithm(self):
        """Тест алгоритма поиска в глубину."""
        result = self.traversal.find_path(
            source_id=self.nodes[0].id,
            target_id=self.nodes[3].id,
            algorithm=TraversalType.DFS
        )
        
        assert result is not None
        assert result.path[0] == self.nodes[0].id
        assert result.path[-1] == self.nodes[3].id
        assert len(result.path) >= 4
    
    def test_dijkstra_algorithm(self):
        """Тест алгоритма Дейкстры."""
        result = self.traversal.find_path(
            source_id=self.nodes[0].id,
            target_id=self.nodes[5].id,
            algorithm=TraversalType.DIJKSTRA
        )
        
        assert result is not None
        assert result.path[0] == self.nodes[0].id
        assert result.path[-1] == self.nodes[5].id
        assert result.cost > 0
    
    def test_find_all_paths(self):
        """Тест поиска всех путей."""
        # Добавляем дополнительные связи для создания альтернативных путей
        # Создаем альтернативный путь: 0 -> 2 -> 3 -> 4 -> 5
        
        alt_relationship1 = Relationship(
            source_id=self.nodes[0].id,
            target_id=self.nodes[2].id,
            relationship_type=RelationshipType.RELATED_TO,
            strength=0.5
        )
        
        alt_relationship2 = Relationship(
            source_id=self.nodes[2].id,
            target_id=self.nodes[3].id,
            relationship_type=RelationshipType.RELATED_TO,
            strength=0.8
        )
        
        self.graph.add_relationship(alt_relationship1)
        self.graph.add_relationship(alt_relationship2)
        
        all_paths = self.traversal.find_all_paths(
            source_id=self.nodes[0].id,
            target_id=self.nodes[3].id,
            max_paths=5,
            max_depth=10
        )
        
        assert len(all_paths) >= 2
        for path_result in all_paths:
            assert len(path_result.path) > 0
            assert path_result.path[0] == self.nodes[0].id
            assert path_result.path[-1] == self.nodes[3].id
    
    def test_analyze_connectivity(self):
        """Тест анализа связности графа."""
        connectivity = self.traversal.analyze_connectivity()
        
        assert 'is_connected' in connectivity
        assert 'components' in connectivity
        assert 'largest_component_size' in connectivity
        assert 'isolated_nodes' in connectivity
        
        # Граф должен быть связным (все узлы связаны)
        assert connectivity['is_connected'] is True
        assert connectivity['components'] == 1
        assert connectivity['largest_component_size'] == 6
        assert connectivity['isolated_nodes'] == 0
    
    def test_get_centrality_metrics(self):
        """Тест вычисления метрик центральности."""
        centrality = self.traversal.get_centrality_metrics()
        
        assert 'degree_centrality' in centrality
        assert 'betweenness_centrality' in centrality
        assert 'closeness_centrality' in centrality
        
        # Узел 2 должен иметь высокую центральность (связывает концы)
        node_2_id = self.nodes[2].id
        assert node_2_id in centrality['degree_centrality']
        
        # Проверяем, что значения центральности находятся в разумных пределах
        for metric in centrality.values():
            for node_id, value in metric.items():
                assert 0.0 <= value <= 1.0


class TestGraphPersistence:
    """Тесты для системы персистентности."""
    
    def setup_method(self):
        """Настройка для каждого теста."""
        self.temp_dir = tempfile.mkdtemp()
        self.graph = KAGGraph(persistence_path=self.temp_dir)
        self.persistence = GraphPersistence(self.graph, self.temp_dir)
        
        # Создаем тестовые данные
        self.test_concept = Concept(
            name="Тестовый Концепт",
            description="Для тестирования персистентности",
            category="test",
            tags=["persistence", "test"]
        )
        
        self.test_relationship = Relationship(
            source_id=self.test_concept.id,
            target_id=self.test_concept.id,  # Самоотношение для простоты
            relationship_type=RelationshipType.SIMILAR_TO,
            strength=1.0
        )
    
    def teardown_method(self):
        """Очистка после тестов."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_load_json(self):
        """Тест сохранения и загрузки в JSON."""
        # Добавляем данные в граф
        self.graph.add_concept(self.test_concept)
        self.graph.add_relationship(self.test_relationship)
        
        # Сохраняем
        filename = self.persistence.save_graph("json")
        assert filename.endswith(".json")
        
        # Создаем новый граф и загружаем
        new_graph = KAGGraph()
        new_persistence = GraphPersistence(new_graph, self.temp_dir)
        
        success = new_persistence.load_graph(Path(filename).name)
        assert success is True
        
        # Проверяем, что данные загружены корректно
        assert len(new_graph.concepts) == 1
        assert len(new_graph.relationships) == 1
        
        loaded_concept = new_graph.get_concept(self.test_concept.id)
        assert loaded_concept is not None
        assert loaded_concept.name == self.test_concept.name
    
    def test_create_restore_backup(self):
        """Тест создания и восстановления резервных копий."""
        # Добавляем данные
        self.graph.add_concept(self.test_concept)
        self.graph.add_relationship(self.test_relationship)
        
        # Создаем резервную копию
        backup_name = self.persistence.create_backup("test_backup")
        assert backup_name.startswith("backup_")
        assert "test_backup" in backup_name
        
        # Проверяем список резервных копий
        backups = self.persistence.list_backups()
        assert len(backups) >= 1
        
        backup_info = backups[0]  # Самая новая
        assert backup_info['filename'] == backup_name
        assert backup_info['concepts_count'] == 1
        assert backup_info['relationships_count'] == 1
        
        # Восстанавливаем из резервной копии
        new_graph = KAGGraph()
        new_persistence = GraphPersistence(new_graph, self.temp_dir)
        
        success = new_persistence.restore_from_backup(backup_name)
        assert success is True
        assert len(new_graph.concepts) == 1
        
        restored_concept = new_graph.get_concept(self.test_concept.id)
        assert restored_concept is not None
        assert restored_concept.name == self.test_concept.name
    
    def test_validate_graph_data(self):
        """Тест валидации данных графа."""
        # Корректные данные
        valid_data = {
            'concepts': [
                {
                    'id': 'valid-concept-1',
                    'name': 'Валидный концепт',
                    'description': 'Описание',
                    'category': 'test',
                    'confidence_score': 1.0,
                    'tags': [],
                    'metadata': {}
                }
            ],
            'relationships': [
                {
                    'id': 'valid-rel-1',
                    'source_id': 'valid-concept-1',
                    'target_id': 'valid-concept-1',
                    'relationship_type': 'related_to',
                    'strength': 1.0,
                    'description': '',
                    'metadata': {}
                }
            ]
        }
        
        validation_result = self.persistence.validate_graph_data(valid_data)
        assert validation_result['is_valid'] is True
        assert len(validation_result['errors']) == 0
        
        # Некорректные данные (отсутствует секция concepts)
        invalid_data = {
            'relationships': []
        }
        
        validation_result = self.persistence.validate_graph_data(invalid_data)
        assert validation_result['is_valid'] is False
        assert len(validation_result['errors']) > 0
        assert "Отсутствует секция 'concepts'" in validation_result['errors']
    
    def test_get_storage_stats(self):
        """Тест получения статистики хранилища."""
        # Добавляем данные
        self.graph.add_concept(self.test_concept)
        
        # Сохраняем файл
        self.persistence.save_graph("json")
        
        stats = self.persistence.get_storage_stats()
        
        assert 'storage_path' in stats
        assert 'main_files' in stats
        assert 'backups' in stats
        assert 'total_size' in stats
        assert 'backup_count' in stats
        
        assert stats['total_size'] > 0
        assert len(stats['main_files']) >= 1


class TestQueryEngine:
    """Тесты для системы запросов."""
    
    def setup_method(self):
        """Настройка для каждого теста."""
        self.graph = KAGGraph()
        self.query_engine = KAGQueryEngine(self.graph)
        
        # Создаем тестовые данные
        self.ai_concept = Concept(
            name="Искусственный Интеллект",
            description="Технология создания умных систем",
            category="технология",
            tags=["AI", "машинное обучение", "интеллект"],
            confidence_score=0.95
        )
        
        self.ml_concept = Concept(
            name="Машинное Обучение",
            description="Методы обучения компьютеров",
            category="технология",
            tags=["ML", "обучение", "алгоритмы"],
            confidence_score=0.90
        )
        
        self.python_concept = Concept(
            name="Python",
            description="Популярный язык программирования",
            category="язык программирования",
            tags=["программирование", "язык", "AI"],
            confidence_score=0.85
        )
        
        # Добавляем в граф
        self.graph.add_concept(self.ai_concept)
        self.graph.add_concept(self.ml_concept)
        self.graph.add_concept(self.python_concept)
        
        # Добавляем отношения
        self.ai_ml_rel = Relationship(
            source_id=self.ai_concept.id,
            target_id=self.ml_concept.id,
            relationship_type=RelationshipType.CONTAINS,
            strength=0.9
        )
        
        self.ml_python_rel = Relationship(
            source_id=self.ml_concept.id,
            target_id=self.python_concept.id,
            relationship_type=RelationshipType.USED_FOR,
            strength=0.8
        )
        
        self.graph.add_relationship(self.ai_ml_rel)
        self.graph.add_relationship(self.ml_python_rel)
    
    def test_find_concepts_by_search_term(self):
        """Тест поиска концептов по поисковому термину."""
        result = self.query_engine.find_concepts(search_term="интеллект")
        
        assert result.total_results >= 1
        assert self.ai_concept in result.results
        
        # Проверяем, что результат отсортирован по релевантности
        if len(result.scores) > 1:
            assert result.scores[0] >= result.scores[1]
    
    def test_find_concepts_by_category(self):
        """Тест поиска концептов по категории."""
        filters = SearchFilters(categories={"технология"})
        result = self.query_engine.find_concepts(filters=filters)
        
        assert result.total_results == 2
        assert self.ai_concept in result.results
        assert self.ml_concept in result.results
        assert self.python_concept not in result.results
    
    def test_find_concepts_by_tags(self):
        """Тест поиска концептов по тегам."""
        filters = SearchFilters(tags={"AI", "машинное обучение"})
        result = self.query_engine.find_concepts(filters=filters)
        
        # Должен найти концепт, который содержит оба тега
        assert result.total_results >= 1
        assert self.ai_concept in result.results
    
    def test_find_relationships(self):
        """Тест поиска отношений."""
        result = self.query_engine.find_relationships(
            source_concept_id=self.ai_concept.id
        )
        
        assert result.total_results >= 1
        
        # Проверяем статистику запроса
        stats = self.query_engine.get_query_stats()
        assert stats['total_queries'] > 0
    
    def test_find_path(self):
        """Тест поиска пути между концептами."""
        result = self.query_engine.find_path(
            source_concept_id=self.ai_concept.id,
            target_concept_id=self.python_concept.id
        )
        
        assert result.total_results >= 1
        assert len(result.paths) >= 1
        assert len(result.paths[0]) >= 3  # A -> ML -> Python
        
        # Проверяем, что путь корректный
        path = result.paths[0]
        assert path[0] == self.ai_concept.id
        assert path[-1] == self.python_concept.id
    
    def test_find_similar(self):
        """Тест поиска похожих концептов."""
        result = self.query_engine.find_similar(
            target_concept_id=self.ai_concept.id,
            similarity_threshold=0.1
        )
        
        assert result.total_results >= 0
        
        # Проверяем, что ML концепт найден как похожий
        similar_concepts = result.results
        assert len(similar_concepts) >= 1
        
        # ML должен быть наиболее похожим на AI
        if self.ml_concept in similar_concepts:
            ml_index = similar_concepts.index(self.ml_concept)
            python_index = similar_concepts.index(self.python_concept) if self.python_concept in similar_concepts else -1
            
            if python_index != -1:
                assert ml_index < python_index  # ML должен быть выше в списке
    
    def test_analyze_graph(self):
        """Тест анализа графа."""
        result = self.query_engine.analyze_graph("metrics")
        
        assert result.total_results >= 1
        
        # Проверяем, что есть концепты-метрики
        metric_concepts = result.results
        assert len(metric_concepts) > 0
        
        # Каждый концепт-метрика должен иметь свойства со значением
        for metric_concept in metric_concepts:
            assert 'value' in metric_concept.properties or len(metric_concept.properties) > 0
    
    def test_query_statistics(self):
        """Тест статистики выполнения запросов."""
        initial_stats = self.query_engine.get_query_stats()
        
        # Выполняем несколько запросов
        self.query_engine.find_concepts(search_term="тест")
        self.query_engine.find_path(
            source_concept_id=self.ai_concept.id,
            target_concept_id=self.ml_concept.id
        )
        
        updated_stats = self.query_engine.get_query_stats()
        
        assert updated_stats['total_queries'] > initial_stats['total_queries']
        assert 'find_concepts' in updated_stats['queries_by_type']
        assert 'traverse_path' in updated_stats['queries_by_type']
    
    def test_advanced_filters(self):
        """Тест расширенных фильтров."""
        # Фильтр по диапазону уверенности
        filters = SearchFilters(
            confidence_range=(0.8, 1.0)
        )
        
        result = self.query_engine.find_concepts(filters=filters)
        
        # Все результаты должны соответствовать фильтру уверенности
        for concept in result.results:
            assert 0.8 <= concept.confidence_score <= 1.0
        
        # Фильтр по времени создания
        current_time = time.time()
        hour_ago = current_time - 3600
        
        filters = SearchFilters(
            time_range=(hour_ago, current_time)
        )
        
        result = self.query_engine.find_concepts(filters=filters)
        
        # Все концепты должны быть созданы в указанный период
        for concept in result.results:
            assert hour_ago <= concept.created_at <= current_time


class TestIntegration:
    """Интеграционные тесты для всей системы."""
    
    def setup_method(self):
        """Настройка для каждого теста."""
        self.temp_dir = tempfile.mkdtemp()
        self.graph = KAGGraph(persistence_path=self.temp_dir)
        self.query_engine = KAGQueryEngine(self.graph)
        self.persistence = GraphPersistence(self.graph, self.temp_dir)
    
    def teardown_method(self):
        """Очистка после тестов."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_workflow(self):
        """Тест полного рабочего процесса."""
        # 1. Создание концептов
        concept1 = Concept(
            name="Технология Блокчейн",
            description="Децентрализованная технология",
            category="технология",
            tags=["блокчейн", "криптография", "децентрализация"],
            confidence_score=0.95
        )
        
        concept2 = Concept(
            name="Криптовалюта",
            description="Цифровая валюта на основе блокчейна",
            category="финансы",
            tags=["криптовалюта", "биткойн", "финансы"],
            confidence_score=0.90
        )
        
        concept3 = Concept(
            name="Биткойн",
            description="Первая и самая известная криптовалюта",
            category="криптовалюта",
            tags=["биткойн", "Satoshi", "peer-to-peer"],
            confidence_score=0.92
        )
        
        # 2. Добавление в граф
        self.graph.add_concept(concept1)
        self.graph.add_concept(concept2)
        self.graph.add_concept(concept3)
        
        # 3. Создание отношений
        rel1 = Relationship(
            source_id=concept1.id,
            target_id=concept2.id,
            relationship_type=RelationshipType.BASES_FOR,
            strength=0.9,
            description="Блокчейн является основой для криптовалют"
        )
        
        rel2 = Relationship(
            source_id=concept2.id,
            target_id=concept3.id,
            relationship_type=RelationshipType.EXEMPLIFIES,
            strength=0.95,
            description="Биткойн - пример криптовалюты"
        )
        
        self.graph.add_relationship(rel1)
        self.graph.add_relationship(rel2)
        
        # 4. Поиск концептов
        search_result = self.query_engine.find_concepts(search_term="крипт")
        assert search_result.total_results >= 2
        
        # 5. Поиск пути
        path_result = self.query_engine.find_path(
            source_concept_id=concept1.id,
            target_concept_id=concept3.id
        )
        assert path_result.total_results >= 1
        assert len(path_result.paths[0]) == 3  # Blockchain -> Crypto -> Bitcoin
        
        # 6. Поиск похожих концептов
        similar_result = self.query_engine.find_similar(
            target_concept_id=concept1.id,
            similarity_threshold=0.1
        )
        assert similar_result.total_results >= 0
        
        # 7. Анализ графа
        analysis_result = self.query_engine.analyze_graph("metrics")
        assert analysis_result.total_results >= 1
        
        # 8. Сохранение графа
        filename = self.persistence.save_graph("json")
        assert filename.endswith(".json")
        
        # 9. Создание резервной копии
        backup_name = self.persistence.create_backup("integration_test")
        assert backup_name.startswith("backup_")
        
        # 10. Восстановление из резервной копии
        new_graph = KAGGraph()
        new_query_engine = KAGQueryEngine(new_graph)
        new_persistence = GraphPersistence(new_graph, self.temp_dir)
        
        success = new_persistence.restore_from_backup(backup_name)
        assert success is True
        
        # 11. Проверка восстановленных данных
        assert len(new_graph.concepts) == 3
        assert len(new_graph.relationships) == 2
        
        # 12. Поиск в восстановленном графе
        restored_search = new_query_engine.find_concepts(search_term="блокчейн")
        assert restored_search.total_results >= 1
        
        # 13. Проверка метрик восстановленного графа
        restored_metrics = new_graph.calculate_graph_metrics()
        assert restored_metrics['total_concepts'] == 3
        assert restored_metrics['total_relationships'] == 2
    
    def test_performance_stress_test(self):
        """Тест производительности с большим количеством данных."""
        import random
        
        # Создаем большое количество концептов
        num_concepts = 100
        concepts = []
        
        for i in range(num_concepts):
            concept = Concept(
                name=f"Концепт_{i}",
                description=f"Описание концепта {i}",
                category=f"category_{i % 10}",  # 10 категорий
                tags=[f"tag_{j}" for j in range(5)],  # 5 тегов на концепт
                confidence_score=random.uniform(0.5, 1.0)
            )
            concepts.append(concept)
            self.graph.add_concept(concept)
        
        # Создаем случайные отношения
        num_relationships = 200
        for i in range(num_relationships):
            source_idx = random.randint(0, num_concepts - 1)
            target_idx = random.randint(0, num_concepts - 1)
            
            if source_idx != target_idx:
                relationship = Relationship(
                    source_id=concepts[source_idx].id,
                    target_id=concepts[target_idx].id,
                    relationship_type=random.choice(list(RelationshipType)),
                    strength=random.uniform(0.5, 1.0)
                )
                self.graph.add_relationship(relationship)
        
        # Тест производительности поиска
        start_time = time.time()
        search_result = self.query_engine.find_concepts(
            search_term="Концепт",
            max_results=50
        )
        search_time = time.time() - start_time
        
        assert search_time < 5.0  # Поиск должен выполняться менее чем за 5 секунд
        assert search_result.total_results > 0
        
        # Тест производительности анализа графа
        start_time = time.time()
        metrics = self.graph.calculate_graph_metrics()
        analysis_time = time.time() - start_time
        
        assert analysis_time < 10.0  # Анализ должен выполняться менее чем за 10 секунд
        assert metrics['total_concepts'] == num_concepts
        
        # Тест производительности поиска пути
        start_time = time.time()
        path_result = self.query_engine.find_path(
            source_concept_id=concepts[0].id,
            target_concept_id=concepts[-1].id,
            max_depth=10
        )
        path_time = time.time() - start_time
        
        assert path_time < 3.0  # Поиск пути должен выполняться менее чем за 3 секунды


def run_comprehensive_tests():
    """Запуск всех тестов."""
    print("Запуск комплексных тестов KAGGraph...")
    
    # Создаем тестовые классы
    test_classes = [
        TestConcept,
        TestRelationship,
        TestKAGGraph,
        TestGraphTraversal,
        TestGraphPersistence,
        TestQueryEngine,
        TestIntegration
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for test_class in test_classes:
        print(f"\n=== Тестирование {test_class.__name__} ===")
        test_instance = test_class()
        
        # Получаем все методы тестов
        test_methods = [
            method for method in dir(test_instance)
            if method.startswith('test_') and callable(getattr(test_instance, method))
        ]
        
        class_total = 0
        class_passed = 0
        class_failed = 0
        
        for test_method_name in test_methods:
            try:
                # Выполняем setup_method если есть
                if hasattr(test_instance, 'setup_method'):
                    test_instance.setup_method()
                
                # Выполняем тест
                test_method = getattr(test_instance, test_method_name)
                test_method()
                
                print(f"  ✓ {test_method_name}")
                class_passed += 1
                passed_tests += 1
                
            except Exception as e:
                print(f"  ✗ {test_method_name}: {str(e)}")
                class_failed += 1
                failed_tests += 1
                
            finally:
                # Выполняем teardown_method если есть
                if hasattr(test_instance, 'teardown_method'):
                    try:
                        test_instance.teardown_method()
                    except Exception as e:
                        print(f"  ⚠ Ошибка в teardown: {e}")
                
                class_total += 1
                total_tests += 1
        
        print(f"Результаты {test_class.__name__}: {class_passed}/{class_total} пройдено, {class_failed} провалено")
    
    print(f"\n=== ОБЩИЕ РЕЗУЛЬТАТЫ ===")
    print(f"Всего тестов: {total_tests}")
    print(f"Пройдено: {passed_tests}")
    print(f"Провалено: {failed_tests}")
    print(f"Успешность: {(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "Нет тестов")
    
    return passed_tests, failed_tests, total_tests


if __name__ == "__main__":
    # Запуск тестов при прямом вызове
    run_comprehensive_tests()
