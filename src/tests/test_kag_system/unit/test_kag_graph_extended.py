"""
Comprehensive Unit Tests for KAGGraph Component
Тестирование основного класса графа знаний KAG
"""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any, Set, Tuple
import tempfile
import shutil
from pathlib import Path

# KAG компоненты
from knowledge_graph.kag_graph import KAGGraph, Concept, Relationship, RelationshipType, QueryResult


class TestKAGGraphCore:
    """Основные тесты KAGGraph"""
    
    @pytest.mark.unit
    def test_graph_initialization(self, kag_graph):
        """Тест инициализации графа"""
        assert kag_graph is not None
        assert len(kag_graph.concepts) == 0
        assert len(kag_graph.relationships) == 0
        assert len(kag_graph.name_index) == 0
        assert len(kag_graph.category_index) == 0
        assert len(kag_graph.tag_index) == 0
        assert len(kag_graph.type_index) == 0
        
        # Проверка статистики
        stats = kag_graph.get_stats()
        assert stats['total_concepts'] == 0
        assert stats['total_relationships'] == 0
        assert stats['queries_executed'] == 0
    
    @pytest.mark.unit
    def test_add_concept_basic(self, kag_graph):
        """Тест базового добавления концепта"""
        concept = Concept(
            id="test_concept_1",
            name="Test Concept",
            description="A concept for testing",
            category="test",
            confidence_score=0.9,
            tags=["test", "unit"],
            properties={"importance": "high"}
        )
        
        result_id = kag_graph.add_concept(concept)
        
        assert result_id == "test_concept_1"
        assert "test_concept_1" in kag_graph.concepts
        assert kag_graph.concepts["test_concept_1"] == concept
        assert "Test Concept" in kag_graph.name_index
        assert kag_graph.name_index["Test Concept"] == "test_concept_1"
        assert "test" in kag_graph.category_index
        assert "test_concept_1" in kag_graph.category_index["test"]
    
    @pytest.mark.unit
    def test_add_concept_duplicate_name_warning(self, kag_graph):
        """Тест предупреждения при дублировании имен"""
        concept1 = Concept(id="id1", name="Same Name", description="First")
        concept2 = Concept(id="id2", name="Same Name", description="Second")
        
        kag_graph.add_concept(concept1)
        
        # Должно выдать предупреждение при добавлении второго с тем же именем
        with patch('builtins.print') as mock_print:
            kag_graph.add_concept(concept2)
            mock_print.assert_called_once()
            assert "Предупреждение: Концепт с именем 'Same Name' уже существует" in str(mock_print.call_args)
    
    @pytest.mark.unit
    def test_add_relationship_basic(self, kag_graph, sample_concepts, sample_relationships):
        """Тест базового добавления отношения"""
        # Сначала добавляем концепты
        for concept in sample_concepts:
            kag_graph.add_concept(concept)
        
        relationship = sample_relationships[0]
        result_id = kag_graph.add_relationship(relationship)
        
        assert result_id == "rel_1"
        assert "rel_1" in kag_graph.relationships
        assert kag_graph.relationships["rel_1"] == relationship
        assert relationship.source == kag_graph.concepts["concept_2"]
        assert relationship.target == kag_graph.concepts["concept_1"]
        assert "rel_1" in relationship.source.outgoing_relationships
        assert "rel_1" in relationship.target.incoming_relationships
    
    @pytest.mark.unit
    def test_add_relationship_nonexistent_concepts(self, kag_graph):
        """Тест ошибки при добавлении отношения между несуществующими концептами"""
        relationship = Relationship(
            id="test_rel",
            source_id="nonexistent_source",
            target_id="nonexistent_target",
            relationship_type=RelationshipType.RELATED_TO
        )
        
        with pytest.raises(ValueError, match="Исходный концепт nonexistent_source не найден"):
            kag_graph.add_relationship(relationship)
    
    @pytest.mark.unit
    def test_remove_concept(self, kag_graph, sample_concepts, sample_relationships):
        """Тест удаления концепта"""
        # Заполняем граф
        for concept in sample_concepts:
            kag_graph.add_concept(concept)
        for relationship in sample_relationships:
            kag_graph.add_relationship(relationship)
        
        initial_concept_count = len(kag_graph.concepts)
        initial_edge_count = len(kag_graph.relationships)
        
        # Удаляем концепт, который связан с другими
        result = kag_graph.remove_concept("concept_2")
        
        assert result is True
        assert "concept_2" not in kag_graph.concepts
        assert len(kag_graph.concepts) == initial_concept_count - 1
        
        # Проверяем, что связанные отношения удалены
        assert len(kag_graph.relationships) == initial_edge_count - 1
        assert "rel_1" not in kag_graph.relationships
        
        # Проверяем, что индексы обновлены
        assert "concept_2" not in kag_graph.name_index
        assert "concept_2" not in kag_graph.category_index["technology"]
    
    @pytest.mark.unit
    def test_remove_concept_nonexistent(self, kag_graph):
        """Тест удаления несуществующего концепта"""
        result = kag_graph.remove_concept("nonexistent")
        assert result is False
    
    @pytest.mark.unit
    def test_remove_relationship(self, kag_graph, sample_concepts, sample_relationships):
        """Тест удаления отношения"""
        # Заполняем граф
        for concept in sample_concepts:
            kag_graph.add_concept(concept)
        for relationship in sample_relationships:
            kag_graph.add_relationship(relationship)
        
        initial_edge_count = len(kag_graph.relationships)
        
        # Удаляем отношение
        result = kag_graph.remove_relationship("rel_1")
        
        assert result is True
        assert "rel_1" not in kag_graph.relationships
        assert len(kag_graph.relationships) == initial_edge_count - 1
        
        # Проверяем, что связи в концептах удалены
        concept_1 = kag_graph.concepts["concept_1"]
        concept_2 = kag_graph.concepts["concept_2"]
        assert "rel_1" not in concept_1.incoming_relationships
        assert "rel_1" not in concept_2.outgoing_relationships
    
    @pytest.mark.unit
    def test_get_concept_by_methods(self, kag_graph, sample_concepts):
        """Тест получения концептов различными способами"""
        concept = sample_concepts[0]
        kag_graph.add_concept(concept)
        
        # По ID
        retrieved = kag_graph.get_concept("concept_1")
        assert retrieved == concept
        
        # По имени
        retrieved_by_name = kag_graph.get_concept_by_name("Artificial Intelligence")
        assert retrieved_by_name == concept
        
        # Несуществующий концепт
        nonexistent = kag_graph.get_concept("nonexistent")
        assert nonexistent is None
        
        nonexistent_by_name = kag_graph.get_concept_by_name("Nonexistent")
        assert nonexistent_by_name is None


class TestKAGGraphSearch:
    """Тесты поиска и фильтрации в графе"""
    
    @pytest.mark.unit
    def test_find_concepts_by_name(self, kag_graph, sample_concepts):
        """Тест поиска концептов по имени"""
        for concept in sample_concepts:
            kag_graph.add_concept(concept)
        
        # Поиск по частичному совпадению
        results = kag_graph.find_concepts(name="Intelligence")
        assert len(results) == 1
        assert results[0].name == "Artificial Intelligence"
        
        # Поиск без совпадений
        results = kag_graph.find_concepts(name="Nonexistent")
        assert len(results) == 0
    
    @pytest.mark.unit
    def test_find_concepts_by_category(self, kag_graph, sample_concepts):
        """Тест поиска концептов по категории"""
        for concept in sample_concepts:
            kag_graph.add_concept(concept)
        
        # Поиск по категории technology
        results = kag_graph.find_concepts(category="technology")
        assert len(results) == 2
        assert all(c.category == "technology" for c in results)
        
        # Поиск по категории psychology
        results = kag_graph.find_concepts(category="psychology")
        assert len(results) == 1
        assert results[0].name == "Confirmation Bias"
    
    @pytest.mark.unit
    def test_find_concepts_by_tags(self, kag_graph, sample_concepts):
        """Тест поиска концептов по тегам"""
        for concept in sample_concepts:
            kag_graph.add_concept(concept)
        
        # Поиск по тегу AI
        results = kag_graph.find_concepts(tags={"AI"})
        assert len(results) == 1
        assert results[0].name == "Artificial Intelligence"
        
        # Поиск по множеству тегов
        results = kag_graph.find_concepts(tags={"AI", "computer_science"})
        assert len(results) == 1
        
        # Поиск по несуществующему тегу
        results = kag_graph.find_concepts(tags={"nonexistent"})
        assert len(results) == 0
    
    @pytest.mark.unit
    def test_find_concepts_by_confidence(self, kag_graph, sample_concepts):
        """Тест поиска концептов по уверенности"""
        for concept in sample_concepts:
            kag_graph.add_concept(concept)
        
        # Поиск с высокой уверенностью
        results = kag_graph.find_concepts(min_confidence=0.85)
        assert len(results) == 2
        assert all(c.confidence_score >= 0.85 for c in results)
        
        # Поиск с низкой уверенностью
        results = kag_graph.find_concepts(min_confidence=0.5)
        assert len(results) == 3
    
    @pytest.mark.unit
    def test_get_connected_concepts(self, kag_graph, sample_concepts, sample_relationships):
        """Тест получения связанных концептов"""
        # Заполняем граф
        for concept in sample_concepts:
            kag_graph.add_concept(concept)
        for relationship in sample_relationships:
            kag_graph.add_relationship(relationship)
        
        # Получаем связанные концепты
        connected = kag_graph.get_connected_concepts("concept_2")
        
        assert 1 in connected
        assert len(connected[1]) == 1
        assert connected[1][0].id == "concept_1"
        
        # Тест с фильтром типов отношений
        connected_filtered = kag_graph.get_connected_concepts(
            "concept_2", 
            relationship_types={RelationshipType.IS_A}
        )
        assert len(connected_filtered[1]) == 1
    
    @pytest.mark.unit
    def test_find_shortest_path(self, kag_graph, sample_concepts, sample_relationships):
        """Тест поиска кратчайшего пути"""
        # Заполняем граф
        for concept in sample_concepts:
            kag_graph.add_concept(concept)
        for relationship in sample_relationships:
            kag_graph.add_relationship(relationship)
        
        # Поиск пути между concept_3 и concept_1 через concept_2
        path = kag_graph.find_shortest_path("concept_3", "concept_1")
        
        assert path is not None
        assert "concept_3" in path
        assert "concept_2" in path
        assert "concept_1" in path
        assert len(path) == 3
        
        # Поиск несуществующего пути
        concept_no_path = Concept(id="isolated", name="Isolated", description="No connections")
        kag_graph.add_concept(concept_no_path)
        
        path = kag_graph.find_shortest_path("concept_1", "isolated")
        assert path is None


class TestKAGGraphAnalytics:
    """Тесты аналитических функций графа"""
    
    @pytest.mark.unit
    def test_calculate_graph_metrics_empty(self, kag_graph):
        """Тест метрик пустого графа"""
        metrics = kag_graph.calculate_graph_metrics()
        
        assert metrics['total_concepts'] == 0
        assert metrics['total_relationships'] == 0
        assert metrics['density'] == 0.0
        assert metrics['avg_degree'] == 0.0
        assert metrics['clustering_coefficient'] == 0.0
    
    @pytest.mark.unit
    def test_calculate_graph_metrics_populated(self, kag_graph, sample_concepts, sample_relationships):
        """Тест метрик заполненного графа"""
        # Заполняем граф
        for concept in sample_concepts:
            kag_graph.add_concept(concept)
        for relationship in sample_relationships:
            kag_graph.add_relationship(relationship)
        
        metrics = kag_graph.calculate_graph_metrics()
        
        assert metrics['total_concepts'] == 3
        assert metrics['total_relationships'] == 2
        assert metrics['density'] > 0
        assert metrics['avg_degree'] > 0
        assert 0 <= metrics['clustering_coefficient'] <= 1
        assert metrics['categories'] == 2  # technology, psychology
        assert metrics['unique_tags'] > 0
        assert metrics['relationship_types'] == 2  # IS_A, INFLUENCES


class TestKAGGraphSerialization:
    """Тесты сериализации и десериализации"""
    
    @pytest.mark.unit
    def test_export_graph_json(self, kag_graph, sample_concepts, sample_relationships):
        """Тест экспорта графа в JSON"""
        # Заполняем граф
        for concept in sample_concepts:
            kag_graph.add_concept(concept)
        for relationship in sample_relationships:
            kag_graph.add_relationship(relationship)
        
        # Экспортируем
        export_data = kag_graph.export_graph(format_type="json")
        
        # Проверяем структуру
        assert 'concepts' in export_data
        assert 'relationships' in export_data
        assert 'metadata' in export_data
        
        assert len(export_data['concepts']) == 3
        assert len(export_data['relationships']) == 2
        
        metadata = export_data['metadata']
        assert metadata['total_concepts'] == 3
        assert metadata['total_relationships'] == 2
        assert metadata['version'] == '1.0'
        assert 'exported_at' in metadata
    
    @pytest.mark.unit
    def test_import_graph_json(self, kag_graph, sample_concepts, sample_relationships):
        """Тест импорта графа из JSON"""
        # Создаем данные для импорта
        export_data = {
            'concepts': [concept.to_dict() for concept in sample_concepts],
            'relationships': [rel.to_dict() for rel in sample_relationships],
            'metadata': {
                'exported_at': time.time(),
                'total_concepts': len(sample_concepts),
                'total_relationships': len(sample_relationships),
                'version': '1.0'
            }
        }
        
        # Импортируем
        result = kag_graph.import_graph(export_data)
        
        assert result is True
        assert len(kag_graph.concepts) == 3
        assert len(kag_graph.relationships) == 2
        assert "concept_1" in kag_graph.concepts
        assert "rel_1" in kag_graph.relationships
    
    @pytest.mark.unit
    def test_import_graph_invalid_data(self, kag_graph):
        """Тест импорта с некорректными данными"""
        invalid_data = {
            'concepts': [{'id': 'invalid', 'name': 'Test'}],
            'relationships': [{'source_id': 'nonexistent', 'target_id': 'nonexistent'}]
        }
        
        result = kag_graph.import_graph(invalid_data)
        
        assert result is True  # Частичный импорт должен работать
        assert len(kag_graph.concepts) == 1
        assert len(kag_graph.relationships) == 0  # Отношения не добавлены из-за отсутствия концептов


class TestKAGGraphPerformance:
    """Тесты производительности"""
    
    @pytest.mark.performance
    def test_large_graph_creation(self, performance_test_graph):
        """Тест создания большого графа"""
        assert performance_test_graph.get_node_count() == 1000
        assert performance_test_graph.get_edge_count() == 3000
        
        # Проверяем некоторые метрики
        metrics = performance_test_graph.calculate_graph_metrics()
        assert metrics['total_concepts'] == 1000
        assert metrics['total_relationships'] == 3000
        assert metrics['density'] > 0
    
    @pytest.mark.performance
    def test_large_graph_search_performance(self, performance_test_graph):
        """Тест производительности поиска в большом графе"""
        # Тест поиска
        start_time = time.time()
        results = performance_test_graph.find_concepts(name="Performance", min_confidence=0.0)
        search_time = time.time() - start_time
        
        assert search_time < 2.0  # Поиск должен занимать менее 2 секунд
        assert len(results) > 0
    
    @pytest.mark.performance
    def test_large_graph_traversal_performance(self, performance_test_graph):
        """Тест производительности обхода большого графа"""
        # Тест обхода
        start_time = time.time()
        connected = performance_test_graph.get_connected_concepts("perf_node_0", max_depth=3)
        traversal_time = time.time() - start_time
        
        assert traversal_time < 3.0  # Обход должен занимать менее 3 секунд
        assert len(connected) > 0
    
    @pytest.mark.performance
    def test_graph_metrics_performance(self, performance_test_graph):
        """Тест производительности вычисления метрик"""
        start_time = time.time()
        metrics = performance_test_graph.calculate_graph_metrics()
        metrics_time = time.time() - start_time
        
        assert metrics_time < 5.0  # Метрики должны вычисляться менее чем за 5 секунд
        assert metrics['total_concepts'] == 1000


class TestKAGGraphEdgeCases:
    """Тесты граничных случаев"""
    
    @pytest.mark.unit
    def test_self_referencing_relationship(self, kag_graph):
        """Тест отношения с самим собой"""
        concept = Concept(id="self_ref", name="Self Reference", description="References itself")
        kag_graph.add_concept(concept)
        
        relationship = Relationship(
            id="self_rel",
            source_id="self_ref",
            target_id="self_ref",
            relationship_type=RelationshipType.RELATED_TO
        )
        
        kag_graph.add_relationship(relationship)
        assert "self_rel" in kag_graph.relationships
    
    @pytest.mark.unit
    def test_concept_with_special_characters(self, kag_graph):
        """Тест концептов со специальными символами"""
        concept = Concept(
            id="special",
            name="Concept with émojis 🚀 and spëcial çhars",
            description="Test with special characters: @#$%^&*()",
            category="test"
        )
        
        kag_graph.add_concept(concept)
        assert "special" in kag_graph.concepts
        
        # Поиск должен работать
        results = kag_graph.find_concepts(name="émojis")
        assert len(results) == 1
    
    @pytest.mark.unit
    def test_empty_name_concept(self, kag_graph):
        """Тест концепта с пустым именем"""
        concept = Concept(id="empty_name", name="", description="Empty name test")
        
        kag_graph.add_concept(concept)
        assert "empty_name" in kag_graph.concepts
        assert "" in kag_graph.name_index
        assert kag_graph.name_index[""] == "empty_name"
    
    @pytest.mark.unit
    def test_concept_similarity_calculation(self, sample_concepts):
        """Тест вычисления схожести концептов"""
        concept1, concept2 = sample_concepts[0], sample_concepts[1]  # AI and ML
        
        # Схожие концепты должны иметь высокую схожесть
        similarity = concept1.calculate_similarity(concept2)
        assert similarity > 0  # Должна быть хотя бы минимальная схожесть
        
        # Схожесть должна быть симметричной
        similarity_reverse = concept2.calculate_similarity(concept1)
        assert similarity == similarity_reverse
        
        # Концепт должен быть полностью схож с самим собой
        similarity_self = concept1.calculate_similarity(concept1)
        assert similarity_self == 1.0
    
    @pytest.mark.unit
    def test_relationship_metadata(self, kag_graph):
        """Тест метаданных отношений"""
        concept1 = Concept(id="c1", name="Concept 1")
        concept2 = Concept(id="c2", name="Concept 2")
        kag_graph.add_concept(concept1)
        kag_graph.add_concept(concept2)
        
        relationship = Relationship(
            id="rel",
            source_id="c1",
            target_id="c2",
            relationship_type=RelationshipType.RELATED_TO,
            strength=0.85,
            description="Test relationship",
            metadata={"confidence": 0.9, "source": "manual"}
        )
        
        kag_graph.add_relationship(relationship)
        saved_rel = kag_graph.relationships["rel"]
        
        assert saved_rel.strength == 0.85
        assert saved_rel.description == "Test relationship"
        assert saved_rel.metadata["confidence"] == 0.9
        assert saved_rel.metadata["source"] == "manual"


class TestKAGGraphConcurrency:
    """Тесты конкурентного доступа"""
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, kag_graph):
        """Тест конкурентных операций с графом"""
        async def add_concept_task(i):
            concept = Concept(
                id=f"concurrent_{i}",
                name=f"Concurrent Concept {i}",
                description=f"Added concurrently {i}"
            )
            return kag_graph.add_concept(concept)
        
        async def add_relationship_task(i):
            source = f"concurrent_{i}"
            target = f"concurrent_{(i+1) % 50}"
            relationship = Relationship(
                id=f"concurrent_rel_{i}",
                source_id=source,
                target_id=target,
                relationship_type=RelationshipType.RELATED_TO
            )
            return kag_graph.add_relationship(relationship)
        
        # Создаем 50 концептов и 25 отношений конкурентно
        concept_tasks = [add_concept_task(i) for i in range(50)]
        relationship_tasks = [add_relationship_task(i) for i in range(25)]
        
        all_tasks = concept_tasks + relationship_tasks
        results = await asyncio.gather(*all_tasks, return_exceptions=True)
        
        # Проверяем результаты
        assert len([r for r in results if not isinstance(r, Exception)]) == len(all_tasks)
        assert kag_graph.get_node_count() == 50
        assert kag_graph.get_edge_count() == 25


# =============================================================================
# Тесты интеграции с Memory Manager
# =============================================================================

class TestKAGGraphMemoryIntegration:
    """Тесты интеграции с менеджером памяти"""
    
    @pytest.mark.asyncio
    async def test_memory_persistence_mock(self, kag_graph, mock_memory_manager):
        """Тест сохранения в память (mock)"""
        concept = Concept(id="mem_test", name="Memory Test", description="Test memory persistence")
        
        # Мок должен быть вызван при добавлении концепта
        kag_graph.add_concept(concept)
        
        # Проверяем, что mock методы были вызваны
        assert mock_memory_manager.store.called
    
    @pytest.mark.asyncio
    async def test_load_from_memory_mock(self, kag_graph, mock_memory_manager):
        """Тест загрузки из памяти (mock)"""
        # Настраиваем mock для возврата данных
        mock_memory_manager.retrieve.return_value = [
            Mock(data={'id': 'loaded_concept', 'name': 'Loaded', 'description': 'From memory'})
        ]
        
        result = await kag_graph.load_from_memory()
        
        # Проверяем результат загрузки
        assert result is True
        assert 'loaded_concept' in kag_graph.concepts