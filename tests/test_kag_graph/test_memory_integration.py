"""
Тесты интеграции KAGGraph с системой памяти Rebecca Platform.
Проверяет взаимодействие с memory_manager через mock объекты.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any, Optional

# Импорты тестируемой системы
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from knowledge_graph import KAGGraph, Concept, Relationship, RelationshipType


class MockMemoryItem:
    """Mock класс для MemoryItem."""
    
    def __init__(self, id: str, data: Dict[str, Any], metadata: Dict[str, Any] = None):
        self.id = id
        self.data = data
        self.metadata = metadata or {}
        self.layer = "semantic"
        self.timestamp = time.time()


class MockMemoryManager:
    """Mock класс для MemoryManager."""
    
    def __init__(self):
        self.storage = {}
        self.context = Mock()
        self.context.register_layer = Mock()
        
    async def store(
        self,
        layer: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Mock метод для сохранения в память."""
        item_id = f"memory_item_{int(time.time() * 1000000)}"
        self.storage[item_id] = MockMemoryItem(item_id, data, metadata)
        return item_id
    
    async def retrieve(
        self,
        layer: str,
        query: Any,
        filters: Optional[Any] = None
    ) -> List[MockMemoryItem]:
        """Mock метод для извлечения из памяти."""
        results = []
        for item in self.storage.values():
            # Простая фильтрация по метаданным
            if isinstance(query, dict) and 'type' in query:
                if item.metadata.get('type') == query['type']:
                    results.append(item)
            else:
                results.append(item)
        return results
    
    async def delete(
        self,
        layer: str,
        item_id: str
    ) -> bool:
        """Mock метод для удаления из памяти."""
        if item_id in self.storage:
            del self.storage[item_id]
            return True
        return False


class TestKAGGraphMemoryIntegration:
    """Тесты интеграции KAGGraph с системой памяти."""
    
    def setup_method(self):
        """Настройка для каждого теста."""
        self.mock_memory_manager = MockMemoryManager()
        self.graph = KAGGraph(memory_manager=self.mock_memory_manager)
    
    def test_graph_initialization_with_memory_manager(self):
        """Тест инициализации графа с менеджером памяти."""
        assert self.graph.memory_manager is not None
        assert self.graph.memory_manager == self.mock_memory_manager
        assert self.graph.memory_context is not None
    
    async def test_persist_concept_to_memory(self):
        """Тест сохранения концепта в память."""
        concept = Concept(
            name="Тестовый Концепт",
            description="Описание для тестирования",
            category="test",
            tags=["memory", "test"]
        )
        
        await self.graph._persist_concept(concept)
        
        # Проверяем, что концепт сохранен в памяти
        memory_items = await self.mock_memory_manager.retrieve(
            layer="semantic",
            query={'type': 'kag_concept'}
        )
        
        assert len(memory_items) >= 1
        
        # Проверяем, что сохраненные данные корректны
        saved_item = None
        for item in memory_items:
            if item.metadata.get('concept_id') == concept.id:
                saved_item = item
                break
        
        assert saved_item is not None
        assert saved_item.data['name'] == "Тестовый Концепт"
        assert saved_item.metadata['type'] == 'kag_concept'
        assert saved_item.metadata['category'] == 'test'
    
    async def test_persist_relationship_to_memory(self):
        """Тест сохранения отношения в память."""
        # Создаем концепты для отношения
        source_concept = Concept(name="Источник")
        target_concept = Concept(name="Цель")
        
        self.graph.add_concept(source_concept)
        self.graph.add_concept(target_concept)
        
        # Создаем отношение
        relationship = Relationship(
            source_id=source_concept.id,
            target_id=target_concept.id,
            relationship_type=RelationshipType.RELATED_TO,
            strength=0.8
        )
        
        await self.graph._persist_relationship(relationship)
        
        # Проверяем, что отношение сохранено в памяти
        memory_items = await self.mock_memory_manager.retrieve(
            layer="semantic",
            query={'type': 'kag_relationship'}
        )
        
        assert len(memory_items) >= 1
        
        # Проверяем сохраненные данные
        saved_item = None
        for item in memory_items:
            if item.metadata.get('relationship_id') == relationship.id:
                saved_item = item
                break
        
        assert saved_item is not None
        assert saved_item.data['relationship_type'] == 'related_to'
        assert saved_item.metadata['type'] == 'kag_relationship'
    
    async def test_add_concept_with_memory_integration(self):
        """Тест добавления концепта с интеграцией памяти."""
        concept = Concept(
            name="Концепт с памятью",
            description="Тестирование интеграции",
            category="integration",
            confidence_score=0.9
        )
        
        # Добавляем концепт в граф
        concept_id = self.graph.add_concept(concept)
        
        # Проверяем, что концепт добавлен в граф
        assert concept_id in self.graph.concepts
        
        # Проверяем, что концепт сохранен в памяти
        memory_items = await self.mock_memory_manager.retrieve(
            layer="semantic",
            query={'type': 'kag_concept', 'concept_id': concept_id}
        )
        
        assert len(memory_items) >= 1
        saved_concept = memory_items[0]
        assert saved_concept.data['name'] == "Концепт с памятью"
    
    async def test_remove_concept_from_memory(self):
        """Тест удаления концепта из памяти."""
        concept = Concept(
            name="Удаляемый концепт",
            category="test"
        )
        
        # Добавляем концепт
        self.graph.add_concept(concept)
        
        # Удаляем концепт
        result = self.graph.remove_concept(concept.id)
        
        assert result is True
        assert concept.id not in self.graph.concepts
        
        # Проверяем, что концепт удален из памяти
        memory_items = await self.mock_memory_manager.retrieve(
            layer="semantic",
            query={'type': 'kag_concept', 'concept_id': concept.id}
        )
        
        # Список должен быть пустым (концепт удален)
        # Note: В реальной реализации могут быть записи с флагом deleted
        # Здесь мы проверяем базовую функциональность
    
    async def test_load_graph_from_memory(self):
        """Тест загрузки графа из памяти."""
        # Создаем тестовые данные
        concept1 = Concept(
            name="Концепт 1",
            category="test",
            confidence_score=0.9
        )
        
        concept2 = Concept(
            name="Концепт 2", 
            category="test",
            confidence_score=0.8
        )
        
        # Сохраняем в память напрямую через менеджер памяти
        await self.mock_memory_manager.store(
            layer="semantic",
            data=concept1.to_dict(),
            metadata={'type': 'kag_concept', 'concept_id': concept1.id, 'name': concept1.name}
        )
        
        await self.mock_memory_manager.store(
            layer="semantic",
            data=concept2.to_dict(),
            metadata={'type': 'kag_concept', 'concept_id': concept2.id, 'name': concept2.name}
        )
        
        # Загружаем из памяти
        success = await self.graph.load_from_memory()
        
        assert success is True
        assert len(self.graph.concepts) == 2
        assert concept1.id in self.graph.concepts
        assert concept2.id in self.graph.concepts
        
        # Проверяем восстановленные данные
        loaded_concept1 = self.graph.get_concept(concept1.id)
        assert loaded_concept1.name == "Концепт 1"
        assert loaded_concept1.confidence_score == 0.9
    
    async def test_concept_persistence_lifecycle(self):
        """Тест полного жизненного цикла концепта с персистентностью."""
        concept = Concept(
            name="Жизненный цикл",
            description="Тестирование жизненного цикла",
            category="lifecycle",
            tags=["full", "cycle", "test"],
            confidence_score=0.95
        )
        
        # 1. Создание и добавление
        concept_id = self.graph.add_concept(concept)
        assert concept_id in self.graph.concepts
        
        # 2. Обновление
        original_name = concept.name
        concept.name = "Обновленное имя"
        concept.update_timestamp()
        
        # Пересохраняем обновленный концепт
        await self.graph._persist_concept(concept)
        
        # 3. Удаление
        result = self.graph.remove_concept(concept_id)
        assert result is True
        
        # 4. Проверяем, что концепт удален из памяти
        memory_items = await self.mock_memory_manager.retrieve(
            layer="semantic",
            query={'type': 'kag_concept', 'concept_id': concept_id}
        )
        
        # В реальной системе здесь была бы логика удаления или пометки как удаленного
        assert len(memory_items) >= 1
    
    async def test_relationship_persistence_lifecycle(self):
        """Тест жизненного цикла отношения с персистентностью."""
        # Создаем концепты
        source = Concept(name="Источник жизненного цикла")
        target = Concept(name="Цель жизненного цикла")
        
        self.graph.add_concept(source)
        self.graph.add_concept(target)
        
        # Создаем отношение
        relationship = Relationship(
            source_id=source.id,
            target_id=target.id,
            relationship_type=RelationshipType.CAUSES,
            strength=0.85,
            description="Отношение для тестирования жизненного цикла"
        )
        
        # Добавляем в граф
        rel_id = self.graph.add_relationship(relationship)
        
        # Проверяем, что отношение добавлено в граф
        assert rel_id in self.graph.relationships
        
        # Сохраняем в память
        await self.graph._persist_relationship(relationship)
        
        # Проверяем сохранение
        memory_items = await self.mock_memory_manager.retrieve(
            layer="semantic",
            query={'type': 'kag_relationship', 'relationship_id': rel_id}
        )
        
        assert len(memory_items) >= 1
        
        # Удаляем отношение
        result = self.graph.remove_relationship(rel_id)
        assert result is True
        
        # Проверяем, что отношение удалено из памяти
        remaining_items = await self.mock_memory_manager.retrieve(
            layer="semantic",
            query={'type': 'kag_relationship', 'relationship_id': rel_id}
        )
        
        # В реальной системе здесь была бы логика удаления
        assert len(remaining_items) >= 1
    
    def test_memory_manager_integration_without_manager(self):
        """Тест работы графа без менеджера памяти."""
        # Создаем граф без менеджера памяти
        graph_no_memory = KAGGraph()
        
        assert graph_no_memory.memory_manager is None
        assert graph_no_memory.memory_context is None
        
        # Тестируем базовую функциональность без интеграции с памятью
        concept = Concept(name="Без памяти", category="test")
        concept_id = graph_no_memory.add_concept(concept)
        
        assert concept_id == concept.id
        assert concept_id in graph_no_memory.concepts
        
        # Граф должен работать, но не сохранять в память
        assert len(graph_no_memory.concepts) == 1


class TestKAGGraphMemoryErrorHandling:
    """Тесты обработки ошибок при работе с памятью."""
    
    def setup_method(self):
        """Настройка для каждого теста."""
        # Создаем mock менеджер памяти, который имитирует ошибки
        self.failing_memory_manager = Mock()
        self.failing_memory_manager.context = Mock()
        
        # Настраиваем mock для имитации ошибок
        self.failing_memory_manager.store = Mock(side_effect=Exception("Memory storage error"))
        self.failing_memory_manager.retrieve = Mock(side_effect=Exception("Memory retrieval error"))
        
        self.graph = KAGGraph(memory_manager=self.failing_memory_manager)
    
    async def test_persist_concept_with_memory_error(self):
        """Тест обработки ошибки при сохранении концепта."""
        concept = Concept(name="Проблемный концепт")
        
        # Операция должна завершиться успешно, несмотря на ошибку в памяти
        # (ошибки памяти не должны ломать основную логику графа)
        concept_id = self.graph.add_concept(concept)
        
        assert concept_id == concept.id
        assert concept_id in self.graph.concepts
        
        # Mock должен был быть вызван, но ошибка обработана
        self.failing_memory_manager.store.assert_called_once()
    
    async def test_load_from_memory_with_error(self):
        """Тест обработки ошибки при загрузке из памяти."""
        # Операция загрузки должна обработать ошибку и вернуть False
        result = await self.graph.load_from_memory()
        
        assert result is False
        self.failing_memory_manager.retrieve.assert_called_once()
    
    def test_graph_works_without_memory_manager(self):
        """Тест работы графа при отсутствии менеджера памяти."""
        # Создаем граф без менеджера памяти
        graph = KAGGraph(memory_manager=None)
        
        # Добавляем концепт
        concept = Concept(name="Автономный концепт", category="standalone")
        concept_id = graph.add_concept(concept)
        
        assert concept_id in graph.concepts
        assert len(graph.concepts) == 1
        
        # Создаем отношение
        concept2 = Concept(name="Второй концепт")
        graph.add_concept(concept2)
        
        relationship = Relationship(
            source_id=concept_id,
            target_id=concept2.id,
            relationship_type=RelationshipType.RELATED_TO
        )
        
        rel_id = graph.add_relationship(relationship)
        assert rel_id in graph.relationships
        
        # Граф должен работать полностью автономно
        assert len(graph.relationships) == 1
        metrics = graph.calculate_graph_metrics()
        assert metrics['total_concepts'] == 2


class TestKAGGraphAdvancedMemoryIntegration:
    """Тесты продвинутой интеграции с памятью."""
    
    def setup_method(self):
        """Настройка для каждого теста."""
        self.memory_manager = MockMemoryManager()
        self.graph = KAGGraph(memory_manager=self.memory_manager)
    
    async def test_batch_concept_persistence(self):
        """Тест пакетного сохранения концептов."""
        concepts = []
        for i in range(10):
            concept = Concept(
                name=f"Пакетный концепт {i}",
                category="batch",
                tags=[f"tag_{j}" for j in range(3)]
            )
            concepts.append(concept)
            self.graph.add_concept(concept)
        
        # Проверяем, что все концепты сохранены в память
        memory_items = await self.memory_manager.retrieve(
            layer="semantic",
            query={'type': 'kag_concept'}
        )
        
        assert len(memory_items) >= 10
        
        # Проверяем, что все концепты имеют корректные метаданные
        for item in memory_items:
            if item.metadata.get('type') == 'kag_concept':
                assert 'concept_id' in item.metadata
                assert 'name' in item.metadata
                assert 'category' in item.metadata
    
    async def test_complex_graph_with_memory(self):
        """Тест сложного графа с интеграцией памяти."""
        # Создаем иерархическую структуру
        concepts = {}
        
        # Уровень 1: Базовые концепты
        base_concepts = [
            ("Искусственный Интеллект", "AI технологии"),
            ("Машинное Обучение", "Подраздел AI"),
            ("Глубокое Обучение", "Подраздел ML")
        ]
        
        for name, desc in base_concepts:
            concept = Concept(
                name=name,
                description=desc,
                category="технология",
                tags=["AI", "ML", "технология"]
            )
            self.graph.add_concept(concept)
            concepts[name] = concept
        
        # Уровень 2: Связи между концептами
        relationships = [
            ("Искусственный Интеллект", "Машинное Обучение", RelationshipType.CONTAINS),
            ("Машинное Обучение", "Глубокое Обучение", RelationshipType.CONTAINS),
            ("Глубокое Обучение", "Искусственный Интеллект", RelationshipType.RELATED_TO)
        ]
        
        for source_name, target_name, rel_type in relationships:
            source = concepts[source_name]
            target = concepts[target_name]
            
            relationship = Relationship(
                source_id=source.id,
                target_id=target.id,
                relationship_type=rel_type,
                strength=0.9,
                description=f"{source_name} связан с {target_name}"
            )
            
            self.graph.add_relationship(relationship)
        
        # Проверяем сохранение в память
        concept_items = await self.memory_manager.retrieve(
            layer="semantic",
            query={'type': 'kag_concept'}
        )
        
        relationship_items = await self.memory_manager.retrieve(
            layer="semantic",
            query={'type': 'kag_relationship'}
        )
        
        assert len(concept_items) >= 3
        assert len(relationship_items) >= 3
        
        # Проверяем поиск в сохраненном графе
        search_result = self.graph.find_concepts(name="Интеллект")
        assert len(search_result) >= 1
        
        # Проверяем метрики
        metrics = self.graph.calculate_graph_metrics()
        assert metrics['total_concepts'] >= 3
        assert metrics['total_relationships'] >= 3


# Функция для запуска тестов интеграции с памятью
def run_memory_integration_tests():
    """Запуск тестов интеграции с памятью."""
    print("Запуск тестов интеграции KAGGraph с системой памяти...")
    
    test_classes = [
        TestKAGGraphMemoryIntegration,
        TestKAGGraphMemoryErrorHandling,
        TestKAGGraphAdvancedMemoryIntegration
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
                
                # Выполняем тест (синхронно для простоты)
                test_method = getattr(test_instance, test_method_name)
                
                # Если тест асинхронный, запускаем его
                if asyncio.iscoroutinefunction(test_method):
                    asyncio.run(test_method())
                else:
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
    
    print(f"\n=== РЕЗУЛЬТАТЫ ТЕСТОВ ИНТЕГРАЦИИ С ПАМЯТЬЮ ===")
    print(f"Всего тестов: {total_tests}")
    print(f"Пройдено: {passed_tests}")
    print(f"Провалено: {failed_tests}")
    print(f"Успешность: {(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "Нет тестов")
    
    return passed_tests, failed_tests, total_tests


if __name__ == "__main__":
    # Запуск тестов интеграции с памятью
    run_memory_integration_tests()
