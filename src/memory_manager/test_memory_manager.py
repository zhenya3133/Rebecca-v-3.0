"""Тесты для полноценного MemoryManager.

Проверяет основную функциональность системы управления памятью.
"""

import asyncio
import pytest
import time
from typing import Dict, Any

from memory_manager import (
    MemoryManager, 
    create_memory_manager, 
    CORE, 
    EPISODIC, 
    SEMANTIC, 
    PROCEDURAL, 
    VAULT, 
    SECURITY,
    quick_memory_test
)
from logger import setup_logger

# Настройка логгера для тестов
logger = setup_logger(__name__)


class TestMemoryManager:
    """Тесты MemoryManager."""
    
    @pytest.fixture
    async def manager(self):
        """Фикстура для создания MemoryManager."""
        manager = create_memory_manager({
            "cache_size": 100,
            "cache_ttl": 300,
            "optimization_interval": 60
        })
        
        await manager.start()
        yield manager
        await manager.stop()
    
    async def test_basic_operations(self, manager):
        """Тест базовых операций с памятью."""
        
        # Сохранение данных
        core_id = await manager.store(
            layer=CORE,
            data="Тестовый факт",
            metadata={"test": True},
            tags=["test"],
            priority=5
        )
        
        assert core_id is not None
        assert isinstance(core_id, str)
        
        # Извлечение данных
        core_data = await manager.retrieve(CORE)
        assert len(core_data) > 0
        
        # Проверяем, что сохраненные данные есть в результате
        found = any(item["id"] == core_id for item in core_data)
        assert found, "Сохраненная запись не найдена при извлечении"
        
        # Обновление данных
        update_success = await manager.update(
            layer=CORE,
            item_id=core_id,
            data="Обновленный факт",
            metadata={"test": True, "updated": True}
        )
        
        assert update_success, "Обновление должно быть успешным"
        
        # Удаление данных
        delete_success = await manager.delete(CORE, core_id)
        assert delete_success, "Удаление должно быть успешным"
        
        # Проверяем, что данные удалены
        core_data_after_delete = await manager.retrieve(CORE)
        deleted = all(item["id"] != core_id for item in core_data_after_delete)
        assert deleted, "Запись должна быть удалена"
    
    async def test_all_layers(self, manager):
        """Тест всех слоев памяти."""
        
        layer_data = {
            CORE: "Базовый факт",
            EPISODIC: "Событие системы",
            SEMANTIC: "Концепция знания",
            PROCEDURAL: "Процедура выполнения",
            VAULT: "Секретная информация", 
            SECURITY: "Событие безопасности"
        }
        
        stored_ids = {}
        
        # Сохраняем данные во все слои
        for layer, data in layer_data.items():
            item_id = await manager.store(
                layer=layer,
                data=data,
                metadata={"layer": layer, "test": True},
                tags=["test", layer.lower()],
                priority=7
            )
            
            stored_ids[layer] = item_id
            assert item_id is not None
        
        # Проверяем извлечение из каждого слоя
        for layer in layer_data.keys():
            layer_data_retrieved = await manager.retrieve(layer)
            assert len(layer_data_retrieved) > 0, f"Данные не найдены в слое {layer}"
            
            # Проверяем, что сохраненные данные есть
            found = any(item["id"] == stored_ids[layer] for item in layer_data_retrieved)
            assert found, f"Сохраненные данные не найдены в слое {layer}"
    
    async def test_search_across_layers(self, manager):
        """Тест поиска по слоям памяти."""
        
        # Сохраняем данные с разными ключевыми словами
        await manager.store(CORE, "Система реактора работает", {"source": "core"})
        await manager.store(EPISODIC, "Система была обновлена вчера", {"source": "episodic"})
        await manager.store(SEMANTIC, "ИИ система интеллекта", {"source": "semantic"})
        await manager.store(SEMANTIC, "Реакторная система охлаждения", {"source": "semantic"})
        
        # Поиск по слову "система"
        results = await manager.search_across_layers(
            query="система",
            layers=[CORE, EPISODIC, SEMANTIC],
            limit=10
        )
        
        assert len(results) > 0, "Поиск должен найти результаты"
        
        # Все результаты должны содержать слово "система"
        for result in results:
            content = str(result["content"]).lower()
            assert "система" in content, f"Результат не содержит искомое слово: {result}"
        
        # Поиск только в одном слое
        semantic_only = await manager.search_across_layers(
            query="система",
            layers=[SEMANTIC],
            limit=10
        )
        
        assert len(semantic_only) > 0, "Поиск в семантической памяти должен найти результаты"
        
        # Результаты должны быть только из семантической памяти
        for result in semantic_only:
            # ID должен присутствовать в семантической памяти
            semantic_data = await manager.retrieve(SEMANTIC)
            found_in_semantic = any(item["id"] == result["id"] for item in semantic_data)
            assert found_in_semantic, "Результат должен быть из семантической памяти"
    
    async def test_blueprint_tracker(self, manager):
        """Тест AdaptiveBlueprintTracker."""
        
        # Записываем первую версию архитектуры
        blueprint1 = {
            "version": "1.0",
            "components": {
                "memory": {"status": "active"},
                "vector_store": {"status": "active"}
            }
        }
        
        version1 = await manager.blueprint_tracker.record_blueprint(
            blueprint=blueprint1,
            metadata={"author": "test"},
            change_type="initial"
        )
        
        assert version1 == 1
        
        # Записываем вторую версию
        blueprint2 = {
            "version": "1.1", 
            "components": {
                "memory": {"status": "enhanced"},
                "vector_store": {"status": "active"},
                "ai_processor": {"status": "active"}
            }
        }
        
        version2 = await manager.blueprint_tracker.record_blueprint(
            blueprint=blueprint2,
            metadata={"author": "test"},
            change_type="enhancement",
            change_description="Добавлен AI процессор"
        )
        
        assert version2 == 2
        
        # Сравниваем версии
        comparison = await manager.blueprint_tracker.compare_blueprints(1, 2, detailed=True)
        assert comparison["version1"] == 1
        assert comparison["version2"] == 2
        assert comparison["change_type"] == "enhancement"
        
        # Анализируем влияние изменений
        impact = await manager.blueprint_tracker.analyze_impact(1, 2)
        assert impact.from_version == 1
        assert impact.to_version == 2
        assert impact.impact_score >= 0.0
        assert impact.impact_score <= 1.0
        assert impact.risk_assessment in ["minimal", "low", "medium", "high"]
        
        # Связываем ресурс
        await manager.blueprint_tracker.link_resource(
            identifier="test_resource",
            resource={"type": "service", "endpoint": "/api/test"},
            resource_type="service",
            dependency_level=2
        )
        
        # Получаем связанные ресурсы
        resources = await manager.blueprint_tracker.get_resource_links()
        assert len(resources) > 0
        
        resource_found = any(r.identifier == "test_resource" for r in resources)
        assert resource_found, "Связанный ресурс должен быть найден"
        
        # Получаем последнюю версию
        latest = await manager.blueprint_tracker.get_latest_blueprint()
        assert latest is not None
        assert latest.version == 2
    
    async def test_statistics(self, manager):
        """Тест получения статистики."""
        
        # Сохраняем несколько записей
        await manager.store(CORE, "Факт 1", {"test": 1})
        await manager.store(CORE, "Факт 2", {"test": 2})
        await manager.store(SEMANTIC, "Концепция 1", {"test": 3})
        
        # Получаем статистику
        stats = await manager.get_layer_statistics()
        
        assert "memory_context" in stats
        assert "cache" in stats
        assert "vector_store" in stats
        assert "layer_statistics" in stats
        assert "blueprint_tracker" in stats
        
        # Проверяем статистику по слоям
        assert CORE in stats["layer_statistics"]
        assert SEMANTIC in stats["layer_statistics"]
        
        core_stats = stats["layer_statistics"][CORE]
        assert core_stats["total_items"] >= 2
        
        semantic_stats = stats["layer_statistics"][SEMANTIC]
        assert semantic_stats["total_items"] >= 1
        
        # Проверяем статистику кэша
        cache_stats = stats["cache"]
        assert "size" in cache_stats
        assert "max_size" in cache_stats
        assert "utilization" in cache_stats
    
    async def test_optimization(self, manager):
        """Тест оптимизации памяти."""
        
        # Сохраняем данные
        await manager.store(CORE, "Данные для оптимизации", {"priority": 1})
        
        # Запускаем оптимизацию
        results = await manager.optimize_memory()
        
        assert "memory_optimization" in results
        assert "cache_optimization" in results
        assert "duration" in results
        assert results["duration"] > 0
        
        # Проверяем, что структура результатов правильная
        memory_opt = results["memory_optimization"]
        assert "total_optimized" in memory_opt
        
        cache_opt = results["cache_optimization"]
        assert "expired_entries_removed" in cache_opt
    
    async def test_sync_with_orchestrator(self, manager):
        """Тест синхронизации с оркестратором."""
        
        # Сохраняем данные
        await manager.store(CORE, "Тестовые данные", {"sync_test": True})
        
        # Синхронизируем
        sync_result = await manager.sync_with_orchestrator()
        
        assert "success" in sync_result
        assert "sync_timestamp" in sync_result
        assert "trace_id" in sync_result
        
        if sync_result["success"]:
            assert isinstance(sync_result["trace_id"], str)
            assert len(sync_result["trace_id"]) > 0
    
    async def test_error_handling(self, manager):
        """Тест обработки ошибок."""
        
        # Попытка работы с несуществующим слоем
        with pytest.raises(ValueError):
            await manager.store("INVALID_LAYER", "data")
        
        # Извлечение несуществующих данных не должно вызывать ошибку
        results = await manager.retrieve(CORE, query="nonexistent")
        assert isinstance(results, list)
        
        # Попытка обновления несуществующей записи
        update_success = await manager.update(CORE, "nonexistent-id", "data")
        assert update_success is False
        
        # Попытка удаления несуществующей записи  
        delete_success = await manager.delete(CORE, "nonexistent-id")
        assert delete_success is False
    
    async def test_performance(self, manager):
        """Тест производительности."""
        
        # Сохраняем 10 записей и замеряем время
        start_time = time.time()
        
        for i in range(10):
            await manager.store(
                layer=CORE if i % 2 == 0 else SEMANTIC,
                data=f"Performance test {i}",
                metadata={"test_id": i},
                priority=i % 10
            )
        
        store_time = time.time() - start_time
        avg_store_time = store_time / 10
        
        assert avg_store_time < 0.1, f"Сохранение слишком медленное: {avg_store_time:.3f}s"
        
        # Тест извлечения
        start_time = time.time()
        
        results = await manager.retrieve(CORE)
        
        retrieve_time = time.time() - start_time
        
        assert retrieve_time < 0.5, f"Извлечение слишком медленное: {retrieve_time:.3f}s"
        
        # Тест поиска по слоям
        start_time = time.time()
        
        search_results = await manager.search_across_layers(
            query="performance",
            layers=[CORE, SEMANTIC],
            limit=20
        )
        
        search_time = time.time() - start_time
        
        assert search_time < 1.0, f"Поиск слишком медленный: {search_time:.3f}s"


async def test_quick_memory_test():
    """Тест быстрой функции тестирования."""
    
    result = await quick_memory_test()
    
    assert "success" in result
    assert "stored_items" in result
    assert "retrieved_core" in result
    assert "retrieved_semantic" in result
    assert "statistics" in result
    
    assert result["success"] is True
    assert len(result["stored_items"]) == 2
    assert result["retrieved_core"] > 0
    assert result["retrieved_semantic"] > 0


if __name__ == "__main__":
    # Запуск тестов напрямую
    async def run_tests():
        print("🧪 Запуск тестов MemoryManager...")
        
        test_manager = TestMemoryManager()
        
        # Создаем менеджер для тестов
        manager = create_memory_manager({
            "cache_size": 100,
            "cache_ttl": 300,
            "optimization_interval": 60
        })
        
        try:
            await manager.start()
            
            print("\\n1. Тест базовых операций...")
            await test_manager.test_basic_operations(manager)
            print("✅ Базовые операции: OK")
            
            print("\\n2. Тест всех слоев памяти...")
            await test_manager.test_all_layers(manager)
            print("✅ Все слои: OK")
            
            print("\\n3. Тест поиска по слоям...")
            await test_manager.test_search_across_layers(manager)
            print("✅ Поиск по слоям: OK")
            
            print("\\n4. Тест Blueprint Tracker...")
            await test_manager.test_blueprint_tracker(manager)
            print("✅ Blueprint Tracker: OK")
            
            print("\\n5. Тест статистики...")
            await test_manager.test_statistics(manager)
            print("✅ Статистика: OK")
            
            print("\\n6. Тест оптимизации...")
            await test_manager.test_optimization(manager)
            print("✅ Оптимизация: OK")
            
            print("\\n7. Тест синхронизации...")
            await test_manager.test_sync_with_orchestrator(manager)
            print("✅ Синхронизация: OK")
            
            print("\\n8. Тест обработки ошибок...")
            await test_manager.test_error_handling(manager)
            print("✅ Обработка ошибок: OK")
            
            print("\\n9. Тест производительности...")
            await test_manager.test_performance(manager)
            print("✅ Производительность: OK")
            
            print("\\n10. Тест быстрой функции...")
            result = await quick_memory_test()
            assert result["success"]
            print("✅ Быстрая функция: OK")
            
            print("\\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
            
        finally:
            await manager.stop()
    
    asyncio.run(run_tests())
