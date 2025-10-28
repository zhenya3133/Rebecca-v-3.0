"""Пример использования полноценного MemoryManager с 6 слоями памяти.

Демонстрирует основные возможности системы управления памятью:
- Сохранение и извлечение данных из разных слоев
- Использование AdaptiveBlueprintTracker
- Поиск по слоям памяти
- Оптимизация и синхронизация
"""

import asyncio
import json
from typing import Dict, Any

from memory_manager import MemoryManager, create_memory_manager, CORE, EPISODIC, SEMANTIC, PROCEDURAL, VAULT, SECURITY
from logger import setup_logger

# Настройка логгера
logger = setup_logger(__name__)


async def demonstrate_basic_operations():
    """Демонстрация базовых операций с памятью."""
    
    logger.info("=== ДЕМОНСТРАЦИЯ БАЗОВЫХ ОПЕРАЦИЙ ===")
    
    # Создаем MemoryManager
    manager = create_memory_manager({
        "cache_size": 500,
        "cache_ttl": 1800,
        "optimization_interval": 60
    })
    
    try:
        # Запускаем менеджер
        await manager.start()
        logger.info("MemoryManager запущен")
        
        # 1. Сохраняем данные в разные слои
        logger.info("\\n1. Сохранение данных в слои памяти...")
        
        # Core Memory - базовые факты
        core_fact_id = await manager.store(
            layer=CORE,
            data="Реактор Rebecca работает на нейтронах",
            metadata={"source": "manual", "domain": "physics"},
            tags=["physics", "reactor"],
            priority=8
        )
        
        # Episodic Memory - события
        episodic_event_id = await manager.store(
            layer=EPISODIC,
            data="Сегодня была успешно проведена диагностика системы",
            metadata={"timestamp": "2025-10-28T03:55:14", "status": "completed"},
            tags=["diagnostics", "system"],
            priority=6
        )
        
        # Semantic Memory - концепции
        semantic_concept_id = await manager.store(
            layer=SEMANTIC,
            data="Искусственный интеллект - это технология имитации человеческого интеллекта",
            metadata={"category": "AI", "complexity": "medium"},
            tags=["AI", "technology", "definition"],
            priority=9
        )
        
        # Procedural Memory - процедуры
        procedural_workflow_id = await manager.store(
            layer=PROCEDURAL,
            data="Процедура запуска: 1) Проверить охлаждение 2) Активировать контроллеры 3) Запустить реактор",
            metadata={"steps": 3, "duration_minutes": 15},
            tags=["procedure", "startup", "safety"],
            priority=7
        )
        
        # Vault Memory - секреты (демо данные)
        secret_id = await manager.store(
            layer=VAULT,
            data="API ключ: sk-1234567890abcdef",
            metadata={"type": "api_key", "service": "external_api"},
            tags=["security", "api"],
            priority=10
        )
        
        # Security Memory - события безопасности
        security_event_id = await manager.store(
            layer=SECURITY,
            data="Обнаружена попытка несанкционированного доступа",
            metadata={"severity": "high", "source_ip": "192.168.1.100"},
            tags=["security", "intrusion"],
            priority=10
        )
        
        logger.info(f"Сохранено: {core_fact_id}, {episodic_event_id}, {semantic_concept_id}, "
                   f"{procedural_workflow_id}, {secret_id}, {security_event_id}")
        
        # 2. Извлекаем данные
        logger.info("\\n2. Извлечение данных из слоев...")
        
        core_data = await manager.retrieve(CORE)
        episodic_data = await manager.retrieve(EPISODIC)
        semantic_data = await manager.retrieve(SEMANTIC)
        
        logger.info(f"Core Memory: {len(core_data)} записей")
        logger.info(f"Episodic Memory: {len(episodic_data)} записей")
        logger.info(f"Semantic Memory: {len(semantic_data)} записей")
        
        # 3. Обновляем запись
        logger.info("\\n3. Обновление записи...")
        
        update_success = await manager.update(
            layer=CORE,
            item_id=core_fact_id,
            data="Реактор Rebecca работает на нейтронах с КПД 95%",
            metadata={"source": "manual", "domain": "physics", "efficiency": "95%"}
        )
        
        logger.info(f"Обновление Core Memory: {'успешно' if update_success else 'неудачно'}")
        
        # 4. Поиск по слоям
        logger.info("\\n4. Поиск по слоям памяти...")
        
        search_results = await manager.search_across_layers(
            query="реактор",
            layers=[CORE, EPISODIC, SEMANTIC],
            limit=10
        )
        
        logger.info(f"Найдено записей по запросу 'реактор': {len(search_results)}")
        
        # 5. Получаем статистику
        logger.info("\\n5. Статистика слоев памяти...")
        
        stats = await manager.get_layer_statistics()
        logger.info(f"Всего записей в памяти: {stats['memory_context']['total_memory_entries']}")
        logger.info(f"Использование кэша: {stats['cache']['utilization']:.2%}")
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка в демонстрации базовых операций: {e}")
        return False
    finally:
        await manager.stop()


async def demonstrate_blueprint_tracker():
    """Демонстрация работы с AdaptiveBlueprintTracker."""
    
    logger.info("\\n=== ДЕМОНСТРАЦИЯ ADAPTIVE BLUEPRINT TRACKER ===")
    
    # Создаем MemoryManager
    manager = create_memory_manager()
    
    try:
        await manager.start()
        
        # 1. Записываем начальное состояние архитектуры
        logger.info("\\n1. Запись начального состояния архитектуры...")
        
        initial_blueprint = {
            "version": "1.0",
            "components": {
                "core": {"name": "CoreMemory", "status": "active"},
                "episodic": {"name": "EpisodicMemory", "status": "active"},
                "semantic": {"name": "SemanticMemory", "status": "active"},
                "procedural": {"name": "ProceduralMemory", "status": "active"},
                "vault": {"name": "VaultMemory", "status": "active"},
                "security": {"name": "SecurityMemory", "status": "active"}
            },
            "config": {
                "cache_size": 1000,
                "optimization_interval": 300
            }
        }
        
        version1 = await manager.blueprint_tracker.record_blueprint(
            blueprint=initial_blueprint,
            metadata={"author": "system", "description": "Initial architecture"},
            change_type="initial"
        )
        
        logger.info(f"Записана версия архитектуры: {version1}")
        
        # 2. Связываем ресурсы
        logger.info("\\n2. Связывание ресурсов с архитектурой...")
        
        await manager.blueprint_tracker.link_resource(
            identifier="core_memory_service",
            resource={
                "type": "service",
                "endpoint": "/api/core",
                "version": "1.0"
            },
            resource_type="microservice",
            dependency_level=3
        )
        
        await manager.blueprint_tracker.link_resource(
            identifier="vector_store_db",
            resource={
                "type": "database",
                "connection": "postgresql://localhost:5432",
                "version": "14.2"
            },
            resource_type="database",
            dependency_level=2
        )
        
        # 3. Записываем изменения
        logger.info("\\n3. Запись изменений архитектуры...")
        
        updated_blueprint = initial_blueprint.copy()
        updated_blueprint["components"]["core"]["status"] = "enhanced"
        updated_blueprint["components"]["ai_processor"] = {
            "name": "AIProcessor", 
            "status": "active"
        }
        updated_blueprint["config"]["cache_size"] = 2000
        
        version2 = await manager.blueprint_tracker.record_blueprint(
            blueprint=updated_blueprint,
            metadata={"author": "developer", "changes": "added AI processor, increased cache"},
            change_type="enhancement",
            change_description="Добавлен AI процессор и увеличен кэш"
        )
        
        logger.info(f"Записана версия архитектуры: {version2}")
        
        # 4. Сравниваем версии
        logger.info("\\n4. Сравнение версий архитектуры...")
        
        comparison = await manager.blueprint_tracker.compare_blueprints(1, 2, detailed=True)
        logger.info(f"Тип изменения: {comparison['change_type']}")
        logger.info(f"Описание: {comparison['change_description']}")
        logger.info(f"Добавлено элементов: {len(comparison.get('detailed_changes', {}).get('added', []))}")
        logger.info(f"Изменено элементов: {len(comparison.get('detailed_changes', {}).get('modified', []))}")
        
        # 5. Анализируем влияние
        logger.info("\\n5. Анализ влияния изменений...")
        
        impact = await manager.blueprint_tracker.analyze_impact(1, 2)
        logger.info(f"Оценка влияния: {impact.impact_score:.2f}")
        logger.info(f"Уровень риска: {impact.risk_assessment}")
        logger.info(f"Рекомендации: {len(impact.recommendations)}")
        
        # 6. Получаем связи ресурсов
        logger.info("\\n6. Связанные ресурсы...")
        
        resources = await manager.blueprint_tracker.get_resource_links()
        logger.info(f"Всего связанных ресурсов: {len(resources)}")
        
        for resource in resources:
            logger.info(f"Ресурс: {resource.identifier} (тип: {resource.resource_type}, "
                       f"зависимость: {resource.dependency_level})")
        
        # 7. Получаем историю архитектуры
        logger.info("\\n7. История архитектуры...")
        
        lineage = await manager.blueprint_tracker.get_blueprint_lineage()
        logger.info(f"Всего версий в истории: {len(lineage)}")
        
        # 8. Проверяем целостность
        logger.info("\\n8. Проверка целостности...")
        
        integrity = await manager.blueprint_tracker.validate_blueprint_integrity(2)
        logger.info(f"Целостность версии 2: {'OK' if integrity['valid'] else 'ОШИБКА'}")
        
        # 9. Статистика трекера
        logger.info("\\n9. Статистика трекера...")
        
        tracker_stats = manager.blueprint_tracker.get_statistics()
        logger.info(f"Всего версий: {tracker_stats['total_versions']}")
        logger.info(f"Текущая версия: {tracker_stats['current_version']}")
        logger.info(f"Связей ресурсов: {tracker_stats['total_resource_links']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка в демонстрации Blueprint Tracker: {e}")
        return False
    finally:
        await manager.stop()


async def demonstrate_advanced_features():
    """Демонстрация продвинутых возможностей."""
    
    logger.info("\\n=== ДЕМОНСТРАЦИЯ ПРОДВИНУТЫХ ВОЗМОЖНОСТЕЙ ===")
    
    manager = create_memory_manager({
        "cache_size": 100,
        "cache_ttl": 300,
        "optimization_interval": 30
    })
    
    try:
        await manager.start()
        
        # 1. Создаем много данных для демонстрации оптимизации
        logger.info("\\n1. Создание данных для демонстрации оптимизации...")
        
        # Добавляем много записей с разными приоритетами
        for i in range(50):
            priority = (i % 11)  # 0-10
            
            await manager.store(
                layer=CORE if i % 3 == 0 else SEMANTIC if i % 3 == 1 else EPISODIC,
                data=f"Тестовая запись {i} с приоритетом {priority}",
                metadata={"test_id": i, "priority": priority},
                tags=["test", f"priority_{priority}"],
                priority=priority
            )
        
        logger.info("Создано 50 тестовых записей")
        
        # 2. Демонстрируем поиск с фильтрами
        logger.info("\\n2. Поиск с фильтрами...")
        
        semantic_results = await manager.retrieve(
            layer=SEMANTIC,
            filters={"test_id": {"$gte": 10}},
            limit=5
        )
        
        logger.info(f"Найдено семантических записей с фильтром: {len(semantic_results)}")
        
        # 3. Удаляем некоторые записи
        logger.info("\\n3. Удаление записей...")
        
        # Получаем первую запись для удаления
        all_core = await manager.retrieve(CORE)
        if all_core:
            delete_id = all_core[0]["id"]
            delete_success = await manager.delete(CORE, delete_id)
            logger.info(f"Удаление записи {delete_id}: {'успешно' if delete_success else 'неудачно'}")
        
        # 4. Запускаем оптимизацию
        logger.info("\\n4. Запуск оптимизации памяти...")
        
        optimization_results = await manager.optimize_memory()
        logger.info(f"Оптимизация завершена за {optimization_results['duration']:.2f}s")
        logger.info(f"Удалено записей: {optimization_results['total_items_removed']}")
        
        # 5. Синхронизация с оркестратором
        logger.info("\\n5. Синхронизация с оркестратором...")
        
        sync_result = await manager.sync_with_orchestrator()
        logger.info(f"Синхронизация: {'успешно' if sync_result['success'] else 'неудачно'}")
        if sync_result['success']:
            logger.info(f"Trace ID: {sync_result['trace_id']}")
        
        # 6. Подробная статистика
        logger.info("\\n6. Подробная статистика...")
        
        detailed_stats = await manager.get_layer_statistics()
        
        logger.info("\\nСтатистика по слоям:")
        for layer, stats in detailed_stats['layer_statistics'].items():
            logger.info(f"  {layer}:")
            logger.info(f"    Всего элементов: {stats['total_items']}")
            logger.info(f"    Коэффициент попаданий в кэш: {stats['cache_hit_ratio']:.2%}")
            logger.info(f"    Среднее время доступа: {stats['average_access_time']:.4f}s")
        
        logger.info("\\nИнформация о векторном хранилище:")
        vector_info = detailed_stats['vector_store']
        logger.info(f"  Провайдер: {vector_info['current_provider']}")
        
        logger.info("\\nКэш:")
        cache_stats = detailed_stats['cache']
        logger.info(f"  Использование: {cache_stats['utilization']:.2%}")
        logger.info(f"  Размер: {cache_stats['size']}/{cache_stats['max_size']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка в демонстрации продвинутых возможностей: {e}")
        return False
    finally:
        await manager.stop()


async def demonstrate_error_handling():
    """Демонстрация обработки ошибок."""
    
    logger.info("\\n=== ДЕМОНСТРАЦИЯ ОБРАБОТКИ ОШИБОК ===")
    
    manager = create_memory_manager()
    
    try:
        await manager.start()
        
        # 1. Попытка работы с несуществующим слоем
        logger.info("\\n1. Тест работы с несуществующим слоем...")
        
        try:
            await manager.store("INVALID_LAYER", "data")
            logger.error("Ошибка: должно было возникнуть исключение")
        except ValueError as e:
            logger.info(f"Корректно обработана ошибка: {e}")
        
        # 2. Попытка извлечения несуществующей записи
        logger.info("\\n2. Тест извлечения несуществующей записи...")
        
        nonexistent_results = await manager.retrieve(CORE, query="nonexistent")
        logger.info(f"Результат поиска несуществующих данных: {len(nonexistent_results)} записей")
        
        # 3. Попытка обновления несуществующей записи
        logger.info("\\n3. Тест обновления несуществующей записи...")
        
        update_result = await manager.update(CORE, "nonexistent-id", "new_data")
        logger.info(f"Результат обновления несуществующей записи: {update_result}")
        
        # 4. Попытка удаления несуществующей записи
        logger.info("\\n4. Тест удаления несуществующей записи...")
        
        delete_result = await manager.delete(CORE, "nonexistent-id")
        logger.info(f"Результат удаления несуществующей записи: {delete_result}")
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка в демонстрации обработки ошибок: {e}")
        return False
    finally:
        await manager.stop()


async def main():
    """Главная функция демонстрации."""
    
    logger.info("🚀 ЗАПУСК ДЕМОНСТРАЦИИ MEMORYMANAGER")
    logger.info("=" * 60)
    
    success_count = 0
    total_tests = 4
    
    try:
        # Базовые операции
        if await demonstrate_basic_operations():
            success_count += 1
        
        # Blueprint Tracker
        if await demonstrate_blueprint_tracker():
            success_count += 1
        
        # Продвинутые возможности
        if await demonstrate_advanced_features():
            success_count += 1
        
        # Обработка ошибок
        if await demonstrate_error_handling():
            success_count += 1
        
    except Exception as e:
        logger.error(f"Критическая ошибка в демонстрации: {e}")
    
    # Итоговый отчет
    logger.info("\\n" + "=" * 60)
    logger.info("📊 ИТОГОВЫЙ ОТЧЕТ")
    logger.info("=" * 60)
    logger.info(f"Пройдено тестов: {success_count}/{total_tests}")
    logger.info(f"Успешность: {success_count/total_tests*100:.1f}%")
    
    if success_count == total_tests:
        logger.info("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    else:
        logger.warning(f"⚠️ {total_tests - success_count} тестов завершились с ошибками")
    
    logger.info("\\n🎯 ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")


if __name__ == "__main__":
    asyncio.run(main())
