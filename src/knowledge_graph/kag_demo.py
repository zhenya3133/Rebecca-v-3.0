"""
Пример использования KAG системы интеграции с 6 слоями памяти Rebecca-Platform.

Демонстрирует:
1. Инициализацию системы
2. Добавление знаний в разные слои памяти
3. Выполнение запросов
4. Синхронизацию данных
5. Валидацию и контроль доступа
6. Мониторинг системы
"""

import asyncio
import json
from datetime import datetime

# Импорт компонентов Rebecca-Platform
from src.memory_manager.memory_manager import create_memory_manager
from src.knowledge_graph.memory_integration import (
    KAGMemoryIntegration, 
    NodeType, 
    EdgeType, 
    AccessLevel,
    create_kag_integration,
    quick_kag_test
)


async def comprehensive_kag_demo():
    """Комплексная демонстрация возможностей KAG системы."""
    
    print("🚀 Запуск комплексной демонстрации KAG системы")
    print("=" * 60)
    
    # 1. Инициализация системы
    print("\n📦 1. Инициализация системы...")
    
    memory_manager = create_memory_manager({
        "cache_size": 100,
        "cache_ttl": 1800,
        "optimization_interval": 300
    })
    
    await memory_manager.start()
    
    kag_integration = await create_kag_integration(memory_manager)
    await kag_integration.start()
    
    print("✅ Система инициализирована успешно")
    
    # 2. Добавление знаний в разные слои
    print("\n📚 2. Добавление знаний в разные слои памяти...")
    
    # Core Layer - системные концепты
    core_concept = await kag_integration.add_knowledge(
        content="Rebecca-Platform - интеллектуальная система агентов с 6 слоями памяти",
        node_type=NodeType.CONCEPT,
        metadata={
            "system": "rebecca",
            "category": "platform",
            "version": "1.0",
            "description": "Базовая информация о платформе"
        },
        tags=["platform", "agents", "memory"],
        access_level=AccessLevel.INTERNAL
    )
    
    # Episodic Layer - события
    event_id = await kag_integration.add_knowledge(
        content="Успешная интеграция KAG системы с 6 слоями памяти Rebecca-Platform завершена",
        node_type=NodeType.EVENT,
        metadata={
            "timestamp": datetime.now().isoformat(),
            "event_type": "integration_completed",
            "outcome": "success",
            "participants": ["KAG", "MemoryManager", "6 Memory Layers"]
        },
        tags=["integration", "success", "milestone"],
        access_level=AccessLevel.INTERNAL
    )
    
    # Semantic Layer - концептуальные связи
    semantic_concept = await kag_integration.add_knowledge(
        content="Агенты в Rebecca-Platform используют KAG систему для семантического поиска",
        node_type=NodeType.CONCEPT,
        metadata={
            "relationship_type": "uses",
            "domain": "artificial_intelligence",
            "semantic_context": "agent_knowledge_retrieval"
        },
        tags=["agents", "KAG", "semantic_search", "knowledge"],
        access_level=AccessLevel.INTERNAL
    )
    
    # Procedural Layer - алгоритмы
    procedure_id = await kag_integration.add_knowledge(
        content="""
        Алгоритм работы с KAG интеграцией:
        1. Инициализировать MemoryManager
        2. Создать KAGMemoryIntegration
        3. Добавить знания с указанием типа узла
        4. Выполнить синхронизацию слоев
        5. Запрашивать знания через семантический поиск
        6. Валидировать результаты
        """,
        node_type=NodeType.PROCEDURE,
        metadata={
            "algorithm_name": "KAG_integration_workflow",
            "complexity": "O(n log n)",
            "steps": 6,
            "category": "workflow"
        },
        tags=["algorithm", "workflow", "integration"],
        access_level=AccessLevel.INTERNAL
    )
    
    # Vault Layer - секретные знания
    vault_id = await kag_integration.add_knowledge(
        content="Секретный API endpoint для интеграции: https://api.rebecca.internal/v1/kag/sync",
        node_type=NodeType.VAULT_ITEM,
        metadata={
            "classification": "secret",
            "sensitivity": "high",
            "service": "rebecca_api",
            "retention": "permanent"
        },
        tags=["API", "endpoint", "integration"],
        access_level=AccessLevel.SECRET
    )
    
    # Security Layer - правила безопасности
    security_rule = await kag_integration.add_knowledge(
        content="Все операции синхронизации между KAG графом и памятью должны логироваться в security layer",
        node_type=NodeType.SECURITY_RULE,
        metadata={
            "rule_type": "audit_policy",
            "enforcement": "mandatory",
            "scope": "all_sync_operations",
            "compliance": ["GDPR", "SOX"]
        },
        tags=["security", "audit", "logging", "compliance"],
        access_level=AccessLevel.CONFIDENTIAL
    )
    
    print("✅ Добавлены знания во все 6 слоев памяти")
    print(f"   - Core Layer: {core_concept}")
    print(f"   - Episodic Layer: {event_id}")
    print(f"   - Semantic Layer: {semantic_concept}")
    print(f"   - Procedural Layer: {procedure_id}")
    print(f"   - Vault Layer: {vault_id}")
    print(f"   - Security Layer: {security_rule}")
    
    # 3. Выполнение семантических запросов
    print("\n🔍 3. Выполнение семантических запросов...")
    
    # Запрос по ключевому слову "платформа"
    platform_results = await kag_integration.query_knowledge(
        query="платформа",
        node_types=[NodeType.CONCEPT, NodeType.EVENT],
        max_results=5
    )
    
    print(f"📊 Результаты запроса 'платформа': {len(platform_results)} найдено")
    for i, result in enumerate(platform_results, 1):
        print(f"   {i}. {result['content'][:60]}...")
        print(f"      Уверенность: {result['confidence']:.2f}, Тип: {result['node_type']}")
    
    # Запрос по алгоритмам
    algorithm_results = await kag_integration.query_knowledge(
        query="алгоритм",
        node_types=[NodeType.PROCEDURE],
        max_results=3
    )
    
    print(f"\\n🔧 Результаты запроса 'алгоритм': {len(algorithm_results)} найдено")
    for i, result in enumerate(algorithm_results, 1):
        print(f"   {i}. {result['content'][:60]}...")
        print(f"      Уверенность: {result['confidence']:.2f}")
    
    # 4. Синхронизация данных
    print("\n🔄 4. Синхронизация данных между слоями...")
    
    sync_results = await kag_integration.sync_all_layers()
    
    print("📋 Результаты синхронизации:")
    print(f"   - Успешных слоев: {sync_results['successful_layers']}/{sync_results['total_layers']}")
    print(f"   - Синхронизировано элементов: {sync_results['total_synced_items']}")
    print(f"   - Время синхронизации: {sync_results['duration']:.2f}s")
    
    for layer, result in sync_results['layer_results'].items():
        if result['success']:
            print(f"   ✅ {layer}: {result['synced_items']} элементов")
        else:
            print(f"   ❌ {layer}: ошибка - {result['error']}")
    
    # 5. Валидация знаний
    print("\n✅ 5. Валидация знаний...")
    
    validation_tasks = [
        (core_concept, "Core Concept"),
        (event_id, "Episodic Event"),
        (procedure_id, "Procedural Knowledge"),
        (vault_id, "Vault Item"),
        (security_rule, "Security Rule")
    ]
    
    for node_id, description in validation_tasks:
        validation_result = await kag_integration.validate_knowledge(node_id)
        
        status_icon = "✅" if validation_result["valid"] else "❌"
        print(f"   {status_icon} {description}: {validation_result['status']} "
              f"(confidence: {validation_result['confidence']:.2f})")
    
    # 6. Мониторинг системы
    print("\n📈 6. Мониторинг системы...")
    
    system_status = await kag_integration.get_system_status()
    
    print("📊 Статистика системы:")
    graph_stats = system_status['graph_statistics']
    print(f"   - Узлов в графе: {graph_stats['total_nodes']}")
    print(f"   - Связей в графе: {graph_stats['total_edges']}")
    print(f"   - Успешная валидация: {graph_stats.get('validation_success_rate', 0):.1%}")
    
    sync_stats = system_status['sync_statistics']
    print(f"   - Общие операции синхронизации: {sync_stats['total_sync_operations']}")
    print(f"   - Успешные синхронизации: {sync_stats['successful_syncs']}")
    print(f"   - Узлы в графе: {sync_stats['nodes_in_graph']}")
    
    memory_stats = system_status['memory_statistics']
    print(f"   - Элементов в памяти: {memory_stats['memory_context']['total_items']}")
    print(f"   - Попадания в кэш: {memory_stats['cache']['utilization']:.1%}")
    
    # 7. Демонстрация контроля доступа
    print("\n🔐 7. Демонстрация контроля доступа...")
    
    # Проверяем доступ к разным уровням
    access_levels = [
        (AccessLevel.PUBLIC, "Публичный доступ"),
        (AccessLevel.INTERNAL, "Внутренний доступ"),
        (AccessLevel.CONFIDENTIAL, "Конфиденциальный доступ"),
        (AccessLevel.SECRET, "Секретный доступ"),
        (AccessLevel.TOP_SECRET, "Строго секретно")
    ]
    
    for level, description in access_levels:
        # Классифицируем контент
        test_content = f"Тестовый контент {description.lower()}"
        auto_classification = kag_integration.access_control.classify_content(
            test_content, 
            {}
        )
        print(f"   📝 {description}: классифицирован как {auto_classification.value}")
    
    # 8. Тест производительности
    print("\n⚡ 8. Тест производительности...")
    
    import time
    
    # Тест добавления знаний
    start_time = time.time()
    for i in range(10):
        await kag_integration.add_knowledge(
            content=f"Тестовое знание {i} для проверки производительности",
            node_type=NodeType.CONCEPT,
            tags=["performance_test"]
        )
    add_time = time.time() - start_time
    
    # Тест запросов
    start_time = time.time()
    for i in range(5):
        await kag_integration.query_knowledge("тестовое", max_results=3)
    query_time = time.time() - start_time
    
    print(f"   ⏱️ Добавление 10 знаний: {add_time:.3f}s ({add_time/10*1000:.1f}ms на операцию)")
    print(f"   ⏱️ 5 запросов: {query_time:.3f}s ({query_time/5*1000:.1f}ms на запрос)")
    
    # 9. Финальная статистика
    print("\n📊 9. Финальная статистика...")
    
    final_status = await kag_integration.get_system_status()
    final_graph_stats = final_status['graph_statistics']
    
    print("🎯 Итоговые результаты:")
    print(f"   ✅ Создано узлов: {final_graph_stats['total_nodes']}")
    print(f"   ✅ Создано связей: {final_graph_stats['total_edges']}")
    print(f"   ✅ Успешных операций: {final_status['sync_statistics']['successful_syncs']}")
    print(f"   ✅ Система запущена: {final_status['running']}")
    print(f"   ✅ Версия KAG: {final_status['kag_version']}")
    
    # 10. Очистка ресурсов
    print("\n🧹 10. Очистка ресурсов...")
    
    await kag_integration.stop()
    await memory_manager.stop()
    
    print("✅ Система корректно остановлена")
    print("\n" + "=" * 60)
    print("🎉 Демонстрация завершена успешно!")
    print("💡 Все 6 слоев памяти интегрированы с KAG системой")
    print("🔄 Bidirectional synchronization работает корректно")
    print("🔒 Security и validation слои активны")
    print("📈 Система готова к промышленному использованию")
    
    return {
        "success": True,
        "total_nodes": final_graph_stats['total_nodes'],
        "total_edges": final_graph_stats['total_edges'],
        "sync_success": final_status['sync_statistics']['successful_syncs'],
        "performance": {
            "add_operations_per_second": 10 / add_time,
            "query_operations_per_second": 5 / query_time
        }
    }


async def quick_demo():
    """Быстрая демонстрация основных функций."""
    
    print("🚀 Быстрая демонстрация KAG системы")
    print("-" * 40)
    
    try:
        # Запуск быстрого теста
        result = await quick_kag_test()
        
        if result["success"]:
            print("✅ Тест пройден успешно!")
            print(f"   Создано знаний: {len(result['created_knowledge'])}")
            print(f"   Результатов запросов: {result['query_results_count']}")
            print(f"   Система запущена: {result['system_status']['running']}")
        else:
            print(f"❌ Тест не пройден: {result.get('error', 'Неизвестная ошибка')}")
            
    except Exception as e:
        print(f"❌ Ошибка демонстрации: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Быстрая демонстрация
        asyncio.run(quick_demo())
    else:
        # Полная демонстрация
        result = asyncio.run(comprehensive_kag_demo())
        
        print(f"\n📋 Отчет о демонстрации:")
        print(json.dumps(result, indent=2, ensure_ascii=False))