"""
Пример использования интерфейса MemoryManager.
Демонстрирует все основные возможности системы управления памятью.
"""

import asyncio
from typing import Dict, Any

from memory_manager_interface import (
    MemoryManager,
    MemoryLayer,
    MemoryFilter,
    MemoryContext,
    VectorStoreClient
)


async def demo_basic_operations():
    """Демонстрация основных операций с памятью."""
    print("=== Демонстрация основных операций с памятью ===\n")
    
    # Создание менеджера памяти
    memory_manager = MemoryManager(
        cache_ttl=300.0,
        max_cache_size=1000
    )
    
    print(f"Доступные слои памяти: {[layer.value for layer in memory_manager.list_layers()]}")
    print(f"Статистика памяти: {memory_manager.get_memory_stats()}\n")
    
    # Сохранение данных в разные слои
    print("1. Сохранение данных в слои памяти:")
    
    # Сохранение в Core память
    core_data = {"fact": "Python - это язык программирования", "confidence": 0.9}
    core_id = await memory_manager.store(
        MemoryLayer.CORE, 
        core_data, 
        {"category": "programming", "language": "python"}
    )
    print(f"   ✓ Core память - ID: {core_id}")
    
    # Сохранение в Semantic память
    semantic_data = {"concept": "машинное обучение", "definition": "подраздел ИИ"}
    semantic_id = await memory_manager.store(
        MemoryLayer.SEMANTIC,
        semantic_data,
        {"domain": "AI", "importance": "high"}
    )
    print(f"   ✓ Semantic память - ID: {semantic_id}")
    
    # Сохранение в Episodic память
    episodic_data = {"event": "изучение Python", "date": "2024-01-15", "result": "успешно"}
    episodic_id = await memory_manager.store(
        MemoryLayer.EPISODIC,
        episodic_data,
        {"context": "learning", "duration": "2 часа"}
    )
    print(f"   ✓ Episodic память - ID: {episodic_id}")
    
    print("\n2. Поиск данных:")
    
    # Поиск в Core памяти
    core_results = await memory_manager.retrieve(
        MemoryLayer.CORE,
        "Python",
        MemoryFilter(metadata={"language": "python"})
    )
    print(f"   ✓ Поиск в Core памяти: {len(core_results)} результатов")
    for result in core_results:
        print(f"     - {result.data}")
    
    # Поиск в Semantic памяти
    semantic_results = await memory_manager.retrieve(
        MemoryLayer.SEMANTIC,
        "обучение",
        MemoryFilter(metadata={"domain": "AI"})
    )
    print(f"   ✓ Поиск в Semantic памяти: {len(semantic_results)} результатов")
    for result in semantic_results:
        print(f"     - {result.data}")
    
    print("\n3. Поиск по всем слоям:")
    
    # Поиск по всем слоям одновременно
    all_results = await memory_manager.search_across_layers(
        "обучение",
        [MemoryLayer.CORE, MemoryLayer.SEMANTIC, MemoryLayer.EPISODIC]
    )
    for layer, results in all_results.items():
        print(f"   ✓ {layer.value}: {len(results)} результатов")
    
    print("\n4. Обновление данных:")
    
    # Обновление данных
    success = await memory_manager.update(
        MemoryLayer.CORE,
        core_id,
        {"confidence": 0.95, "tags": ["python", "programming", "ai"]}
    )
    print(f"   ✓ Обновление Core памяти: {'успешно' if success else 'ошибка'}")
    
    print("\n5. Статистика после операций:")
    stats = memory_manager.get_memory_stats()
    print(f"   - Закэшированных запросов: {stats['cache_stats']['hit_count']}")
    print(f"   - Кэш-промахов: {stats['cache_stats']['miss_count']}")
    print(f"   - Процент попаданий в кэш: {stats['cache_stats']['hit_rate']:.2%}")
    
    print("\n6. Удаление данных:")
    
    # Удаление данных
    success = await memory_manager.delete(MemoryLayer.EPISODIC, episodic_id)
    print(f"   ✓ Удаление из Episodic памяти: {'успешно' if success else 'ошибка'}")
    
    return memory_manager


async def demo_advanced_features(memory_manager: MemoryManager):
    """Демонстрация продвинутых возможностей."""
    print("\n=== Демонстрация продвинутых возможностей ===\n")
    
    print("1. Работа с фильтрами по времени:")
    
    # Добавление данных с разными временными метками
    import time
    
    # Данные "вчера"
    yesterday = time.time() - 86400
    data1 = await memory_manager.store(
        MemoryLayer.EPISODIC,
        {"event": "вчерашнее событие"},
        {"date": "yesterday"}
    )
    
    # Данные "сегодня"
    today_data = await memory_manager.store(
        MemoryLayer.EPISODIC,
        {"event": "сегодняшнее событие"},
        {"date": "today"}
    )
    
    # Фильтрация по времени
    time_filter = MemoryFilter(
        time_range=(yesterday, time.time()),
        metadata={"date": "today"}
    )
    
    results = await memory_manager.retrieve(
        MemoryLayer.EPISODIC,
        "событие",
        time_filter
    )
    print(f"   ✓ Найдено событий за сегодня: {len(results)}")
    
    print("\n2. Работа с метаданными:")
    
    # Сложная фильтрация по метаданным
    complex_filter = MemoryFilter(
        metadata={
            "domain": "AI",
            "importance": "high",
            "category": "programming"
        }
    )
    
    semantic_results = await memory_manager.retrieve(
        MemoryLayer.SEMANTIC,
        "",
        complex_filter
    )
    print(f"   ✓ Найдено результатов с комплексными фильтрами: {len(semantic_results)}")
    
    print("\n3. Очистка кэша:")
    memory_manager.clear_cache()
    print("   ✓ Кэш очищен")
    
    print("\n4. Финальная статистика:")
    final_stats = memory_manager.get_memory_stats()
    print(f"   - Общее количество проиндексированных элементов: {final_stats['indexed_items']}")
    print(f"   - Ключи метаданных в индексе: {final_stats['metadata_keys']}")
    print(f"   - Размер кэша: {final_stats['cache_stats']['cache_size']}")


async def demo_factory_pattern():
    """Демонстрация паттерна Factory."""
    print("\n=== Демонстрация паттерна Factory ===\n")
    
    from .memory_manager_interface import LayerFactory, MemoryLayer
    
    # Регистрация пользовательского слоя памяти
    class CustomMemoryLayer:
        def __init__(self):
            self.data = {}
        
        def store(self, data: Dict[str, Any], metadata: Dict[str, Any]):
            self.data[str(hash(str(data)))] = {"data": data, "metadata": metadata}
    
    # Регистрация нового слоя
    LayerFactory.register_layer(MemoryLayer.VAULT, CustomMemoryLayer)
    
    # Создание слоя через фабрику
    custom_layer = LayerFactory.create_layer(MemoryLayer.VAULT)
    print(f"   ✓ Создан пользовательский слой: {type(custom_layer).__name__}")
    
    # Тестирование функциональности
    custom_layer.store({"test": "data"}, {"test": "metadata"})
    print(f"   ✓ Данные в пользовательском слое: {list(custom_layer.data.keys())}")


def main():
    """Основная функция демонстрации."""
    print("🚀 Демонстрация работы с MemoryManager интерфейсом\n")
    print("=" * 60)
    
    async def run_demo():
        try:
            # Основные операции
            memory_manager = await demo_basic_operations()
            
            # Продвинутые возможности
            await demo_advanced_features(memory_manager)
            
            # Паттерн Factory
            await demo_factory_pattern()
            
            print("\n" + "=" * 60)
            print("✅ Демонстрация завершена успешно!")
            
        except Exception as e:
            print(f"\n❌ Ошибка во время демонстрации: {e}")
            import traceback
            traceback.print_exc()
    
    # Запуск демонстрации
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()