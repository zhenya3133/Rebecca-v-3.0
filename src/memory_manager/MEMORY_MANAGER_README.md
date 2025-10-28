# MemoryManager - Полноценная система управления памятью

## Обзор

MemoryManager - это комплексная система управления памятью, обеспечивающая эффективное хранение, извлечение и оптимизацию данных в многослойной архитектуре. Система интегрирована с векторными хранилищами, кэшированием и отслеживанием изменений архитектуры.

## Основные возможности

### 🧠 Многослойная архитектура памяти

1. **CoreMemory (CORE)** - Базовые факты и константы
2. **EpisodicMemory (EPISODIC)** - События и временные данные
3. **SemanticMemory (SEMANTIC)** - Концепции и знания
4. **ProceduralMemory (PROCEDURAL)** - Процедуры и рабочие процессы
5. **VaultMemory (VAULT)** - Секретная информация
6. **SecurityMemory (SECURITY)** - События безопасности

### ⚡ Ключевые функции

- **Высокопроизводительное кэширование** с LRU алгоритмом
- **Интеграция с векторными хранилищами** (Qdrant, ChromaDB, Weaviate)
- **AdaptiveBlueprintTracker** для отслеживания изменений архитектуры
- **Автоматическая оптимизация** памяти
- **Синхронизация с оркестратором**
- **Comprehensive логирование** и обработка ошибок
- **Параллельная обработка** запросов

## Быстрый старт

### Базовая инициализация

```python
from memory_manager import MemoryManager, create_memory_manager, CORE, EPISODIC

# Создание с конфигурацией по умолчанию
manager = create_memory_manager()

# Запуск менеджера
await manager.start()

# Сохранение данных
core_id = await manager.store(
    layer=CORE,
    data="Базовый факт системы",
    metadata={"source": "system"},
    tags=["core", "system"],
    priority=8
)

# Извлечение данных
core_data = await manager.retrieve(CORE)
print(f"Найдено {len(core_data)} записей")

# Остановка
await manager.stop()
```

### Продвинутая конфигурация

```python
from memory_manager import MemoryManager
from vector_store_client import VectorStoreConfig

# Создание конфигурации
config = {
    "cache_size": 2000,
    "cache_ttl": 7200,  # 2 часа
    "optimization_interval": 300,  # 5 минут
    "vector_store": {
        "provider": "qdrant",
        "base_url": "http://localhost:6333",
        "vector_size": 384,
        "distance_metric": "cosine"
    }
}

# Создание менеджера
manager = create_memory_manager(config)
await manager.start()
```

## API Reference

### MemoryManager основные методы

#### store(layer, data, metadata=None, tags=None, priority=5)

Сохраняет данные в указанный слой памяти.

**Параметры:**
- `layer` (str): Слой памяти (CORE, EPISODIC, SEMANTIC, etc.)
- `data` (Any): Данные для сохранения
- `metadata` (Dict): Дополнительные метаданные
- `tags` (List[str]): Список тегов
- `priority` (int): Приоритет записи (0-10)

**Возвращает:** ID сохраненной записи

**Пример:**
```python
semantic_id = await manager.store(
    layer=SEMANTIC,
    data="Искусственный интеллект - это технология",
    metadata={"category": "AI", "complexity": "medium"},
    tags=["AI", "definition"],
    priority=9
)
```

#### retrieve(layer, query=None, filters=None, limit=10)

Извлекает данные из указанного слоя памяти.

**Параметры:**
- `layer` (str): Слой памяти
- `query` (str): Поисковый запрос
- `filters` (Dict): Фильтры для поиска
- `limit` (int): Максимальное количество результатов

**Возвращает:** Список записей

**Пример:**
```python
# Быстрое извлечение всех записей слоя
all_core = await manager.retrieve(CORE)

# Поиск с фильтрами
filtered_results = await manager.retrieve(
    layer=SEMANTIC,
    query="искусственный интеллект",
    filters={"complexity": "medium"},
    limit=5
)
```

#### update(layer, item_id, data, metadata=None)

Обновляет запись в памяти.

**Параметры:**
- `layer` (str): Слой памяти
- `item_id` (str): ID записи
- `data` (Any): Новые данные
- `metadata` (Dict): Новые метаданные

**Возвращает:** True при успешном обновлении

**Пример:**
```python
success = await manager.update(
    layer=CORE,
    item_id=core_id,
    data="Обновленный факт",
    metadata={"updated": True}
)
```

#### delete(layer, item_id)

Удаляет запись из памяти.

**Параметры:**
- `layer` (str): Слой памяти
- `item_id` (str): ID записи

**Возвращает:** True при успешном удалении

**Пример:**
```python
success = await manager.delete(CORE, item_id)
```

#### search_across_layers(query, layers=None, limit=20)

Ищет данные по всем слоям памяти.

**Параметры:**
- `query` (str): Поисковый запрос
- `layers` (List[str]): Список слоев (None = все слои)
- `limit` (int): Максимальное количество результатов

**Возвращает:** Список найденных записей из всех слоев

**Пример:**
```python
results = await manager.search_across_layers(
    query="реактор",
    layers=[CORE, SEMANTIC],
    limit=10
)
```

#### get_layer_statistics()

Возвращает подробную статистику по всем слоям памяти.

**Возвращает:** Словарь со статистикой

**Пример:**
```python
stats = await manager.get_layer_statistics()
print(f"Всего записей: {stats['memory_context']['total_memory_entries']}")
print(f"Использование кэша: {stats['cache']['utilization']:.2%}")
```

#### optimize_memory()

Оптимизирует память и кэш.

**Возвращает:** Словарь с результатами оптимизации

**Пример:**
```python
results = await manager.optimize_memory()
print(f"Удалено записей: {results['total_items_removed']}")
print(f"Время оптимизации: {results['duration']:.2f}s")
```

#### sync_with_orchestrator()

Синхронизирует состояние с оркестратором.

**Возвращает:** Словарь с результатами синхронизации

**Пример:**
```python
sync_result = await manager.sync_with_orchestrator()
if sync_result['success']:
    print(f"Синхронизировано, trace_id: {sync_result['trace_id']}")
```

## AdaptiveBlueprintTracker

Трекер изменений архитектуры для отслеживания версионирования и анализа влияния изменений.

### Основные методы

#### record_blueprint(blueprint, metadata=None, change_type="update")

Записывает новое состояние архитектуры.

**Пример:**
```python
blueprint = {
    "version": "2.0",
    "components": {
        "core": {"status": "enhanced"},
        "ai_processor": {"status": "active"}
    }
}

version = await manager.blueprint_tracker.record_blueprint(
    blueprint=blueprint,
    metadata={"author": "developer"},
    change_type="enhancement",
    change_description="Добавлен AI процессор"
)
```

#### compare_blueprints(version1, version2, detailed=True)

Сравнивает две версии архитектуры.

**Пример:**
```python
comparison = await manager.blueprint_tracker.compare_blueprints(1, 2)
print(f"Добавлено: {len(comparison['detailed_changes']['added'])}")
print(f"Изменено: {len(comparison['detailed_changes']['modified'])}")
```

#### link_resource(identifier, resource, resource_type="unknown")

Связывает ресурс с архитектурой.

**Пример:**
```python
await manager.blueprint_tracker.link_resource(
    identifier="api_service",
    resource={"endpoint": "/api/v1", "version": "1.0"},
    resource_type="microservice",
    dependency_level=3
)
```

#### analyze_impact(from_version, to_version)

Анализирует влияние изменений между версиями.

**Пример:**
```python
impact = await manager.blueprint_tracker.analyze_impact(1, 2)
print(f"Оценка влияния: {impact.impact_score:.2f}")
print(f"Уровень риска: {impact.risk_assessment}")
print(f"Рекомендации: {impact.recommendations}")
```

## Конфигурация

### Параметры MemoryManager

```python
config = {
    # Кэш
    "cache_size": 1000,              # Размер кэша
    "cache_ttl": 3600,               # TTL кэша в секундах
    
    # Оптимизация
    "optimization_interval": 300,    # Интервал оптимизации в секундах
    
    # Векторное хранилище
    "vector_store": {
        "provider": "qdrant",        # qdrant, chroma, weaviate
        "base_url": "http://localhost:6333",
        "api_key": None,
        "vector_size": 384,
        "distance_metric": "cosine", # cosine, l2, dot
        "embedding_provider": "local", # local, openai, ollama
        "embedding_model": "all-MiniLM-L6-v2",
        "max_retries": 3,
        "timeout": 30.0
    }
}
```

### Векторные хранилища

Система поддерживает несколько векторных хранилищ:

1. **Qdrant** - Рекомендуется для продакшена
2. **ChromaDB** - Хорошо для разработки
3. **Weaviate** - Для облачных решений
4. **In-Memory** - Fallback для простых случаев

### Настройка кэширования

```python
# Автоматическое управление TTL
# TTL зависит от слоя и приоритета записи

TTL_BASE_VALUES = {
    CORE: 7200,        # 2 часа
    EPISODIC: 86400,   # 24 часа  
    SEMANTIC: 604800,  # 7 дней
    PROCEDURAL: 2592000, # 30 дней
    VAULT: 31536000,   # 1 год
    SECURITY: 7776000  # 90 дней
}

# Модификаторы приоритета:
# priority >= 8: TTL * 2
# priority <= 2: TTL * 0.5
```

## Мониторинг и статистика

### Получение статистики

```python
stats = await manager.get_layer_statistics()

# Статистика по слоям
for layer, layer_stats in stats['layer_statistics'].items():
    print(f"{layer}:")
    print(f"  Всего элементов: {layer_stats['total_items']}")
    print(f"  Cache hit ratio: {layer_stats['cache_hit_ratio']:.2%}")
    print(f"  Среднее время доступа: {layer_stats['average_access_time']:.4f}s")

# Информация о кэше
cache_stats = stats['cache']
print(f"Использование кэша: {cache_stats['utilization']:.2%}")

# Информация о векторном хранилище
vector_stats = stats['vector_store']
print(f"Провайдер: {vector_stats['current_provider']}")
```

### Логирование

Система предоставляет детальное логирование:

```python
# Уровни логирования:
# - INFO: Основные операции
# - DEBUG: Подробная информация о работе
# - WARNING: Предупреждения
# - ERROR: Ошибки

import logging

# Настройка уровня логирования
logging.basicConfig(level=logging.DEBUG)

# Логи сохраняются с указанием:
# - Времени операции
# - ID записей
# - Слоев памяти
# - Размера данных
# - Результатов операций
```

## Обработка ошибок

Система обеспечивает robust обработку ошибок:

```python
try:
    # Сохранение данных
    item_id = await manager.store(CORE, "data", metadata={})
    
    # Извлечение данных
    results = await manager.retrieve(CORE, query="search")
    
    # Обновление данных
    success = await manager.update(CORE, item_id, "new_data")
    
    # Удаление данных
    deleted = await manager.delete(CORE, item_id)
    
except ValueError as e:
    # Недопустимые параметры
    print(f"Ошибка параметров: {e}")
    
except RuntimeError as e:
    # Ошибки выполнения операций
    print(f"Ошибка выполнения: {e}")
    
except Exception as e:
    # Неожиданные ошибки
    print(f"Неожиданная ошибка: {e}")
```

## Производительность

### Рекомендации по оптимизации

1. **Настройка кэша:**
   - Размер кэша: 1000-5000 записей для средних нагрузок
   - TTL: Соответствует частоте изменения данных

2. **Векторное хранилище:**
   - Qdrant для высоких нагрузок
   - ChromaDB для разработки и тестирования
   - Weaviate для облачных развертываний

3. **Оптимизация:**
   - Автоматическая оптимизация каждые 5-30 минут
   - Ручная оптимизация перед большими нагрузками

4. **Мониторинг:**
   - Регулярная проверка статистики
   - Отслеживание cache hit ratio
   - Мониторинг времени ответа

### Бенчмарки

Типичные показатели производительности:

- **Сохранение записи:** 10-50ms
- **Извлечение записи:** 5-25ms  
- **Поиск по слоям:** 50-200ms
- **Оптимизация:** 100-1000ms (зависит от объема данных)
- **Синхронизация:** 200-500ms

## Примеры использования

### Полный пример

```python
import asyncio
from memory_manager import create_memory_manager, CORE, SEMANTIC, EPISODIC

async def main():
    # Создание менеджера
    manager = create_memory_manager({
        "cache_size": 1000,
        "cache_ttl": 3600,
        "optimization_interval": 300
    })
    
    try:
        # Запуск
        await manager.start()
        
        # Сохранение данных
        core_id = await manager.store(
            layer=CORE,
            data="Система инициализирована",
            metadata={"init_time": "2025-10-28T03:55:14"},
            tags=["system", "init"],
            priority=8
        )
        
        # Поиск и извлечение
        results = await manager.search_across_layers(
            query="система",
            layers=[CORE, EPISODIC],
            limit=10
        )
        
        # Получение статистики
        stats = await manager.get_layer_statistics()
        print(f"Всего записей: {stats['memory_context']['total_memory_entries']}")
        
        # Оптимизация
        await manager.optimize_memory()
        
        # Синхронизация
        sync_result = await manager.sync_with_orchestrator()
        print(f"Синхронизация: {sync_result['success']}")
        
    finally:
        await manager.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### Работа с Blueprint Tracker

```python
async def manage_architecture():
    manager = create_memory_manager()
    await manager.start()
    
    try:
        # Запись архитектуры
        blueprint = {
            "version": "1.0",
            "components": {
                "memory": {"status": "active"},
                "vector_store": {"status": "active"}
            }
        }
        
        version = await manager.blueprint_tracker.record_blueprint(
            blueprint=blueprint,
            change_type="initial",
            change_description="Начальная архитектура"
        )
        
        # Связывание ресурса
        await manager.blueprint_tracker.link_resource(
            identifier="main_db",
            resource={"type": "postgresql", "version": "14"},
            resource_type="database",
            dependency_level=2
        )
        
        # Анализ изменений (при обновлении)
        impact = await manager.blueprint_tracker.analyze_impact(1, 2)
        print(f"Влияние изменений: {impact.impact_score:.2f}")
        
    finally:
        await manager.stop()
```

## Расширение системы

### Добавление нового слоя памяти

1. Создайте новый класс слоя памяти
2. Зарегистрируйте его в MemoryContext
3. Добавьте константу для слоя
4. Обновите конфигурацию TTL

```python
# 1. Новый слой памяти
class CustomMemory:
    def __init__(self):
        self.data = {}
    
    async def store_custom(self, key, value):
        self.data[key] = value
    
    async def get_custom(self, key):
        return self.data.get(key)

# 2. Регистрация в MemoryContext
CUSTOM = "CUSTOM"
MEMORY_LAYERS[CUSTOM] = "CUSTOM"

# 3. Добавление в фабрику
class MemoryLayerFactory:
    @staticmethod
    def create_layer(layer_type: str):
        if layer_type == CUSTOM:
            return CustomMemory()
```

### Интеграция с внешними системами

```python
class CustomSyncHandler:
    async def sync_with_external_system(self, envelope):
        # Интеграция с внешними системами
        pass
    
    async def receive_external_updates(self):
        # Получение обновлений извне
        pass

# Добавление в MemoryManager
custom_handler = CustomSyncHandler()
```

## Заключение

MemoryManager предоставляет комплексное решение для управления многослойной памятью с возможностями:

- ✅ 6 слоев памяти для разных типов данных
- ✅ Высокопроизводительное кэширование
- ✅ Интеграция с векторными хранилищами  
- ✅ Отслеживание изменений архитектуры
- ✅ Автоматическая оптимизация
- ✅ Comprehensive логирование
- ✅ Robust обработка ошибок
- ✅ Синхронизация с оркестратором

Система готова к продакшн использованию и может быть легко расширена для специфических потребностей проекта.
