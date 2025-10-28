# Knowledge Graph модуль для Rebecca-Platform

Этот модуль предоставляет комплексную систему управления знаниями с интеграцией KAG (Knowledge-Augmented Generation) системы и 6 слоев памяти Rebecca-Platform.

## 🎯 Основные возможности

- **Интеграция с 6 слоями памяти**: Core, Episodic, Semantic, Procedural, Vault, Security
- **Bidirectional Synchronization**: Двунаправленная синхронизация между графом знаний и памятью
- **Knowledge Validation**: Автоматическая валидация знаний с настраиваемыми правилами
- **Access Control**: Многоуровневый контроль доступа к знаниям
- **Semantic Search**: Семантический поиск по графу знаний
- **Performance Optimization**: Кэширование и оптимизация производительности

## 📦 Установка и настройка

### Требования

```python
# Основные зависимости уже установлены в Rebecca-Platform
import asyncio
import networkx as nx
import numpy as np
from src.memory_manager.memory_manager import MemoryManager
```

### Быстрый старт

```python
import asyncio
from src.memory_manager.memory_manager import create_memory_manager
from src.knowledge_graph.memory_integration import create_kag_integration, NodeType, AccessLevel

async def main():
    # 1. Создаем MemoryManager
    memory_manager = create_memory_manager()
    await memory_manager.start()
    
    # 2. Создаем KAG интеграцию
    kag_integration = await create_kag_integration(memory_manager)
    await kag_integration.start()
    
    # 3. Добавляем знания
    concept_id = await kag_integration.add_knowledge(
        content="Rebecca-Platform - интеллектуальная система агентов",
        node_type=NodeType.CONCEPT,
        tags=["platform", "agents"],
        access_level=AccessLevel.INTERNAL
    )
    
    # 4. Выполняем запрос
    results = await kag_integration.query_knowledge("платформа")
    
    # 5. Получаем статус системы
    status = await kag_integration.get_system_status()
    
    print(f"Создан концепт: {concept_id}")
    print(f"Найдено результатов: {len(results)}")
    
    # 6. Останавливаем систему
    await kag_integration.stop()
    await memory_manager.stop()

# Запуск
asyncio.run(main())
```

## 🏗️ Архитектура

### 6 слоев памяти

| Слой | Назначение | TTL | Тип узлов | Уровень доступа |
|------|------------|-----|-----------|-----------------|
| **Core** | Системные концепты | 2 часа | CONCEPT | INTERNAL |
| **Episodic** | События и временные связи | 24 часа | EVENT | INTERNAL |
| **Semantic** | Концептуальные знания | 7 дней | CONCEPT, RELATION | INTERNAL |
| **Procedural** | Процессы и алгоритмы | 30 дней | PROCEDURE | INTERNAL |
| **Vault** | Секретные знания | 1 год | VAULT_ITEM | SECRET+ |
| **Security** | Правила безопасности | 90 дней | SECURITY_RULE | CONFIDENTIAL+ |

### Основные компоненты

```python
# KAGMemoryIntegration - главный класс интеграции
kag_integration = await create_kag_integration(memory_manager)

# KAGGraphManager - управление графом знаний
graph_manager = kag_integration.graph_manager

# MemoryLayerIntegration - интеграция с памятью
memory_integration = kag_integration.memory_integration

# KnowledgeValidator - валидатор знаний
validator = kag_integration.validator

# AccessControl - контроль доступа
access_control = kag_integration.access_control
```

## 📚 Примеры использования

### 1. Добавление знаний

```python
# Концепт (Core/Semantic слой)
concept_id = await kag_integration.add_knowledge(
    content="Искусственный интеллект - технология имитации человеческого интеллекта",
    node_type=NodeType.CONCEPT,
    metadata={
        "domain": "technology",
        "parent_concepts": ["machine_learning"],
        "confidence": 0.9
    },
    tags=["AI", "intelligence", "technology"],
    access_level=AccessLevel.INTERNAL
)

# Событие (Episodic слой)
event_id = await kag_integration.add_knowledge(
    content="Запуск новой KAG системы в production",
    node_type=NodeType.EVENT,
    metadata={
        "timestamp": "2025-10-28T06:56:14",
        "event_type": "system_launch",
        "outcome": "success"
    },
    tags=["launch", "production", "kag"],
    access_level=AccessLevel.INTERNAL
)

# Процедура (Procedural слой)
procedure_id = await kag_integration.add_knowledge(
    content="""
    Алгоритм добавления знания:
    1. Валидация входных данных
    2. Классификация уровня доступа
    3. Создание KAG узла
    4. Добавление в граф
    5. Синхронизация с памятью
    """,
    node_type=NodeType.PROCEDURE,
    metadata={
        "algorithm": "add_knowledge",
        "steps": 5,
        "complexity": "O(1)"
    },
    tags=["algorithm", "knowledge", "procedure"],
    access_level=AccessLevel.INTERNAL
)

# Секретное знание (Vault слой)
vault_id = await kag_integration.add_knowledge(
    content="API ключ для интеграции: sk_live_abc123...",
    node_type=NodeType.VAULT_ITEM,
    metadata={
        "classification": "secret",
        "service": "api_integration",
        "sensitivity": "high"
    },
    tags=["api", "key", "secret"],
    access_level=AccessLevel.SECRET
)

# Правило безопасности (Security слой)
security_id = await kag_integration.add_knowledge(
    content="Все секретные данные должны быть зашифрованы",
    node_type=NodeType.SECURITY_RULE,
    metadata={
        "rule_type": "encryption_policy",
        "enforcement": "mandatory",
        "compliance": ["GDPR", "SOX"]
    },
    tags=["security", "encryption", "compliance"],
    access_level=AccessLevel.CONFIDENTIAL
)
```

### 2. Семантический поиск

```python
# Поиск по ключевому слову
results = await kag_integration.query_knowledge(
    query="искусственный интеллект",
    max_results=10
)

# Поиск с фильтрацией по типам
ai_results = await kag_integration.query_knowledge(
    query="AI",
    node_types=[NodeType.CONCEPT, NodeType.PROCEDURE],
    max_results=5
)

# Анализ результатов
for result in results:
    print(f"Найден: {result['content'][:100]}...")
    print(f"Тип: {result['node_type']}")
    print(f"Уверенность: {result['confidence']:.2f}")
    print(f"Релевантность: {result['relevance']:.2f}")
    print("-" * 50)
```

### 3. Синхронизация данных

```python
# Полная синхронизация всех слоев
sync_results = await kag_integration.sync_all_layers()

print(f"Синхронизировано слоев: {sync_results['successful_layers']}/{sync_results['total_layers']}")
print(f"Элементов обработано: {sync_results['total_synced_items']}")
print(f"Время выполнения: {sync_results['duration']:.2f}s")

# Детали по слоям
for layer, result in sync_results['layer_results'].items():
    if result['success']:
        print(f"✅ {layer}: {result['synced_items']} элементов")
    else:
        print(f"❌ {layer}: {result['error']}")
```

### 4. Валидация знаний

```python
# Валидация отдельного узла
validation_result = await kag_integration.validate_knowledge(concept_id)

if validation_result["valid"]:
    print(f"✅ Валидация пройдена (confidence: {validation_result['confidence']:.2f})")
else:
    print(f"❌ Валидация не пройдена: {validation_result.get('error', 'Неизвестная ошибка')}")

# Массовая валидация
all_nodes = await graph_manager.find_nodes_by_type(NodeType.CONCEPT)
validation_tasks = []

for node in all_nodes:
    result = await kag_integration.validate_knowledge(node.id)
    validation_tasks.append({
        "node_id": node.id,
        "valid": result["valid"],
        "confidence": result["confidence"]
    })

# Статистика валидации
valid_count = sum(1 for task in validation_tasks if task["valid"])
total_count = len(validation_tasks)
success_rate = valid_count / total_count

print(f"Статистика валидации: {valid_count}/{total_count} ({success_rate:.1%})")
```

### 5. Контроль доступа

```python
# Настройка разрешений пользователя
kag_integration.access_control.add_user_permission(
    user_id="user_123",
    access_level=AccessLevel.CONFIDENTIAL
)

# Проверка доступа к узлу
node = await graph_manager.get_node(concept_id)
has_access = kag_integration.access_control.check_access("user_123", node)

if has_access:
    print(f"✅ Доступ разрешен к узлу {concept_id}")
else:
    print(f"❌ Доступ запрещен к узлу {concept_id}")

# Автоматическая классификация контента
test_content = "Секретный пароль администратора"
classification = kag_integration.access_control.classify_content(
    test_content, 
    {}
)
print(f"Классификация: {classification.value}")
```

### 6. Мониторинг системы

```python
# Получение полного статуса системы
status = await kag_integration.get_system_status()

print("📊 Статистика системы:")
print(f"Версия KAG: {status['kag_version']}")
print(f"Система запущена: {status['running']}")

# Статистика графа
graph_stats = status['graph_statistics']
print(f"Узлов в графе: {graph_stats['total_nodes']}")
print(f"Связей в графе: {graph_stats['total_edges']}")

# Статистика памяти
memory_stats = status['memory_statistics']
print(f"Элементов в памяти: {memory_stats['memory_context']['total_items']}")
print(f"Использование кэша: {memory_stats['cache']['utilization']:.1%}")

# Статистика синхронизации
sync_stats = status['sync_statistics']
print(f"Операций синхронизации: {sync_stats['total_sync_operations']}")
print(f"Успешных синхронизаций: {sync_stats['successful_syncs']}")
```

## 🧪 Тестирование

### Быстрый тест

```python
# Запуск встроенного теста
result = await quick_kag_test()
print(f"Тест пройден: {result['success']}")
```

### Комплексная демонстрация

```bash
cd /workspace/Rebecca-Platform/src/knowledge_graph
python kag_demo.py
```

### Настройка тестов

```python
# Создание тестовых данных
async def create_test_data(kag_integration):
    test_data = []
    
    for i in range(50):
        concept_id = await kag_integration.add_knowledge(
            content=f"Тестовый концепт {i}",
            node_type=NodeType.CONCEPT,
            tags=[f"test_{i % 5}"]
        )
        test_data.append(concept_id)
    
    return test_data

# Запуск нагрузочного теста
async def performance_test():
    memory_manager = create_memory_manager()
    await memory_manager.start()
    
    kag_integration = await create_kag_integration(memory_manager)
    await kag_integration.start()
    
    # Создание тестовых данных
    test_ids = await create_test_data(kag_integration)
    
    # Тест производительности
    import time
    
    # Добавление знаний
    start_time = time.time()
    for i in range(100):
        await kag_integration.add_knowledge(
            content=f"Значение для теста {i}",
            node_type=NodeType.CONCEPT
        )
    add_time = time.time() - start_time
    
    # Поиск
    start_time = time.time()
    for i in range(50):
        await kag_integration.query_knowledge("тест", max_results=10)
    search_time = time.time() - start_time
    
    print(f"Добавление 100 знаний: {add_time:.3f}s")
    print(f"50 поисковых запросов: {search_time:.3f}s")
    
    await kag_integration.stop()
    await memory_manager.stop()
```

## ⚙️ Конфигурация

### Настройка через YAML

```yaml
# config/kag_integration.yaml
kag_system:
  version: "1.0.0"
  
  performance:
    max_graph_nodes: 10000
    cache_size: 1000
    sync_interval: 30
    validation_threshold: 0.7
  
  layers:
    core:
      ttl: 7200
      priority: 8
      max_items: 10000
    
    episodic:
      ttl: 86400
      priority: 6
      max_items: 50000
    
    semantic:
      ttl: 604800
      priority: 5
      max_items: 25000
    
    procedural:
      ttl: 2592000
      priority: 4
      max_items: 15000
    
    vault:
      ttl: 31536000
      priority: 9
      max_items: 5000
      encryption: true
    
    security:
      ttl: 7776000
      priority: 9
      max_items: 10000
      audit_enabled: true
  
  validation:
    confidence_thresholds:
      concept: 0.8
      event: 0.9
      procedure: 0.85
      vault_item: 0.9
      security_rule: 0.99
  
  access_control:
    auto_classification: true
    default_level: "internal"
    secret_keywords: ["пароль", "ключ", "секрет"]
    confidential_keywords: ["приватный", "конфиденциально"]
```

### Программная настройка

```python
# Создание с кастомной конфигурацией
memory_manager = MemoryManager(
    cache_size=2000,
    cache_ttl=7200,
    optimization_interval=600
)

kag_integration = KAGMemoryIntegration(memory_manager)

# Настройка правил валидации
kag_integration.validator.confidence_thresholds[NodeType.CONCEPT] = 0.85
kag_integration.validator.confidence_thresholds[NodeType.VAULT_ITEM] = 0.95

# Настройка контроля доступа
kag_integration.access_control.classification_rules["api_key"] = AccessLevel.SECRET
kag_integration.access_control.classification_rules["user_data"] = AccessLevel.CONFIDENTIAL
```

## 🚨 Отладка и логирование

### Настройка логирования

```python
import logging

# Включение подробного логирования
logging.basicConfig(level=logging.DEBUG)

# Логирование KAG системы
kag_logger = logging.getLogger('src.knowledge_graph.memory_integration')
kag_logger.setLevel(logging.INFO)

# Логирование MemoryManager
memory_logger = logging.getLogger('src.memory_manager.memory_manager')
memory_logger.setLevel(logging.INFO)
```

### Диагностика проблем

```python
# Проверка состояния системы
async def diagnose_system():
    status = await kag_integration.get_system_status()
    
    print("🔍 Диагностика системы:")
    
    # Проверка графовой структуры
    graph_stats = status['graph_statistics']
    if graph_stats['total_nodes'] == 0:
        print("⚠️ Граф пуст - проблемы с добавлением узлов")
    
    if graph_stats['total_edges'] == 0:
        print("⚠️ Нет связей в графе")
    
    # Проверка синхронизации
    sync_stats = status['sync_statistics']
    if sync_stats['failed_syncs'] > sync_stats['successful_syncs']:
        print("⚠️ Много неудачных синхронизаций")
    
    # Проверка памяти
    memory_stats = status['memory_statistics']
    cache_hit_ratio = memory_stats['cache']['utilization']
    if cache_hit_ratio < 0.5:
        print("⚠️ Низкая эффективность кэша")
    
    # Проверка валидации
    validation_success = graph_stats.get('validation_success_rate', 1.0)
    if validation_success < 0.8:
        print("⚠️ Низкий процент успешной валидации")
    
    print("✅ Диагностика завершена")

# Мониторинг в реальном времени
async def monitor_system():
    while True:
        status = await kag_integration.get_system_status()
        
        # Выводим ключевые метрики
        graph_nodes = status['graph_statistics']['total_nodes']
        sync_operations = status['sync_statistics']['total_sync_operations']
        
        print(f"📊 Узлы: {graph_nodes}, Синхронизации: {sync_operations}")
        
        await asyncio.sleep(10)  # Обновляем каждые 10 секунд
```

## 🔧 API Reference

### Основные классы

#### KAGMemoryIntegration
Главный класс интеграции KAG системы с памятью.

**Методы:**
- `add_knowledge()` - добавление знания
- `query_knowledge()` - поиск знаний
- `sync_all_layers()` - синхронизация слоев
- `validate_knowledge()` - валидация знания
- `get_system_status()` - статус системы

#### KAGGraphManager
Управление графом знаний.

**Методы:**
- `add_node()` - добавление узла
- `add_edge()` - добавление связи
- `get_node()` - получение узла
- `find_related_nodes()` - поиск связанных узлов
- `query_graph()` - запрос к графу

#### KnowledgeValidator
Валидатор знаний.

**Методы:**
- `validate_node()` - валидация узла
- `add_validation_rule()` - добавление правила

#### AccessControl
Контроль доступа.

**Методы:**
- `check_access()` - проверка доступа
- `classify_content()` - классификация контента
- `add_user_permission()` - добавление разрешения

### Типы узлов (NodeType)

- `CONCEPT` - концепты и идеи
- `ENTITY` - сущности и объекты
- `RELATION` - отношения и связи
- `EVENT` - события и эпизоды
- `PROCEDURE` - процессы и алгоритмы
- `RULE` - правила и законы
- `VAULT_ITEM` - секретные элементы
- `SECURITY_RULE` - правила безопасности

### Типы связей (EdgeType)

- `IS_A` - является (иерархия)
- `PART_OF` - часть чего-либо
- `RELATED_TO` - связано с
- `CAUSES` - причинно-следственная связь
- `ENABLES` - обеспечивает, позволяет
- `VALIDATES` - валидирует
- `CONFLICTS` - конфликтует
- `DEPENDS_ON` - зависит от

### Уровни доступа (AccessLevel)

- `PUBLIC` - публичный доступ
- `INTERNAL` - внутреннее использование
- `CONFIDENTIAL` - конфиденциальная информация
- `SECRET` - секретная информация
- `TOP_SECRET` - строго секретно

## 🤝 Вклад в разработку

1. Форкните репозиторий
2. Создайте ветку для новой функции (`git checkout -b feature/amazing-feature`)
3. Закоммитьте изменения (`git commit -m 'Add amazing feature'`)
4. Запушьте в ветку (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

### Стандарты кода

- Используйте типизацию Python (type hints)
- Добавляйте docstrings ко всем публичным методам
- Следуйте PEP 8
- Добавляйте тесты для новой функциональности
- Обновляйте документацию

## 📄 Лицензия

Этот проект лицензирован под MIT License - см. файл LICENSE для деталей.

## 📞 Поддержка

Для получения поддержки:

1. Проверьте документацию выше
2. Изучите примеры в `kag_demo.py`
3. Запустите тесты с `--quick` флагом
4. Откройте issue в репозитории

---

**Версия документации:** 1.0.0  
**Последнее обновление:** 28.10.2025  
**Совместимость:** Rebecca-Platform v1.0+