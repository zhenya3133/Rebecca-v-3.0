# Memory Manager

Модуль управления памятью для агентной платформы Rebecca. Предоставляет единый интерфейс для работы с различными типами памяти и их оркестрации.

## Архитектура

```
memory_manager/
├── __init__.py                 # Экспорт основных компонентов
├── memory_context.py           # ⭐ Главный класс MemoryContext
├── memory_manager.py           # Основной менеджер памяти
├── memory_manager_main.py      # Фасад для MemoryManager
├── memory_manager_interface.py # Интерфейсы и абстракции
├── core_memory.py              # Основная системная память
├── episodic_memory.py          # Эпизодическая память
├── semantic_memory.py          # Семантическая память
├── procedural_memory.py        # Процедурная память
├── vault_memory.py             # Защищенная память (секреты)
├── security_memory.py          # Память безопасности
├── vector_store_client.py      # Клиент векторного хранилища
├── document_ingest.py          # Загрузка документов
├── adaptive_blueprint.py       # Адаптивные схемы памяти
├── logger.py                   # Логирование модуля
└── test_main.py                # Тесты модуля
```

## Основные компоненты

### MemoryContext (Главный компонент)

Единая точка входа для работы с памятью:

```python
from memory_manager import MemoryContext, CORE, EPISODIC, SEMANTIC

# Создание контекста
context = MemoryContext()

# Регистрация слоев происходит автоматически
core = context.get_layer(CORE)
episodic = context.get_layer(EPISODIC)
semantic = context.get_layer(SEMANTIC)

# Добавление записей
memory_id = context.add_memory(
    layer_type=SEMANTIC,
    content="Концепт ИИ",
    tags=["ai", "concept"],
    priority=8
)

# Поиск
results = context.search_memories("ИИ")

# Создание контекстной оболочки
envelope = context.build_context_envelope("session_123")
```

### Типы слоев памяти

| Тип | Назначение | Примеры данных |
|-----|------------|----------------|
| **CORE** | Основная системная информация | Константы, конфигурация |
| **EPISODIC** | События и временные данные | Логи, взаимодействия |
| **SEMANTIC** | Концепты и знания | Определения, факты |
| **PROCEDURAL** | Алгоритмы и процессы | Workflows, инструкции |
| **VAULT** | Секреты и конфиденциальные данные | Пароли, ключи |
| **SECURITY** | Правила и аудит безопасности | Политики, события |

### Основные методы MemoryContext

#### Управление слоями
- `register_layer(name, adapter)` - регистрация адаптера слоя
- `get_layer(name)` - получение адаптера слоя
- `clear_layer(layer_type)` - очистка слоя

#### Работа с записями
- `add_memory(layer_type, content, ...)` - добавление записи
- `get_memory(memory_id)` - получение записи по ID
- `remove_memory(memory_id)` - удаление записи
- `get_memories_by_layer(layer_type)` - получение записей слоя

#### Поиск и фильтрация
- `search_memories(query, ...)` - поиск по содержимому
- Фильтрация по слоям, тегам, приоритету

#### Контекстные оболочки
- `build_context_envelope(trace_id)` - создание оболочки
- `hydrate_from_envelope(envelope)` - восстановление состояния

#### Мониторинг и оптимизация
- `get_memory_statistics()` - статистика использования
- `optimize_memory()` - автоматическая оптимизация
- `get_trace_history(trace_id)` - история trace

## Использование

### Базовый пример

```python
from memory_manager import MemoryContext, CORE, SEMANTIC

# Создаем контекст памяти
context = MemoryContext()

# Сохраняем факт в CORE памяти
fact_id = context.add_memory(
    CORE,
    "REBECCA_VERSION = '2.0'",
    tags=["version", "system"],
    priority=10
)

# Сохраняем знание в SEMANTIC памяти
knowledge_id = context.add_memory(
    SEMANTIC,
    "Multi-agent systems enable distributed problem solving",
    metadata={"source": "research", "confidence": 0.95},
    tags=["agents", "theory"],
    priority=7
)

# Ищем информацию
results = context.search_memories("agent")
for entry in results:
    print(f"Найдено: {entry.content}")
```

### Создание контекстной оболочки

```python
# Для сохранения состояния между сессиями
envelope = context.build_context_envelope("user_session_123")

# Сериализация в JSON
import json
with open("context_backup.json", "w") as f:
    json.dump(envelope, f, indent=2)

# Восстановление состояния
with open("context_backup.json", "r") as f:
    saved_envelope = json.load(f)

new_context = MemoryContext()
new_context.hydrate_from_envelope(saved_envelope)
```

## Интеграция

### С Orchestrator

```python
from orchestrator.context_handler import ContextHandler
from memory_manager import MemoryContext

class Agent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.memory = MemoryContext()
        self.context_handler = ContextHandler()
    
    def get_context_for_trace(self, trace_id):
        # Получаем оболочку для текущего trace
        envelope = self.memory.build_context_envelope(trace_id)
        return self.context_handler.create_context(envelope)
```

### С другими компонентами

- **Logger**: Все операции логируются с разными уровнями
- **VectorStoreClient**: Интеграция для векторного поиска
- **DocumentIngestor**: Загрузка документов в память

## Логирование

Модуль использует Python logging с уровнями:
- **INFO**: Основные операции
- **DEBUG**: Детали восстановления состояния
- **WARNING**: Предупреждения
- **ERROR**: Ошибки

## Производительность

- O(1) доступ к слоям памяти
- O(n) поиск по содержимому
- Эффективная сериализация оболочек
- Автоматическая оптимизация

## Тестирование

```bash
# Запуск тестов модуля
cd /workspace/Rebecca-Platform
python -m pytest src/memory_manager/test_main.py -v
```

## Разработка

### Добавление нового слоя памяти

1. Создайте класс адаптера в отдельном файле
2. Реализуйте необходимые методы
3. Зарегистрируйте в MemoryContext:

```python
from memory_manager import MemoryContext

class CustomMemoryLayer:
    def store_custom_data(self, key, value):
        # Реализация
        pass

context = MemoryContext()
context.register_layer("CUSTOM", CustomMemoryLayer())
```

## Лицензия

Часть агентной платформы Rebecca-Platform.
