# VectorStoreClient - Документация

VectorStoreClient предоставляет единый интерфейс для работы с векторными хранилищами в Rebecca Platform.

## Поддерживаемые провайдеры

- **Qdrant** - высокопроизводительное векторное хранилище
- **ChromaDB** - локальное векторное хранилище
- **Weaviate** - облачное векторное хранилище
- **Memory** - in-memory хранилище (fallback)

## Установка зависимостей

```bash
pip install qdrant-client chromadb weaviate-client numpy
```

## Быстрый старт

### Базовое использование

```python
import asyncio
from vector_store_client import VectorStoreClient, VectorStoreConfig

async def main():
    # Создаем конфигурацию
    config = VectorStoreConfig(
        provider="qdrant",
        base_url="http://localhost:6333",
        vector_size=384
    )
    
    # Создаем клиент
    client = VectorStoreClient(config)
    
    # Сохраняем векторы
    items = [
        {
            "text": "Это пример текста для векторизации",
            "metadata": {"category": "example"}
        }
    ]
    await client.store_vectors("semantic", items)
    
    # Поиск векторов
    results = await client.retrieve_vectors(
        "semantic", 
        {"text": "пример поиска", "limit": 10}
    )
    
    await client.close()

asyncio.run(main())
```

### Создание embeddings

```python
# Создание embeddings для списка текстов
texts = ["Текст 1", "Текст 2", "Текст 3"]
embeddings = await client.create_embeddings(texts)

# Создание embedding для одного текста
vector = await client.vectorize_text("Один текст для векторизации")
```

### Работа с множественными слоями

```python
# Семантическая память
semantic_items = [
    {"text": "Концепция ИИ", "metadata": {"type": "concept"}}
]
await client.store_vectors("semantic", semantic_items)

# Эпизодическая память
episodic_items = [
    {"text": "Вчера изучал новый алгоритм", "metadata": {"date": "2025-10-27"}}
]
await client.store_vectors("episodic", episodic_items)
```

### Обновление векторов

```python
# Сначала нужно найти вектор для обновления
results = await client.retrieve_vectors("semantic", {"text": "старый текст", "limit": 1})
if results:
    vector_id = results[0]['id']
    
    # Обновляем
    await client.update_vector(
        "semantic",
        vector_id,
        {
            "text": "Новый текст",
            "metadata": {"updated": True}
        }
    )
```

## Конфигурация

### VectorStoreConfig

```python
config = VectorStoreConfig(
    # Основные настройки
    provider="qdrant",                    # Провайдер хранилища
    base_url="http://localhost:6333",     # URL хранилища
    api_key=None,                         # API ключ
    collection_name="rebecca_vectors",    # Название коллекции
    
    # Настройки векторов
    vector_size=384,                      # Размерность векторов
    distance_metric="cosine",             # Метрика расстояния
    
    # Embeddings настройки
    embedding_provider="local",           # Провайдер embeddings
    embedding_model="all-MiniLM-L6-v2",  # Модель embeddings
    embedding_api_key=None,               # API ключ для embeddings
    
    # Retry настройки
    max_retries=3,                        # Максимум попыток
    retry_delay=1.0,                      # Задержка между попытками
    timeout=30.0,                         # Таймаут запросов
    
    # Fallback настройки
    fallback_enabled=True,                # Включить fallback
    fallback_providers=["chroma", "memory"]  # Список fallback провайдеров
)
```

## Слои памяти

VectorStoreClient поддерживает работу с различными слоями памяти Rebecca:

- **semantic** - семантическая память (концепты, знания)
- **episodic** - эпизодическая память (события, опыт)
- **procedural** - процедурная память (алгоритмы, процедуры)
- **core** - основная память (базовые данные)

## Embeddings провайдеры

### Локальные embeddings
Используют встроенные алгоритмы для создания embeddings:
```python
config = VectorStoreConfig(embedding_provider="local")
```

### OpenAI embeddings
```python
config = VectorStoreConfig(
    embedding_provider="openai",
    embedding_api_key="your-api-key",
    embedding_model="text-embedding-ada-002"
)
```

### Ollama embeddings
```python
config = VectorStoreConfig(
    embedding_provider="ollama",
    embedding_base_url="http://localhost:11434",
    embedding_model="nomic-embed-text"
)
```

## Мониторинг и диагностика

### Информация о хранилище
```python
info = client.get_store_info()
print(f"Провайдер: {info['provider']}")
print(f"Размер векторов: {info['vector_size']}")
```

### Проверка состояния
```python
health = await client.health_check()
for provider, status in health['stores'].items():
    print(f"{provider}: {status}")
```

## Обработка ошибок

VectorStoreClient включает встроенную систему retry для обработки временных сбоев:

```python
try:
    await client.store_vectors("layer", items)
except Exception as e:
    print(f"Ошибка сохранения: {e}")
    # Клиент автоматически делает повторные попытки
```

## Fallback механизм

Если основной провайдер недоступен, клиент автоматически переключается на fallback провайдеры:

```python
# Если Qdrant недоступен, будет использован ChromaDB или Memory
config = VectorStoreConfig(
    provider="qdrant",
    fallback_enabled=True,
    fallback_providers=["chroma", "memory"]
)
```

## Интеграция с существующей архитектурой

VectorStoreClient интегрируется с существующими компонентами Rebecca:

```python
from memory_manager import MemoryManager
from vector_store_client import VectorStoreClient

class EnhancedMemoryManager(MemoryManager):
    def __init__(self):
        super().__init__()
        self.vector_client = VectorStoreClient()
        
    async def store_semantic_with_vectors(self, concept, description):
        # Сохраняем в семантическую память
        self.semantic.store_concept(concept, description)
        
        # Сохраняем векторы
        await self.vector_client.store_vectors("semantic", [
            {"text": description, "metadata": {"concept": concept}}
        ])
```

## Производительность

- **Batch операции**: Используйте `create_embeddings` для множественных текстов
- **Connection pooling**: Клиенты переиспользуют соединения
- **Async/await**: Все операции асинхронные для лучшей производительности
- **Retry логика**: Автоматические повторные попытки при сбоях

## Примеры

Запустите файл `vector_store_example.py` для демонстрации возможностей:

```bash
cd src/memory_manager
python vector_store_example.py
```

Этот файл содержит следующие примеры:
- Базовое использование
- Работа с множественными слоями
- Создание embeddings
- Обновление векторов
- Fallback механизм

## Троублшутинг

### Проблемы с подключением
1. Проверьте URL и доступность хранилища
2. Убедитесь, что хранилище запущено
3. Проверьте API ключи

### Проблемы с embeddings
1. Убедитесь, что указан корректный провайдер
2. Проверьте настройки модели
3. Проверьте API ключи для внешних сервисов

### Медленная работа
1. Уменьшите размерность векторов
2. Используйте batch операции
3. Настройте индексы в векторном хранилище

## Поддержка

Для получения поддержки создайте issue в репозитории Rebecca Platform.
