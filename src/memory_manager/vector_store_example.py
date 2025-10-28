"""Пример использования VectorStoreClient."""

import asyncio
from vector_store_client import (
    VectorStoreClient, 
    VectorStoreConfig, 
    create_vector_client_from_config
)


async def basic_example():
    """Базовый пример использования."""
    
    print("=== Базовый пример VectorStoreClient ===\n")
    
    # Создаем конфигурацию
    config = VectorStoreConfig(
        provider="memory",  # Используем memory для примера
        vector_size=384,
        collection_name="example_vectors"
    )
    
    # Создаем клиент
    client = VectorStoreClient(config)
    
    # Подготавливаем данные
    sample_items = [
        {
            "text": "Rebecca - это интеллектуальная платформа",
            "metadata": {"type": "description", "category": "platform"}
        },
        {
            "text": "Vector store позволяет хранить эмбеддинги",
            "metadata": {"type": "explanation", "category": "feature"}
        },
        {
            "text": "Поддерживаются Qdrant, ChromaDB и Weaviate",
            "metadata": {"type": "feature", "category": "integration"}
        }
    ]
    
    # Сохраняем векторы
    print("1. Сохранение векторов...")
    await client.store_vectors("semantic", sample_items)
    print("✓ Векторы сохранены\n")
    
    # Извлекаем векторы
    print("2. Поиск векторов...")
    results = await client.retrieve_vectors(
        "semantic", 
        {"text": "платформа", "limit": 5}
    )
    
    for result in results:
        print(f"   ID: {result['id']}")
        print(f"   Текст: {result['text']}")
        print(f"   Метаданные: {result['metadata']}")
        print(f"   Вектор размер: {len(result['vector'])}")
        print()
    
    # Получаем информацию о хранилище
    print("3. Информация о хранилище:")
    info = client.get_store_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    print("\n4. Проверка состояния:")
    health = await client.health_check()
    for key, value in health.items():
        print(f"   {key}: {value}")
    
    await client.close()


async def multi_layer_example():
    """Пример работы с множественными слоями."""
    
    print("\n=== Пример работы с множественными слоями ===\n")
    
    config = VectorStoreConfig(provider="memory")
    client = VectorStoreClient(config)
    
    # Семантическая память
    semantic_items = [
        {"text": "Концепция машинного обучения", "metadata": {"topic": "AI"}},
        {"text": "Нейронные сети и глубокое обучение", "metadata": {"topic": "AI"}}
    ]
    
    # Эпизодическая память
    episodic_items = [
        {"text": "Вчера изучал новый алгоритм", "metadata": {"date": "2025-10-27"}},
        {"text": "Сегодня работал над проектом", "metadata": {"date": "2025-10-28"}}
    ]
    
    # Сохраняем в разные слои
    print("1. Сохранение в семантический слой...")
    await client.store_vectors("semantic", semantic_items)
    
    print("2. Сохранение в эпизодический слой...")
    await client.store_vectors("episodic", episodic_items)
    
    # Поиск в семантическом слое
    print("\n3. Поиск в семантическом слое...")
    semantic_results = await client.retrieve_vectors(
        "semantic",
        {"text": "обучение", "limit": 5}
    )
    
    for result in semantic_results:
        print(f"   Семантический: {result['text']}")
    
    # Поиск в эпизодическом слое
    print("\n4. Поиск в эпизодическом слое...")
    episodic_results = await client.retrieve_vectors(
        "episodic",
        {"text": "работал", "limit": 5}
    )
    
    for result in episodic_results:
        print(f"   Эпизодический: {result['text']}")
    
    await client.close()


async def embedding_example():
    """Пример создания embeddings."""
    
    print("\n=== Пример создания embeddings ===\n")
    
    config = VectorStoreConfig(vector_size=128)  # Меньший размер для примера
    client = VectorStoreClient(config)
    
    texts = [
        "Привет, мир!",
        "Это пример векторизации текста",
        "Vector embeddings позволяют найти похожие тексты"
    ]
    
    print("1. Создание embeddings для списка текстов...")
    embeddings = await client.create_embeddings(texts)
    
    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        print(f"\nТекст {i+1}: {text}")
        print(f"Вектор: {embedding[:5]}... (первые 5 элементов)")
        print(f"Размер вектора: {len(embedding)}")
    
    # Векторизация одного текста
    print("\n2. Векторизация одного текста...")
    single_vector = await client.vectorize_text("Отдельный текст для векторизации")
    print(f"Вектор: {single_vector[:5]}... (первые 5 элементов)")
    
    await client.close()


async def update_example():
    """Пример обновления векторов."""
    
    print("\n=== Пример обновления векторов ===\n")
    
    config = VectorStoreConfig(provider="memory")
    client = VectorStoreClient(config)
    
    # Сохраняем исходные данные
    original_item = {
        "text": "Оригинальный текст",
        "metadata": {"version": 1}
    }
    
    print("1. Сохранение исходного элемента...")
    await client.store_vectors("test", [original_item])
    
    # Извлекаем для получения ID
    results = await client.retrieve_vectors("test", {"text": "Оригинальный", "limit": 1})
    
    if results:
        vector_id = results[0]['id']
        print(f"2. Обновление элемента с ID: {vector_id}")
        
        # Обновляем
        await client.update_vector(
            "test",
            vector_id,
            {
                "text": "Обновленный текст",
                "metadata": {"version": 2, "updated": True}
            }
        )
        
        print("3. Проверяем обновление...")
        updated_results = await client.retrieve_vectors("test", {"text": "Обновленный", "limit": 1})
        
        if updated_results:
            updated = updated_results[0]
            print(f"   Обновленный текст: {updated['text']}")
            print(f"   Обновленные метаданные: {updated['metadata']}")
    
    await client.close()


async def fallback_example():
    """Пример работы с fallback провайдерами."""
    
    print("\n=== Пример работы с fallback провайдерами ===\n")
    
    # Пробуем недоступный провайдер с fallback
    config = VectorStoreConfig(
        provider="nonexistent",  # Недоступный провайдер
        fallback_enabled=True,
        fallback_providers=["memory"]
    )
    
    client = VectorStoreClient(config)
    
    print(f"Основной провайдер: {config.provider}")
    print(f"Фактический провайдер: {client.config.provider}")
    print(f"Доступные провайдеры: {list(client.stores.keys())}")
    
    # Тестируем работу
    test_item = {"text": "Тестовый текст для fallback", "metadata": {"test": True}}
    await client.store_vectors("test", [test_item])
    
    results = await client.retrieve_vectors("test", {"text": "fallback", "limit": 1})
    print(f"Найдено результатов: {len(results)}")
    
    if results:
        print(f"Результат: {results[0]['text']}")
    
    await client.close()


async def main():
    """Запускает все примеры."""
    
    print("🚀 Примеры использования VectorStoreClient\n")
    
    try:
        await basic_example()
        await multi_layer_example()
        await embedding_example()
        await update_example()
        await fallback_example()
        
        print("\n✅ Все примеры выполнены успешно!")
        
    except Exception as e:
        print(f"\n❌ Ошибка выполнения: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Устанавливаем уровень логирования
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Запускаем примеры
    asyncio.run(main())
