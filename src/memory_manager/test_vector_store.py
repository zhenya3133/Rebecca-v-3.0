"""Тесты для VectorStoreClient."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from vector_store_client import (
    VectorStoreClient,
    VectorStoreConfig,
    VectorItem,
    MemoryVectorStore,
    EmbeddingProvider,
    RetryHandler
)


class TestVectorStoreConfig:
    """Тесты конфигурации."""
    
    def test_default_config(self):
        """Тест конфигурации по умолчанию."""
        config = VectorStoreConfig()
        
        assert config.provider == "qdrant"
        assert config.vector_size == 384
        assert config.distance_metric == "cosine"
        assert config.embedding_provider == "local"
        assert config.max_retries == 3
        assert config.fallback_enabled is True
    
    def test_custom_config(self):
        """Тест кастомной конфигурации."""
        config = VectorStoreConfig(
            provider="chroma",
            vector_size=512,
            collection_name="test_collection"
        )
        
        assert config.provider == "chroma"
        assert config.vector_size == 512
        assert config.collection_name == "test_collection"


class TestVectorItem:
    """Тесты VectorItem."""
    
    def test_vector_item_creation(self):
        """Тест создания VectorItem."""
        item = VectorItem(
            id="test-1",
            text="Test text",
            metadata={"category": "test"},
            layer="semantic"
        )
        
        assert item.id == "test-1"
        assert item.text == "Test text"
        assert item.metadata["category"] == "test"
        assert item.layer == "semantic"
        assert item.timestamp is not None
    
    def test_vector_item_auto_id(self):
        """Тест автоматической генерации ID."""
        item = VectorItem(
            text="Test text",
            metadata={"category": "test"}
        )
        
        assert item.id is not None
        assert len(item.id) == 32  # MD5 hash length


class TestMemoryVectorStore:
    """Тесты in-memory хранилища."""
    
    @pytest.fixture
    def memory_store(self):
        """Создает MemoryVectorStore для тестов."""
        config = VectorStoreConfig()
        return MemoryVectorStore(config)
    
    @pytest.mark.asyncio
    async def test_store_vectors(self, memory_store):
        """Тест сохранения векторов."""
        items = [
            VectorItem(
                id="test-1",
                text="Test text 1",
                vector=[0.1, 0.2, 0.3],
                layer="semantic"
            ),
            VectorItem(
                id="test-2",
                text="Test text 2",
                vector=[0.4, 0.5, 0.6],
                layer="semantic"
            )
        ]
        
        await memory_store.store(items)
        
        assert len(memory_store.vectors) == 2
        assert "test-1" in memory_store.vectors
        assert "test-2" in memory_store.vectors
    
    @pytest.mark.asyncio
    async def test_retrieve_vectors(self, memory_store):
        """Тест извлечения векторов."""
        # Сохраняем тестовые данные
        items = [
            VectorItem(id="test-1", text="Apple fruit", layer="semantic"),
            VectorItem(id="test-2", text="Orange fruit", layer="semantic"),
            VectorItem(id="test-3", text="Car vehicle", layer="semantic")
        ]
        await memory_store.store(items)
        
        # Ищем фрукты
        results = await memory_store.retrieve(
            "semantic",
            {"text": "fruit", "limit": 10}
        )
        
        assert len(results) == 2
        texts = [item.text for item in results]
        assert "Apple fruit" in texts
        assert "Orange fruit" in texts
        assert "Car vehicle" not in texts
    
    @pytest.mark.asyncio
    async def test_update_vector(self, memory_store):
        """Тест обновления вектора."""
        # Сохраняем вектор
        item = VectorItem(id="test-1", text="Old text", layer="semantic")
        await memory_store.store([item])
        
        # Обновляем
        await memory_store.update("test-1", {"text": "New text"})
        
        # Проверяем обновление
        results = await memory_store.retrieve("semantic", {"text": "New", "limit": 1})
        assert len(results) == 1
        assert results[0].text == "New text"


class TestEmbeddingProvider:
    """Тесты провайдера embeddings."""
    
    @pytest.fixture
    def config(self):
        """Создает конфигурацию для тестов."""
        return VectorStoreConfig(vector_size=10)
    
    @pytest.fixture
    def provider(self, config):
        """Создает EmbeddingProvider для тестов."""
        return EmbeddingProvider(config)
    
    @pytest.mark.asyncio
    async def test_create_embedding_local(self, provider):
        """Тест создания локального embedding."""
        text = "Test text for embedding"
        vector = await provider.create_embedding(text)
        
        assert isinstance(vector, list)
        assert len(vector) == 10  # vector_size from config
        assert all(isinstance(x, float) for x in vector)
    
    @pytest.mark.asyncio
    async def test_create_embedding_empty_text(self, provider):
        """Тест создания embedding для пустого текста."""
        vector = await provider.create_embedding("")
        
        assert isinstance(vector, list)
        assert len(vector) == 10
        # Все элементы должны быть 0.0 для пустого текста
        assert all(x == 0.0 for x in vector)
    
    @pytest.mark.asyncio
    async def test_create_embeddings_batch(self, provider):
        """Тест создания embeddings для батча текстов."""
        texts = ["Text 1", "Text 2", "Text 3"]
        vectors = await provider.create_embeddings_batch(texts)
        
        assert isinstance(vectors, list)
        assert len(vectors) == 3
        assert all(len(v) == 10 for v in vectors)


class TestRetryHandler:
    """Тесты обработчика повторных попыток."""
    
    @pytest.mark.asyncio
    async def test_successful_execution(self):
        """Тест успешного выполнения."""
        call_count = 0
        
        async def test_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await RetryHandler.execute_with_retry(test_func, max_retries=3)
        
        assert result == "success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_failed_execution_with_retry(self):
        """Тест неудачного выполнения с повторными попытками."""
        call_count = 0
        
        async def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"
        
        result = await RetryHandler.execute_with_retry(test_func, max_retries=3, delay=0.01)
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_permanent_failure(self):
        """Тест постоянного сбоя."""
        call_count = 0
        
        async def test_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Permanent error")
        
        with pytest.raises(ValueError, match="Permanent error"):
            await RetryHandler.execute_with_retry(test_func, max_retries=2, delay=0.01)
        
        assert call_count == 3  # 1 initial + 2 retries


class TestVectorStoreClient:
    """Тесты основного клиента."""
    
    @pytest.fixture
    def config(self):
        """Создает конфигурацию для тестов."""
        return VectorStoreConfig(
            provider="memory",
            vector_size=10,
            fallback_enabled=False
        )
    
    @pytest.fixture
    def client(self, config):
        """Создает VectorStoreClient для тестов."""
        return VectorStoreClient(config)
    
    @pytest.mark.asyncio
    async def test_store_vectors(self, client):
        """Тест сохранения векторов."""
        items = [
            {
                "text": "Test text 1",
                "metadata": {"category": "test"}
            },
            {
                "text": "Test text 2",
                "metadata": {"category": "example"}
            }
        ]
        
        await client.store_vectors("semantic", items)
        
        # Проверяем, что векторы сохранены
        results = await client.retrieve_vectors("semantic", {"text": "Test", "limit": 10})
        assert len(results) == 2
        
        # Проверяем данные
        texts = [result["text"] for result in results]
        assert "Test text 1" in texts
        assert "Test text 2" in texts
    
    @pytest.mark.asyncio
    async def test_retrieve_vectors(self, client):
        """Тест извлечения векторов."""
        # Сохраняем данные
        items = [
            {"text": "Apple fruit", "metadata": {"type": "fruit"}},
            {"text": "Banana fruit", "metadata": {"type": "fruit"}},
            {"text": "Car vehicle", "metadata": {"type": "transport"}}
        ]
        await client.store_vectors("episodic", items)
        
        # Ищем фрукты
        results = await client.retrieve_vectors(
            "episodic",
            {"text": "fruit", "limit": 10}
        )
        
        assert len(results) == 2
        for result in results:
            assert result["layer"] == "episodic"
            assert "fruit" in result["text"].lower()
    
    @pytest.mark.asyncio
    async def test_update_vector(self, client):
        """Тест обновления вектора."""
        # Сохраняем вектор
        items = [{"text": "Original text", "metadata": {"version": 1}}]
        await client.store_vectors("test", items)
        
        # Находим вектор для обновления
        results = await client.retrieve_vectors("test", {"text": "Original", "limit": 1})
        assert len(results) == 1
        
        vector_id = results[0]["id"]
        
        # Обновляем
        await client.update_vector(
            "test",
            vector_id,
            {
                "text": "Updated text",
                "metadata": {"version": 2}
            }
        )
        
        # Проверяем обновление
        updated_results = await client.retrieve_vectors("test", {"text": "Updated", "limit": 1})
        assert len(updated_results) == 1
        assert updated_results[0]["text"] == "Updated text"
        assert updated_results[0]["metadata"]["version"] == 2
    
    @pytest.mark.asyncio
    async def test_sync_schema(self, client):
        """Тест синхронизации схемы."""
        # MemoryVectorStore не требует синхронизации схемы,
        # но тест должен пройти без ошибок
        await client.sync_schema()
        # Если не возникло исключений, тест пройден
    
    @pytest.mark.asyncio
    async def test_create_embeddings(self, client):
        """Тест создания embeddings."""
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = await client.create_embeddings(texts)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        assert all(len(emb) == 10 for emb in embeddings)  # vector_size
    
    @pytest.mark.asyncio
    async def test_vectorize_text(self, client):
        """Тест векторизации одного текста."""
        text = "Single text for vectorization"
        vector = await client.vectorize_text(text)
        
        assert isinstance(vector, list)
        assert len(vector) == 10
        assert all(isinstance(x, float) for x in vector)
    
    def test_get_store_info(self, client):
        """Тест получения информации о хранилище."""
        info = client.get_store_info()
        
        assert "provider" in info
        assert "available_providers" in info
        assert "current_store" in info
        assert "collection_name" in info
        assert "vector_size" in info
        assert info["provider"] == "memory"
    
    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Тест проверки состояния."""
        health = await client.health_check()
        
        assert "current_provider" in health
        assert "stores" in health
        assert "memory" in health["stores"]
    
    @pytest.mark.asyncio
    async def test_close(self, client):
        """Тест закрытия клиента."""
        # MemoryVectorStore не требует закрытия,
        # но метод должен работать без ошибок
        await client.close()


class TestFallbackMechanism:
    """Тесты fallback механизма."""
    
    @pytest.mark.asyncio
    async def test_fallback_to_memory(self):
        """Тест fallback на MemoryVectorStore."""
        config = VectorStoreConfig(
            provider="nonexistent",  # Недоступный провайдер
            fallback_enabled=True,
            fallback_providers=["memory"]
        )
        
        client = VectorStoreClient(config)
        
        # Должен fallback на memory
        assert client.config.provider == "memory"
        assert "memory" in client.stores
        
        # Должен работать
        items = [{"text": "Test fallback", "metadata": {"test": True}}]
        await client.store_vectors("test", items)
        
        results = await client.retrieve_vectors("test", {"text": "fallback", "limit": 1})
        assert len(results) == 1
        
        await client.close()


@pytest.mark.asyncio
async def test_integration_example():
    """Интеграционный тест с полным циклом операций."""
    config = VectorStoreConfig(
        provider="memory",
        vector_size=16,
        max_retries=2
    )
    
    client = VectorStoreClient(config)
    
    try:
        # 1. Создаем embeddings для текстов
        texts = [
            "Rebecca - интеллектуальная платформа",
            "Векторные хранилища для поиска",
            "Множественные провайдеры поддерживаются"
        ]
        
        embeddings = await client.create_embeddings(texts)
        assert len(embeddings) == 3
        assert all(len(emb) == 16 for emb in embeddings)
        
        # 2. Сохраняем в разные слои
        semantic_items = [
            {"text": texts[0], "metadata": {"type": "description"}},
            {"text": texts[1], "metadata": {"type": "feature"}}
        ]
        
        episodic_items = [
            {"text": texts[2], "metadata": {"type": "capability"}}
        ]
        
        await client.store_vectors("semantic", semantic_items)
        await client.store_vectors("episodic", episodic_items)
        
        # 3. Поиск в семантическом слое
        semantic_results = await client.retrieve_vectors(
            "semantic",
            {"text": "платформа", "limit": 10}
        )
        assert len(semantic_results) >= 1
        
        # 4. Поиск в эпизодическом слое
        episodic_results = await client.retrieve_vectors(
            "episodic", 
            {"text": "поддерживаются", "limit": 10}
        )
        assert len(episodic_results) >= 1
        
        # 5. Обновление вектора
        if semantic_results:
            vector_id = semantic_results[0]["id"]
            await client.update_vector(
                "semantic",
                vector_id,
                {"metadata": {"updated": True, "type": "feature"}}
            )
            
            # Проверяем обновление
            updated_results = await client.retrieve_vectors(
                "semantic",
                {"text": "платформа", "limit": 1}
            )
            assert updated_results[0]["metadata"]["updated"] is True
        
        # 6. Проверяем информацию о хранилище
        info = client.get_store_info()
        assert info["provider"] == "memory"
        assert info["vector_size"] == 16
        
        # 7. Проверяем состояние
        health = await client.health_check()
        assert "memory" in health["stores"]
        
    finally:
        await client.close()


if __name__ == "__main__":
    # Запуск тестов
    pytest.main([__file__, "-v"])
