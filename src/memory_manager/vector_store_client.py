"""Vector store client с поддержкой множественных провайдеров и embeddings."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import json
import hashlib
from pathlib import Path

# Внешние зависимости с fallback
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import qdrant_client
    from qdrant_client.http import models as qdrant_models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    import weaviate
    from weaviate.classes.config import Configure, Property, DataType
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False


@dataclass
class VectorStoreConfig:
    """Конфигурация векторного хранилища."""
    
    # Основные настройки
    provider: str = "qdrant"  # qdrant, weaviate, chroma
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    collection_name: str = "rebecca_vectors"
    
    # Настройки коллекции
    vector_size: int = 384
    distance_metric: str = "cosine"  # cosine, l2, dot
    
    # Embeddings настройки
    embedding_provider: str = "local"  # local, openai, ollama
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_api_key: Optional[str] = None
    embedding_base_url: Optional[str] = None
    
    # Retry настройки
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0
    
    # Fallback настройки
    fallback_enabled: bool = True
    fallback_providers: List[str] = field(default_factory=lambda: ["chroma", "memory"])
    
    # Локальные настройки
    local_storage_path: Optional[str] = None


@dataclass
class VectorItem:
    """Элемент векторного хранилища."""
    
    id: str
    vector: Optional[List[float]] = None
    text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    layer: str = "default"
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        
        # Генерируем ID, если не задан
        if not self.id:
            content = f"{self.text or ''}{json.dumps(self.metadata, sort_keys=True)}"
            self.id = hashlib.md5(content.encode()).hexdigest()


class RetryHandler:
    """Обработчик повторных попыток."""
    
    @staticmethod
    async def execute_with_retry(func, max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
        """Выполняет функцию с повторными попытками."""
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                # Проверяем, является ли func корутиной
                if asyncio.iscoroutine(func):
                    return await func
                elif asyncio.iscoroutinefunction(func):
                    return await func()
                else:
                    # Синхронная функция - запускаем в event loop
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, func)
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    wait_time = delay * (backoff ** attempt)
                    logging.warning(f"Попытка {attempt + 1} неудачна: {e}. Повтор через {wait_time}s")
                    await asyncio.sleep(wait_time)
                else:
                    logging.error(f"Все {max_retries + 1} попытки неудачны")
        
        raise last_exception


class EmbeddingProvider:
    """Провайдер для создания embeddings."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self._client = None
    
    async def create_embedding(self, text: str) -> List[float]:
        """Создает embedding для текста."""
        
        if not text or not text.strip():
            return [0.0] * self.config.vector_size
        
        try:
            if self.config.embedding_provider == "local":
                return await self._create_local_embedding(text)
            elif self.config.embedding_provider == "openai":
                return await self._create_openai_embedding(text)
            elif self.config.embedding_provider == "ollama":
                return await self._create_ollama_embedding(text)
            else:
                logging.warning(f"Неподдерживаемый провайдер: {self.config.embedding_provider}")
                return await self._create_local_embedding(text)
        except Exception as e:
            logging.error(f"Ошибка создания embedding: {e}")
            return [0.0] * self.config.vector_size
    
    async def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Создает embeddings для списка текстов."""
        
        if not texts:
            return []
        
        tasks = [self.create_embedding(text) for text in texts]
        return await asyncio.gather(*tasks)
    
    async def _create_local_embedding(self, text: str) -> List[float]:
        """Создает локальный embedding (заглушка)."""
        
        # Простой хеш-базированный embedding для демонстрации
        # В реальном проекте здесь был бы sentence-transformers или подобное
        
        if not NUMPY_AVAILABLE:
            logging.warning("NumPy не доступен, используем простой embedding")
            return self._simple_hash_embedding(text)
        
        # Простой embedding на основе хеша
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Преобразуем в вектор нужного размера
        vector = [0.0] * self.config.vector_size
        
        for i, byte_val in enumerate(hash_bytes):
            if i < self.config.vector_size:
                vector[i] = (byte_val / 255.0) * 2 - 1
        
        # Заполняем остальные измерения
        for i in range(len(hash_bytes), self.config.vector_size):
            vector[i] = (hash_obj.digest()[i % len(hash_bytes)] / 255.0) * 2 - 1
        
        return vector
    
    def _simple_hash_embedding(self, text: str) -> List[float]:
        """Простое embedding на основе хеша."""
        
        hash_obj = hashlib.md5(text.encode())
        vector = []
        
        for i in range(self.config.vector_size):
            byte_val = hash_obj.digest()[i % 16] if i < 16 else 0
            vector.append((byte_val / 255.0) * 2 - 1)
        
        return vector
    
    async def _create_openai_embedding(self, text: str) -> List[float]:
        """Создает embedding через OpenAI API."""
        
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx не доступен для OpenAI API")
        
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.config.embedding_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "input": text,
                    "model": self.config.embedding_model
                }
            )
            
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
    
    async def _create_ollama_embedding(self, text: str) -> List[float]:
        """Создает embedding через Ollama API."""
        
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx не доступен для Ollama API")
        
        url = f"{self.config.embedding_base_url}/api/embeddings"
        
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(
                url,
                json={
                    "model": self.config.embedding_model,
                    "prompt": text
                }
            )
            
            response.raise_for_status()
            data = response.json()
            return data["embedding"]


class MemoryVectorStore:
    """In-memory векторное хранилище для fallback."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.vectors: Dict[str, VectorItem] = {}
        self.collections: Dict[str, List[str]] = {}
    
    def store(self, items: List[VectorItem]) -> None:
        """Сохраняет векторы в память."""
        
        for item in items:
            self.vectors[item.id] = item
            
            # Добавляем в коллекцию
            if item.layer not in self.collections:
                self.collections[item.layer] = []
            
            if item.id not in self.collections[item.layer]:
                self.collections[item.layer].append(item.id)
    
    def retrieve(self, layer: str, query: Dict[str, Any]) -> List[VectorItem]:
        """Извлекает векторы из памяти."""
        
        results = []
        query_text = query.get("text", "")
        limit = query.get("limit", 10)
        
        if layer in self.collections:
            for vector_id in self.collections[layer][:limit]:
                if vector_id in self.vectors:
                    item = self.vectors[vector_id]
                    
                    # Простая фильтрация по тексту
                    if query_text and item.text:
                        if query_text.lower() in item.text.lower():
                            results.append(item)
                    else:
                        results.append(item)
        
        return results[:limit]
    
    def update(self, vector_id: str, changes: Dict[str, Any]) -> None:
        """Обновляет вектор в памяти."""
        
        if vector_id in self.vectors:
            item = self.vectors[vector_id]
            
            for key, value in changes.items():
                if hasattr(item, key):
                    setattr(item, key, value)
                
            # Обновляем timestamp
            item.timestamp = time.time()
    
    def create_collection(self, name: str) -> None:
        """Создает коллекцию."""
        
        if name not in self.collections:
            self.collections[name] = []


class QdrantStore:
    """Qdrant векторное хранилище."""
    
    def __init__(self, config: VectorStoreConfig):
        if not QDRANT_AVAILABLE:
            raise RuntimeError("Qdrant клиент не установлен")
        
        self.config = config
        
        # Инициализация клиента
        if config.base_url:
            self.client = qdrant_client.QdrantClient(url=config.base_url, api_key=config.api_key)
        else:
            self.client = qdrant_client.QdrantClient(":memory:")
        
        self.collection_name = config.collection_name
    
    async def store(self, items: List[VectorItem]) -> None:
        """Сохраняет векторы в Qdrant."""
        
        points = []
        
        for item in items:
            if item.vector is None:
                continue
            
            point = qdrant_models.PointStruct(
                id=item.id,
                vector=item.vector,
                payload={
                    "text": item.text or "",
                    "layer": item.layer,
                    "metadata": json.dumps(item.metadata),
                    "timestamp": item.timestamp
                }
            )
            points.append(point)
        
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
    
    async def retrieve(self, layer: str, query: Dict[str, Any]) -> List[VectorItem]:
        """Извлекает векторы из Qdrant."""
        
        query_vector = query.get("vector")
        query_text = query.get("text", "")
        limit = query.get("limit", 10)
        
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="layer",
                        match=qdrant_models.ValueMatch(value=layer)
                    )
                ]
            )
        )
        
        results = []
        for result in search_results:
            try:
                metadata = json.loads(result.payload.get("metadata", "{}"))
            except json.JSONDecodeError:
                metadata = {}
            
            item = VectorItem(
                id=str(result.id),
                vector=result.vector,
                text=result.payload.get("text", ""),
                metadata=metadata,
                layer=layer,
                timestamp=result.payload.get("timestamp")
            )
            results.append(item)
        
        return results
    
    async def update(self, vector_id: str, changes: Dict[str, Any]) -> None:
        """Обновляет вектор в Qdrant."""
        
        update_payload = {}
        
        for key, value in changes.items():
            if key in ["text", "metadata", "layer"]:
                update_payload[key] = value
        
        self.client.update_vectors(
            collection_name=self.collection_name,
            points=[{
                "id": vector_id,
                "vector": changes.get("vector")
            }]
        )
    
    async def create_collection(self, name: str) -> None:
        """Создает коллекцию в Qdrant."""
        
        self.client.create_collection(
            collection_name=name,
            vectors_config=qdrant_models.VectorParams(
                size=self.config.vector_size,
                distance=qdrant_models.Distance.COSINE if self.config.distance_metric == "cosine" 
                        else qdrant_models.Distance.EUCLID
            )
        )


class ChromaStore:
    """ChromaDB векторное хранилище."""
    
    def __init__(self, config: VectorStoreConfig):
        if not CHROMA_AVAILABLE:
            raise RuntimeError("ChromaDB клиент не установлен")
        
        self.config = config
        
        # Инициализация клиента
        if config.local_storage_path:
            self.client = chromadb.PersistentClient(path=config.local_storage_path)
        else:
            self.client = chromadb.Client()
        
        self.collection_name = config.collection_name
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"layer": "default"}
        )
    
    async def store(self, items: List[VectorItem]) -> None:
        """Сохраняет векторы в ChromaDB."""
        
        ids = []
        documents = []
        embeddings = []
        metadatas = []
        
        for item in items:
            ids.append(item.id)
            documents.append(item.text or "")
            embeddings.append(item.vector or [0.0] * self.config.vector_size)
            metadatas.append({
                **item.metadata,
                "layer": item.layer,
                "timestamp": item.timestamp
            })
        
        if ids:
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
    
    async def retrieve(self, layer: str, query: Dict[str, Any]) -> List[VectorItem]:
        """Извлекает векторы из ChromaDB."""
        
        query_text = query.get("text", "")
        query_embedding = query.get("vector")
        limit = query.get("limit", 10)
        
        where_clause = {"layer": layer} if layer else None
        
        results = self.collection.query(
            query_texts=[query_text] if query_text else None,
            query_embeddings=[query_embedding] if query_embedding else None,
            n_results=limit,
            where=where_clause
        )
        
        items = []
        if results["ids"]:
            for i, doc_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                
                item = VectorItem(
                    id=doc_id,
                    text=results["documents"][0][i] if results["documents"] else "",
                    vector=results["embeddings"][0][i] if results["embeddings"] else None,
                    metadata=metadata,
                    layer=layer,
                    timestamp=metadata.get("timestamp")
                )
                items.append(item)
        
        return items
    
    async def update(self, vector_id: str, changes: Dict[str, Any]) -> None:
        """Обновляет вектор в ChromaDB."""
        
        update_data = {}
        
        if "text" in changes:
            update_data["documents"] = [changes["text"]]
        if "vector" in changes:
            update_data["embeddings"] = [changes["vector"]]
        
        metadata = changes.get("metadata", {})
        metadata["layer"] = changes.get("layer", "default")
        update_data["metadatas"] = [metadata]
        
        self.collection.update(
            ids=[vector_id],
            **update_data
        )
    
    async def create_collection(self, name: str) -> None:
        """Создает коллекцию в ChromaDB."""
        
        self.client.create_collection(name=name)


class WeaviateStore:
    """Weaviate векторное хранилище."""
    
    def __init__(self, config: VectorStoreConfig):
        if not WEAVIATE_AVAILABLE:
            raise RuntimeError("Weaviate клиент не установлен")
        
        self.config = config
        
        # Инициализация клиента
        if config.base_url:
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=config.base_url,
                api_key=config.api_key
            )
        else:
            self.client = weaviate.connect_to_local()
        
        self.collection_name = config.collection_name
        self._ensure_schema()
    
    def _ensure_schema(self):
        """Создает схему если не существует."""
        
        if self.client.collections.exists(self.collection_name):
            return
        
        self.client.collections.create(
            name=self.collection_name,
            vectorizer_config=Configure.Vectorizer.text2vec_transformers(),
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="layer", data_type=DataType.TEXT),
                Property(name="metadata", data_type=DataType.TEXT),
                Property(name="timestamp", data_type=DataType.NUMBER)
            ]
        )
    
    async def store(self, items: List[VectorItem]) -> None:
        """Сохраняет векторы в Weaviate."""
        
        collection = self.client.collections.get(self.collection_name)
        
        with collection.batch.dynamic() as batch:
            for item in items:
                batch.add_object(
                    properties={
                        "text": item.text or "",
                        "layer": item.layer,
                        "metadata": json.dumps(item.metadata),
                        "timestamp": item.timestamp
                    },
                    vector=item.vector
                )
    
    async def retrieve(self, layer: str, query: Dict[str, Any]) -> List[VectorItem]:
        """Извлекает векторы из Weaviate."""
        
        collection = self.client.collections.get(self.collection_name)
        
        query_text = query.get("text", "")
        query_embedding = query.get("vector")
        limit = query.get("limit", 10)
        
        response = collection.query.near_text(
            query=query_text,
            limit=limit,
            filters=f"layer == '{layer}'"
        )
        
        items = []
        for obj in response.objects:
            metadata = obj.properties.get("metadata", "{}")
            
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
            
            item = VectorItem(
                id=obj.uuid,
                text=obj.properties.get("text", ""),
                vector=obj.vector,
                metadata=metadata,
                layer=layer,
                timestamp=obj.properties.get("timestamp")
            )
            items.append(item)
        
        return items
    
    async def update(self, vector_id: str, changes: Dict[str, Any]) -> None:
        """Обновляет вектор в Weaviate."""
        
        collection = self.client.collections.get(self.collection_name)
        
        properties = {}
        for key, value in changes.items():
            if key in ["text", "layer", "metadata", "timestamp"]:
                if key == "metadata" and isinstance(value, dict):
                    properties[key] = json.dumps(value)
                else:
                    properties[key] = value
        
        collection.data.update(
            uuid=vector_id,
            properties=properties
        )
    
    async def create_collection(self, name: str) -> None:
        """Создает коллекцию в Weaviate."""
        
        self.client.collections.create(
            name=name,
            vectorizer_config=Configure.Vectorizer.text2vec_transformers(),
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="layer", data_type=DataType.TEXT),
                Property(name="metadata", data_type=DataType.TEXT),
                Property(name="timestamp", data_type=DataType.NUMBER)
            ]
        )


class VectorStoreClient:
    """Клиент векторного хранилища с поддержкой множественных провайдеров."""

    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """Инициализирует клиент векторного хранилища."""
        
        self.config = config or VectorStoreConfig()
        self.logger = logging.getLogger(__name__)
        
        # Инициализируем embedding провайдер
        self.embedding_provider = EmbeddingProvider(self.config)
        
        # Инициализируем хранилища
        self._initialize_stores()
    
    def _initialize_stores(self):
        """Инициализирует доступные векторные хранилища."""
        
        self.stores = {}
        
        # Qdrant
        if QDRANT_AVAILABLE and self.config.provider == "qdrant":
            try:
                self.stores["qdrant"] = QdrantStore(self.config)
                self.logger.info("Инициализирован Qdrant store")
            except Exception as e:
                self.logger.error(f"Не удалось инициализировать Qdrant: {e}")
        
        # ChromaDB
        if CHROMA_AVAILABLE and self.config.provider == "chroma":
            try:
                self.stores["chroma"] = ChromaStore(self.config)
                self.logger.info("Инициализирован ChromaDB store")
            except Exception as e:
                self.logger.error(f"Не удалось инициализировать ChromaDB: {e}")
        
        # Weaviate
        if WEAVIATE_AVAILABLE and self.config.provider == "weaviate":
            try:
                self.stores["weaviate"] = WeaviateStore(self.config)
                self.logger.info("Инициализирован Weaviate store")
            except Exception as e:
                self.logger.error(f"Не удалось инициализировать Weaviate: {e}")
        
        # Fallback память
        self.stores["memory"] = MemoryVectorStore(self.config)
        
        # Проверяем основной провайдер
        if self.config.provider not in self.stores:
            self.logger.warning(f"Провайдер {self.config.provider} недоступен")
            
            if self.config.fallback_enabled:
                for fallback in self.config.fallback_providers:
                    if fallback in self.stores:
                        self.config.provider = fallback
                        self.logger.info(f"Используется fallback: {fallback}")
                        break
                else:
                    self.config.provider = "memory"
                    self.logger.info("Используется memory fallback")
        
        self.current_store = self.stores.get(self.config.provider, self.stores["memory"])
    
    async def store_vectors(self, layer: str, items: List[Dict[str, Any]]) -> None:
        """Сохраняет векторы для заданного слоя.
        
        Args:
            layer: Слой памяти (semantic, episodic, etc.)
            items: Список элементов для сохранения
        """
        
        try:
            vector_items = []
            
            for item_data in items:
                # Создаем VectorItem
                vector_item = VectorItem(
                    id=item_data.get("id"),
                    text=item_data.get("text", ""),
                    metadata=item_data.get("metadata", {}),
                    layer=layer
                )
                
                # Создаем embedding если нужен
                if "vector" not in item_data and vector_item.text:
                    vector_item.vector = await self.embedding_provider.create_embedding(vector_item.text)
                elif "vector" in item_data:
                    vector_item.vector = item_data["vector"]
                
                vector_items.append(vector_item)
            
            # Сохраняем через retry handler
            await RetryHandler.execute_with_retry(
                lambda: self.current_store.store(vector_items),
                self.config.max_retries,
                self.config.retry_delay
            )
            
            self.logger.info(f"Сохранено {len(vector_items)} векторов в слой {layer}")
            
        except Exception as e:
            self.logger.error(f"Ошибка сохранения векторов в слой {layer}: {e}")
            raise
    
    async def retrieve_vectors(self, layer: str, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Извлекает векторы для заданного слоя.
        
        Args:
            layer: Слой памяти для поиска
            query: Параметры запроса (text, vector, limit, etc.)
            
        Returns:
            Список найденных векторов
        """
        
        try:
            # Создаем embedding для текстового запроса
            if "text" in query and "vector" not in query:
                query = query.copy()
                query["vector"] = await self.embedding_provider.create_embedding(query["text"])
            
            # Извлекаем через retry handler
            results = await RetryHandler.execute_with_retry(
                lambda: self.current_store.retrieve(layer, query),
                self.config.max_retries,
                self.config.retry_delay
            )
            
            # Преобразуем в словари
            return [
                {
                    "id": item.id,
                    "text": item.text,
                    "vector": item.vector,
                    "metadata": item.metadata,
                    "layer": item.layer,
                    "timestamp": item.timestamp
                }
                for item in results
            ]
            
        except Exception as e:
            self.logger.error(f"Ошибка извлечения векторов из слоя {layer}: {e}")
            return []
    
    async def update_vector(self, layer: str, vector_id: str, changes: Dict[str, Any]) -> None:
        """Обновляет вектор в хранилище.
        
        Args:
            layer: Слой памяти
            vector_id: ID вектора для обновления
            changes: Словарь изменений
        """
        
        try:
            # Обновляем embedding если изменился текст
            if "text" in changes:
                changes = changes.copy()
                changes["vector"] = await self.embedding_provider.create_embedding(changes["text"])
            
            # Обновляем через retry handler
            await RetryHandler.execute_with_retry(
                lambda: self.current_store.update(vector_id, changes),
                self.config.max_retries,
                self.config.retry_delay
            )
            
            self.logger.info(f"Обновлен вектор {vector_id} в слое {layer}")
            
        except Exception as e:
            self.logger.error(f"Ошибка обновления вектора {vector_id} в слое {layer}: {e}")
            raise
    
    async def sync_schema(self) -> None:
        """Синхронизирует схему с векторным хранилищем."""
        
        try:
            # Создаем коллекцию
            await RetryHandler.execute_with_retry(
                lambda: self.current_store.create_collection(self.config.collection_name),
                self.config.max_retries,
                self.config.retry_delay
            )
            
            self.logger.info(f"Схема синхронизирована для коллекции {self.config.collection_name}")
            
        except Exception as e:
            self.logger.error(f"Ошибка синхронизации схемы: {e}")
            raise
    
    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Создает embeddings для списка текстов.
        
        Args:
            texts: Список текстов для векторизации
            
        Returns:
            Список embedding векторов
        """
        
        try:
            return await self.embedding_provider.create_embeddings_batch(texts)
        except Exception as e:
            self.logger.error(f"Ошибка создания embeddings: {e}")
            return [[0.0] * self.config.vector_size for _ in texts]
    
    async def vectorize_text(self, text: str) -> List[float]:
        """Векторизует один текст.
        
        Args:
            text: Текст для векторизации
            
        Returns:
            Векторное представление текста
        """
        
        try:
            return await self.embedding_provider.create_embedding(text)
        except Exception as e:
            self.logger.error(f"Ошибка векторизации текста: {e}")
            return [0.0] * self.config.vector_size
    
    def get_store_info(self) -> Dict[str, Any]:
        """Возвращает информацию о текущем хранилище."""
        
        return {
            "provider": self.config.provider,
            "available_providers": list(self.stores.keys()),
            "current_store": type(self.current_store).__name__,
            "collection_name": self.config.collection_name,
            "vector_size": self.config.vector_size,
            "distance_metric": self.config.distance_metric,
            "embedding_provider": self.config.embedding_provider
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Проверяет состояние векторных хранилищ."""
        
        health_status = {
            "current_provider": self.config.provider,
            "stores": {}
        }
        
        for name, store in self.stores.items():
            try:
                # Простая проверка доступности
                if hasattr(store, 'client') and store.client:
                    health_status["stores"][name] = "healthy"
                else:
                    health_status["stores"][name] = "ok"
            except Exception as e:
                health_status["stores"][name] = f"error: {str(e)}"
        
        return health_status
    
    async def close(self):
        """Закрывает соединения с хранилищами."""
        
        try:
            if hasattr(self.current_store, 'client'):
                if hasattr(self.current_store.client, 'close'):
                    self.current_store.client.close()
                elif hasattr(self.current_store.client, 'disconnect'):
                    await self.current_store.client.disconnect()
        except Exception as e:
            self.logger.warning(f"Ошибка закрытия соединения: {e}")


# Функции для удобного использования

def create_vector_client_from_config(config_path: Optional[str] = None) -> VectorStoreClient:
    """Создает VectorStoreClient из конфигурационного файла."""
    
    # Здесь можно добавить чтение из config.yaml
    # Пока возвращаем клиент с конфигурацией по умолчанию
    
    config = VectorStoreConfig(
        provider="qdrant",
        base_url="http://localhost:6333",
        vector_size=384,
        collection_name="rebecca_vectors"
    )
    
    return VectorStoreClient(config)


async def quick_embed(texts: List[str]) -> List[List[float]]:
    """Быстрое создание embeddings для списка текстов."""
    
    config = VectorStoreConfig()
    client = VectorStoreClient(config)
    
    try:
        return await client.create_embeddings(texts)
    finally:
        await client.close()


# Экспорт основных классов
__all__ = [
    "VectorStoreClient",
    "VectorStoreConfig", 
    "VectorItem",
    "EmbeddingProvider",
    "MemoryVectorStore",
    "QdrantStore",
    "ChromaStore",
    "WeaviateStore",
    "create_vector_client_from_config",
    "quick_embed"
]
