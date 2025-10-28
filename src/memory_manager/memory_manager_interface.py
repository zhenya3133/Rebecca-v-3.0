"""
Интерфейс и реализация MemoryManager для Rebecca Platform.
Обеспечивает унифицированный доступ к различным слоям памяти.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union
from enum import Enum

from .memory_context import MemoryContext
from .vector_store_client import VectorStoreClient


class MemoryLayer(Enum):
    """Поддерживаемые слои памяти."""
    CORE = "core"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    VAULT = "vault"
    SECURITY = "security"


@dataclass
class MemoryItem:
    """Структура элемента памяти."""
    id: str
    layer: MemoryLayer
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    vector_embedding: Optional[List[float]] = None


@dataclass
class MemoryFilter:
    """Фильтр для поиска элементов памяти."""
    metadata: Dict[str, Any] = field(default_factory=dict)
    time_range: Optional[tuple[float, float]] = None
    vector_similarity: Optional[Dict[str, Any]] = None


@dataclass
class CacheEntry:
    """Запись в кэше."""
    data: Any
    timestamp: float = field(default_factory=time.time)
    ttl: float = 300.0  # 5 минут по умолчанию


class IMemoryManager(ABC):
    """Интерфейс менеджера памяти."""
    
    @abstractmethod
    async def store(
        self,
        layer: MemoryLayer,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Сохранить данные в указанный слой памяти."""
        pass
    
    @abstractmethod
    async def retrieve(
        self,
        layer: MemoryLayer,
        query: Union[str, Dict[str, Any]],
        filters: Optional[MemoryFilter] = None
    ) -> List[MemoryItem]:
        """Извлечь данные из указанного слоя памяти."""
        pass
    
    @abstractmethod
    async def update(
        self,
        layer: MemoryLayer,
        item_id: str,
        data: Dict[str, Any]
    ) -> bool:
        """Обновить элемент памяти."""
        pass
    
    @abstractmethod
    async def delete(
        self,
        layer: MemoryLayer,
        item_id: str
    ) -> bool:
        """Удалить элемент памяти."""
        pass
    
    @abstractmethod
    def list_layers(self) -> List[MemoryLayer]:
        """Получить список доступных слоев памяти."""
        pass


class LayerFactory:
    """Фабрика для создания слоев памяти."""
    
    _layer_classes: Dict[MemoryLayer, Type] = {}
    
    @classmethod
    def register_layer(cls, layer_type: MemoryLayer, layer_class: Type):
        """Регистрация класса слоя памяти."""
        cls._layer_classes[layer_type] = layer_class
    
    @classmethod
    def create_layer(cls, layer_type: MemoryLayer) -> Any:
        """Создание экземпляра слоя памяти."""
        if layer_type not in cls._layer_classes:
            raise ValueError(f"Неизвестный тип слоя памяти: {layer_type}")
        
        return cls._layer_classes[layer_type]()


class PerformanceOptimizer:
    """Оптимизатор производительности с кэшированием."""
    
    def __init__(self, max_cache_size: int = 1000, default_ttl: float = 300.0):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_cache_size = max_cache_size
        self.default_ttl = default_ttl
        self.hit_count = 0
        self.miss_count = 0
    
    def _generate_cache_key(self, *args, **kwargs) -> str:
        """Генерация ключа кэша."""
        return str(hash((str(args), str(sorted(kwargs.items())))))
    
    def get(self, key: str) -> Optional[Any]:
        """Получение данных из кэша."""
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry.timestamp < entry.ttl:
                self.hit_count += 1
                return entry.data
            else:
                del self.cache[key]
        
        self.miss_count += 1
        return None
    
    def set(self, key: str, data: Any, ttl: Optional[float] = None) -> None:
        """Сохранение данных в кэш."""
        if len(self.cache) >= self.max_cache_size:
            # Удаляем самые старые записи
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k].timestamp)
            del self.cache[oldest_key]
        
        self.cache[key] = CacheEntry(data=data, ttl=ttl or self.default_ttl)
    
    def clear(self) -> None:
        """Очистка кэша."""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики кэша."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
        
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "max_cache_size": self.max_cache_size
        }


class MemoryManager(IMemoryManager):
    """Основная реализация менеджера памяти."""
    
    def __init__(
        self,
        context: Optional[MemoryContext] = None,
        vector_store: Optional[VectorStoreClient] = None,
        cache_ttl: float = 300.0,
        max_cache_size: int = 1000
    ):
        # Инициализация контекста и векторного хранилища
        self.context = context or MemoryContext()
        self.vector_store = vector_store or VectorStoreClient()
        
        # Инициализация оптимизатора производительности
        self.optimizer = PerformanceOptimizer(
            max_cache_size=max_cache_size,
            default_ttl=cache_ttl
        )
        
        # Инициализация слоев памяти
        self.layers: Dict[MemoryLayer, Any] = {}
        self._initialize_layers()
        
        # Индексы для оптимизации поиска
        self.metadata_index: Dict[str, Dict[str, List[str]]] = {}
        self.id_index: Dict[str, MemoryItem] = {}
    
    def _initialize_layers(self) -> None:
        """Инициализация всех слоев памяти."""
        try:
            from .core_memory import CoreMemory
            from .episodic_memory import EpisodicMemory
            from .semantic_memory import SemanticMemory
            from .procedural_memory import ProceduralMemory
            from .vault_memory import VaultMemory
            from .security_memory import SecurityMemory
            
            self.layers = {
                MemoryLayer.CORE: CoreMemory(),
                MemoryLayer.EPISODIC: EpisodicMemory(),
                MemoryLayer.SEMANTIC: SemanticMemory(),
                MemoryLayer.PROCEDURAL: ProceduralMemory(),
                MemoryLayer.VAULT: VaultMemory(),
                MemoryLayer.SECURITY: SecurityMemory(),
            }
            
            # Регистрация слоев в контексте
            for layer_type, layer_instance in self.layers.items():
                self.context.register_layer(layer_type.value, layer_instance)
        
        except ImportError as e:
            print(f"Предупреждение: Не удалось импортировать слои памяти: {e}")
    
    async def store(
        self,
        layer: MemoryLayer,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Сохранить данные в указанный слой памяти."""
        # Генерация уникального ID
        item_id = f"{layer.value}_{int(time.time() * 1000000)}"
        
        # Создание элемента памяти
        memory_item = MemoryItem(
            id=item_id,
            layer=layer,
            data=data,
            metadata=metadata or {}
        )
        
        # Сохранение в соответствующий слой
        if layer in self.layers:
            layer_instance = self.layers[layer]
            await self._store_in_layer(layer_instance, memory_item)
        
        # Сохранение в векторное хранилище
        await self._store_in_vector_store(memory_item)
        
        # Обновление индексов
        self._update_indexes(memory_item)
        
        # Инвалидация кэша
        self._invalidate_cache(layer, "store")
        
        return item_id
    
    async def retrieve(
        self,
        layer: MemoryLayer,
        query: Union[str, Dict[str, Any]],
        filters: Optional[MemoryFilter] = None
    ) -> List[MemoryItem]:
        """Извлечь данные из указанного слоя памяти."""
        # Генерация ключа кэша
        cache_key = self.optimizer._generate_cache_key("retrieve", layer, query, filters)
        
        # Проверка кэша
        cached_result = self.optimizer.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        results = []
        
        # Поиск в соответствующем слое
        if layer in self.layers:
            layer_instance = self.layers[layer]
            results = await self._retrieve_from_layer(layer_instance, query, filters)
        
        # Поиск в векторном хранилище
        vector_results = await self._retrieve_from_vector_store(layer, query, filters)
        results.extend(vector_results)
        
        # Применение дополнительных фильтров
        if filters:
            results = self._apply_filters(results, filters)
        
        # Кэширование результата
        self.optimizer.set(cache_key, results)
        
        return results
    
    async def update(
        self,
        layer: MemoryLayer,
        item_id: str,
        data: Dict[str, Any]
    ) -> bool:
        """Обновить элемент памяти."""
        try:
            # Поиск элемента
            memory_item = self.id_index.get(item_id)
            if not memory_item:
                return False
            
            # Обновление данных
            memory_item.data.update(data)
            memory_item.timestamp = time.time()
            
            # Обновление в слое
            if layer in self.layers:
                layer_instance = self.layers[layer]
                await self._update_in_layer(layer_instance, memory_item)
            
            # Обновление в векторном хранилище
            await self._update_in_vector_store(memory_item)
            
            # Обновление индексов
            self._update_indexes(memory_item)
            
            # Инвалидация кэша
            self._invalidate_cache(layer, "update", item_id)
            
            return True
        
        except Exception as e:
            print(f"Ошибка при обновлении элемента памяти: {e}")
            return False
    
    async def delete(
        self,
        layer: MemoryLayer,
        item_id: str
    ) -> bool:
        """Удалить элемент памяти."""
        try:
            memory_item = self.id_index.get(item_id)
            if not memory_item:
                return False
            
            # Удаление из слоя
            if layer in self.layers:
                layer_instance = self.layers[layer]
                await self._delete_from_layer(layer_instance, item_id)
            
            # Удаление из векторного хранилища
            await self._delete_from_vector_store(item_id)
            
            # Удаление из индексов
            self._remove_from_indexes(memory_item)
            self.id_index.pop(item_id, None)
            
            # Инвалидация кэша
            self._invalidate_cache(layer, "delete", item_id)
            
            return True
        
        except Exception as e:
            print(f"Ошибка при удалении элемента памяти: {e}")
            return False
    
    def list_layers(self) -> List[MemoryLayer]:
        """Получить список доступных слоев памяти."""
        return list(self.layers.keys())
    
    # Вспомогательные методы для работы с кэшем
    
    def _invalidate_cache(self, layer: MemoryLayer, operation: str, item_id: Optional[str] = None) -> None:
        """Инвалидация связанных записей кэша."""
        # Простая стратегия: очистка всего кэша при изменениях
        # В продакшене можно реализовать более точечную инвалидацию
        self.optimizer.clear()
    
    def _apply_filters(self, items: List[MemoryItem], filters: MemoryFilter) -> List[MemoryItem]:
        """Применение фильтров к результатам поиска."""
        filtered_items = items
        
        # Фильтрация по метаданным
        if filters.metadata:
            filtered_items = [
                item for item in filtered_items
                if all(
                    item.metadata.get(key) == value
                    for key, value in filters.metadata.items()
                )
            ]
        
        # Фильтрация по времени
        if filters.time_range:
            start_time, end_time = filters.time_range
            filtered_items = [
                item for item in filtered_items
                if start_time <= item.timestamp <= end_time
            ]
        
        return filtered_items
    
    def _update_indexes(self, item: MemoryItem) -> None:
        """Обновление поисковых индексов."""
        # Индекс по ID
        self.id_index[item.id] = item
        
        # Индекс по метаданным
        for key, value in item.metadata.items():
            if key not in self.metadata_index:
                self.metadata_index[key] = {}
            
            if value not in self.metadata_index[key]:
                self.metadata_index[key][value] = []
            
            if item.id not in self.metadata_index[key][value]:
                self.metadata_index[key][value].append(item.id)
    
    def _remove_from_indexes(self, item: MemoryItem) -> None:
        """Удаление элемента из поисковых индексов."""
        # Удаление из индекса метаданных
        for key, value in item.metadata.items():
            if key in self.metadata_index and value in self.metadata_index[key]:
                if item.id in self.metadata_index[key][value]:
                    self.metadata_index[key][value].remove(item.id)
                
                if not self.metadata_index[key][value]:
                    del self.metadata_index[key][value]
    
    # Адаптеры для работы с конкретными слоями памяти
    
    async def _store_in_layer(self, layer_instance: Any, item: MemoryItem) -> None:
        """Сохранение в конкретный слой памяти."""
        try:
            if hasattr(layer_instance, 'store'):
                await layer_instance.store(item.data, item.metadata)
            elif hasattr(layer_instance, 'store_fact'):
                for key, value in item.data.items():
                    layer_instance.store_fact(key, value)
        except Exception as e:
            print(f"Ошибка при сохранении в слой памяти: {e}")
    
    async def _retrieve_from_layer(
        self,
        layer_instance: Any,
        query: Union[str, Dict[str, Any]],
        filters: Optional[MemoryFilter]
    ) -> List[MemoryItem]:
        """Извлечение из конкретного слоя памяти."""
        results = []
        try:
            if hasattr(layer_instance, 'retrieve'):
                raw_results = await layer_instance.retrieve(query, filters)
                for result in raw_results:
                    results.append(MemoryItem(
                        id=str(hash(str(result))),
                        layer=MemoryLayer.CORE,  # Будет переопределен вызывающим методом
                        data=result,
                        metadata=filters.metadata if filters else {}
                    ))
        except Exception as e:
            print(f"Ошибка при извлечении из слоя памяти: {e}")
        
        return results
    
    async def _update_in_layer(self, layer_instance: Any, item: MemoryItem) -> None:
        """Обновление в конкретном слое памяти."""
        try:
            if hasattr(layer_instance, 'update'):
                await layer_instance.update(item.id, item.data)
        except Exception as e:
            print(f"Ошибка при обновлении в слое памяти: {e}")
    
    async def _delete_from_layer(self, layer_instance: Any, item_id: str) -> None:
        """Удаление из конкретного слоя памяти."""
        try:
            if hasattr(layer_instance, 'delete'):
                await layer_instance.delete(item_id)
        except Exception as e:
            print(f"Ошибка при удалении из слоя памяти: {e}")
    
    async def _store_in_vector_store(self, item: MemoryItem) -> None:
        """Сохранение в векторное хранилище."""
        try:
            if self.vector_store:
                self.vector_store.store_vectors(
                    item.layer.value,
                    [{
                        'id': item.id,
                        'data': item.data,
                        'metadata': item.metadata,
                        'timestamp': item.timestamp
                    }]
                )
        except Exception as e:
            print(f"Ошибка при сохранении в векторное хранилище: {e}")
    
    async def _retrieve_from_vector_store(
        self,
        layer: MemoryLayer,
        query: Union[str, Dict[str, Any]],
        filters: Optional[MemoryFilter]
    ) -> List[MemoryItem]:
        """Извлечение из векторного хранилища."""
        results = []
        try:
            if self.vector_store:
                raw_results = self.vector_store.retrieve_vectors(
                    layer.value,
                    {'query': query, 'filters': filters}
                )
                for result in raw_results:
                    results.append(MemoryItem(
                        id=result.get('id', ''),
                        layer=layer,
                        data=result.get('data', {}),
                        metadata=result.get('metadata', {}),
                        timestamp=result.get('timestamp', time.time())
                    ))
        except Exception as e:
            print(f"Ошибка при извлечении из векторного хранилища: {e}")
        
        return results
    
    async def _update_in_vector_store(self, item: MemoryItem) -> None:
        """Обновление в векторном хранилище."""
        try:
            if self.vector_store:
                self.vector_store.update_vector(
                    item.layer.value,
                    item.id,
                    {
                        'data': item.data,
                        'metadata': item.metadata,
                        'timestamp': item.timestamp
                    }
                )
        except Exception as e:
            print(f"Ошибка при обновлении в векторном хранилище: {e}")
    
    async def _delete_from_vector_store(self, item_id: str) -> None:
        """Удаление из векторного хранилища."""
        try:
            if self.vector_store:
                # Удаление из всех слоев
                for layer in self.layers.keys():
                    try:
                        self.vector_store.update_vector(
                            layer.value,
                            item_id,
                            {'deleted': True}
                        )
                    except:
                        continue
        except Exception as e:
            print(f"Ошибка при удалении из векторного хранилища: {e}")
    
    # Дополнительные методы для аналитики и мониторинга
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Получение статистики использования памяти."""
        return {
            "layers_count": len(self.layers),
            "available_layers": [layer.value for layer in self.layers.keys()],
            "cache_stats": self.optimizer.get_stats(),
            "indexed_items": len(self.id_index),
            "metadata_keys": list(self.metadata_index.keys())
        }
    
    async def search_across_layers(
        self,
        query: str,
        layers: Optional[List[MemoryLayer]] = None
    ) -> Dict[MemoryLayer, List[MemoryItem]]:
        """Поиск по нескольким слоям памяти одновременно."""
        if layers is None:
            layers = list(self.layers.keys())
        
        results = {}
        for layer in layers:
            layer_results = await self.retrieve(layer, query)
            results[layer] = layer_results
        
        return results
    
    def clear_cache(self) -> None:
        """Очистка кэша."""
        self.optimizer.clear()
    
    def clear_all_data(self) -> None:
        """Очистка всех данных (использовать с осторожностью!)."""
        # Очистка всех слоев памяти
        for layer_instance in self.layers.values():
            try:
                if hasattr(layer_instance, 'clear'):
                    layer_instance.clear()
            except:
                pass
        
        # Очистка индексов
        self.metadata_index.clear()
        self.id_index.clear()
        
        # Очистка кэша
        self.clear_cache()
        
        print("Все данные памяти были очищены!")