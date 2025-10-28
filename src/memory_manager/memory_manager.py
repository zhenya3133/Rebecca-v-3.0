"""Полноценный MemoryManager с 6 слоями памяти и AdaptiveBlueprintTracker.

Реализует комплексную систему управления памятью с:
- 6 слоями памяти (Core, Episodic, Semantic, Procedural, Vault, Security)
- Интеграцией с MemoryContext и VectorStoreClient
- Factory паттерном для создания слоев
- Кэшированием и оптимизацией
- Comprehensive логированием и error handling
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from .core_memory import CoreMemory
from .episodic_memory import EpisodicMemory
from .semantic_memory import SemanticMemory
from .procedural_memory import ProceduralMemory
from .vault_memory import VaultMemory
from .security_memory import SecurityMemory
from .memory_context import MemoryContext, MemoryEntry, CORE, EPISODIC, SEMANTIC, PROCEDURAL, VAULT, SECURITY
from .vector_store_client import VectorStoreClient, VectorStoreConfig
from .adaptive_blueprint import AdaptiveBlueprintTracker
import logging

# Константы для слоев памяти
MEMORY_LAYERS = {
    CORE: "CORE",
    EPISODIC: "EPISODIC", 
    SEMANTIC: "SEMANTIC",
    PROCEDURAL: "PROCEDURAL",
    VAULT: "VAULT",
    SECURITY: "SECURITY"
}

# Настройка логгера
logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Запись кэша памяти."""
    data: Any
    metadata: Dict[str, Any]
    timestamp: datetime
    access_count: int = 0
    last_access: datetime = field(default_factory=datetime.now)
    ttl: Optional[float] = None

    def is_expired(self) -> bool:
        """Проверяет, истек ли TTL записи."""
        if self.ttl is None:
            return False
        return (datetime.now() - self.timestamp).total_seconds() > self.ttl


@dataclass
class LayerStats:
    """Статистика слоя памяти."""
    layer_type: str
    total_items: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_access_count: int = 0
    average_access_time: float = 0.0
    last_optimization: Optional[datetime] = None
    optimization_count: int = 0
    
    @property
    def hit_ratio(self) -> float:
        """Возвращает коэффициент попаданий в кэш."""
        total_requests = self.cache_hits + self.cache_misses
        return self.cache_hits / max(1, total_requests)


class MemoryLayerFactory:
    """Фабрика для создания слоев памяти."""
    
    @staticmethod
    def create_layer(layer_type: str) -> Any:
        """Создает слой памяти указанного типа.
        
        Args:
            layer_type: Тип слоя памяти
            
        Returns:
            Экземпляр слоя памяти
            
        Raises:
            ValueError: Если тип слоя не поддерживается
        """
        if layer_type == CORE:
            return CoreMemory()
        elif layer_type == EPISODIC:
            return EpisodicMemory()
        elif layer_type == SEMANTIC:
            return SemanticMemory()
        elif layer_type == PROCEDURAL:
            return ProceduralMemory()
        elif layer_type == VAULT:
            return VaultMemory()
        elif layer_type == SECURITY:
            return SecurityMemory()
        else:
            raise ValueError(f"Неподдерживаемый тип слоя памяти: {layer_type}")


class MemoryCache:
    """Кэш для оптимизации доступа к памяти."""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        
        logger.info(f"MemoryCache инициализирован (max_size={max_size}, ttl={default_ttl}s)")
    
    async def get(self, key: str) -> Optional[Any]:
        """Получает данные из кэша."""
        async with self._lock:
            entry = self._cache.get(key)
            
            if entry and not entry.is_expired():
                # Обновляем статистику доступа
                entry.access_count += 1
                entry.last_access = datetime.now()
                
                # Перемещаем в конец (LRU)
                self._cache.move_to_end(key)
                
                logger.debug(f"Кэш попадание для ключа {key}")
                return entry.data
            else:
                # Удаляем истекшую запись
                if entry:
                    del self._cache[key]
                    logger.debug(f"Удалена истекшая запись кэша: {key}")
                
                logger.debug(f"Кэш промах для ключа {key}")
                return None
    
    async def set(self, key: str, data: Any, metadata: Dict[str, Any], ttl: Optional[float] = None) -> None:
        """Сохраняет данные в кэш."""
        async with self._lock:
            # Удаляем старую запись если есть
            if key in self._cache:
                del self._cache[key]
            
            # Создаем новую запись
            entry = CacheEntry(
                data=data,
                metadata=metadata,
                timestamp=datetime.now(),
                ttl=ttl or self.default_ttl
            )
            
            self._cache[key] = entry
            
            # Проверяем размер кэша
            while len(self._cache) > self.max_size:
                # Удаляем наименее используемую запись
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                logger.debug(f"Кэш переполнение, удален ключ {oldest_key}")
    
    async def delete(self, key: str) -> bool:
        """Удаляет данные из кэша."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"Удален ключ кэша: {key}")
                return True
            return False
    
    async def clear(self) -> int:
        """Очищает кэш."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Кэш очищен, удалено {count} записей")
            return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику кэша."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "utilization": len(self._cache) / self.max_size,
            "entries": [
                {
                    "key": key,
                    "access_count": entry.access_count,
                    "age": (datetime.now() - entry.timestamp).total_seconds(),
                    "last_access": (datetime.now() - entry.last_access).total_seconds(),
                    "ttl": entry.ttl,
                    "expired": entry.is_expired()
                }
                for key, entry in list(self._cache.items())[-10:]  # Последние 10 записей
            ]
        }


class AdaptiveBlueprintTracker:
    """Расширенный трекер изменений архитектуры."""
    
    def __init__(self, semantic_layer: SemanticMemory):
        self.semantic_layer = semantic_layer
        self.blueprint_history: List[Dict[str, Any]] = []
        self.resource_links: Dict[str, Dict[str, Any]] = {}
        self.version_counter = 0
        self._lock = asyncio.Lock()
        
        logger.info("AdaptiveBlueprintTracker инициализирован")
    
    async def record_blueprint(self, blueprint: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
        """Записывает новое состояние архитектуры.
        
        Args:
            blueprint: Данные архитектуры
            metadata: Дополнительные метаданные
            
        Returns:
            ID записанного blueprint
        """
        async with self._lock:
            self.version_counter += 1
            
            versioned_blueprint = {
                "version": self.version_counter,
                "blueprint": blueprint,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat(),
                "hash": self._calculate_blueprint_hash(blueprint)
            }
            
            # Добавляем в историю
            self.blueprint_history.append(versioned_blueprint)
            
            # Сохраняем в семантическую память
            blueprint_id = f"blueprint_v{self.version_counter}"
            await self.semantic_layer.store_concept_async(
                blueprint_id, 
                versioned_blueprint
            )
            
            logger.info(f"Записан blueprint версии {self.version_counter}")
            return blueprint_id
    
    async def get_latest_blueprint(self) -> Optional[Dict[str, Any]]:
        """Возвращает последнее состояние архитектуры."""
        if self.blueprint_history:
            return self.blueprint_history[-1]
        return None
    
    async def compare_blueprints(self, version1: int, version2: int) -> Dict[str, Any]:
        """Сравнивает две версии архитектуры.
        
        Args:
            version1: Первая версия
            version2: Вторая версия
            
        Returns:
            Словарь с результатами сравнения
        """
        if version1 < 1 or version2 < 1 or version1 > len(self.blueprint_history) or version2 > len(self.blueprint_history):
            raise ValueError(f"Неверные номера версий: {version1}, {version2}")
        
        bp1 = self.blueprint_history[version1 - 1]["blueprint"]
        bp2 = self.blueprint_history[version2 - 1]["blueprint"]
        
        # Вычисляем различия
        changes = {
            "added": self._find_added_items(bp1, bp2),
            "removed": self._find_removed_items(bp1, bp2),
            "modified": self._find_modified_items(bp1, bp2)
        }
        
        logger.info(f"Сравнены blueprint версий {version1} и {version2}")
        return changes
    
    async def link_resource(self, identifier: str, resource: Dict[str, Any], 
                          resource_type: str = "unknown") -> None:
        """Связывает ресурс с архитектурой.
        
        Args:
            identifier: Идентификатор ресурса
            resource: Данные ресурса
            resource_type: Тип ресурса
        """
        self.resource_links[identifier] = {
            "resource": resource,
            "resource_type": resource_type,
            "linked_at": datetime.now().isoformat(),
            "hash": self._calculate_blueprint_hash(resource)
        }
        
        # Сохраняем связь в семантической памяти
        link_id = f"resource::{identifier}"
        await self.semantic_layer.store_concept_async(
            link_id, 
            self.resource_links[identifier]
        )
        
        logger.info(f"Связан ресурс {identifier} типа {resource_type}")
    
    async def get_blueprint_lineage(self, max_versions: int = 10) -> List[Dict[str, Any]]:
        """Возвращает историю изменений архитектуры.
        
        Args:
            max_versions: Максимальное количество версий
            
        Returns:
            Список версий архитектуры
        """
        return self.blueprint_history[-max_versions:]
    
    def _calculate_blueprint_hash(self, blueprint: Dict[str, Any]) -> str:
        """Вычисляет хеш для blueprint."""
        blueprint_str = json.dumps(blueprint, sort_keys=True)
        return hashlib.sha256(blueprint_str.encode()).hexdigest()
    
    def _find_added_items(self, old_bp: Dict[str, Any], new_bp: Dict[str, Any]) -> List[str]:
        """Находит добавленные элементы."""
        old_keys = set(str(k) for k in self._flatten_dict(old_bp).keys())
        new_keys = set(str(k) for k in self._flatten_dict(new_bp).keys())
        return list(new_keys - old_keys)
    
    def _find_removed_items(self, old_bp: Dict[str, Any], new_bp: Dict[str, Any]) -> List[str]:
        """Находит удаленные элементы."""
        old_keys = set(str(k) for k in self._flatten_dict(old_bp).keys())
        new_keys = set(str(k) for k in self._flatten_dict(new_bp).keys())
        return list(old_keys - new_keys)
    
    def _find_modified_items(self, old_bp: Dict[str, Any], new_bp: Dict[str, Any]) -> List[str]:
        """Находит измененные элементы."""
        old_dict = self._flatten_dict(old_bp)
        new_dict = self._flatten_dict(new_bp)
        
        common_keys = set(old_dict.keys()) & set(new_dict.keys())
        modified = []
        
        for key in common_keys:
            if old_dict[key] != new_dict[key]:
                modified.append(str(key))
        
        return modified
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Разворачивает вложенный словарь."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


class MemoryManager:
    """Полноценный менеджер памяти с 6 слоями и интеграцией компонентов."""
    
    def __init__(self, 
                 cache_size: int = 1000,
                 cache_ttl: float = 3600,
                 vector_store_config: Optional[VectorStoreConfig] = None,
                 optimization_interval: float = 300):
        """Инициализирует MemoryManager.
        
        Args:
            cache_size: Размер кэша
            cache_ttl: TTL кэша в секундах
            vector_store_config: Конфигурация векторного хранилища
            optimization_interval: Интервал оптимизации в секундах
        """
        # Инициализация компонентов
        self.memory_context = MemoryContext()
        self.vector_store_client = VectorStoreClient(vector_store_config)
        self.cache = MemoryCache(cache_size, cache_ttl)
        self.blueprint_tracker = AdaptiveBlueprintTracker(self.memory_context.get_layer(SEMANTIC))
        
        # Статистика по слоям
        self.layer_stats: Dict[str, LayerStats] = {
            layer: LayerStats(layer_type=layer) for layer in MEMORY_LAYERS.keys()
        }
        
        # Настройки
        self.optimization_interval = optimization_interval
        self.last_optimization = datetime.now()
        
        # Внутренние структуры
        self._cache_index: Dict[str, str] = {}  # mapping cache_key -> original_key
        self._memory_mapping: Dict[str, Dict[str, Any]] = {}  # item_id -> memory info
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        
        # Состояние
        self._optimization_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        logger.info("MemoryManager успешно инициализирован")
    
    async def start(self) -> None:
        """Запускает MemoryManager."""
        logger.info("Запуск MemoryManager...")
        
        # Синхронизируем схему векторного хранилища
        await self.vector_store_client.sync_schema()
        
        # Запускаем задачу оптимизации
        self._optimization_task = asyncio.create_task(self._optimization_loop())
        
        logger.info("MemoryManager запущен")
    
    async def stop(self) -> None:
        """Останавливает MemoryManager."""
        logger.info("Остановка MemoryManager...")
        
        self._shutdown = True
        
        # Останавливаем задачу оптимизации
        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass
        
        # Закрываем соединения
        await self.vector_store_client.close()
        
        logger.info("MemoryManager остановлен")
    
    async def store(self, layer: str, data: Any, metadata: Optional[Dict[str, Any]] = None,
                   tags: Optional[List[str]] = None, priority: int = 5) -> str:
        """Сохраняет данные в указанный слой памяти.
        
        Args:
            layer: Слой памяти
            data: Данные для сохранения
            metadata: Дополнительные метаданные
            tags: Список тегов
            priority: Приоритет (0-10)
            
        Returns:
            ID сохраненной записи
            
        Raises:
            ValueError: Если слой памяти недопустим
        """
        if layer not in MEMORY_LAYERS:
            raise ValueError(f"Недопустимый слой памяти: {layer}")
        
        async with self._locks[layer]:
            try:
                # Создаем запись через MemoryContext
                memory_id = self.memory_context.add_memory(
                    layer_type=layer,
                    content=data,
                    metadata=metadata or {},
                    tags=tags or [],
                    priority=max(0, min(10, priority))
                )
                
                # Сохраняем в векторное хранилище
                await self._store_in_vector_store(layer, memory_id, data, metadata or {})
                
                # Кэшируем данные
                cache_key = f"{layer}:{memory_id}"
                await self.cache.set(
                    key=cache_key,
                    data=data,
                    metadata=metadata or {},
                    ttl=self._calculate_ttl(layer, priority)
                )
                
                # Обновляем статистику
                self.layer_stats[layer].total_items += 1
                self.layer_stats[layer].total_access_count += 1
                
                # Сохраняем маппинг для поиска
                self._memory_mapping[memory_id] = {
                    "layer": layer,
                    "cache_key": cache_key,
                    "metadata": metadata or {},
                    "created_at": datetime.now().isoformat()
                }
                
                logger.info(f"Данные сохранены в слой {layer}, ID: {memory_id}")
                return memory_id
                
            except Exception as e:
                logger.error(f"Ошибка сохранения в слой {layer}: {e}")
                raise RuntimeError(f"Не удалось сохранить данные в слой {layer}: {e}")
    
    async def retrieve(self, layer: str, query: Optional[str] = None, 
                      filters: Optional[Dict[str, Any]] = None,
                      limit: int = 10) -> List[Dict[str, Any]]:
        """Извлекает данные из указанного слоя памяти.
        
        Args:
            layer: Слой памяти
            query: Поисковый запрос
            filters: Фильтры для поиска
            limit: Максимальное количество результатов
            
        Returns:
            Список найденных записей
        """
        if layer not in MEMORY_LAYERS:
            raise ValueError(f"Недопустимый слой памяти: {layer}")
        
        start_time = time.time()
        async with self._locks[layer]:
            try:
                results = []
                
                # Сначала проверяем кэш
                if not query and not filters:
                    # Быстрое извлечение из MemoryContext
                    memory_entries = self.memory_context.get_memories_by_layer(layer)
                    results = [
                        {
                            "id": entry.id,
                            "content": entry.content,
                            "metadata": entry.metadata,
                            "tags": entry.tags,
                            "priority": entry.priority,
                            "timestamp": entry.timestamp.isoformat(),
                            "access_count": entry.access_count
                        }
                        for entry in memory_entries[:limit]
                    ]
                else:
                    # Поиск через векторное хранилище
                    vector_results = await self.vector_store_client.retrieve_vectors(
                        layer=layer,
                        query={
                            "text": query or "",
                            "limit": limit,
                            **(filters or {})
                        }
                    )
                    
                    # Получаем полные данные из MemoryContext
                    for vector_item in vector_results:
                        entry = self.memory_context.get_memory(vector_item["id"])
                        if entry:
                            results.append({
                                "id": entry.id,
                                "content": entry.content,
                                "metadata": entry.metadata,
                                "tags": entry.tags,
                                "priority": entry.priority,
                                "timestamp": entry.timestamp.isoformat(),
                                "access_count": entry.access_count
                            })
                
                # Обновляем статистику
                access_time = time.time() - start_time
                self._update_access_stats(layer, True, access_time)
                
                logger.debug(f"Извлечено {len(results)} записей из слоя {layer}")
                return results
                
            except Exception as e:
                self._update_access_stats(layer, False, 0)
                logger.error(f"Ошибка извлечения из слоя {layer}: {e}")
                raise RuntimeError(f"Не удалось извлечь данные из слоя {layer}: {e}")
    
    async def update(self, layer: str, item_id: str, data: Any, 
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Обновляет запись в памяти.
        
        Args:
            layer: Слой памяти
            item_id: ID записи
            data: Новые данные
            metadata: Новые метаданные
            
        Returns:
            True, если обновление прошло успешно
        """
        if layer not in MEMORY_LAYERS:
            raise ValueError(f"Недопустимый слой памяти: {layer}")
        
        async with self._locks[f"{layer}:{item_id}"]:
            try:
                # Получаем запись
                entry = self.memory_context.get_memory(item_id)
                if not entry or entry.layer_type != layer:
                    logger.warning(f"Запись {item_id} не найдена в слое {layer}")
                    return False
                
                # Обновляем данные
                updated_metadata = metadata or entry.metadata
                updated_metadata.update({
                    "updated_at": datetime.now().isoformat(),
                    "update_count": updated_metadata.get("update_count", 0) + 1
                })
                
                # Создаем новую запись с обновленными данными
                new_entry = MemoryEntry(
                    id=item_id,
                    layer_type=layer,
                    content=data,
                    metadata=updated_metadata,
                    tags=entry.tags,
                    priority=entry.priority
                )
                
                # Обновляем в MemoryContext
                self.memory_context.memory_entries[item_id] = new_entry
                
                # Обновляем в векторном хранилище
                await self.vector_store_client.update_vector(
                    layer=layer,
                    vector_id=item_id,
                    changes={
                        "text": str(data),
                        "metadata": updated_metadata
                    }
                )
                
                # Обновляем кэш
                cache_key = f"{layer}:{item_id}"
                await self.cache.set(
                    key=cache_key,
                    data=data,
                    metadata=updated_metadata
                )
                
                logger.info(f"Запись {item_id} обновлена в слое {layer}")
                return True
                
            except Exception as e:
                logger.error(f"Ошибка обновления записи {item_id} в слое {layer}: {e}")
                return False
    
    async def delete(self, layer: str, item_id: str) -> bool:
        """Удаляет запись из памяти.
        
        Args:
            layer: Слой памяти
            item_id: ID записи
            
        Returns:
            True, если удаление прошло успешно
        """
        if layer not in MEMORY_LAYERS:
            raise ValueError(f"Недопустимый слой памяти: {layer}")
        
        async with self._locks[f"{layer}:{item_id}"]:
            try:
                # Удаляем из MemoryContext
                success = self.memory_context.remove_memory(item_id)
                if not success:
                    logger.warning(f"Запись {item_id} не найдена для удаления")
                    return False
                
                # Удаляем из кэша
                cache_key = f"{layer}:{item_id}"
                await self.cache.delete(cache_key)
                
                # Удаляем из маппинга
                self._memory_mapping.pop(item_id, None)
                
                # Обновляем статистику
                if layer in self.layer_stats and self.layer_stats[layer].total_items > 0:
                    self.layer_stats[layer].total_items -= 1
                
                logger.info(f"Запись {item_id} удалена из слоя {layer}")
                return True
                
            except Exception as e:
                logger.error(f"Ошибка удаления записи {item_id} из слоя {layer}: {e}")
                return False
    
    async def search_across_layers(self, query: str, 
                                  layers: Optional[List[str]] = None,
                                  limit: int = 20) -> List[Dict[str, Any]]:
        """Ищет данные по всем слоям памяти.
        
        Args:
            query: Поисковый запрос
            layers: Список слоев для поиска (None - все слои)
            limit: Максимальное количество результатов
            
        Returns:
            Список найденных записей из всех слоев
        """
        target_layers = layers or list(MEMORY_LAYERS.keys())
        
        # Проверяем валидность слоев
        invalid_layers = [l for l in target_layers if l not in MEMORY_LAYERS]
        if invalid_layers:
            raise ValueError(f"Недопустимые слои памяти: {invalid_layers}")
        
        start_time = time.time()
        try:
            all_results = []
            per_layer_results = {}
            
            # Ищем параллельно во всех слоях
            tasks = []
            for layer in target_layers:
                task = asyncio.create_task(
                    self.retrieve(layer=layer, query=query, limit=limit)
                )
                tasks.append((layer, task))
            
            # Собираем результаты
            for layer, task in tasks:
                try:
                    results = await task
                    per_layer_results[layer] = results
                    all_results.extend(results)
                except Exception as e:
                    logger.warning(f"Ошибка поиска в слое {layer}: {e}")
                    per_layer_results[layer] = []
            
            # Сортируем результаты по релевантности (приоритет + частота доступа)
            all_results.sort(
                key=lambda x: (x.get("priority", 0), x.get("access_count", 0)),
                reverse=True
            )
            
            # Ограничиваем результаты
            final_results = all_results[:limit]
            
            # Обновляем статистику
            access_time = time.time() - start_time
            for layer in target_layers:
                self._update_access_stats(layer, True, access_time)
            
            logger.info(f"Найдено {len(final_results)} записей по запросу '{query}' "
                       f"в {len(target_layers)} слоях")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Ошибка поиска по слоям: {e}")
            raise RuntimeError(f"Не удалось выполнить поиск: {e}")
    
    async def get_layer_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику по всем слоям памяти.
        
        Returns:
            Словарь со статистикой
        """
        try:
            memory_stats = self.memory_context.get_memory_statistics()
            cache_stats = self.cache.get_stats()
            vector_health = await self.vector_store_client.health_check()
            
            # Добавляем статистику по слоям
            layer_details = {}
            for layer, stats in self.layer_stats.items():
                layer_details[layer] = {
                    "total_items": stats.total_items,
                    "cache_hit_ratio": stats.hit_ratio,
                    "total_access_count": stats.total_access_count,
                    "average_access_time": stats.average_access_time,
                    "last_optimization": stats.last_optimization.isoformat() if stats.last_optimization else None,
                    "optimization_count": stats.optimization_count
                }
            
            return {
                "memory_context": memory_stats,
                "cache": cache_stats,
                "vector_store": vector_health,
                "layer_statistics": layer_details,
                "blueprint_tracker": {
                    "version_count": self.blueprint_tracker.version_counter,
                    "resource_links": len(self.blueprint_tracker.resource_links),
                    "history_size": len(self.blueprint_tracker.blueprint_history)
                },
                "performance": {
                    "total_optimizations": sum(s.optimization_count for s in self.layer_stats.values()),
                    "cache_utilization": cache_stats["utilization"],
                    "last_optimization": self.last_optimization.isoformat()
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            raise RuntimeError(f"Не удалось получить статистику: {e}")
    
    async def optimize_memory(self) -> Dict[str, Any]:
        """Оптимизирует память и кэш.
        
        Returns:
            Словарь с результатами оптимизации
        """
        logger.info("Начало оптимизации памяти...")
        
        start_time = datetime.now()
        optimization_results = {
            "memory_optimization": {},
            "cache_optimization": {},
            "vector_optimization": {},
            "total_items_processed": 0,
            "total_items_removed": 0,
            "duration": 0
        }
        
        try:
            # Оптимизация MemoryContext
            memory_opt = self.memory_context.optimize_memory()
            optimization_results["memory_optimization"] = memory_opt
            
            # Очистка устаревших записей кэша
            cache_entries = list(self.cache._cache.items())
            expired_count = 0
            
            for key, entry in cache_entries:
                if entry.is_expired():
                    await self.cache.delete(key)
                    expired_count += 1
            
            optimization_results["cache_optimization"] = {
                "expired_entries_removed": expired_count,
                "cache_size_before": len(cache_entries),
                "cache_size_after": len(self.cache._cache)
            }
            
            # Обновляем время последней оптимизации
            self.last_optimization = start_time
            
            # Обновляем статистику по слоям
            for layer in self.layer_stats:
                self.layer_stats[layer].last_optimization = start_time
                self.layer_stats[layer].optimization_count += 1
            
            # Вычисляем общее время
            duration = (datetime.now() - start_time).total_seconds()
            optimization_results["duration"] = duration
            
            # Подсчитываем общую статистику
            optimization_results["total_items_removed"] = (
                memory_opt.get("total_optimized", 0) + 
                optimization_results["cache_optimization"]["expired_entries_removed"]
            )
            
            logger.info(f"Оптимизация завершена за {duration:.2f}s, "
                       f"удалено {optimization_results['total_items_removed']} записей")
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Ошибка оптимизации памяти: {e}")
            raise RuntimeError(f"Не удалось оптимизировать память: {e}")
    
    async def sync_with_orchestrator(self) -> Dict[str, Any]:
        """Синхронизирует состояние с оркестратором.
        
        Returns:
            Словарь с результатами синхронизации
        """
        logger.info("Синхронизация с оркестратором...")
        
        try:
            # Создаем контекстную оболочку
            trace_id = f"sync_{int(time.time())}"
            envelope = self.memory_context.build_context_envelope(trace_id)
            
            # Получаем текущий blueprint
            latest_blueprint = await self.blueprint_tracker.get_latest_blueprint()
            
            # Формируем данные для синхронизации
            sync_data = {
                "envelope": envelope,
                "blueprint": latest_blueprint,
                "statistics": await self.get_layer_statistics(),
                "vector_store_info": self.vector_store_client.get_store_info(),
                "sync_timestamp": datetime.now().isoformat(),
                "trace_id": trace_id
            }
            
            # Здесь можно добавить отправку данных в оркестратор
            # Например, через message bus или API
            
            logger.info(f"Синхронизация завершена, trace_id: {trace_id}")
            
            return {
                "success": True,
                "trace_id": trace_id,
                "sync_timestamp": sync_data["sync_timestamp"],
                "envelope_size": len(str(envelope)),
                "blueprint_version": latest_blueprint["version"] if latest_blueprint else None
            }
            
        except Exception as e:
            logger.error(f"Ошибка синхронизации с оркестратором: {e}")
            return {
                "success": False,
                "error": str(e),
                "sync_timestamp": datetime.now().isoformat()
            }
    
    # Вспомогательные методы
    
    async def _store_in_vector_store(self, layer: str, item_id: str, 
                                   data: Any, metadata: Dict[str, Any]) -> None:
        """Сохраняет данные в векторное хранилище."""
        try:
            text_content = str(data)
            vector_metadata = {
                **metadata,
                "layer": layer,
                "item_id": item_id,
                "stored_at": datetime.now().isoformat()
            }
            
            await self.vector_store_client.store_vectors(
                layer=layer,
                items=[
                    {
                        "id": item_id,
                        "text": text_content,
                        "metadata": vector_metadata
                    }
                ]
            )
        except Exception as e:
            logger.warning(f"Не удалось сохранить в векторное хранилище: {e}")
    
    def _calculate_ttl(self, layer: str, priority: int) -> float:
        """Вычисляет TTL для записи в зависимости от слоя и приоритета."""
        base_ttl = {
            CORE: 7200,       # 2 часа
            EPISODIC: 86400,  # 24 часа
            SEMANTIC: 604800, # 7 дней
            PROCEDURAL: 2592000, # 30 дней
            VAULT: 31536000,  # 1 год
            SECURITY: 7776000 # 90 дней
        }
        
        layer_ttl = base_ttl.get(layer, 3600)
        
        # Увеличиваем TTL для высокоприоритетных записей
        if priority >= 8:
            return layer_ttl * 2
        elif priority <= 2:
            return layer_ttl * 0.5
        else:
            return layer_ttl
    
    def _update_access_stats(self, layer: str, is_hit: bool, access_time: float) -> None:
        """Обновляет статистику доступа к слою."""
        if layer in self.layer_stats:
            stats = self.layer_stats[layer]
            
            if is_hit:
                stats.cache_hits += 1
            else:
                stats.cache_misses += 1
            
            # Обновляем среднее время доступа
            if access_time > 0:
                current_avg = stats.average_access_time
                total_requests = stats.cache_hits + stats.cache_misses
                stats.average_access_time = (current_avg * (total_requests - 1) + access_time) / total_requests
    
    async def _optimization_loop(self) -> None:
        """Цикл автоматической оптимизации."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.optimization_interval)
                
                if self._shutdown:
                    break
                
                logger.debug("Выполнение автоматической оптимизации...")
                await self.optimize_memory()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ошибка в цикле оптимизации: {e}")
                await asyncio.sleep(60)  # Ждем минуту перед повтором


# Функции для удобного использования

def create_memory_manager(config: Optional[Dict[str, Any]] = None) -> MemoryManager:
    """Создает MemoryManager с конфигурацией.
    
    Args:
        config: Конфигурация MemoryManager
        
    Returns:
        Экземпляр MemoryManager
    """
    config = config or {}
    
    vector_config = None
    if "vector_store" in config:
        vector_config = VectorStoreConfig(**config["vector_store"])
    
    return MemoryManager(
        cache_size=config.get("cache_size", 1000),
        cache_ttl=config.get("cache_ttl", 3600),
        vector_store_config=vector_config,
        optimization_interval=config.get("optimization_interval", 300)
    )


async def quick_memory_test() -> Dict[str, Any]:
    """Быстрый тест MemoryManager."""
    manager = create_memory_manager()
    
    try:
        await manager.start()
        
        # Сохраняем тестовые данные
        core_id = await manager.store(CORE, "Тестовый факт", {"source": "test"})
        semantic_id = await manager.store(SEMANTIC, "Концепция AI", {"domain": "technology"})
        
        # Извлекаем данные
        core_data = await manager.retrieve(CORE)
        semantic_data = await manager.retrieve(SEMANTIC)
        
        # Получаем статистику
        stats = await manager.get_layer_statistics()
        
        return {
            "success": True,
            "stored_items": [core_id, semantic_id],
            "retrieved_core": len(core_data),
            "retrieved_semantic": len(semantic_data),
            "statistics": stats
        }
        
    finally:
        await manager.stop()


# Экспорт основных классов
__all__ = [
    "MemoryManager",
    "AdaptiveBlueprintTracker", 
    "MemoryLayerFactory",
    "MemoryCache",
    "LayerStats",
    "CacheEntry",
    "create_memory_manager",
    "quick_memory_test"
]