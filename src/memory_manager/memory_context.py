"""Context orchestration for layered memories.

Inspired by mem0's modular memory abstractions and Weaviate's object access
patterns. This module handles routing requests across Rebecca's
Core, Episodic, Semantic, Procedural, Vault, and Security layers.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

# Константы для типов memory layers
CORE = "CORE"
EPISODIC = "EPISODIC"
SEMANTIC = "SEMANTIC"
PROCEDURAL = "PROCEDURAL"
VAULT = "VAULT"
SECURITY = "SECURITY"

# Все доступные типы слоев памяти
MEMORY_LAYER_TYPES = {
    CORE,
    EPISODIC,
    SEMANTIC,
    PROCEDURAL,
    VAULT,
    SECURITY
}

# Настройка логгера
logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """Запись в памяти для слоев CORE, EPISODIC и других."""
    id: str
    layer_type: str
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    priority: int = 0
    access_count: int = 0


@dataclass
class MemoryContext:
    """Контекст для оркестрации слоев памяти.
    
    Управляет регистрацией адаптеров слоев памяти, созданием контекстных
    оболочек для отслеживания состояния и обеспечивает единый интерфейс
    для работы с различными типами памяти.
    
    TODO: Интеграция с orchestrator ContextHandler для синхронизации
    состояния между агентами по trace_id.
    TODO: Реализация cache/TTL поведения аналогично модулям памяти mem0.
    """

    layers: Dict[str, Any] = field(default_factory=dict)
    memory_entries: Dict[str, MemoryEntry] = field(default_factory=dict)
    trace_history: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _registered_layer_types: set = field(default_factory=set)

    def __post_init__(self):
        """Инициализация после создания объекта."""
        self._initialize_default_layers()
        logger.info("MemoryContext инициализирован")

    def _initialize_default_layers(self):
        """Инициализация стандартных слоев памяти."""
        try:
            # Импортируем и регистрируем стандартные адаптеры памяти
            from .core_memory import CoreMemory
            from .episodic_memory import EpisodicMemory
            from .semantic_memory import SemanticMemory
            from .procedural_memory import ProceduralMemory
            from .vault_memory import VaultMemory
            from .security_memory import SecurityMemory

            # Регистрируем стандартные слои
            self.register_layer(CORE, CoreMemory())
            self.register_layer(EPISODIC, EpisodicMemory())
            self.register_layer(SEMANTIC, SemanticMemory())
            self.register_layer(PROCEDURAL, ProceduralMemory())
            self.register_layer(VAULT, VaultMemory())
            self.register_layer(SECURITY, SecurityMemory())

            logger.info("Стандартные слои памяти успешно инициализированы")
            
        except ImportError as e:
            logger.warning(f"Не удалось импортировать стандартные слои памяти: {e}")
        except Exception as e:
            logger.error(f"Ошибка инициализации стандартных слоев: {e}")

    def register_layer(self, name: str, adapter: Any) -> None:
        """Регистрация адаптера для слоя памяти.
        
        Args:
            name: Имя слоя памяти (должно быть из MEMORY_LAYER_TYPES)
            adapter: Объект адаптера для работы с данным слоем
            
        Raises:
            ValueError: Если имя слоя недопустимо
            TypeError: Если адаптер не является объектом
        """
        if name not in MEMORY_LAYER_TYPES:
            raise ValueError(f"Недопустимый тип слоя памяти: {name}. "
                           f"Допустимые типы: {MEMORY_LAYER_TYPES}")
        
        if not isinstance(adapter, object):
            raise TypeError(f"Адаптер должен быть объектом, получен: {type(adapter)}")
        
        if name in self.layers:
            logger.warning(f"Слой памяти '{name}' уже зарегистрирован, перезапись")
        
        self.layers[name] = adapter
        self._registered_layer_types.add(name)
        logger.info(f"Слой памяти '{name}' успешно зарегистрирован")

    def get_layer(self, name: str) -> Optional[Any]:
        """Получение адаптера для слоя памяти.
        
        Args:
            name: Имя слоя памяти
            
        Returns:
            Адаптер слоя памяти или None, если слой не найден
            
        Raises:
            ValueError: Если имя слоя пустое или недопустимое
        """
        if not name or not isinstance(name, str):
            raise ValueError("Имя слоя памяти должно быть непустой строкой")
        
        if name not in MEMORY_LAYER_TYPES:
            logger.warning(f"Запрос неизвестного слоя памяти: {name}")
            return None
        
        layer = self.layers.get(name)
        if layer is None:
            logger.warning(f"Слой памяти '{name}' не зарегистрирован")
            return None
        
        logger.debug(f"Слой памяти '{name}' успешно получен")
        return layer

    def add_memory(self, layer_type: str, content: Any, 
                   metadata: Optional[Dict[str, Any]] = None,
                   tags: Optional[List[str]] = None,
                   priority: int = 0) -> str:
        """Добавление записи в память.
        
        Args:
            layer_type: Тип слоя памяти
            content: Содержимое записи
            metadata: Дополнительные метаданные
            tags: Список тегов
            priority: Приоритет записи (0-10)
            
        Returns:
            ID созданной записи
            
        Raises:
            ValueError: Если тип слоя памяти недопустим
            RuntimeError: Если слой памяти не зарегистрирован
        """
        if layer_type not in MEMORY_LAYER_TYPES:
            raise ValueError(f"Недопустимый тип слоя памяти: {layer_type}")
        
        layer = self.get_layer(layer_type)
        if layer is None:
            raise RuntimeError(f"Слой памяти '{layer_type}' не зарегистрирован")
        
        # Создаем запись в памяти
        memory_id = str(uuid.uuid4())
        entry = MemoryEntry(
            id=memory_id,
            layer_type=layer_type,
            content=content,
            metadata=metadata or {},
            tags=tags or [],
            priority=max(0, min(10, priority))
        )
        
        # Сохраняем запись
        self.memory_entries[memory_id] = entry
        
        # Добавляем в соответствующий слой памяти
        try:
            if layer_type == CORE:
                # CoreMemory
                layer.store_fact(memory_id, content)
            elif layer_type == EPISODIC:
                # EpisodicMemory
                layer.store_event({
                    'id': memory_id,
                    'content': content,
                    'metadata': metadata or {},
                    'timestamp': entry.timestamp.isoformat()
                })
            elif layer_type == SEMANTIC:
                # SemanticMemory
                layer.store_concept(memory_id, content)
            elif layer_type == PROCEDURAL:
                # ProceduralMemory
                layer.store_workflow(memory_id, content)
            elif layer_type == VAULT:
                # VaultMemory
                layer.store_secret(memory_id, content)
            elif layer_type == SECURITY:
                # SecurityMemory
                layer.store_audit({
                    'id': memory_id,
                    'content': content,
                    'metadata': metadata or {},
                    'timestamp': entry.timestamp.isoformat()
                })
            
            logger.info(f"Запись памяти '{memory_id}' добавлена в слой '{layer_type}'")
            return memory_id
            
        except Exception as e:
            # Удаляем запись в случае ошибки
            self.memory_entries.pop(memory_id, None)
            logger.error(f"Ошибка добавления записи в слой '{layer_type}': {e}")
            raise RuntimeError(f"Не удалось добавить запись в слой '{layer_type}': {e}")

    def get_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Получение записи из памяти.
        
        Args:
            memory_id: ID записи в памяти
            
        Returns:
            Запись из памяти или None, если не найдена
        """
        entry = self.memory_entries.get(memory_id)
        if entry:
            entry.access_count += 1
            logger.debug(f"Запись памяти '{memory_id}' получена")
        else:
            logger.warning(f"Запись памяти '{memory_id}' не найдена")
        return entry

    def get_memories_by_layer(self, layer_type: str) -> List[MemoryEntry]:
        """Получение всех записей для определенного слоя памяти.
        
        Args:
            layer_type: Тип слоя памяти
            
        Returns:
            Список записей для указанного слоя
        """
        if layer_type not in MEMORY_LAYER_TYPES:
            logger.warning(f"Неизвестный тип слоя памяти: {layer_type}")
            return []
        
        entries = [
            entry for entry in self.memory_entries.values()
            if entry.layer_type == layer_type
        ]
        
        logger.info(f"Найдено {len(entries)} записей для слоя '{layer_type}'")
        return sorted(entries, key=lambda x: (x.priority, x.timestamp), reverse=True)

    def remove_memory(self, memory_id: str) -> bool:
        """Удаление записи из памяти.
        
        Args:
            memory_id: ID записи в памяти
            
        Returns:
            True, если запись была удалена, False иначе
        """
        entry = self.memory_entries.pop(memory_id, None)
        if entry:
            logger.info(f"Запись памяти '{memory_id}' удалена из слоя '{entry.layer_type}'")
            return True
        else:
            logger.warning(f"Запись памяти '{memory_id}' не найдена для удаления")
            return False

    def search_memories(self, query: str, layer_type: Optional[str] = None,
                       tags: Optional[List[str]] = None,
                       limit: int = 10) -> List[MemoryEntry]:
        """Поиск записей в памяти.
        
        Args:
            query: Поисковый запрос (ищется в содержимом)
            layer_type: Фильтр по типу слоя памяти
            tags: Фильтр по тегам
            limit: Максимальное количество результатов
            
        Returns:
            Список найденных записей, отсортированных по релевантности
        """
        results = []
        
        for entry in self.memory_entries.values():
            # Фильтр по типу слоя
            if layer_type and entry.layer_type != layer_type:
                continue
            
            # Фильтр по тегам
            if tags:
                if not any(tag in entry.tags for tag in tags):
                    continue
            
            # Поиск в содержимом
            query_lower = query.lower()
            content_str = str(entry.content).lower()
            metadata_str = str(entry.metadata).lower()
            
            if query_lower in content_str or query_lower in metadata_str:
                results.append(entry)
        
        # Сортировка по приоритету и времени доступа
        results.sort(key=lambda x: (x.priority, x.access_count, x.timestamp), reverse=True)
        
        # Ограничение количества результатов
        limited_results = results[:limit]
        
        logger.info(f"Найдено {len(limited_results)} записей по запросу '{query}'")
        return limited_results

    def build_context_envelope(self, trace_id: str) -> Dict[str, Any]:
        """Создание контекстной оболочки для trace_id.
        
        Собирает текущее состояние всех слоев памяти в единую структуру,
        которая может быть передана между агентами и компонентами системы.
        
        Args:
            trace_id: Идентификатор трассировки
            
        Returns:
            Словарь с контекстной информацией
            
        Raises:
            ValueError: Если trace_id пустой
        """
        if not trace_id or not isinstance(trace_id, str):
            raise ValueError("trace_id должен быть непустой строкой")
        
        try:
            # Собираем состояние всех слоев памяти
            layers_state = {}
            for layer_name, layer_adapter in self.layers.items():
                layer_data = {
                    'adapter_type': type(layer_adapter).__name__,
                    'state': {},
                    'last_updated': datetime.now().isoformat()
                }
                
                try:
                    # Получаем состояние конкретного слоя
                    if hasattr(layer_adapter, 'data'):
                        layer_data['state'] = layer_adapter.data
                    elif hasattr(layer_adapter, '_events'):
                        layer_data['state'] = layer_adapter._events
                    elif hasattr(layer_adapter, 'concepts'):
                        layer_data['state'] = layer_adapter.concepts
                    elif hasattr(layer_adapter, 'procedures'):
                        layer_data['state'] = layer_adapter.procedures
                    elif hasattr(layer_adapter, 'secrets'):
                        layer_data['state'] = layer_adapter.secrets
                    elif hasattr(layer_adapter, 'rules'):
                        layer_data['state'] = layer_adapter.rules
                    
                    layers_state[layer_name] = layer_data
                    
                except Exception as e:
                    logger.warning(f"Ошибка получения состояния слоя '{layer_name}': {e}")
                    layers_state[layer_name] = layer_data
            
            # Создаем контекстную оболочку
            envelope = {
                'trace_id': trace_id,
                'timestamp': datetime.now().isoformat(),
                'layers_state': layers_state,
                'memory_entries': {
                    entry_id: {
                        'id': entry.id,
                        'layer_type': entry.layer_type,
                        'content': entry.content,
                        'metadata': entry.metadata,
                        'timestamp': entry.timestamp.isoformat(),
                        'tags': entry.tags,
                        'priority': entry.priority,
                        'access_count': entry.access_count
                    }
                    for entry_id, entry in self.memory_entries.items()
                },
                'context_metadata': {
                    'total_layers': len(self.layers),
                    'total_memory_entries': len(self.memory_entries),
                    'registered_layer_types': list(self._registered_layer_types),
                    'envelope_version': '1.0'
                }
            }
            
            # Сохраняем историю trace
            self.trace_history[trace_id] = envelope
            
            logger.info(f"Контекстная оболочка создана для trace_id '{trace_id}'")
            return envelope
            
        except Exception as e:
            logger.error(f"Ошибка создания контекстной оболочки для trace_id '{trace_id}': {e}")
            raise RuntimeError(f"Не удалось создать контекстную оболочку: {e}")

    def hydrate_from_envelope(self, envelope: Dict[str, Any]) -> None:
        """Восстановление состояния из контекстной оболочки.
        
        Загружает состояние слоев памяти и записей из контекстной оболочки,
        синхронизируя состояние с другими компонентами системы.
        
        Args:
            envelope: Контекстная оболочка для восстановления состояния
            
        Raises:
            ValueError: Если оболочка имеет неправильный формат
            RuntimeError: Если произошла ошибка восстановления
        """
        if not isinstance(envelope, dict):
            raise ValueError("Оболочка должна быть словарем")
        
        required_keys = ['layers_state', 'memory_entries', 'trace_id']
        if not all(key in envelope for key in required_keys):
            raise ValueError(f"Оболочка должна содержать ключи: {required_keys}")
        
        try:
            trace_id = envelope['trace_id']
            logger.info(f"Начало восстановления состояния для trace_id '{trace_id}'")
            
            # Восстанавливаем состояние слоев памяти
            layers_state = envelope.get('layers_state', {})
            for layer_name, layer_data in layers_state.items():
                layer_adapter = self.get_layer(layer_name)
                if layer_adapter and 'state' in layer_data:
                    try:
                        # Восстанавливаем состояние слоя в зависимости от типа
                        if layer_name == CORE and hasattr(layer_adapter, 'data'):
                            layer_adapter.data.update(layer_data['state'])
                        elif layer_name == EPISODIC and hasattr(layer_adapter, '_events'):
                            layer_adapter._events = layer_data['state']
                        elif layer_name == SEMANTIC and hasattr(layer_adapter, 'concepts'):
                            layer_adapter.concepts.update(layer_data['state'])
                        elif layer_name == PROCEDURAL and hasattr(layer_adapter, 'workflows'):
                            layer_adapter.workflows.update(layer_data['state'])
                        elif layer_name == VAULT and hasattr(layer_adapter, 'secrets'):
                            layer_adapter.secrets.update(layer_data['state'])
                        elif layer_name == SECURITY and hasattr(layer_adapter, 'audits'):
                            layer_adapter.audits = layer_data['state']
                        
                        logger.debug(f"Состояние слоя '{layer_name}' восстановлено")
                        
                    except Exception as e:
                        logger.warning(f"Ошибка восстановления состояния слоя '{layer_name}': {e}")
            
            # Восстанавливаем записи памяти
            memory_entries_data = envelope.get('memory_entries', {})
            restored_count = 0
            
            for entry_data in memory_entries_data.values():
                try:
                    entry = MemoryEntry(
                        id=entry_data['id'],
                        layer_type=entry_data['layer_type'],
                        content=entry_data['content'],
                        metadata=entry_data.get('metadata', {}),
                        timestamp=datetime.fromisoformat(entry_data['timestamp']),
                        tags=entry_data.get('tags', []),
                        priority=entry_data.get('priority', 0),
                        access_count=entry_data.get('access_count', 0)
                    )
                    self.memory_entries[entry.id] = entry
                    restored_count += 1
                    
                except Exception as e:
                    logger.warning(f"Ошибка восстановления записи памяти '{entry_data.get('id', 'unknown')}': {e}")
            
            # Сохраняем восстановленную оболочку в историю
            self.trace_history[trace_id] = envelope
            
            logger.info(f"Состояние восстановлено для trace_id '{trace_id}': "
                       f"{len(layers_state)} слоев, {restored_count} записей памяти")
            
        except Exception as e:
            logger.error(f"Ошибка восстановления состояния из оболочки: {e}")
            raise RuntimeError(f"Не удалось восстановить состояние: {e}")

    def get_trace_history(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Получение истории trace_id.
        
        Args:
            trace_id: Идентификатор трассировки
            
        Returns:
            История trace или None, если не найдена
        """
        history = self.trace_history.get(trace_id)
        if history:
            logger.debug(f"История trace_id '{trace_id}' найдена")
        else:
            logger.warning(f"История trace_id '{trace_id}' не найдена")
        return history

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Получение статистики по памяти.
        
        Returns:
            Словарь со статистикой использования памяти
        """
        stats = {
            'total_memory_entries': len(self.memory_entries),
            'total_layers': len(self.layers),
            'registered_layer_types': list(self._registered_layer_types),
            'total_traces': len(self.trace_history),
            'layer_statistics': {},
            'memory_distribution': {}
        }
        
        # Статистика по слоям памяти
        for layer_type in MEMORY_LAYER_TYPES:
            layer_entries = [
                entry for entry in self.memory_entries.values()
                if entry.layer_type == layer_type
            ]
            stats['layer_statistics'][layer_type] = {
                'count': len(layer_entries),
                'total_access_count': sum(entry.access_count for entry in layer_entries),
                'avg_priority': sum(entry.priority for entry in layer_entries) / max(1, len(layer_entries))
            }
            stats['memory_distribution'][layer_type] = len(layer_entries)
        
        logger.debug("Статистика памяти получена")
        return stats

    def clear_layer(self, layer_type: str) -> int:
        """Очистка всех записей в указанном слое памяти.
        
        Args:
            layer_type: Тип слоя памяти для очистки
            
        Returns:
            Количество удаленных записей
        """
        if layer_type not in MEMORY_LAYER_TYPES:
            raise ValueError(f"Неизвестный тип слоя памяти: {layer_type}")
        
        # Находим и удаляем записи
        entries_to_remove = [
            entry_id for entry_id, entry in self.memory_entries.items()
            if entry.layer_type == layer_type
        ]
        
        removed_count = 0
        for entry_id in entries_to_remove:
            if self.remove_memory(entry_id):
                removed_count += 1
        
        logger.info(f"Слой памяти '{layer_type}' очищен: {removed_count} записей удалено")
        return removed_count

    def optimize_memory(self) -> Dict[str, int]:
        """Оптимизация памяти: удаление старых и редко используемых записей.
        
        Returns:
            Словарь с информацией об оптимизации
        """
        current_time = datetime.now()
        optimization_results = {
            'expired_entries_removed': 0,
            'low_priority_removed': 0,
            'unused_entries_removed': 0,
            'total_optimized': 0
        }
        
        entries_to_remove = []
        
        for entry in self.memory_entries.values():
            should_remove = False
            reason = None
            
            # Удаляем записи с нулевым приоритетом и без доступа
            if entry.priority == 0 and entry.access_count == 0:
                should_remove = True
                reason = 'unused'
            
            # Удаляем очень старые записи (старше 30 дней)
            if (current_time - entry.timestamp).days > 30 and entry.priority <= 2:
                should_remove = True
                reason = 'expired'
            
            if should_remove:
                entries_to_remove.append(entry.id)
                if reason == 'expired':
                    optimization_results['expired_entries_removed'] += 1
                elif reason == 'unused':
                    optimization_results['unused_entries_removed'] += 1
        
        # Удаляем найденные записи
        for entry_id in entries_to_remove:
            self.remove_memory(entry_id)
        
        optimization_results['total_optimized'] = len(entries_to_remove)
        
        logger.info(f"Оптимизация памяти завершена: {optimization_results['total_optimized']} записей удалено")
        return optimization_results
