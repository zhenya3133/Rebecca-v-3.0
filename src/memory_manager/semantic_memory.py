"""
Расширенная семантическая память для Rebecca-Platform

Содержит концепты и их взаимосвязи, извлеченные из текстовых данных.
Интегрируется с системой извлечения концептов для автоматического
обновления и структурирования знаний.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Set
from datetime import datetime

logger = logging.getLogger(__name__)


class SemanticMemory:
    """Расширенная семантическая память с асинхронными методами."""
    
    def __init__(self):
        """Инициализация семантической памяти."""
        self.concepts: Dict[str, Any] = {}
        self.concept_relationships: Dict[str, List[Dict[str, Any]]] = {}
        self.concept_frequencies: Dict[str, int] = {}
        self.last_access: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()
        
        logger.info("SemanticMemory инициализирована")
    
    def store_concept(self, name: str, description: Any) -> None:
        """Синхронное сохранение концепта (legacy метод).
        
        Args:
            name: Имя концепта
            description: Описание концепта
        """
        self.concepts[name] = description
        self.concept_frequencies[name] = self.concept_frequencies.get(name, 0) + 1
        self.last_access[name] = datetime.now()
        logger.debug(f"Сохранен концепт: {name}")
    
    def get_concept(self, name: str) -> Optional[Any]:
        """Синхронное получение концепта (legacy метод).
        
        Args:
            name: Имя концепта
            
        Returns:
            Описание концепта или None
        """
        if name in self.concepts:
            self.last_access[name] = datetime.now()
        return self.concepts.get(name)
    
    async def store_concept_async(self, name: str, data: Any, 
                                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """Асинхронное сохранение концепта с метаданными.
        
        Args:
            name: Имя концепта
            data: Данные концепта
            metadata: Дополнительные метаданные
            
        Returns:
            ID сохраненного концепта
        """
        async with self._lock:
            try:
                concept_id = f"concept_{len(self.concepts) + 1}"
                
                concept_data = {
                    'id': concept_id,
                    'name': name,
                    'data': data,
                    'metadata': metadata or {},
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat(),
                    'access_count': 0
                }
                
                self.concepts[concept_id] = concept_data
                self.concept_frequencies[concept_id] = 1
                self.last_access[concept_id] = datetime.now()
                
                logger.info(f"Сохранен концепт {name} с ID {concept_id}")
                return concept_id
                
            except Exception as e:
                logger.error(f"Ошибка сохранения концепта {name}: {e}")
                raise
    
    async def get_concept_async(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Асинхронное получение концепта по ID.
        
        Args:
            concept_id: ID концепта
            
        Returns:
            Данные концепта или None
        """
        async with self._lock:
            if concept_id in self.concepts:
                concept = self.concepts[concept_id]
                concept['access_count'] += 1
                self.last_access[concept_id] = datetime.now()
                return concept
            
            return None
    
    async def update_concept(self, concept_id: str, data: Any, 
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Обновление концепта.
        
        Args:
            concept_id: ID концепта
            data: Новые данные
            metadata: Новые метаданные
            
        Returns:
            True, если обновление прошло успешно
        """
        async with self._lock:
            if concept_id not in self.concepts:
                return False
            
            try:
                concept = self.concepts[concept_id]
                concept['data'] = data
                if metadata:
                    concept['metadata'].update(metadata)
                concept['updated_at'] = datetime.now().isoformat()
                
                self.concept_frequencies[concept_id] = self.concept_frequencies.get(concept_id, 0) + 1
                self.last_access[concept_id] = datetime.now()
                
                logger.debug(f"Обновлен концепт {concept_id}")
                return True
                
            except Exception as e:
                logger.error(f"Ошибка обновления концепта {concept_id}: {e}")
                return False
    
    async def delete_concept(self, concept_id: str) -> bool:
        """Удаление концепта.
        
        Args:
            concept_id: ID концепта
            
        Returns:
            True, если удаление прошло успешно
        """
        async with self._lock:
            if concept_id not in self.concepts:
                return False
            
            try:
                del self.concepts[concept_id]
                self.concept_frequencies.pop(concept_id, None)
                self.last_access.pop(concept_id, None)
                self.concept_relationships.pop(concept_id, None)
                
                # Удаляем связи, где этот концепт является источником или целью
                for cid, relationships in self.concept_relationships.items():
                    self.concept_relationships[cid] = [
                        rel for rel in relationships 
                        if rel.get('source_id') != concept_id and rel.get('target_id') != concept_id
                    ]
                
                logger.info(f"Удален концепт {concept_id}")
                return True
                
            except Exception as e:
                logger.error(f"Ошибка удаления концепта {concept_id}: {e}")
                return False
    
    async def add_relationship(self, source_id: str, target_id: str, 
                             relationship_type: str, 
                             metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Добавление связи между концептами.
        
        Args:
            source_id: ID исходного концепта
            target_id: ID целевого концепта
            relationship_type: Тип связи
            metadata: Дополнительные метаданные
            
        Returns:
            True, если связь добавлена успешно
        """
        async with self._lock:
            if source_id not in self.concepts or target_id not in self.concepts:
                return False
            
            try:
                relationship = {
                    'id': f"rel_{len(self.concept_relationships.get(source_id, [])) + 1}",
                    'source_id': source_id,
                    'target_id': target_id,
                    'type': relationship_type,
                    'metadata': metadata or {},
                    'created_at': datetime.now().isoformat()
                }
                
                if source_id not in self.concept_relationships:
                    self.concept_relationships[source_id] = []
                
                self.concept_relationships[source_id].append(relationship)
                
                logger.debug(f"Добавлена связь {source_id} -> {target_id} типа {relationship_type}")
                return True
                
            except Exception as e:
                logger.error(f"Ошибка добавления связи: {e}")
                return False
    
    async def get_concept_relationships(self, concept_id: str) -> List[Dict[str, Any]]:
        """Получение всех связей концепта.
        
        Args:
            concept_id: ID концепта
            
        Returns:
            Список связей
        """
        async with self._lock:
            return self.concept_relationships.get(concept_id, [])
    
    async def search_concepts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Поиск концептов по запросу.
        
        Args:
            query: Поисковый запрос
            limit: Максимальное количество результатов
            
        Returns:
            Список найденных концептов
        """
        async with self._lock:
            results = []
            query_lower = query.lower()
            
            for concept_id, concept in self.concepts.items():
                # Поиск в имени
                if query_lower in concept.get('name', '').lower():
                    results.append(concept)
                    continue
                
                # Поиск в данных (если это строка)
                data = concept.get('data')
                if isinstance(data, str) and query_lower in data.lower():
                    results.append(concept)
                    continue
                
                # Поиск в метаданных
                metadata = concept.get('metadata', {})
                for value in metadata.values():
                    if isinstance(value, str) and query_lower in value.lower():
                        results.append(concept)
                        break
            
            # Сортируем по частоте доступа и времени последнего доступа
            results.sort(
                key=lambda x: (
                    self.concept_frequencies.get(x['id'], 0),
                    self.last_access.get(x['id'], datetime.min)
                ),
                reverse=True
            )
            
            return results[:limit]
    
    async def get_concept_statistics(self) -> Dict[str, Any]:
        """Получение статистики семантической памяти.
        
        Returns:
            Словарь со статистикой
        """
        async with self._lock:
            total_concepts = len(self.concepts)
            total_relationships = sum(len(rels) for rels in self.concept_relationships.values())
            
            # Топ концептов по частоте
            top_concepts = sorted(
                self.concept_frequencies.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            # Недавно используемые концепты
            recent_concepts = sorted(
                self.last_access.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            return {
                'total_concepts': total_concepts,
                'total_relationships': total_relationships,
                'avg_relationships_per_concept': (
                    total_relationships / total_concepts if total_concepts > 0 else 0
                ),
                'top_concepts_by_frequency': [
                    {
                        'concept_id': cid,
                        'frequency': freq,
                        'name': self.concepts[cid].get('name', 'Unknown')
                    }
                    for cid, freq in top_concepts
                    if cid in self.concepts
                ],
                'recent_concepts': [
                    {
                        'concept_id': cid,
                        'last_access': access_time.isoformat(),
                        'name': self.concepts[cid].get('name', 'Unknown')
                    }
                    for cid, access_time in recent_concepts
                    if cid in self.concepts
                ],
                'memory_utilization': {
                    'concepts_size': len(str(self.concepts)),
                    'relationships_size': len(str(self.concept_relationships))
                },
                'timestamp': datetime.now().isoformat()
            }
    
    async def optimize_memory(self) -> Dict[str, Any]:
        """Оптимизация семантической памяти.
        
        Returns:
            Словарь с результатами оптимизации
        """
        async with self._lock:
            optimization_results = {
                'concepts_optimized': 0,
                'relationships_optimized': 0,
                'old_concepts_removed': 0,
                'orphaned_relationships_removed': 0
            }
            
            # Удаляем концепты, к которым нет обращений более 30 дней
            cutoff_date = datetime.now()
            concepts_to_remove = []
            
            for concept_id, last_access in self.last_access.items():
                if (cutoff_date - last_access).days > 30:
                    concepts_to_remove.append(concept_id)
            
            for concept_id in concepts_to_remove:
                await self.delete_concept(concept_id)
                optimization_results['old_concepts_removed'] += 1
            
            # Очищаем висячие связи
            valid_concept_ids = set(self.concepts.keys())
            for source_id, relationships in self.concept_relationships.items():
                original_count = len(relationships)
                self.concept_relationships[source_id] = [
                    rel for rel in relationships
                    if rel.get('target_id') in valid_concept_ids
                ]
                optimization_results['orphaned_relationships_removed'] += (
                    original_count - len(self.concept_relationships[source_id])
                )
            
            optimization_results['concepts_optimized'] = len(self.concepts)
            optimization_results['total_relationships'] = sum(
                len(rels) for rels in self.concept_relationships.values()
            )
            
            logger.info(f"Оптимизация завершена: {optimization_results}")
            return optimization_results
    
    async def export_knowledge_graph(self) -> Dict[str, Any]:
        """Экспорт графа знаний в формате для визуализации.
        
        Returns:
            Словарь с узлами и связями графа
        """
        async with self._lock:
            # Создаем список узлов
            nodes = []
            for concept_id, concept in self.concepts.items():
                node = {
                    'id': concept_id,
                    'label': concept.get('name', 'Unknown'),
                    'data': concept.get('data', {}),
                    'metadata': concept.get('metadata', {}),
                    'frequency': self.concept_frequencies.get(concept_id, 0)
                }
                nodes.append(node)
            
            # Создаем список связей
            edges = []
            for source_id, relationships in self.concept_relationships.items():
                for rel in relationships:
                    edge = {
                        'source': source_id,
                        'target': rel.get('target_id'),
                        'type': rel.get('type', 'UNKNOWN'),
                        'metadata': rel.get('metadata', {}),
                        'created_at': rel.get('created_at')
                    }
                    edges.append(edge)
            
            return {
                'nodes': nodes,
                'edges': edges,
                'statistics': {
                    'total_nodes': len(nodes),
                    'total_edges': len(edges),
                    'export_timestamp': datetime.now().isoformat()
                }
            }


# Функции для совместимости с существующим API

async def create_semantic_memory() -> SemanticMemory:
    """Создает экземпляр SemanticMemory.
    
    Returns:
        Экземпляр SemanticMemory
    """
    return SemanticMemory()


async def quick_semantic_test() -> Dict[str, Any]:
    """Быстрый тест SemanticMemory."""
    semantic_memory = await create_semantic_memory()
    
    try:
        # Сохраняем тестовые концепты
        concept_id1 = await semantic_memory.store_concept_async(
            "Искусственный интеллект", 
            {"definition": "Системы, имитирующие человеческий интеллект"},
            {"domain": "technology", "importance": "high"}
        )
        
        concept_id2 = await semantic_memory.store_concept_async(
            "Машинное обучение",
            {"definition": "Подраздел ИИ для обучения алгоритмов"},
            {"domain": "technology", "importance": "high"}
        )
        
        # Добавляем связь
        await semantic_memory.add_relationship(
            concept_id2, concept_id1, "PART_OF"
        )
        
        # Получаем статистику
        stats = await semantic_memory.get_concept_statistics()
        
        return {
            "success": True,
            "concept_ids": [concept_id1, concept_id2],
            "statistics": stats
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# Экспорт основных классов и функций
__all__ = [
    'SemanticMemory',
    'create_semantic_memory',
    'quick_semantic_test'
]
