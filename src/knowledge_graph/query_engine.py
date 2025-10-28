"""
Модуль QueryEngine - обеспечивает интерфейс для запросов к графу знаний.
Реализует различные типы запросов для извлечения структурированных знаний.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Any, Union, Tuple
from abc import ABC, abstractmethod
from enum import Enum

from .kag_graph import KAGGraph, Concept, Relationship, RelationshipType, QueryResult
from .graph_traversal import GraphTraversal, TraversalType


class QueryType(Enum):
    """Типы запросов к графу."""
    FIND_CONCEPTS = "find_concepts"           # Поиск концептов
    FIND_RELATIONSHIPS = "find_relationships" # Поиск отношений
    TRAVERSE_PATH = "traverse_path"          # Поиск пути
    SIMILARITY_SEARCH = "similarity_search"  # Поиск похожих концептов
    RELATIONSHIP_DISCOVERY = "relationship_discovery"  # Открытие новых отношений
    CONCEPT_ANALYSIS = "concept_analysis"    # Анализ концепта
    GRAPH_ANALYSIS = "graph_analysis"        # Анализ графа


@dataclass
class QueryContext:
    """Контекст выполнения запроса."""
    query_type: QueryType
    parameters: Dict[str, Any]
    timestamp: float
    execution_time: float = 0.0
    confidence_threshold: float = 0.0
    max_results: int = 100


@dataclass
class SearchFilters:
    """Фильтры для поиска."""
    categories: Optional[Set[str]] = None
    tags: Optional[Set[str]] = None
    confidence_range: Optional[Tuple[float, float]] = None
    time_range: Optional[Tuple[float, float]] = None
    source: Optional[str] = None
    relationship_types: Optional[Set[RelationshipType]] = None
    exclude_ids: Optional[Set[str]] = None


class QueryEngine(ABC):
    """Базовый класс для движков запросов."""
    
    def __init__(self, graph: KAGGraph):
        self.graph = graph
    
    @abstractmethod
    def execute(self, query: QueryContext) -> QueryResult:
        """Выполняет запрос."""
        pass


class ConceptSearchEngine(QueryEngine):
    """Движок для поиска концептов."""
    
    def execute(self, query: QueryContext) -> QueryResult:
        """Выполняет поиск концептов."""
        start_time = time.time()
        
        parameters = query.parameters
        results = []
        
        # Извлекаем параметры поиска
        search_term = parameters.get('search_term', '')
        filters = parameters.get('filters', SearchFilters())
        similarity_threshold = parameters.get('similarity_threshold', 0.0)
        search_mode = parameters.get('search_mode', 'fuzzy')  # exact, fuzzy, regex
        
        # Начинаем с базового поиска
        if search_term:
            results = self._search_by_term(search_term, search_mode, filters)
        else:
            # Если нет поискового термина, используем фильтры
            results = self._search_by_filters(filters)
        
        # Применяем дополнительные фильтры
        results = self._apply_filters(results, filters)
        
        # Сортируем по релевантности и уверенности
        results = self._rank_results(results, search_term, similarity_threshold)
        
        # Ограничиваем количество результатов
        max_results = min(query.max_results, len(results))
        final_results = results[:max_results]
        
        execution_time = time.time() - start_time
        
        return QueryResult(
            query=f"search_term='{search_term}', filters={filters}",
            results=final_results,
            paths=[],  # Не применимо для поиска концептов
            scores=[self._calculate_relevance_score(concept, search_term) 
                   for concept in final_results],
            execution_time=execution_time,
            total_results=len(results)
        )
    
    def _search_by_term(self, search_term: str, search_mode: str, filters: SearchFilters) -> List[Concept]:
        """Поиск по термину с различными режимами."""
        results = []
        
        for concept in self.graph.concepts.values():
            match = False
            
            if search_mode == 'exact':
                match = (search_term.lower() == concept.name.lower() or
                        search_term.lower() in concept.description.lower())
            
            elif search_mode == 'fuzzy':
                # Простая нечеткая логика: проверяем вхождение в имя, описание, теги
                search_lower = search_term.lower()
                match = (search_lower in concept.name.lower() or
                        search_lower in concept.description.lower() or
                        any(search_lower in tag.lower() for tag in concept.tags))
            
            elif search_mode == 'regex':
                try:
                    pattern = re.compile(search_term, re.IGNORECASE)
                    match = (pattern.search(concept.name) or
                            pattern.search(concept.description) or
                            any(pattern.search(tag) for tag in concept.tags))
                except re.error:
                    # Если regex некорректен, используем fuzzy поиск
                    search_lower = search_term.lower()
                    match = (search_lower in concept.name.lower() or
                            search_lower in concept.description.lower())
            
            if match:
                results.append(concept)
        
        return results
    
    def _search_by_filters(self, filters: SearchFilters) -> List[Concept]:
        """Поиск только по фильтрам."""
        results = list(self.graph.concepts.values())
        
        if filters.categories:
            results = [c for c in results if c.category in filters.categories]
        
        if filters.tags:
            concept_tags = {c.id: set(c.tags) for c in results}
            results = [c for c in results if filters.tags.issubset(concept_tags[c.id])]
        
        if filters.confidence_range:
            min_conf, max_conf = filters.confidence_range
            results = [c for c in results if min_conf <= c.confidence_score <= max_conf]
        
        if filters.time_range:
            min_time, max_time = filters.time_range
            results = [c for c in results if min_time <= c.created_at <= max_time]
        
        if filters.source:
            results = [c for c in results if c.source == filters.source]
        
        if filters.exclude_ids:
            results = [c for c in results if c.id not in filters.exclude_ids]
        
        return results
    
    def _apply_filters(self, results: List[Concept], filters: SearchFilters) -> List[Concept]:
        """Применяет дополнительные фильтры."""
        filtered_results = results.copy()
        
        # Дополнительные фильтры могут быть добавлены здесь
        return filtered_results
    
    def _rank_results(self, results: List[Concept], search_term: str, similarity_threshold: float) -> List[Concept]:
        """Ранжирует результаты по релевантности."""
        if not results:
            return results
        
        # Вычисляем оценку релевантности для каждого результата
        scored_results = []
        for concept in results:
            relevance_score = self._calculate_relevance_score(concept, search_term)
            if relevance_score >= similarity_threshold:
                scored_results.append((concept, relevance_score))
        
        # Сортируем по убыванию релевантности
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        return [concept for concept, score in scored_results]
    
    def _calculate_relevance_score(self, concept: Concept, search_term: str) -> float:
        """Вычисляет оценку релевантности концепта."""
        if not search_term:
            return concept.confidence_score
        
        search_lower = search_term.lower()
        score = 0.0
        
        # Точное совпадение в имени - максимальный балл
        if search_lower == concept.name.lower():
            score += 1.0
        
        # Частичное совпадение в имени
        elif search_lower in concept.name.lower():
            score += 0.8 * (len(search_lower) / len(concept.name))
        
        # Совпадение в описании
        elif search_lower in concept.description.lower():
            score += 0.5
        
        # Совпадение в тегах
        for tag in concept.tags:
            if search_lower == tag.lower():
                score += 0.7
                break
            elif search_lower in tag.lower():
                score += 0.4
                break
        
        # Бонус за уверенность
        score *= concept.confidence_score
        
        return score


class RelationshipSearchEngine(QueryEngine):
    """Движок для поиска отношений."""
    
    def execute(self, query: QueryContext) -> QueryResult:
        """Выполняет поиск отношений."""
        start_time = time.time()
        
        parameters = query.parameters
        results = []
        
        # Извлекаем параметры поиска
        source_concept_id = parameters.get('source_concept_id')
        target_concept_id = parameters.get('target_concept_id')
        relationship_types = parameters.get('relationship_types', set())
        filters = parameters.get('filters', SearchFilters())
        
        # Определяем стратегию поиска
        if source_concept_id and target_concept_id:
            # Ищем конкретное отношение между двумя концептами
            results = self._find_direct_relationship(source_concept_id, target_concept_id, relationship_types)
        
        elif source_concept_id:
            # Ищем отношения, исходящие от концепта
            results = self._find_outgoing_relationships(source_concept_id, relationship_types)
        
        elif target_concept_id:
            # Ищем отношения, входящие в концепт
            results = self._find_incoming_relationships(target_concept_id, relationship_types)
        
        else:
            # Ищем все отношения с фильтрами
            results = self._find_all_relationships(relationship_types, filters)
        
        # Применяем фильтры
        results = self._apply_relationship_filters(results, filters)
        
        # Ограничиваем количество результатов
        max_results = min(query.max_results, len(results))
        final_results = results[:max_results]
        
        execution_time = time.time() - start_time
        
        return QueryResult(
            query=f"source={source_concept_id}, target={target_concept_id}, types={relationship_types}",
            results=[],  # Возвращаем концепты, а не отношения
            paths=[],
            scores=[rel.strength * rel.confidence_score for rel in final_results],
            execution_time=execution_time,
            total_results=len(results)
        )
    
    def _find_direct_relationship(self, source_id: str, target_id: str, relationship_types: Set[RelationshipType]) -> List[Relationship]:
        """Находит прямое отношение между двумя концептами."""
        results = []
        
        source_concept = self.graph.concepts.get(source_id)
        if source_concept:
            for rel in source_concept.get_relationships(relationship_types):
                if rel.target_id == target_id:
                    results.append(rel)
        
        return results
    
    def _find_outgoing_relationships(self, source_id: str, relationship_types: Set[RelationshipType]) -> List[Relationship]:
        """Находит исходящие отношения от концепта."""
        source_concept = self.graph.concepts.get(source_id)
        if source_concept:
            return source_concept.get_relationships(relationship_types)
        return []
    
    def _find_incoming_relationships(self, target_id: str, relationship_types: Set[RelationshipType]) -> List[Relationship]:
        """Находит входящие отношения к концепту."""
        target_concept = self.graph.concepts.get(target_id)
        if target_concept:
            return target_concept.get_relationships(relationship_types)
        return []
    
    def _find_all_relationships(self, relationship_types: Set[RelationshipType], filters: SearchFilters) -> List[Relationship]:
        """Находит все отношения с фильтрами."""
        results = list(self.graph.relationships.values())
        
        if relationship_types:
            results = [rel for rel in results if rel.relationship_type in relationship_types]
        
        return results
    
    def _apply_relationship_filters(self, results: List[Relationship], filters: SearchFilters) -> List[Relationship]:
        """Применяет фильтры к отношениям."""
        filtered_results = results.copy()
        
        if filters.confidence_range:
            min_conf, max_conf = filters.confidence_range
            filtered_results = [
                rel for rel in filtered_results 
                if min_conf <= rel.confidence_score <= max_conf
            ]
        
        if filters.exclude_ids:
            # Исключаем отношения, где участвуют исключенные концепты
            filtered_results = [
                rel for rel in filtered_results
                if rel.source_id not in filters.exclude_ids and rel.target_id not in filters.exclude_ids
            ]
        
        return filtered_results


class PathTraversalEngine(QueryEngine):
    """Движок для поиска путей между концептами."""
    
    def __init__(self, graph: KAGGraph):
        super().__init__(graph)
        self.traversal = GraphTraversal(graph)
    
    def execute(self, query: QueryContext) -> QueryResult:
        """Выполняет поиск пути."""
        start_time = time.time()
        
        parameters = query.parameters
        source_id = parameters.get('source_concept_id')
        target_id = parameters.get('target_concept_id')
        algorithm = parameters.get('algorithm', TraversalType.BFS)
        relationship_types = parameters.get('relationship_types', set())
        max_depth = parameters.get('max_depth', 10)
        find_all_paths = parameters.get('find_all_paths', False)
        
        if not source_id or not target_id:
            return QueryResult(
                query="path_traversal",
                results=[],
                paths=[],
                scores=[],
                execution_time=time.time() - start_time,
                total_results=0
            )
        
        results = []
        paths = []
        scores = []
        
        try:
            if find_all_paths:
                # Находим все пути
                search_results = self.traversal.find_all_paths(
                    source_id=source_id,
                    target_id=target_id,
                    max_paths=query.max_results,
                    max_depth=max_depth,
                    relationship_types=relationship_types
                )
                
                for search_result in search_results:
                    if search_result.path:
                        # Получаем концепты для пути
                        path_concepts = [self.graph.concepts[cid] for cid in search_result.path if cid in self.graph.concepts]
                        results.extend(path_concepts)
                        paths.append(search_result.path)
                        scores.append(search_result.cost)
            
            else:
                # Находим один лучший путь
                search_result = self.traversal.find_path(
                    source_id=source_id,
                    target_id=target_id,
                    algorithm=algorithm,
                    relationship_types=relationship_types,
                    max_depth=max_depth
                )
                
                if search_result and search_result.path:
                    # Получаем концепты для пути
                    path_concepts = [self.graph.concepts[cid] for cid in search_result.path if cid in self.graph.concepts]
                    results = path_concepts
                    paths = [search_result.path]
                    scores = [search_result.cost]
        
        except Exception as e:
            print(f"Ошибка при поиске пути: {e}")
        
        execution_time = time.time() - start_time
        
        return QueryResult(
            query=f"source={source_id}, target={target_id}, algorithm={algorithm.value}",
            results=results,
            paths=paths,
            scores=scores,
            execution_time=execution_time,
            total_results=len(results)
        )


class SimilarityEngine(QueryEngine):
    """Движок для поиска похожих концептов."""
    
    def execute(self, query: QueryContext) -> QueryResult:
        """Выполняет поиск похожих концептов."""
        start_time = time.time()
        
        parameters = query.parameters
        target_concept_id = parameters.get('target_concept_id')
        similarity_threshold = parameters.get('similarity_threshold', 0.1)
        max_results = parameters.get('max_results', 10)
        exclude_direct_connections = parameters.get('exclude_direct_connections', True)
        
        if not target_concept_id or target_concept_id not in self.graph.concepts:
            return QueryResult(
                query="similarity_search",
                results=[],
                paths=[],
                scores=[],
                execution_time=time.time() - start_time,
                total_results=0
            )
        
        target_concept = self.graph.concepts[target_concept_id]
        similarities = []
        
        # Вычисляем схожесть с другими концептами
        for concept_id, concept in self.graph.concepts.items():
            if concept_id == target_concept_id:
                continue
            
            # Исключаем напрямую связанные концепты если требуется
            if exclude_direct_connections:
                related_ids = {n.id for n in target_concept.get_related_concepts()}
                if concept_id in related_ids:
                    continue
            
            similarity_score = target_concept.calculate_similarity(concept)
            if similarity_score >= similarity_threshold:
                similarities.append((concept, similarity_score))
        
        # Сортируем по убыванию схожести
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Ограничиваем результаты
        similarities = similarities[:max_results]
        
        results = [concept for concept, score in similarities]
        scores = [score for concept, score in similarities]
        
        execution_time = time.time() - start_time
        
        return QueryResult(
            query=f"target_concept={target_concept_id}, threshold={similarity_threshold}",
            results=results,
            paths=[],
            scores=scores,
            execution_time=execution_time,
            total_results=len(similarities)
        )


class GraphAnalysisEngine(QueryEngine):
    """Движок для анализа графа."""
    
    def __init__(self, graph: KAGGraph):
        super().__init__(graph)
        self.traversal = GraphTraversal(graph)
    
    def execute(self, query: QueryContext) -> QueryResult:
        """Выполняет анализ графа."""
        start_time = time.time()
        
        parameters = query.parameters
        analysis_type = parameters.get('analysis_type', 'metrics')
        
        results = []
        
        if analysis_type == 'metrics':
            # Анализ метрик графа
            metrics = self.graph.calculate_graph_metrics()
            results = [self._create_metric_concept(name, value) for name, value in metrics.items()]
        
        elif analysis_type == 'connectivity':
            # Анализ связности
            connectivity = self.traversal.analyze_connectivity()
            results = [self._create_metric_concept(name, value) for name, value in connectivity.items()]
        
        elif analysis_type == 'centrality':
            # Анализ центральности
            centrality = self.traversal.get_centrality_metrics()
            for concept_id, centralities in centrality.items():
                concept = self.graph.concepts.get(concept_id)
                if concept:
                    for metric_name, value in centralities.items():
                        metric_concept = self._create_metric_concept(
                            f"{concept.name}_{metric_name}", 
                            value,
                            f"Центральность {metric_name} для концепта {concept.name}"
                        )
                        results.append(metric_concept)
        
        elif analysis_type == 'communities':
            # Поиск сообществ в графе
            communities = self._detect_communities()
            for i, community in enumerate(communities):
                community_concept = self._create_metric_concept(
                    f"community_{i}", 
                    len(community),
                    f"Сообщество {i}: {', '.join([self.graph.concepts[cid].name for cid in community[:3]])}"
                )
                results.append(community_concept)
        
        execution_time = time.time() - start_time
        
        return QueryResult(
            query=f"analysis_type={analysis_type}",
            results=results,
            paths=[],
            scores=[1.0] * len(results),
            execution_time=execution_time,
            total_results=len(results)
        )
    
    def _create_metric_concept(self, name: str, value: Any, description: str = "") -> Concept:
        """Создает концепт для представления метрики."""
        concept = Concept(
            id=f"metric_{name}_{id(value)}",
            name=name,
            description=description or f"Метрика: {name}",
            category="metric",
            confidence_score=1.0,
            source="graph_analysis"
        )
        
        # Преобразуем значение в свойства
        if isinstance(value, dict):
            concept.properties = value
        else:
            concept.properties = {"value": value, "type": type(value).__name__}
        
        return concept
    
    def _detect_communities(self) -> List[List[str]]:
        """Простое обнаружение сообществ в графе."""
        # Простой алгоритм на основе связности
        visited = set()
        communities = []
        
        for concept_id in self.graph.concepts:
            if concept_id not in visited:
                community = self._find_connected_component(concept_id, visited)
                if len(community) >= 2:  # Сообщества минимум из 2 узлов
                    communities.append(community)
        
        return communities
    
    def _find_connected_component(self, start_id: str, visited: Set[str]) -> List[str]:
        """Находит компоненту связности."""
        component = []
        to_visit = [start_id]
        
        while to_visit:
            current_id = to_visit.pop()
            if current_id not in visited:
                visited.add(current_id)
                component.append(current_id)
                
                current_concept = self.graph.concepts[current_id]
                related = current_concept.get_related_concepts()
                
                for related_concept in related:
                    if related_concept.id not in visited:
                        to_visit.append(related_concept.id)
        
        return component


class KAGQueryEngine:
    """
    Основной класс для выполнения запросов к графу знаний.
    Объединяет различные движки запросов.
    """
    
    def __init__(self, graph: KAGGraph):
        self.graph = graph
        self.engines = {
            QueryType.FIND_CONCEPTS: ConceptSearchEngine(graph),
            QueryType.FIND_RELATIONSHIPS: RelationshipSearchEngine(graph),
            QueryType.TRAVERSE_PATH: PathTraversalEngine(graph),
            QueryType.SIMILARITY_SEARCH: SimilarityEngine(graph),
            QueryType.GRAPH_ANALYSIS: GraphAnalysisEngine(graph)
        }
        
        # Статистика выполнения запросов
        self.query_stats = {
            'total_queries': 0,
            'queries_by_type': {qt.value: 0 for qt in QueryType},
            'avg_execution_time': 0.0,
            'last_query_time': 0.0
        }
    
    def execute_query(
        self,
        query_type: QueryType,
        parameters: Dict[str, Any],
        confidence_threshold: float = 0.0,
        max_results: int = 100
    ) -> QueryResult:
        """
        Выполняет запрос к графу знаний.
        
        Args:
            query_type: Тип запроса
            parameters: Параметры запроса
            confidence_threshold: Минимальный порог уверенности
            max_results: Максимальное количество результатов
            
        Returns:
            Результат запроса
        """
        if query_type not in self.engines:
            raise ValueError(f"Неподдерживаемый тип запроса: {query_type}")
        
        # Создаем контекст запроса
        query_context = QueryContext(
            query_type=query_type,
            parameters=parameters,
            timestamp=time.time(),
            confidence_threshold=confidence_threshold,
            max_results=max_results
        )
        
        # Выполняем запрос
        engine = self.engines[query_type]
        result = engine.execute(query_context)
        
        # Обновляем статистику
        self._update_query_stats(query_type, result.execution_time)
        
        return result
    
    def find_concepts(
        self,
        search_term: str = "",
        filters: Optional[SearchFilters] = None,
        similarity_threshold: float = 0.0,
        search_mode: str = "fuzzy",
        max_results: int = 100
    ) -> QueryResult:
        """Удобный метод для поиска концептов."""
        parameters = {
            'search_term': search_term,
            'filters': filters or SearchFilters(),
            'similarity_threshold': similarity_threshold,
            'search_mode': search_mode
        }
        
        return self.execute_query(
            QueryType.FIND_CONCEPTS,
            parameters,
            max_results=max_results
        )
    
    def find_relationships(
        self,
        source_concept_id: Optional[str] = None,
        target_concept_id: Optional[str] = None,
        relationship_types: Optional[Set[RelationshipType]] = None,
        filters: Optional[SearchFilters] = None,
        max_results: int = 100
    ) -> QueryResult:
        """Удобный метод для поиска отношений."""
        parameters = {
            'source_concept_id': source_concept_id,
            'target_concept_id': target_concept_id,
            'relationship_types': relationship_types or set(),
            'filters': filters or SearchFilters()
        }
        
        return self.execute_query(
            QueryType.FIND_RELATIONSHIPS,
            parameters,
            max_results=max_results
        )
    
    def find_path(
        self,
        source_concept_id: str,
        target_concept_id: str,
        algorithm: TraversalType = TraversalType.BFS,
        relationship_types: Optional[Set[RelationshipType]] = None,
        max_depth: int = 10,
        find_all_paths: bool = False,
        max_paths: int = 10
    ) -> QueryResult:
        """Удобный метод для поиска путей."""
        parameters = {
            'source_concept_id': source_concept_id,
            'target_concept_id': target_concept_id,
            'algorithm': algorithm,
            'relationship_types': relationship_types or set(),
            'max_depth': max_depth,
            'find_all_paths': find_all_paths,
            'max_paths': max_paths
        }
        
        return self.execute_query(
            QueryType.TRAVERSE_PATH,
            parameters,
            max_results=max_paths if find_all_paths else 1
        )
    
    def find_similar(
        self,
        target_concept_id: str,
        similarity_threshold: float = 0.1,
        exclude_direct_connections: bool = True,
        max_results: int = 10
    ) -> QueryResult:
        """Удобный метод для поиска похожих концептов."""
        parameters = {
            'target_concept_id': target_concept_id,
            'similarity_threshold': similarity_threshold,
            'exclude_direct_connections': exclude_direct_connections,
            'max_results': max_results
        }
        
        return self.execute_query(
            QueryType.SIMILARITY_SEARCH,
            parameters,
            max_results=max_results
        )
    
    def analyze_graph(
        self,
        analysis_type: str = "metrics"
    ) -> QueryResult:
        """Удобный метод для анализа графа."""
        parameters = {
            'analysis_type': analysis_type
        }
        
        return self.execute_query(
            QueryType.GRAPH_ANALYSIS,
            parameters,
            max_results=1000  # Анализ может возвращать много результатов
        )
    
    def _update_query_stats(self, query_type: QueryType, execution_time: float) -> None:
        """Обновляет статистику выполнения запросов."""
        self.query_stats['total_queries'] += 1
        self.query_stats['queries_by_type'][query_type.value] += 1
        self.query_stats['last_query_time'] = time.time()
        
        # Обновляем среднее время выполнения
        total_time = (self.query_stats['avg_execution_time'] * 
                     (self.query_stats['total_queries'] - 1) + execution_time)
        self.query_stats['avg_execution_time'] = total_time / self.query_stats['total_queries']
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Возвращает статистику выполнения запросов."""
        return self.query_stats.copy()
