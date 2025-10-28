"""
Модуль GraphTraversal - реализует различные алгоритмы обхода графа знаний.
Обеспечивает поиск путей, анализ связности и извлечение структурированных знаний.
"""

from __future__ import annotations

import heapq
import time
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from abc import ABC, abstractmethod

from .kag_graph import KAGGraph, Concept, Relationship, RelationshipType


class TraversalType(Enum):
    """Типы алгоритмов обхода графа."""
    BFS = "breadth_first_search"        # Поиск в ширину
    DFS = "depth_first_search"          # Поиск в глубину
    DIJKSTRA = "dijkstra"               # Алгоритм Дейкстры
    A_STAR = "a_star"                   # A* алгоритм
    GREEDY_BEST = "greedy_best_first"   # Жадный поиск по лучшему
    BIDIRECTIONAL = "bidirectional"     # Двунаправленный поиск


@dataclass
class SearchResult:
    """
    Результат поиска в графе.
    """
    path: List[str] = None  # Путь как список ID концептов
    cost: float = 0.0       # Стоимость пути
    confidence: float = 0.0 # Уверенность в результате
    visited_nodes: Set[str] = None  # Посещенные узлы
    execution_time: float = 0.0     # Время выполнения
    
    def __post_init__(self):
        if self.path is None:
            self.path = []
        if self.visited_nodes is None:
            self.visited_nodes = set()
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует результат в словарь."""
        return {
            'path': self.path,
            'cost': self.cost,
            'confidence': self.confidence,
            'visited_nodes': list(self.visited_nodes),
            'execution_time': self.execution_time,
            'path_length': len(self.path)
        }


class TraversalAlgorithm(ABC):
    """Базовый класс для алгоритмов обхода."""
    
    def __init__(self, graph: KAGGraph):
        self.graph = graph
    
    @abstractmethod
    def find_path(
        self,
        source_id: str,
        target_id: str,
        relationship_types: Optional[Set[RelationshipType]] = None,
        max_depth: int = 10,
        **kwargs
    ) -> Optional[SearchResult]:
        """Находит путь между концептами."""
        pass


class BreadthFirstSearch(TraversalAlgorithm):
    """Алгоритм поиска в ширину."""
    
    def find_path(
        self,
        source_id: str,
        target_id: str,
        relationship_types: Optional[Set[RelationshipType]] = None,
        max_depth: int = 10,
        **kwargs
    ) -> Optional[SearchResult]:
        start_time = time.time()
        
        if source_id not in self.graph.concepts or target_id not in self.graph.concepts:
            return None
        
        queue = deque([(source_id, [source_id])])
        visited = {source_id}
        min_cost = float('inf')
        best_path = []
        
        while queue:
            current_id, path = queue.popleft()
            current_depth = len(path) - 1
            
            if current_depth > max_depth:
                continue
            
            if current_id == target_id:
                # Вычисляем стоимость пути
                cost = self._calculate_path_cost(path, relationship_types)
                if cost < min_cost:
                    min_cost = cost
                    best_path = path
            
            current_concept = self.graph.concepts[current_id]
            related = current_concept.get_related_concepts(relationship_types)
            
            for related_concept in related:
                if related_concept.id not in visited:
                    visited.add(related_concept.id)
                    new_path = path + [related_concept.id]
                    queue.append((related_concept.id, new_path))
        
        execution_time = time.time() - start_time
        
        if best_path:
            confidence = self._calculate_path_confidence(best_path, relationship_types)
            return SearchResult(
                path=best_path,
                cost=min_cost,
                confidence=confidence,
                visited_nodes=visited,
                execution_time=execution_time
            )
        
        return None
    
    def _calculate_path_cost(self, path: List[str], relationship_types: Optional[Set[RelationshipType]]) -> float:
        """Вычисляет стоимость пути."""
        cost = 0.0
        
        for i in range(len(path) - 1):
            current_id = path[i]
            next_id = path[i + 1]
            
            current_concept = self.graph.concepts[current_id]
            relationships = current_concept.get_relationships(relationship_types)
            
            for rel in relationships:
                if rel.target_id == next_id:
                    # Стоимость = 1 / (strength * confidence)
                    rel_cost = 1.0 / (rel.strength * rel.confidence_score)
                    cost += rel_cost
                    break
            else:
                cost += 10.0  # Штраф за отсутствующую связь
        
        return cost
    
    def _calculate_path_confidence(self, path: List[str], relationship_types: Optional[Set[RelationshipType]]) -> float:
        """Вычисляет уверенность в пути."""
        confidence_values = []
        
        for concept_id in path:
            concept = self.graph.concepts[concept_id]
            confidence_values.append(concept.confidence_score)
        
        # Уверенность = произведение уверенностей концептов / количество концептов
        if confidence_values:
            return (product(confidence_values) ** (1.0 / len(confidence_values)))
        return 0.0


class DepthFirstSearch(TraversalAlgorithm):
    """Алгоритм поиска в глубину."""
    
    def find_path(
        self,
        source_id: str,
        target_id: str,
        relationship_types: Optional[Set[RelationshipType]] = None,
        max_depth: int = 10,
        **kwargs
    ) -> Optional[SearchResult]:
        start_time = time.time()
        
        if source_id not in self.graph.concepts or target_id not in self.graph.concepts:
            return None
        
        visited = set()
        best_result = None
        min_cost = float('inf')
        
        def dfs(current_id: str, path: List[str], depth: int):
            nonlocal best_result, min_cost
            
            if depth > max_depth or current_id in visited:
                return
            
            visited.add(current_id)
            path.append(current_id)
            
            if current_id == target_id:
                cost = self._calculate_path_cost(path, relationship_types)
                if cost < min_cost:
                    min_cost = cost
                    confidence = self._calculate_path_confidence(path, relationship_types)
                    best_result = SearchResult(
                        path=path.copy(),
                        cost=cost,
                        confidence=confidence,
                        visited_nodes=visited.copy(),
                        execution_time=time.time() - start_time
                    )
            else:
                current_concept = self.graph.concepts[current_id]
                related = current_concept.get_related_concepts(relationship_types)
                
                for related_concept in related:
                    if related_concept.id not in visited:
                        dfs(related_concept.id, path, depth + 1)
            
            path.pop()
            visited.remove(current_id)
        
        dfs(source_id, [], 0)
        
        if best_result:
            best_result.execution_time = time.time() - start_time
        
        return best_result
    
    def _calculate_path_cost(self, path: List[str], relationship_types: Optional[Set[RelationshipType]]) -> float:
        """Вычисляет стоимость пути."""
        cost = 0.0
        
        for i in range(len(path) - 1):
            current_id = path[i]
            next_id = path[i + 1]
            
            current_concept = self.graph.concepts[current_id]
            relationships = current_concept.get_relationships(relationship_types)
            
            for rel in relationships:
                if rel.target_id == next_id:
                    cost += 1.0 / rel.strength
                    break
            else:
                cost += 5.0
        
        return cost
    
    def _calculate_path_confidence(self, path: List[str], relationship_types: Optional[Set[RelationshipType]]) -> float:
        """Вычисляет уверенность в пути."""
        if not path:
            return 0.0
        
        confidence_product = 1.0
        for concept_id in path:
            concept = self.graph.concepts[concept_id]
            confidence_product *= concept.confidence_score
        
        return confidence_product ** (1.0 / len(path))


class DijkstraAlgorithm(TraversalAlgorithm):
    """Алгоритм Дейкстры для поиска кратчайшего пути."""
    
    def find_path(
        self,
        source_id: str,
        target_id: str,
        relationship_types: Optional[Set[RelationshipType]] = None,
        max_depth: int = 10,
        **kwargs
    ) -> Optional[SearchResult]:
        start_time = time.time()
        
        if source_id not in self.graph.concepts or target_id not in self.graph.concepts:
            return None
        
        # Инициализация
        distances = {source_id: 0.0}
        previous = {}
        visited = set()
        priority_queue = [(0.0, source_id)]
        
        while priority_queue:
            current_distance, current_id = heapq.heappop(priority_queue)
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            if current_id == target_id:
                # Восстанавливаем путь
                path = []
                while current_id in previous:
                    path.append(current_id)
                    current_id = previous[current_id]
                path.append(source_id)
                path.reverse()
                
                execution_time = time.time() - start_time
                confidence = self._calculate_path_confidence(path, relationship_types)
                
                return SearchResult(
                    path=path,
                    cost=distances[target_id],
                    confidence=confidence,
                    visited_nodes=visited,
                    execution_time=execution_time
                )
            
            if current_distance > max_depth:
                continue
            
            current_concept = self.graph.concepts[current_id]
            related = current_concept.get_related_concepts(relationship_types)
            
            for related_concept in related:
                if related_concept.id in visited:
                    continue
                
                # Вычисляем вес ребра
                edge_weight = self._get_edge_weight(current_id, related_concept.id, relationship_types)
                new_distance = current_distance + edge_weight
                
                if related_concept.id not in distances or new_distance < distances[related_concept.id]:
                    distances[related_concept.id] = new_distance
                    previous[related_concept.id] = current_id
                    heapq.heappush(priority_queue, (new_distance, related_concept.id))
        
        return None
    
    def _get_edge_weight(self, source_id: str, target_id: str, relationship_types: Optional[Set[RelationshipType]]) -> float:
        """Получает вес ребра между концептами."""
        source_concept = self.graph.concepts[source_id]
        relationships = source_concept.get_relationships(relationship_types)
        
        for rel in relationships:
            if rel.target_id == target_id:
                # Вес = 1 / (strength * confidence)
                return 1.0 / (rel.strength * rel.confidence_score)
        
        return float('inf')
    
    def _calculate_path_confidence(self, path: List[str], relationship_types: Optional[Set[RelationshipType]]) -> float:
        """Вычисляет уверенность в пути."""
        if not path:
            return 0.0
        
        confidence_sum = 0.0
        count = 0
        
        for concept_id in path:
            concept = self.graph.concepts[concept_id]
            confidence_sum += concept.confidence_score
            count += 1
        
        return confidence_sum / count if count > 0 else 0.0


class AStarAlgorithm(TraversalAlgorithm):
    """A* алгоритм поиска пути."""
    
    def find_path(
        self,
        source_id: str,
        target_id: str,
        relationship_types: Optional[Set[RelationshipType]] = None,
        max_depth: int = 10,
        heuristic_func: Optional[Callable[[str, str], float]] = None,
        **kwargs
    ) -> Optional[SearchResult]:
        start_time = time.time()
        
        if source_id not in self.graph.concepts or target_id not in self.graph.concepts:
            return None
        
        # Инициализация
        open_set = [(0.0, source_id)]
        came_from = {}
        g_score = {source_id: 0.0}
        f_score = {}
        
        # Вычисляем эвристику для начального узла
        target_concept = self.graph.concepts[target_id]
        f_score[source_id] = heuristic_func(source_id, target_id) if heuristic_func else 0.0
        
        visited = set()
        
        while open_set:
            current_f, current_id = heapq.heappop(open_set)
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            if current_id == target_id:
                # Восстанавливаем путь
                path = []
                while current_id in came_from:
                    path.append(current_id)
                    current_id = came_from[current_id]
                path.append(source_id)
                path.reverse()
                
                execution_time = time.time() - start_time
                confidence = self._calculate_path_confidence(path, relationship_types)
                
                return SearchResult(
                    path=path,
                    cost=g_score[target_id],
                    confidence=confidence,
                    visited_nodes=visited,
                    execution_time=execution_time
                )
            
            if current_f > max_depth * 10:  # Простая проверка глубины
                continue
            
            current_concept = self.graph.concepts[current_id]
            related = current_concept.get_related_concepts(relationship_types)
            
            for related_concept in related:
                if related_concept.id in visited:
                    continue
                
                # Вычисляем стоимость перехода
                edge_weight = self._get_edge_weight(current_id, related_concept.id, relationship_types)
                tentative_g_score = g_score[current_id] + edge_weight
                
                if related_concept.id not in g_score or tentative_g_score < g_score[related_concept.id]:
                    came_from[related_concept.id] = current_id
                    g_score[related_concept.id] = tentative_g_score
                    
                    # Вычисляем f_score = g_score + h_score
                    h_score = self._calculate_heuristic(
                        related_concept.id, 
                        target_id, 
                        heuristic_func
                    )
                    f_score[related_concept.id] = tentative_g_score + h_score
                    
                    heapq.heappush(open_set, (f_score[related_concept.id], related_concept.id))
        
        return None
    
    def _get_edge_weight(self, source_id: str, target_id: str, relationship_types: Optional[Set[RelationshipType]]) -> float:
        """Получает вес ребра между концептами."""
        source_concept = self.graph.concepts[source_id]
        relationships = source_concept.get_relationships(relationship_types)
        
        for rel in relationships:
            if rel.target_id == target_id:
                return 1.0 / (rel.strength * rel.confidence_score)
        
        return float('inf')
    
    def _calculate_heuristic(
        self, 
        current_id: str, 
        target_id: str,
        heuristic_func: Optional[Callable[[str, str], float]]
    ) -> float:
        """Вычисляет эвристическую оценку."""
        if heuristic_func:
            return heuristic_func(current_id, target_id)
        
        # Простая эвристика: расстояние по категориям и тегам
        current_concept = self.graph.concepts[current_id]
        target_concept = self.graph.concepts[target_id]
        
        distance = 0.0
        
        # Дистанция по категориям
        if current_concept.category != target_concept.category:
            distance += 1.0
        
        # Дистанция по тегам
        current_tags = set(current_concept.tags)
        target_tags = set(target_concept.tags)
        if current_tags or target_tags:
            tag_distance = 1.0 - len(current_tags & target_tags) / len(current_tags | target_tags)
            distance += tag_distance
        
        return distance
    
    def _calculate_path_confidence(self, path: List[str], relationship_types: Optional[Set[RelationshipType]]) -> float:
        """Вычисляет уверенность в пути."""
        if not path:
            return 0.0
        
        confidence_product = 1.0
        for concept_id in path:
            concept = self.graph.concepts[concept_id]
            confidence_product *= concept.confidence_score
        
        return confidence_product ** (1.0 / len(path))


class BidirectionalSearch(TraversalAlgorithm):
    """Двунаправленный поиск."""
    
    def find_path(
        self,
        source_id: str,
        target_id: str,
        relationship_types: Optional[Set[RelationshipType]] = None,
        max_depth: int = 10,
        **kwargs
    ) -> Optional[SearchResult]:
        start_time = time.time()
        
        if source_id not in self.graph.concepts or target_id not in self.graph.concepts:
            return None
        
        # Инициализация двух направлений поиска
        forward_queue = deque([(source_id, [source_id])])
        backward_queue = deque([(target_id, [target_id])])
        
        forward_visited = {source_id}
        backward_visited = {target_id}
        forward_parents = {source_id: None}
        backward_parents = {target_id: None}
        
        forward_paths = {source_id: [source_id]}
        backward_paths = {target_id: [target_id]}
        
        while forward_queue and backward_queue:
            # Поиск в прямом направлении
            if forward_queue:
                forward_result = self._bidirectional_step(
                    forward_queue, forward_visited, forward_parents, forward_paths,
                    backward_visited, relationship_types, max_depth, True
                )
                
                if forward_result:
                    execution_time = time.time() - start_time
                    return forward_result
            
            # Поиск в обратном направлении
            if backward_queue:
                backward_result = self._bidirectional_step(
                    backward_queue, backward_visited, backward_parents, backward_paths,
                    forward_visited, relationship_types, max_depth, False
                )
                
                if backward_result:
                    execution_time = time.time() - start_time
                    return backward_result
        
        return None
    
    def _bidirectional_step(
        self,
        queue: deque,
        visited: Set[str],
        parents: Dict[str, Optional[str]],
        paths: Dict[str, List[str]],
        other_visited: Set[str],
        relationship_types: Optional[Set[RelationshipType]],
        max_depth: int,
        is_forward: bool
    ) -> Optional[SearchResult]:
        """Выполняет один шаг двунаправленного поиска."""
        if not queue:
            return None
        
        current_id, current_path = queue.popleft()
        current_depth = len(current_path) - 1
        
        if current_depth > max_depth:
            return None
        
        # Проверяем встречу с другим направлением
        if current_id in other_visited:
            # Найдена встреча, восстанавливаем полный путь
            return self._reconstruct_bidirectional_path(
                current_path, current_id, parents, other_visited, is_forward
            )
        
        current_concept = self.graph.concepts[current_id]
        related = current_concept.get_related_concepts(relationship_types)
        
        for related_concept in related:
            if related_concept.id not in visited:
                visited.add(related_concept.id)
                parents[related_concept.id] = current_id
                new_path = current_path + [related_concept.id]
                paths[related_concept.id] = new_path
                queue.append((related_concept.id, new_path))
        
        return None
    
    def _reconstruct_bidirectional_path(
        self,
        current_path: List[str],
        meeting_id: str,
        parents: Dict[str, Optional[str]],
        other_visited: Set[str],
        is_forward: bool
    ) -> SearchResult:
        """Восстанавливает полный путь при встрече направлений."""
        # Находим встречный путь
        meeting_node = meeting_id
        
        # В реальной реализации здесь должен быть более сложный алгоритм
        # восстановления пути. Для простоты возвращаем текущий путь.
        full_path = current_path
        
        cost = len(full_path) - 1
        confidence = 1.0 / max(1, len(full_path))
        visited_nodes = set(full_path)
        execution_time = 0.0
        
        return SearchResult(
            path=full_path,
            cost=cost,
            confidence=confidence,
            visited_nodes=visited_nodes,
            execution_time=execution_time
        )


class GraphTraversal:
    """
    Основной класс для выполнения различных алгоритмов обхода графа.
    """
    
    def __init__(self, graph: KAGGraph):
        self.graph = graph
        self.algorithms = {
            TraversalType.BFS: BreadthFirstSearch(graph),
            TraversalType.DFS: DepthFirstSearch(graph),
            TraversalType.DIJKSTRA: DijkstraAlgorithm(graph),
            TraversalType.A_STAR: AStarAlgorithm(graph),
            TraversalType.BIDIRECTIONAL: BidirectionalSearch(graph)
        }
    
    def find_path(
        self,
        source_id: str,
        target_id: str,
        algorithm: TraversalType = TraversalType.BFS,
        relationship_types: Optional[Set[RelationshipType]] = None,
        max_depth: int = 10,
        **kwargs
    ) -> Optional[SearchResult]:
        """
        Находит путь между концептами используя указанный алгоритм.
        
        Args:
            source_id: ID исходного концепта
            target_id: ID целевого концепта
            algorithm: Алгоритм поиска
            relationship_types: Типы отношений для поиска
            max_depth: Максимальная глубина поиска
            **kwargs: Дополнительные параметры для алгоритма
            
        Returns:
            Результат поиска или None если путь не найден
        """
        if algorithm not in self.algorithms:
            raise ValueError(f"Неподдерживаемый алгоритм: {algorithm}")
        
        algo = self.algorithms[algorithm]
        return algo.find_path(
            source_id=source_id,
            target_id=target_id,
            relationship_types=relationship_types,
            max_depth=max_depth,
            **kwargs
        )
    
    def find_all_paths(
        self,
        source_id: str,
        target_id: str,
        max_paths: int = 10,
        max_depth: int = 5,
        relationship_types: Optional[Set[RelationshipType]] = None
    ) -> List[SearchResult]:
        """
        Находит все пути между концептами (с ограничениями).
        
        Args:
            source_id: ID исходного концепта
            target_id: ID целевого концепта
            max_paths: Максимальное количество путей
            max_depth: Максимальная глубина
            relationship_types: Типы отношений для поиска
            
        Returns:
            Список найденных путей
        """
        all_paths = []
        
        def find_paths_recursive(
            current_id: str,
            target_id: str,
            path: List[str],
            depth: int
        ):
            if len(all_paths) >= max_paths or depth > max_depth:
                return
            
            if current_id == target_id:
                result = SearchResult(
                    path=path.copy(),
                    cost=len(path) - 1,
                    confidence=1.0 / max(1, len(path)),
                    visited_nodes=set(path)
                )
                all_paths.append(result)
                return
            
            current_concept = self.graph.concepts[current_id]
            related = current_concept.get_related_concepts(relationship_types)
            
            for related_concept in related:
                if related_concept.id not in path:  # Избегаем циклов
                    new_path = path + [related_concept.id]
                    find_paths_recursive(related_concept.id, target_id, new_path, depth + 1)
        
        find_paths_recursive(source_id, target_id, [source_id], 0)
        
        # Сортируем по стоимости
        all_paths.sort(key=lambda p: p.cost)
        
        return all_paths[:max_paths]
    
    def analyze_connectivity(self) -> Dict[str, Any]:
        """
        Анализирует связность графа.
        
        Returns:
            Статистика связности
        """
        if not self.graph.concepts:
            return {
                'is_connected': False,
                'components': 0,
                'largest_component_size': 0,
                'isolated_nodes': 0
            }
        
        # Поиск компонент связности
        visited = set()
        components = []
        
        for concept_id in self.graph.concepts:
            if concept_id not in visited:
                component = self._find_component(concept_id, visited)
                components.append(component)
        
        largest_component = max(components, key=len) if components else []
        isolated_nodes = [cid for cid in self.graph.concepts 
                         if len(self.graph.concepts[cid].get_related_concepts()) == 0]
        
        return {
            'is_connected': len(components) == 1,
            'components': len(components),
            'largest_component_size': len(largest_component),
            'isolated_nodes': len(isolated_nodes),
            'component_sizes': [len(comp) for comp in components],
            'average_clustering': self._calculate_average_clustering()
        }
    
    def _find_component(self, start_id: str, visited: Set[str]) -> List[str]:
        """Находит компоненту связности."""
        component = []
        queue = deque([start_id])
        visited.add(start_id)
        
        while queue:
            current_id = queue.popleft()
            component.append(current_id)
            
            current_concept = self.graph.concepts[current_id]
            related = current_concept.get_related_concepts()
            
            for related_concept in related:
                if related_concept.id not in visited:
                    visited.add(related_concept.id)
                    queue.append(related_concept.id)
        
        return component
    
    def _calculate_average_clustering(self) -> float:
        """Вычисляет средний коэффициент кластеризации."""
        if not self.graph.concepts:
            return 0.0
        
        clustering_coeffs = []
        
        for concept in self.graph.concepts.values():
            neighbors = concept.get_related_concepts()
            if len(neighbors) < 2:
                clustering_coeffs.append(0.0)
                continue
            
            # Подсчет связей между соседями
            neighbor_ids = {n.id for n in neighbors}
            edges_between = 0
            
            for neighbor in neighbors:
                neighbor_neighbors = {n.id for n in neighbor.get_related_concepts()}
                edges_between += len(neighbor_ids & neighbor_neighbors)
            
            possible_edges = len(neighbors) * (len(neighbors) - 1)
            if possible_edges > 0:
                clustering_coeffs.append(edges_between / possible_edges)
            else:
                clustering_coeffs.append(0.0)
        
        return sum(clustering_coeffs) / len(clustering_coeffs) if clustering_coeffs else 0.0
    
    def get_centrality_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Вычисляет метрики центральности узлов.
        
        Returns:
            Словарь с метриками центральности для каждого концепта
        """
        metrics = {
            'degree_centrality': {},
            'betweenness_centrality': {},
            'closeness_centrality': {}
        }
        
        total_concepts = len(self.graph.concepts)
        
        if total_concepts <= 1:
            return metrics
        
        # Степенная центральность
        for concept in self.graph.concepts.values():
            degree = len(concept.get_related_concepts())
            metrics['degree_centrality'][concept.id] = degree / (total_concepts - 1)
        
        # Центральность по посредничеству (упрощенная версия)
        for concept in self.graph.concepts.values():
            betweenness = self._calculate_betweenness(concept.id)
            metrics['betweenness_centrality'][concept.id] = betweenness
        
        # Центральность по близости (упрощенная версия)
        for concept in self.graph.concepts.values():
            closeness = self._calculate_closeness(concept.id)
            metrics['closeness_centrality'][concept.id] = closeness
        
        return metrics
    
    def _calculate_betweenness(self, concept_id: str) -> float:
        """Вычисляет центральность по посредничеству."""
        betweenness = 0.0
        
        for source_id in self.graph.concepts:
            if source_id == concept_id:
                continue
            
            for target_id in self.graph.concepts:
                if target_id == concept_id or target_id == source_id:
                    continue
                
                # Находим кратчайшие пути, проходящие через данный узел
                paths_through = self._count_paths_through(concept_id, source_id, target_id)
                total_paths = self._count_total_paths(source_id, target_id)
                
                if total_paths > 0:
                    betweenness += paths_through / total_paths
        
        return betweenness
    
    def _calculate_closeness(self, concept_id: str) -> float:
        """Вычисляет центральность по близости."""
        total_distance = 0.0
        reachable_nodes = 0
        
        for other_id in self.graph.concepts:
            if other_id == concept_id:
                continue
            
            # Упрощенное вычисление расстояния
            distance = len(self.graph.concepts[concept_id].get_related_concepts())
            if distance > 0:
                total_distance += distance
                reachable_nodes += 1
        
        if reachable_nodes > 0:
            return reachable_nodes / total_distance
        
        return 0.0
    
    def _count_paths_through(self, through_id: str, source_id: str, target_id: str) -> int:
        """Подсчитывает пути, проходящие через указанный узел."""
        # Упрощенная реализация
        if through_id in [source_id, target_id]:
            return 0
        
        # Проверяем, связан ли source с target напрямую через through
        source_concept = self.graph.concepts[source_id]
        through_concept = self.graph.concepts[through_id]
        
        for rel in source_concept.get_relationships():
            if rel.target_id == through_id:
                for through_rel in through_concept.get_relationships():
                    if through_rel.target_id == target_id:
                        return 1
        
        return 0
    
    def _count_total_paths(self, source_id: str, target_id: str) -> int:
        """Подсчитывает общее количество путей между узлами."""
        # Упрощенная реализация
        if source_id == target_id:
            return 1
        
        source_concept = self.graph.concepts[source_id]
        
        for rel in source_concept.get_relationships():
            if rel.target_id == target_id:
                return 1
        
        return 0


# Вспомогательная функция для вычисления произведения
def product(numbers):
    """Вычисляет произведение чисел."""
    result = 1
    for num in numbers:
        result *= num
    return result
