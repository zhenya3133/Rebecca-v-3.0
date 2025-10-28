"""
Unit тесты для KAG Graph компонента
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import time
import json
from typing import List, Dict, Any, Set, Tuple


class TestKAGGraph:
    """Тесты для KAG Graph класса"""
    
    @pytest.fixture
    def graph_instance(self):
        """Создание экземпляра графа для тестирования"""
        # Имитация KAGGraph класса
        class MockKAGGraph:
            def __init__(self, config: Dict[str, Any]):
                self.config = config
                self.nodes: Dict[str, Dict[str, Any]] = {}
                self.edges: List[Dict[str, Any]] = []
                self.adjacency_list: Dict[str, List[Tuple[str, str, float]]] = {}
                self.metadata = {
                    "created": time.time(),
                    "node_count": 0,
                    "edge_count": 0,
                    "max_nodes": config.get("max_nodes", 50000),
                    "max_edges": config.get("max_edges", 200000)
                }
            
            def add_node(self, node_id: str, node_data: Dict[str, Any]) -> bool:
                """Добавить узел в граф"""
                if len(self.nodes) >= self.metadata["max_nodes"]:
                    return False
                
                if node_id in self.nodes:
                    return False  # Узел уже существует
                
                self.nodes[node_id] = node_data
                self.adjacency_list[node_id] = []
                self.metadata["node_count"] += 1
                return True
            
            def add_edge(self, source: str, target: str, relation: str, weight: float = 1.0) -> bool:
                """Добавить ребро в граф"""
                if (source not in self.nodes or target not in self.nodes):
                    return False
                
                if len(self.edges) >= self.metadata["max_edges"]:
                    return False
                
                # Проверяем, не существует ли уже такое ребро
                for edge in self.edges:
                    if (edge["source"] == source and edge["target"] == target and 
                        edge["relation"] == relation):
                        return False
                
                self.edges.append({
                    "source": source,
                    "target": target,
                    "relation": relation,
                    "weight": weight,
                    "timestamp": time.time()
                })
                
                # Обновляем adjacency list
                if source not in self.adjacency_list:
                    self.adjacency_list[source] = []
                self.adjacency_list[source].append((target, relation, weight))
                
                self.metadata["edge_count"] += 1
                return True
            
            def remove_node(self, node_id: str) -> bool:
                """Удалить узел из графа"""
                if node_id not in self.nodes:
                    return False
                
                # Удаляем связанные ребра
                self.edges = [edge for edge in self.edges 
                             if edge["source"] != node_id and edge["target"] != node_id]
                self.adjacency_list.pop(node_id, None)
                
                # Удаляем узел
                del self.nodes[node_id]
                self.metadata["node_count"] -= 1
                return True
            
            def remove_edge(self, source: str, target: str, relation: str) -> bool:
                """Удалить ребро из графа"""
                original_length = len(self.edges)
                self.edges = [edge for edge in self.edges 
                             if not (edge["source"] == source and edge["target"] == target and edge["relation"] == relation)]
                
                if len(self.edges) < original_length:
                    # Обновляем adjacency list
                    if source in self.adjacency_list:
                        self.adjacency_list[source] = [(t, r, w) for t, r, w in self.adjacency_list[source] 
                                                      if not (t == target and r == relation)]
                    self.metadata["edge_count"] -= 1
                    return True
                return False
            
            def get_neighbors(self, node_id: str, relation: str = None) -> List[Dict[str, Any]]:
                """Получить соседей узла"""
                if node_id not in self.adjacency_list:
                    return []
                
                neighbors = []
                for target, rel, weight in self.adjacency_list[node_id]:
                    if relation is None or rel == relation:
                        neighbors.append({
                            "node_id": target,
                            "relation": rel,
                            "weight": weight
                        })
                return neighbors
            
            def search_related(self, query: str, k: int = 40) -> List[Dict[str, Any]]:
                """Поиск связанных узлов по запросу"""
                # Простая реализация поиска по ключевым словам
                results = []
                query_lower = query.lower()
                
                for node_id, node_data in self.nodes.items():
                    score = 0.0
                    
                    # Поиск в метках узлов
                    if "label" in node_data:
                        if query_lower in str(node_data["label"]).lower():
                            score += 0.8
                    
                    # Поиск в определениях
                    if "definition" in node_data:
                        if query_lower in str(node_data["definition"]).lower():
                            score += 0.6
                    
                    # Поиск в свойствах
                    if "properties" in node_data:
                        for prop_key, prop_value in node_data["properties"].items():
                            if query_lower in str(prop_value).lower():
                                score += 0.4
                    
                    if score > 0:
                        results.append({
                            "id": node_id,
                            "score": score,
                            "relation": "related",
                            "node_data": node_data
                        })
                
                # Сортируем по score и возвращаем топ-k
                results.sort(key=lambda x: x["score"], reverse=True)
                return results[:k]
            
            def traverse(self, start_node: str, max_depth: int = 10, relation_filter: str = None) -> List[str]:
                """Обход графа в глубину"""
                visited = set()
                traversal_order = []
                
                def dfs(current_node: str, depth: int):
                    if current_node in visited or depth > max_depth:
                        return
                    
                    visited.add(current_node)
                    traversal_order.append(current_node)
                    
                    for neighbor, rel, weight in self.adjacency_list.get(current_node, []):
                        if relation_filter is None or rel == relation_filter:
                            dfs(neighbor, depth + 1)
                
                dfs(start_node, 0)
                return traversal_order
            
            def get_path(self, source: str, target: str) -> List[str]:
                """Найти путь между двумя узлами (поиск в ширину)"""
                if source == target:
                    return [source]
                
                queue = [(source, [source])]
                visited = {source}
                
                while queue:
                    current_node, path = queue.pop(0)
                    
                    for neighbor, rel, weight in self.adjacency_list.get(current_node, []):
                        if neighbor == target:
                            return path + [neighbor]
                        
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, path + [neighbor]))
                
                return []  # Путь не найден
            
            def get_node_count(self) -> int:
                return self.metadata["node_count"]
            
            def get_edge_count(self) -> int:
                return self.metadata["edge_count"]
            
            def get_graph_stats(self) -> Dict[str, Any]:
                """Получить статистику графа"""
                return {
                    "nodes": self.get_node_count(),
                    "edges": self.get_edge_count(),
                    "avg_connections": len(self.edges) * 2 / max(len(self.nodes), 1),
                    "density": len(self.edges) / max(len(self.nodes) * (len(self.nodes) - 1) / 2, 1),
                    "metadata": self.metadata.copy()
                }
        
        return MockKAGGraph
    
    def test_graph_initialization(self, graph_instance):
        """Тест инициализации графа"""
        config = {
            "max_nodes": 1000,
            "max_edges": 5000,
            "traversal_depth": 5
        }
        
        graph = graph_instance(config)
        
        assert graph.get_node_count() == 0
        assert graph.get_edge_count() == 0
        assert graph.config["max_nodes"] == 1000
        assert graph.config["max_edges"] == 5000
        assert graph.metadata["max_nodes"] == 1000
        assert graph.metadata["max_edges"] == 5000
    
    def test_add_node(self, graph_instance):
        """Тест добавления узлов"""
        graph = graph_instance({})
        
        # Успешное добавление
        node_data = {
            "label": "Artificial Intelligence",
            "type": "concept",
            "definition": "Computer system capable of intelligent behavior"
        }
        
        result = graph.add_node("ai_concept", node_data)
        assert result is True
        assert graph.get_node_count() == 1
        assert "ai_concept" in graph.nodes
        assert graph.nodes["ai_concept"]["label"] == "Artificial Intelligence"
        
        # Попытка добавить существующий узел
        result = graph.add_node("ai_concept", node_data)
        assert result is False
        assert graph.get_node_count() == 1
        
        # Добавление нескольких узлов
        for i in range(5):
            result = graph.add_node(f"concept_{i}", {"label": f"Concept {i}"})
            assert result is True
        
        assert graph.get_node_count() == 6
    
    def test_add_edge(self, graph_instance):
        """Тест добавления рёбер"""
        graph = graph_instance({})
        
        # Добавляем узлы
        graph.add_node("source", {"label": "Source"})
        graph.add_node("target", {"label": "Target"})
        
        # Успешное добавление ребра
        result = graph.add_edge("source", "target", "related_to", 0.8)
        assert result is True
        assert graph.get_edge_count() == 1
        
        # Проверяем adjacency list
        neighbors = graph.get_neighbors("source")
        assert len(neighbors) == 1
        assert neighbors[0]["node_id"] == "target"
        assert neighbors[0]["relation"] == "related_to"
        assert neighbors[0]["weight"] == 0.8
        
        # Добавление дублирующего ребра
        result = graph.add_edge("source", "target", "related_to", 0.9)
        assert result is False
        assert graph.get_edge_count() == 1
        
        # Добавление ребра между несуществующими узлами
        result = graph.add_edge("nonexistent", "target", "related_to")
        assert result is False
    
    def test_remove_node(self, graph_instance):
        """Тест удаления узлов"""
        graph = graph_instance({})
        
        # Добавляем узлы и связи
        graph.add_node("node1", {"label": "Node 1"})
        graph.add_node("node2", {"label": "Node 2"})
        graph.add_node("node3", {"label": "Node 3"})
        graph.add_edge("node1", "node2", "related_to")
        graph.add_edge("node2", "node3", "related_to")
        graph.add_edge("node1", "node3", "related_to")
        
        assert graph.get_node_count() == 3
        assert graph.get_edge_count() == 3
        
        # Удаляем узел
        result = graph.remove_node("node2")
        assert result is True
        assert graph.get_node_count() == 2
        assert "node2" not in graph.nodes
        assert "node2" not in graph.adjacency_list
        
        # Проверяем, что связанные ребра удалены
        assert graph.get_edge_count() == 1  # Осталось только node1 -> node3
        
        # Попытка удалить несуществующий узел
        result = graph.remove_node("nonexistent")
        assert result is False
    
    def test_remove_edge(self, graph_instance):
        """Тест удаления рёбер"""
        graph = graph_instance({})
        
        # Добавляем узлы и связи
        graph.add_node("source", {"label": "Source"})
        graph.add_node("target", {"label": "Target"})
        graph.add_edge("source", "target", "related_to", 0.8)
        graph.add_edge("source", "target", "similar_to", 0.6)
        
        assert graph.get_edge_count() == 2
        
        # Удаляем одно ребро
        result = graph.remove_edge("source", "target", "related_to")
        assert result is True
        assert graph.get_edge_count() == 1
        
        # Проверяем, что осталось другое ребро
        neighbors = graph.get_neighbors("source")
        assert len(neighbors) == 1
        assert neighbors[0]["relation"] == "similar_to"
        
        # Попытка удалить несуществующее ребро
        result = graph.remove_edge("source", "target", "nonexistent")
        assert result is False
    
    def test_get_neighbors(self, graph_instance):
        """Тест получения соседей узла"""
        graph = graph_instance({})
        
        # Добавляем узлы и связи
        graph.add_node("node1", {"label": "Node 1"})
        graph.add_node("node2", {"label": "Node 2"})
        graph.add_node("node3", {"label": "Node 3"})
        graph.add_node("node4", {"label": "Node 4"})
        
        graph.add_edge("node1", "node2", "related_to", 0.8)
        graph.add_edge("node1", "node3", "similar_to", 0.6)
        graph.add_edge("node1", "node4", "related_to", 0.9)
        
        # Получаем всех соседей
        neighbors = graph.get_neighbors("node1")
        assert len(neighbors) == 3
        
        # Получаем соседей по определенному отношению
        related_neighbors = graph.get_neighbors("node1", "related_to")
        assert len(related_neighbors) == 2
        assert all(n["relation"] == "related_to" for n in related_neighbors)
        
        # Несуществующий узел
        neighbors = graph.get_neighbors("nonexistent")
        assert len(neighbors) == 0
    
    def test_search_related(self, graph_instance):
        """Тест поиска связанных узлов"""
        graph = graph_instance({})
        
        # Добавляем узлы с различными данными
        nodes_data = {
            "ai_concept": {
                "label": "Artificial Intelligence",
                "definition": "Computer system that can perform intelligent tasks",
                "properties": {"domain": "technology", "type": "field"}
            },
            "ml_concept": {
                "label": "Machine Learning", 
                "definition": "Subset of AI focused on data learning",
                "properties": {"domain": "technology", "type": "method"}
            },
            "bias_concept": {
                "label": "Cognitive Bias",
                "definition": "Systematic error in thinking patterns",
                "properties": {"domain": "psychology", "type": "concept"}
            }
        }
        
        for node_id, node_data in nodes_data.items():
            graph.add_node(node_id, node_data)
        
        # Поиск по "artificial"
        results = graph.search_related("artificial", k=2)
        assert len(results) >= 1
        assert any(r["id"] == "ai_concept" for r in results)
        
        # Поиск по "system"
        results = graph.search_related("system", k=2)
        assert len(results) >= 1
        assert any(r["id"] == "ai_concept" for r in results)
        
        # Поиск по "learning"
        results = graph.search_related("learning", k=2)
        assert len(results) >= 1
        assert any(r["id"] == "ml_concept" for r in results)
        
        # Поиск с пустым результатом
        results = graph.search_related("nonexistent", k=10)
        assert len(results) == 0
    
    def test_traverse(self, graph_instance):
        """Тест обхода графа"""
        graph = graph_instance({})
        
        # Создаем иерархическую структуру
        nodes = ["root", "child1", "child2", "grandchild1", "grandchild2", "great_grandchild"]
        
        for node in nodes:
            graph.add_node(node, {"label": node})
        
        # Создаем связи
        edges = [
            ("root", "child1"), ("root", "child2"),
            ("child1", "grandchild1"), ("child1", "grandchild2"),
            ("grandchild1", "great_grandchild")
        ]
        
        for source, target in edges:
            graph.add_edge(source, target, "parent_of")
        
        # Тест обхода в глубину
        traversal = graph.traverse("root", max_depth=3)
        assert "root" in traversal
        assert "child1" in traversal
        assert "child2" in traversal
        assert "grandchild1" in traversal
        assert "grandchild2" in traversal
        assert "great_grandchild" in traversal
        
        # Тест ограничения глубины
        shallow_traversal = graph.traverse("root", max_depth=1)
        assert len(shallow_traversal) <= 2  # root + immediate children
        
        # Тест с фильтром отношений
        filtered_traversal = graph.traverse("root", max_depth=3, relation_filter="parent_of")
        assert len(filtered_traversal) == len(traversal)  # Все связи - parent_of
    
    def test_get_path(self, graph_instance):
        """Тест поиска пути"""
        graph = graph_instance({})
        
        # Создаем граф
        nodes = ["A", "B", "C", "D", "E"]
        for node in nodes:
            graph.add_node(node, {"label": node})
        
        edges = [
            ("A", "B"), ("A", "C"),
            ("B", "D"), ("C", "D"),
            ("D", "E")
        ]
        
        for source, target in edges:
            graph.add_edge(source, target, "connected_to")
        
        # Находим пути
        path = graph.get_path("A", "E")
        assert len(path) > 0
        assert path[0] == "A"
        assert path[-1] == "E"
        assert "E" in path
        
        # Путь через B
        path = graph.get_path("A", "D")
        assert "B" in path or "C" in path
        
        # Несуществующий путь
        graph.add_node("isolated", {"label": "Isolated"})
        path = graph.get_path("A", "isolated")
        assert len(path) == 0
        
        # Путь к самому себе
        path = graph.get_path("A", "A")
        assert path == ["A"]
    
    def test_graph_statistics(self, graph_instance):
        """Тест статистики графа"""
        graph = graph_instance({})
        
        # Пустой граф
        stats = graph.get_graph_stats()
        assert stats["nodes"] == 0
        assert stats["edges"] == 0
        assert stats["avg_connections"] == 0
        assert stats["density"] == 0
        
        # Граф с данными
        nodes_data = {
            "node1": {"label": "Node 1"},
            "node2": {"label": "Node 2"},
            "node3": {"label": "Node 3"}
        }
        
        for node_id, node_data in nodes_data.items():
            graph.add_node(node_id, node_data)
        
        graph.add_edge("node1", "node2", "related_to")
        graph.add_edge("node1", "node3", "related_to")
        graph.add_edge("node2", "node3", "related_to")
        
        stats = graph.get_graph_stats()
        assert stats["nodes"] == 3
        assert stats["edges"] == 3
        assert stats["avg_connections"] > 0
        assert stats["density"] > 0
        assert stats["density"] <= 1.0
    
    def test_performance_large_graph(self, graph_instance):
        """Тест производительности на большом графе"""
        graph = graph_instance({"max_nodes": 10000, "max_edges": 50000})
        
        # Создаем большой граф
        start_time = time.time()
        
        # Добавляем 1000 узлов
        for i in range(1000):
            graph.add_node(f"node_{i}", {"label": f"Node {i}", "index": i})
        
        # Добавляем 5000 рёбер
        for i in range(1000):
            for j in range(5):  # Каждый узел связан с 5 другими
                target = f"node_{(i + j + 1) % 1000}"
                graph.add_edge(f"node_{i}", target, "related_to", 0.1 * j)
        
        add_time = time.time() - start_time
        
        # Тестируем поиск
        start_time = time.time()
        results = graph.search_related("node", k=100)
        search_time = time.time() - start_time
        
        # Тестируем обход
        start_time = time.time()
        traversal = graph.traverse("node_0", max_depth=5)
        traverse_time = time.time() - start_time
        
        # Проверяем производительность
        assert add_time < 5.0  # Добавление должно занимать менее 5 секунд
        assert search_time < 2.0  # Поиск должен занимать менее 2 секунд
        assert traverse_time < 3.0  # Обход должен занимать менее 3 секунд
        assert len(results) > 0
        assert len(traversal) > 0
    
    def test_graph_limits(self, graph_instance):
        """Тест ограничений графа"""
        # Граф с маленькими лимитами
        graph = graph_instance({"max_nodes": 3, "max_edges": 2})
        
        # Добавляем 3 узла (лимит)
        assert graph.add_node("node1", {"label": "Node 1"}) is True
        assert graph.add_node("node2", {"label": "Node 2"}) is True
        assert graph.add_node("node3", {"label": "Node 3"}) is True
        
        # 4-й узел не должен добавиться
        assert graph.add_node("node4", {"label": "Node 4"}) is False
        assert graph.get_node_count() == 3
        
        # Добавляем 2 ребра (лимит)
        assert graph.add_edge("node1", "node2", "related_to") is True
        assert graph.add_edge("node2", "node3", "related_to") is True
        
        # 3-е ребро не должно добавиться
        assert graph.add_edge("node1", "node3", "related_to") is False
        assert graph.get_edge_count() == 2
    
    @pytest.mark.asyncio
    async def test_concurrent_graph_operations(self, graph_instance):
        """Тест конкурентных операций с графом"""
        graph = graph_instance({"max_nodes": 1000, "max_edges": 5000})
        
        async def add_node_task(node_id: str, node_data: Dict[str, Any]):
            return graph.add_node(node_id, node_data)
        
        async def add_edge_task(source: str, target: str, relation: str):
            return graph.add_edge(source, target, relation)
        
        async def search_task(query: str):
            return graph.search_related(query, k=50)
        
        # Выполняем конкурентные задачи
        tasks = []
        
        # Задачи добавления узлов
        for i in range(50):
            tasks.append(add_node_task(f"node_{i}", {"label": f"Node {i}"}))
        
        # Задачи добавления рёбер
        for i in range(25):
            tasks.append(add_edge_task(f"node_{i}", f"node_{i+1}", "related_to"))
        
        # Выполняем все задачи
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Проверяем результаты
        assert graph.get_node_count() == 50
        assert graph.get_edge_count() == 25
        
        # Тестируем конкурентный поиск
        search_tasks = [search_task(f"node_{i}") for i in range(10)]
        search_results = await asyncio.gather(*search_tasks)
        
        assert all(len(results) > 0 for results in search_results)
    
    def test_graph_serialization(self, graph_instance):
        """Тест сериализации графа"""
        graph = graph_instance({})
        
        # Добавляем данные
        node_data = {
            "label": "Test Node",
            "type": "concept",
            "definition": "Test definition",
            "properties": {"key": "value", "number": 42}
        }
        
        graph.add_node("test_node", node_data)
        graph.add_edge("test_node", "test_node", "self_reference", 1.0)
        
        # Сериализуем в JSON
        graph_data = {
            "nodes": graph.nodes,
            "edges": graph.edges,
            "metadata": graph.metadata
        }
        
        json_data = json.dumps(graph_data)
        assert len(json_data) > 0
        
        # Десериализуем
        loaded_data = json.loads(json_data)
        assert "nodes" in loaded_data
        assert "edges" in loaded_data
        assert "metadata" in loaded_data
        assert loaded_data["nodes"]["test_node"]["label"] == "Test Node"
        assert len(loaded_data["edges"]) == 1
