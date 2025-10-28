"""
Integration тесты для интеграции KAG системы с семантической памятью
"""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional

from ..fixtures.conftest import base_config, memory_manager_mock
from ..test_data.cognitive_biases import COGNITIVE_BIASES_DATASET


class SemanticMemoryIntegrationTestSuite:
    """Комплексная тестовая подсистема для семантической памяти"""
    
    @pytest.fixture
    def semantic_memory_engine(self):
        """Mock движок семантической памяти для тестирования"""
        class SemanticMemoryEngine:
            def __init__(self, config: Dict[str, Any]):
                self.config = config
                self.max_concepts = config.get("max_concepts", 10000)
                self.embedding_dim = config.get("embedding_dim", 768)
                self.concepts = {}
                self.concept_embeddings = {}
                self.relationships = {}
                self.similarity_threshold = 0.7
                self.access_log = []
                self.performance_metrics = {
                    "insert_time": [],
                    "retrieval_time": [],
                    "similarity_computation_time": []
                }
            
            def store_concept(self, concept_data: Dict[str, Any]) -> bool:
                """Сохранить концепт в семантической памяти"""
                start_time = time.time()
                
                try:
                    concept_id = concept_data.get("id")
                    if not concept_id:
                        concept_id = f"concept_{len(self.concepts)}"
                    
                    # Проверяем емкость
                    if len(self.concepts) >= self.max_concepts:
                        return False
                    
                    # Сохраняем концепт
                    self.concepts[concept_id] = {
                        **concept_data,
                        "stored_at": time.time(),
                        "access_count": 0,
                        "embedding_version": 1
                    }
                    
                    # Генерируем эмбеддинг (mock)
                    self.concept_embeddings[concept_id] = self._generate_mock_embedding(concept_data)
                    
                    # Создаем отношения с существующими концептами
                    self._create_semantic_relations(concept_id, concept_data)
                    
                    # Записываем в лог доступа
                    self.access_log.append({
                        "timestamp": time.time(),
                        "operation": "store",
                        "concept_id": concept_id
                    })
                    
                    elapsed_time = time.time() - start_time
                    self.performance_metrics["insert_time"].append(elapsed_time)
                    
                    return True
                    
                except Exception as e:
                    self.access_log.append({
                        "timestamp": time.time(),
                        "operation": "store_failed",
                        "concept_id": concept_id,
                        "error": str(e)
                    })
                    return False
            
            def retrieve_concept(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
                """Извлечь релевантные концепты по запросу"""
                start_time = time.time()
                
                try:
                    # Генерируем эмбеддинг запроса
                    query_embedding = self._generate_query_embedding(query)
                    
                    # Вычисляем схожесть с существующими концептами
                    similarities = []
                    for concept_id, concept_embedding in self.concept_embeddings.items():
                        similarity = self._compute_similarity(query_embedding, concept_embedding)
                        if similarity >= self.similarity_threshold:
                            similarities.append({
                                "concept_id": concept_id,
                                "concept_data": self.concepts[concept_id],
                                "similarity_score": similarity
                            })
                    
                    # Сортируем по схожести
                    similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
                    
                    # Ограничиваем количество результатов
                    results = similarities[:top_k]
                    
                    # Обновляем счетчики доступа
                    for result in results:
                        concept_id = result["concept_id"]
                        self.concepts[concept_id]["access_count"] += 1
                    
                    # Записываем в лог доступа
                    self.access_log.append({
                        "timestamp": time.time(),
                        "operation": "retrieve",
                        "query": query,
                        "results_count": len(results)
                    })
                    
                    elapsed_time = time.time() - start_time
                    self.performance_metrics["retrieval_time"].append(elapsed_time)
                    
                    return results
                    
                except Exception as e:
                    self.access_log.append({
                        "timestamp": time.time(),
                        "operation": "retrieve_failed",
                        "query": query,
                        "error": str(e)
                    })
                    return []
            
            def update_concept(self, concept_id: str, updates: Dict[str, Any]) -> bool:
                """Обновить концепт в семантической памяти"""
                if concept_id not in self.concepts:
                    return False
                
                try:
                    # Сохраняем предыдущую версию
                    old_data = self.concepts[concept_id].copy()
                    
                    # Обновляем данные
                    self.concepts[concept_id].update(updates)
                    self.concepts[concept_id]["last_updated"] = time.time()
                    self.concepts[concept_id]["embedding_version"] += 1
                    
                    # Перегенерируем эмбеддинг
                    self.concept_embeddings[concept_id] = self._generate_mock_embedding(self.concepts[concept_id])
                    
                    # Записываем в лог доступа
                    self.access_log.append({
                        "timestamp": time.time(),
                        "operation": "update",
                        "concept_id": concept_id,
                        "old_data_keys": list(old_data.keys()),
                        "new_data_keys": list(self.concepts[concept_id].keys())
                    })
                    
                    return True
                    
                except Exception as e:
                    self.access_log.append({
                        "timestamp": time.time(),
                        "operation": "update_failed",
                        "concept_id": concept_id,
                        "error": str(e)
                    })
                    return False
            
            def find_related_concepts(self, concept_id: str, depth: int = 2) -> List[Dict[str, Any]]:
                """Найти связанные концепты через семантические связи"""
                if concept_id not in self.concepts:
                    return []
                
                related = []
                visited = set()
                
                def dfs(current_id: str, current_depth: int, current_path: List[str]):
                    if current_id in visited or current_depth > depth:
                        return
                    
                    visited.add(current_id)
                    
                    # Добавляем текущий концепт
                    if current_id != concept_id:
                        related.append({
                            "concept_id": current_id,
                            "concept_data": self.concepts[current_id],
                            "distance": current_depth,
                            "path": current_path + [current_id]
                        })
                    
                    # Исследуем связи
                    if current_id in self.relationships:
                        for related_id in self.relationships[current_id]:
                            dfs(related_id, current_depth + 1, current_path + [current_id])
                
                dfs(concept_id, 0, [])
                return related
            
            def _generate_mock_embedding(self, concept_data: Dict[str, Any]) -> List[float]:
                """Генерировать mock эмбеддинг для концепта"""
                # Простой хеш-основанный эмбеддинг
                concept_str = json.dumps(concept_data, sort_keys=True)
                import hashlib
                hash_obj = hashlib.md5(concept_str.encode())
                hash_bytes = hash_obj.digest()
                
                # Преобразуем в список чисел
                embedding = []
                for i in range(self.embedding_dim):
                    byte_idx = i % len(hash_bytes)
                    value = (hash_bytes[byte_idx] / 255.0) * 2 - 1  # Нормализация к [-1, 1]
                    embedding.append(value)
                
                return embedding
            
            def _generate_query_embedding(self, query: str) -> List[float]:
                """Генерировать эмбеддинг для запроса"""
                return self._generate_mock_embedding({"query": query})
            
            def _compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
                """Вычислить косинусную схожесть между эмбеддингами"""
                import math
                
                dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
                magnitude1 = math.sqrt(sum(a * a for a in embedding1))
                magnitude2 = math.sqrt(sum(a * a for a in embedding2))
                
                if magnitude1 == 0 or magnitude2 == 0:
                    return 0.0
                
                return dot_product / (magnitude1 * magnitude2)
            
            def _create_semantic_relations(self, new_concept_id: str, concept_data: Dict[str, Any]):
                """Создать семантические отношения с существующими концептами"""
                # Простая логика на основе пересечения ключевых слов
                new_keywords = self._extract_keywords(concept_data)
                
                for existing_id, existing_data in self.concepts.items():
                    if existing_id == new_concept_id:
                        continue
                    
                    existing_keywords = self._extract_keywords(existing_data)
                    
                    # Вычисляем пересечение ключевых слов
                    intersection = new_keywords.intersection(existing_keywords)
                    if len(intersection) > 0:
                        relation_strength = len(intersection) / max(len(new_keywords), len(existing_keywords))
                        
                        if relation_strength > 0.3:  # Минимальная сила связи
                            # Добавляем связь
                            if new_concept_id not in self.relationships:
                                self.relationships[new_concept_id] = []
                            if existing_id not in self.relationships:
                                self.relationships[existing_id] = []
                            
                            self.relationships[new_concept_id].append(existing_id)
                            self.relationships[existing_id].append(new_concept_id)
            
            def _extract_keywords(self, data: Dict[str, Any]) -> set:
                """Извлечь ключевые слова из данных концепта"""
                keywords = set()
                
                # Извлекаем из различных полей
                text_fields = ["name", "definition", "description", "category"]
                for field in text_fields:
                    if field in data:
                        words = str(data[field]).lower().split()
                        keywords.update(words)
                
                # Извлекаем из related_concepts
                if "related_concepts" in data:
                    if isinstance(data["related_concepts"], list):
                        keywords.update(str(c).lower() for c in data["related_concepts"])
                
                # Убираем стоп-слова
                stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
                keywords = keywords - stop_words
                
                return keywords
            
            def get_performance_metrics(self) -> Dict[str, Any]:
                """Получить метрики производительности"""
                return {
                    "total_concepts": len(self.concepts),
                    "total_relationships": len(self.relationships),
                    "avg_insert_time": sum(self.performance_metrics["insert_time"]) / max(len(self.performance_metrics["insert_time"]), 1),
                    "avg_retrieval_time": sum(self.performance_metrics["retrieval_time"]) / max(len(self.performance_metrics["retrieval_time"]), 1),
                    "total_operations": len(self.access_log),
                    "access_count_distribution": self._get_access_distribution(),
                    "capacity_usage": (len(self.concepts) / self.max_concepts) * 100
                }
            
            def _get_access_distribution(self) -> Dict[str, int]:
                """Получить распределение доступа к концептам"""
                distribution = {}
                for access in self.access_log:
                    if access["operation"] in ["retrieve", "update"]:
                        concept_id = access.get("concept_id")
                        if concept_id:
                            distribution[concept_id] = distribution.get(concept_id, 0) + 1
                return distribution
            
            def reset(self):
                """Сбросить состояние семантической памяти"""
                self.concepts.clear()
                self.concept_embeddings.clear()
                self.relationships.clear()
                self.access_log.clear()
                self.performance_metrics = {
                    "insert_time": [],
                    "retrieval_time": [],
                    "similarity_computation_time": []
                }
        
        return SemanticMemoryEngine
    
    def test_semantic_memory_initialization(self, semantic_memory_engine, base_config):
        """Тест инициализации семантической памяти"""
        semantic_config = base_config["memory_layers"]["semantic"]
        memory_engine = semantic_memory_engine(semantic_config)
        
        assert memory_engine.max_concepts == semantic_config["max_concepts"]
        assert memory_engine.embedding_dim == semantic_config["embedding_dim"]
        assert len(memory_engine.concepts) == 0
        assert len(memory_engine.concept_embeddings) == 0
        assert len(memory_engine.relationships) == 0
    
    def test_concept_storage_basic(self, semantic_memory_engine, base_config):
        """Тест базового сохранения концептов"""
        semantic_config = base_config["memory_layers"]["semantic"]
        memory_engine = semantic_memory_engine(semantic_config)
        
        # Тестовый концепт
        test_concept = {
            "id": "ai_concept_001",
            "name": "Artificial Intelligence",
            "definition": "Computer system capable of intelligent behavior",
            "category": "technology",
            "properties": {
                "domain": "computer_science",
                "complexity": "high"
            }
        }
        
        result = memory_engine.store_concept(test_concept)
        
        assert result is True
        assert "ai_concept_001" in memory_engine.concepts
        assert memory_engine.concepts["ai_concept_001"]["name"] == "Artificial Intelligence"
        assert "ai_concept_001" in memory_engine.concept_embeddings
        assert len(memory_engine.concept_embeddings["ai_concept_001"]) == semantic_config["embedding_dim"]
    
    def test_concept_retrieval_basic(self, semantic_memory_engine, base_config):
        """Тест базового извлечения концептов"""
        semantic_config = base_config["memory_layers"]["semantic"]
        memory_engine = semantic_memory_engine(semantic_config)
        
        # Сохраняем тестовые концепты
        concepts = [
            {
                "id": "ai_concept_001",
                "name": "Artificial Intelligence",
                "definition": "Computer system capable of intelligent behavior",
                "category": "technology"
            },
            {
                "id": "ml_concept_001", 
                "name": "Machine Learning",
                "definition": "Subset of AI focused on algorithms that learn from data",
                "category": "technology"
            },
            {
                "id": "bias_concept_001",
                "name": "Cognitive Bias",
                "definition": "Systematic error in thinking patterns",
                "category": "psychology"
            }
        ]
        
        for concept in concepts:
            assert memory_engine.store_concept(concept) is True
        
        # Тестируем извлечение по запросу
        results = memory_engine.retrieve_concept("machine learning", top_k=5)
        
        assert len(results) > 0
        
        # Проверяем что ml_concept_001 найден
        found_ml = any(r["concept_id"] == "ml_concept_001" for r in results)
        assert found_ml
        
        # Проверяем качество ранжирования
        if len(results) >= 2:
            assert results[0]["similarity_score"] >= results[1]["similarity_score"]
    
    def test_semantic_relationships_creation(self, semantic_memory_engine, base_config):
        """Тест создания семантических связей"""
        semantic_config = base_config["memory_layers"]["semantic"]
        memory_engine = semantic_memory_engine(semantic_config)
        
        # Создаем связанные концепты
        concepts = [
            {
                "id": "ai_base",
                "name": "Artificial Intelligence",
                "definition": "Computer system capable of intelligent behavior",
                "category": "technology",
                "related_concepts": ["machine_learning", "neural_networks"]
            },
            {
                "id": "ml_subset",
                "name": "Machine Learning",
                "definition": "Subset of AI focused on learning algorithms",
                "category": "technology", 
                "related_concepts": ["artificial_intelligence", "deep_learning"]
            },
            {
                "id": "bias_psychology",
                "name": "Cognitive Bias",
                "definition": "Systematic error in thinking",
                "category": "psychology",
                "related_concepts": ["confirmation_bias", "availability_heuristic"]
            }
        ]
        
        for concept in concepts:
            memory_engine.store_concept(concept)
        
        # Проверяем создание связей
        assert len(memory_engine.relationships) > 0
        
        # Проверяем конкретную связь между AI и ML
        ai_relations = memory_engine.relationships.get("ai_base", [])
        ml_relations = memory_engine.relationships.get("ml_subset", [])
        
        # Должна быть связь между связанными концептами
        assert "ml_subset" in ai_relations or "ai_base" in ml_relations
    
    def test_concept_update_functionality(self, semantic_memory_engine, base_config):
        """Тест обновления концептов"""
        semantic_config = base_config["memory_layers"]["semantic"]
        memory_engine = semantic_memory_engine(semantic_config)
        
        # Сохраняем исходный концепт
        original_concept = {
            "id": "update_test_001",
            "name": "Original Concept",
            "definition": "Original definition",
            "category": "test",
            "properties": {"version": 1}
        }
        
        memory_engine.store_concept(original_concept)
        
        # Обновляем концепт
        updates = {
            "name": "Updated Concept",
            "definition": "Updated definition",
            "properties": {"version": 2, "updated": True}
        }
        
        result = memory_engine.update_concept("update_test_001", updates)
        
        assert result is True
        
        # Проверяем обновления
        updated_concept = memory_engine.concepts["update_test_001"]
        assert updated_concept["name"] == "Updated Concept"
        assert updated_concept["definition"] == "Updated definition"]
        assert updated_concept["properties"]["version"] == 2
        assert updated_concept["properties"]["updated"] is True
        assert "last_updated" in updated_concept
        assert updated_concept["embedding_version"] == 2
        
        # Проверяем что эмбеддинг обновлен
        new_embedding = memory_engine.concept_embeddings["update_test_001"]
        assert len(new_embedding) == semantic_config["embedding_dim"]
    
    def test_related_concepts_traversal(self, semantic_memory_engine, base_config):
        """Тест обхода связанных концептов"""
        semantic_config = base_config["memory_layers"]["semantic"]
        memory_engine = semantic_memory_engine(semantic_config)
        
        # Создаем сеть связанных концептов
        concepts = [
            {"id": "root", "name": "Root Concept", "definition": "Root", "category": "base"},
            {"id": "child1", "name": "Child 1", "definition": "First child", "category": "level1"},
            {"id": "child2", "name": "Child 2", "definition": "Second child", "category": "level1"},
            {"id": "grandchild", "name": "Grandchild", "definition": "Child of child1", "category": "level2"}
        ]
        
        for concept in concepts:
            memory_engine.store_concept(concept)
        
        # Настраиваем связи вручную для теста
        memory_engine.relationships = {
            "root": ["child1", "child2"],
            "child1": ["root", "grandchild"],
            "child2": ["root"],
            "grandchild": ["child1"]
        }
        
        # Тестируем поиск связанных концептов
        related = memory_engine.find_related_concepts("root", depth=2)
        
        assert len(related) > 0
        
        # Проверяем что найдены прямые и косвенные связи
        found_ids = {r["concept_id"] for r in related}
        assert "child1" in found_ids
        assert "child2" in found_ids
        
        # Проверяем расстояния
        distance_1 = [r for r in related if r["concept_id"] == "child1"][0]["distance"]
        distance_2 = [r for r in related if r["concept_id"] == "child2"][0]["distance"]
        
        assert distance_1 == 1
        assert distance_2 == 1
    
    def test_performance_metrics_collection(self, semantic_memory_engine, base_config):
        """Тест сбора метрик производительности"""
        semantic_config = base_config["memory_layers"]["semantic"]
        memory_engine = semantic_memory_engine(semantic_config)
        
        # Выполняем операции и собираем метрики
        for i in range(10):
            concept = {
                "id": f"perf_test_{i}",
                "name": f"Performance Test Concept {i}",
                "definition": f"Definition {i}",
                "category": "performance"
            }
            memory_engine.store_concept(concept)
        
        for i in range(5):
            memory_engine.retrieve_concept(f"query {i}")
        
        # Получаем метрики
        metrics = memory_engine.get_performance_metrics()
        
        assert metrics["total_concepts"] == 10
        assert metrics["total_operations"] == 15  # 10 stores + 5 retrieves
        assert metrics["avg_insert_time"] > 0
        assert metrics["avg_retrieval_time"] > 0
        assert metrics["capacity_usage"] == (10 / semantic_config["max_concepts"]) * 100
    
    def test_capacity_limit_enforcement(self, semantic_memory_engine, base_config):
        """Тест соблюдения лимитов емкости"""
        small_config = {"max_concepts": 3, "embedding_dim": 128}
        memory_engine = semantic_memory_engine(small_config)
        
        # Заполняем до лимита
        for i in range(3):
            concept = {
                "id": f"capacity_test_{i}",
                "name": f"Capacity Test {i}",
                "definition": f"Definition {i}"
            }
            assert memory_engine.store_concept(concept) is True
        
        # Пытаемся добавить сверх лимита
        overflow_concept = {
            "id": "overflow",
            "name": "Overflow Concept",
            "definition": "Should not be stored"
        }
        
        result = memory_engine.store_concept(overflow_concept)
        assert result is False
        assert len(memory_engine.concepts) == 3  # Количество не изменилось
    
    @pytest.mark.asyncio
    async def test_concurrent_semantic_operations(self, semantic_memory_engine, base_config):
        """Тест конкурентных операций с семантической памятью"""
        semantic_config = base_config["memory_layers"]["semantic"]
        memory_engine = semantic_memory_engine(semantic_config)
        
        async def store_concept_task(concept_id: int):
            concept = {
                "id": f"concurrent_store_{concept_id}",
                "name": f"Concurrent Concept {concept_id}",
                "definition": f"Definition {concept_id}",
                "category": "concurrent"
            }
            return memory_engine.store_concept(concept)
        
        async def retrieve_concept_task(query: str):
            return memory_engine.retrieve_concept(query, top_k=5)
        
        # Выполняем конкурентные операции
        start_time = time.time()
        store_tasks = [store_concept_task(i) for i in range(20)]
        retrieve_tasks = [retrieve_concept_task(f"concept {i}") for i in range(10)]
        
        store_results = await asyncio.gather(*store_tasks)
        retrieve_results = await asyncio.gather(*retrieve_tasks)
        
        concurrent_time = time.time() - start_time
        
        # Проверяем результаты
        assert all(result is True for result in store_results)
        assert all(isinstance(result, list) for result in retrieve_results)
        
        # Проверяем итоговое состояние
        metrics = memory_engine.get_performance_metrics()
        assert metrics["total_concepts"] == 20
        assert concurrent_time < 10.0  # Операции должны завершиться быстро
    
    def test_cognitive_bias_integration(self, semantic_memory_engine, base_config):
        """Тест интеграции с данными о когнитивных искажениях"""
        semantic_config = base_config["memory_layers"]["semantic"]
        memory_engine = semantic_memory_engine(semantic_config)
        
        # Сохраняем концепты когнитивных искажений
        bias_count = 0
        for bias_name, bias_data in COGNITIVE_BIASES_DATASET.items():
            if bias_count >= 10:  # Ограничиваем для теста
                break
                
            concept = {
                "id": bias_data["id"],
                "name": bias_data["name"],
                "definition": bias_data["definition"],
                "category": "cognitive_bias",
                "severity": bias_data["severity"],
                "impact_score": bias_data["impact_score"],
                "related_concepts": bias_data.get("related_concepts", []),
                "examples": bias_data.get("examples", {}),
                "mitigation_techniques": bias_data.get("mitigation_techniques", [])
            }
            
            assert memory_engine.store_concept(concept) is True
            bias_count += 1
        
        # Тестируем извлечение по различным запросам
        test_queries = [
            "cognitive bias psychology",
            "confirmation bias detection",
            "bias mitigation techniques",
            "judgment heuristic errors"
        ]
        
        for query in test_queries:
            results = memory_engine.retrieve_concept(query, top_k=5)
            assert len(results) > 0
            
            # Проверяем качество результатов
            for result in results:
                assert "concept_id" in result
                assert "similarity_score" in result
                assert result["similarity_score"] >= memory_engine.similarity_threshold
    
    def test_semantic_memory_error_handling(self, semantic_memory_engine, base_config):
        """Тест обработки ошибок в семантической памяти"""
        semantic_config = base_config["memory_layers"]["semantic"]
        memory_engine = semantic_memory_engine(semantic_config)
        
        # Тест с некорректными данными
        invalid_concept = None
        result = memory_engine.store_concept(invalid_concept)
        assert result is False
        
        # Тест с пустыми данными
        empty_concept = {}
        result = memory_engine.store_concept(empty_concept)
        assert result is True  # Пустой концепт должен сохраниться с автоматически сгенерированным ID
        
        # Тест извлечения несуществующего концепта
        related = memory_engine.find_related_concepts("nonexistent_id")
        assert related == []
        
        # Тест обновления несуществующего концепта
        result = memory_engine.update_concept("nonexistent_id", {"name": "test"})
        assert result is False
        
        # Проверяем что лог доступа содержит записи об ошибках
        error_logs = [log for log in memory_engine.access_log if "failed" in log["operation"]]
        assert len(error_logs) > 0
    
    def test_access_pattern_analysis(self, semantic_memory_engine, base_config):
        """Тест анализа паттернов доступа"""
        semantic_config = base_config["memory_layers"]["semantic"]
        memory_engine = semantic_memory_engine(semantic_config)
        
        # Создаем концепты с разными паттернами доступа
        concepts = []
        for i in range(10):
            concept = {
                "id": f"access_test_{i}",
                "name": f"Access Test Concept {i}",
                "definition": f"Definition {i}",
                "category": "access_test"
            }
            memory_engine.store_concept(concept)
            concepts.append(f"access_test_{i}")
        
        # Создаем разные паттерны доступа
        # Часто используемые концепты
        for _ in range(5):
            memory_engine.retrieve_concept("access test 1")
        
        # Средне используемые
        for _ in range(3):
            memory_engine.retrieve_concept("access test 2")
            memory_engine.retrieve_concept("access test 3")
        
        # Редко используемые - остальные концепты
        
        # Получаем распределение доступа
        metrics = memory_engine.get_performance_metrics()
        access_distribution = metrics["access_count_distribution"]
        
        assert len(access_distribution) > 0
        assert access_distribution.get("access_test_1", 0) >= 5
        assert access_distribution.get("access_test_2", 0) >= 3
    
    def test_embedding_consistency(self, semantic_memory_engine, base_config):
        """Тест консистентности эмбеддингов"""
        semantic_config = base_config["memory_layers"]["semantic"]
        memory_engine = semantic_memory_engine(semantic_config)
        
        # Создаем одинаковые концепты дважды
        concept_data = {
            "id": "consistency_test",
            "name": "Consistency Test",
            "definition": "Testing embedding consistency",
            "category": "test"
        }
        
        # Первый раз
        result1 = memory_engine.store_concept(concept_data)
        embedding1 = memory_engine.concept_embeddings["consistency_test"].copy()
        
        # Удаляем и создаем заново
        memory_engine.reset()
        result2 = memory_engine.store_concept(concept_data)
        embedding2 = memory_engine.concept_embeddings["consistency_test"].copy()
        
        # Проверяем что эмбеддинги идентичны
        assert result1 is True
        assert result2 is True
        assert embedding1 == embedding2
        assert len(embedding1) == semantic_config["embedding_dim"]
        
        # Проверяем что после обновления эмбеддинг изменился
        memory_engine.update_concept("consistency_test", {"name": "Updated Consistency Test"})
        embedding3 = memory_engine.concept_embeddings["consistency_test"].copy()
        
        assert embedding1 != embedding3  # Эмбеддинг должен измениться после обновления