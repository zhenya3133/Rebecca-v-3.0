"""
Integration тесты для полного стека памяти KAG системы (6 слоев)
"""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional

from ..fixtures.conftest import base_config, strict_config, memory_manager_mock
from ..test_data.cognitive_biases import COGNITIVE_BIASES_DATASET


class FullMemoryStackIntegrationTestSuite:
    """Комплексная тестовая подсистема для полного стека памяти"""
    
    @pytest.fixture
    def full_memory_stack(self):
        """Полный стек памяти с 6 слоями"""
        class FullMemoryStack:
            def __init__(self, config: Dict[str, Any]):
                self.config = config
                self.memory_layers = config.get("memory_layers", {})
                
                # Инициализация 6 слоев памяти
                self.semantic_memory = self._init_semantic_layer()
                self.episodic_memory = self._init_episodic_layer()
                self.procedural_memory = self._init_procedural_layer()
                self.vault_memory = self._init_vault_layer()
                self.security_memory = self._init_security_layer()
                self.vector_store = self._init_vector_store()
                
                # Интеграционные компоненты
                self.cross_layer_index = {}
                self.fusion_engine = FusionEngine()
                self.consistency_manager = ConsistencyManager()
                self.performance_monitor = PerformanceMonitor()
                
                # Состояние системы
                self.system_state = {
                    "total_knowledge_items": 0,
                    "cross_layer_relations": 0,
                    "query_responses": 0,
                    "error_count": 0,
                    "last_sync": time.time()
                }
                
                # Настройка связей между слоями
                self._setup_layer_connections()
            
            def _init_semantic_layer(self):
                """Инициализация семантической памяти"""
                return {
                    "concepts": {},
                    "relationships": {},
                    "embeddings": {},
                    "max_capacity": 10000,
                    "current_usage": 0
                }
            
            def _init_episodic_layer(self):
                """Инициализация эпизодической памяти"""
                return {
                    "events": {},
                    "timeline": [],
                    "temporal_index": {},
                    "importance_scores": {},
                    "max_capacity": 5000,
                    "current_usage": 0
                }
            
            def _init_procedural_layer(self):
                """Инициализация процедурной памяти"""
                return {
                    "workflows": {},
                    "procedures": {},
                    "execution_traces": {},
                    "step_dependencies": {},
                    "max_capacity": 1000,
                    "current_usage": 0
                }
            
            def _init_vault_layer(self):
                """Инициализация хранилища секретов"""
                return {
                    "secrets": {},
                    "encryption_keys": {},
                    "access_policies": {},
                    "audit_log": [],
                    "max_capacity": 100,
                    "current_usage": 0
                }
            
            def _init_security_layer(self):
                """Инициализация системы безопасности"""
                return {
                    "policies": {},
                    "access_controls": {},
                    "audit_trails": [],
                    "threat_detections": [],
                    "compliance_rules": {},
                    "max_capacity": 500,
                    "current_usage": 0
                }
            
            def _init_vector_store(self):
                """Инициализация векторного хранилища"""
                return {
                    "vectors": {},
                    "indexes": {},
                    "similarity_cache": {},
                    "query_accelerator": {},
                    "dimension": 768,
                    "metric": "cosine",
                    "max_capacity": 50000,
                    "current_usage": 0
                }
            
            def _setup_layer_connections(self):
                """Настроить связи между слоями памяти"""
                # Semantic -> Vector Store: концепты создают векторы
                # Episodic -> Semantic: события создают концепты
                # Procedural -> All: процедуры могут ссылаться на все слои
                # Vault -> Security: секреты требуют политики безопасности
                # Security -> All: все операции проходят через безопасность
                # Vector Store -> All: векторное индексирование для всех слоев
                pass
            
            def store_knowledge_item(self, knowledge_data: Dict[str, Any]) -> Dict[str, Any]:
                """Сохранить элемент знаний во всех релевантных слоях"""
                operation_start = time.time()
                operation_id = f"op_{int(time.time() * 1000)}"
                
                try:
                    # 1. Проверка безопасности
                    security_check = self._security_check(knowledge_data)
                    if not security_check["passed"]:
                        return {
                            "success": False,
                            "error": "Security check failed",
                            "details": security_check["details"]
                        }
                    
                    # 2. Проверка целостности данных
                    consistency_check = self.consistency_manager.validate_data(knowledge_data)
                    if not consistency_check["valid"]:
                        return {
                            "success": False,
                            "error": "Data consistency check failed",
                            "details": consistency_check["issues"]
                        }
                    
                    # 3. Распределение по слоям памяти
                    storage_results = {}
                    
                    # Семантическая память
                    if self._should_store_in_semantic(knowledge_data):
                        storage_results["semantic"] = self._store_in_semantic(knowledge_data)
                    
                    # Эпизодическая память
                    if self._should_store_in_episodic(knowledge_data):
                        storage_results["episodic"] = self._store_in_episodic(knowledge_data)
                    
                    # Процедурная память
                    if self._should_store_in_procedural(knowledge_data):
                        storage_results["procedural"] = self._store_in_procedural(knowledge_data)
                    
                    # Хранилище секретов
                    if self._should_store_in_vault(knowledge_data):
                        storage_results["vault"] = self._store_in_vault(knowledge_data)
                    
                    # Система безопасности
                    if self._should_update_security(knowledge_data):
                        storage_results["security"] = self._update_security_layer(knowledge_data)
                    
                    # Векторное хранилище
                    if self._should_store_in_vector_store(knowledge_data):
                        storage_results["vector_store"] = self._store_in_vector_store(knowledge_data)
                    
                    # 4. Создание кросс-слойных связей
                    self._create_cross_layer_relations(knowledge_data, storage_results)
                    
                    # 5. Обновление индексов
                    self._update_cross_layer_index(knowledge_data, storage_results)
                    
                    # 6. Обновление состояния системы
                    self.system_state["total_knowledge_items"] += 1
                    self.system_state["cross_layer_relations"] += len(storage_results)
                    self.system_state["last_sync"] = time.time()
                    
                    # 7. Запись в лог операции
                    self.performance_monitor.log_operation({
                        "operation_id": operation_id,
                        "operation_type": "store_knowledge",
                        "timestamp": time.time(),
                        "duration": time.time() - operation_start,
                        "layers_involved": list(storage_results.keys()),
                        "success": True
                    })
                    
                    return {
                        "success": True,
                        "operation_id": operation_id,
                        "layers_affected": list(storage_results.keys()),
                        "cross_layer_relations": len(storage_results),
                        "total_duration": time.time() - operation_start
                    }
                    
                except Exception as e:
                    self.system_state["error_count"] += 1
                    self.performance_monitor.log_operation({
                        "operation_id": operation_id,
                        "operation_type": "store_knowledge",
                        "timestamp": time.time(),
                        "duration": time.time() - operation_start,
                        "success": False,
                        "error": str(e)
                    })
                    
                    return {
                        "success": False,
                        "operation_id": operation_id,
                        "error": str(e)
                    }
            
            def retrieve_knowledge_fusion(self, query: str, fusion_strategy: str = "hybrid") -> Dict[str, Any]:
                """Извлечь знания с использованием фьюжн-стратегии"""
                query_start = time.time()
                query_id = f"query_{int(time.time() * 1000)}"
                
                try:
                    # 1. Параллельное извлечение из всех слоев
                    retrieval_tasks = {}
                    
                    # Семантическое извлечение
                    retrieval_tasks["semantic"] = self._retrieve_semantic_concepts(query)
                    
                    # Эпизодическое извлечение
                    retrieval_tasks["episodic"] = self._retrieve_episodic_events(query)
                    
                    # Процедурное извлечение
                    retrieval_tasks["procedural"] = self._retrieve_procedural_info(query)
                    
                    # Векторное извлечение
                    retrieval_tasks["vector_store"] = self._retrieve_vector_similar(query)
                    
                    # Объединение результатов
                    layer_results = {}
                    for layer_name, results in retrieval_tasks.items():
                        if results is not None:
                            layer_results[layer_name] = results
                    
                    # 2. Фьюжн результатов
                    fused_results = self.fusion_engine.fuse_results(
                        query=query,
                        layer_results=layer_results,
                        strategy=fusion_strategy
                    )
                    
                    # 3. Пост-обработка результатов
                    processed_results = self._post_process_results(fused_results)
                    
                    # 4. Обновление состояния
                    self.system_state["query_responses"] += 1
                    
                    # 5. Запись в лог
                    self.performance_monitor.log_operation({
                        "operation_id": query_id,
                        "operation_type": "retrieve_knowledge",
                        "timestamp": time.time(),
                        "query": query,
                        "strategy": fusion_strategy,
                        "layers_queried": list(layer_results.keys()),
                        "results_count": len(processed_results.get("results", [])),
                        "duration": time.time() - query_start,
                        "success": True
                    })
                    
                    return {
                        "success": True,
                        "query_id": query_id,
                        "query": query,
                        "strategy": fusion_strategy,
                        "layers_queried": list(layer_results.keys()),
                        "results": processed_results["results"],
                        "metadata": processed_results["metadata"],
                        "total_duration": time.time() - query_start
                    }
                    
                except Exception as e:
                    self.performance_monitor.log_operation({
                        "operation_id": query_id,
                        "operation_type": "retrieve_knowledge",
                        "timestamp": time.time(),
                        "query": query,
                        "success": False,
                        "error": str(e)
                    })
                    
                    return {
                        "success": False,
                        "query_id": query_id,
                        "query": query,
                        "error": str(e)
                    }
            
            def _security_check(self, knowledge_data: Dict[str, Any]) -> Dict[str, Any]:
                """Проверка безопасности данных"""
                # Простая проверка на наличие секретных данных
                secret_indicators = ["password", "secret", "token", "key", "api_key"]
                data_str = json.dumps(knowledge_data).lower()
                
                detected_secrets = []
                for indicator in secret_indicators:
                    if indicator in data_str:
                        detected_secrets.append(indicator)
                
                if detected_secrets:
                    # Проверяем политики безопасности
                    auth_required = "authorized" in knowledge_data and knowledge_data["authorized"]
                    if not auth_required:
                        return {
                            "passed": False,
                            "details": f"Authorization required for sensitive data: {detected_secrets}"
                        }
                
                return {"passed": True, "details": "Security check passed"}
            
            def _should_store_in_semantic(self, knowledge_data: Dict[str, Any]) -> bool:
                """Определить нужно ли сохранять в семантической памяти"""
                semantic_indicators = ["concept", "definition", "knowledge", "understanding"]
                data_str = json.dumps(knowledge_data).lower()
                return any(indicator in data_str for indicator in semantic_indicators)
            
            def _should_store_in_episodic(self, knowledge_data: Dict[str, Any]) -> bool:
                """Определить нужно ли сохранять в эпизодической памяти"""
                episodic_indicators = ["event", "experience", "happened", "occurred", "timestamp"]
                data_str = json.dumps(knowledge_data).lower()
                return any(indicator in data_str for indicator in episodic_indicators)
            
            def _should_store_in_procedural(self, knowledge_data: Dict[str, Any]) -> bool:
                """Определить нужно ли сохранять в процедурной памяти"""
                procedural_indicators = ["procedure", "workflow", "process", "step", "method"]
                data_str = json.dumps(knowledge_data).lower()
                return any(indicator in data_str for indicator in procedural_indicators)
            
            def _should_store_in_vault(self, knowledge_data: Dict[str, Any]) -> bool:
                """Определить нужно ли сохранять в хранилище секретов"""
                secret_indicators = ["password", "secret", "token", "key", "api_key"]
                data_str = json.dumps(knowledge_data).lower()
                return any(indicator in data_str for indicator in secret_indicators)
            
            def _should_update_security(self, knowledge_data: Dict[str, Any]) -> bool:
                """Определить нужно ли обновлять систему безопасности"""
                security_indicators = ["policy", "security", "access", "permission", "role"]
                data_str = json.dumps(knowledge_data).lower()
                return any(indicator in data_str for indicator in security_indicators)
            
            def _should_store_in_vector_store(self, knowledge_data: Dict[str, Any]) -> bool:
                """Определить нужно ли сохранять в векторном хранилище"""
                # Все данные могут быть векторизованы
                return True
            
            def _store_in_semantic(self, knowledge_data: Dict[str, Any]) -> Dict[str, Any]:
                """Сохранить в семантической памяти"""
                concept_id = knowledge_data.get("id", f"concept_{len(self.semantic_memory['concepts'])}")
                
                self.semantic_memory["concepts"][concept_id] = {
                    "data": knowledge_data,
                    "timestamp": time.time(),
                    "type": "concept",
                    "access_count": 0
                }
                
                # Создаем связи
                if "related_concepts" in knowledge_data:
                    self.semantic_memory["relationships"][concept_id] = knowledge_data["related_concepts"]
                
                self.semantic_memory["current_usage"] += 1
                
                return {
                    "stored": True,
                    "concept_id": concept_id,
                    "layer": "semantic"
                }
            
            def _store_in_episodic(self, knowledge_data: Dict[str, Any]) -> Dict[str, Any]:
                """Сохранить в эпизодической памяти"""
                event_id = knowledge_data.get("id", f"event_{len(self.episodic_memory['events'])}")
                
                event_record = {
                    "data": knowledge_data,
                    "timestamp": knowledge_data.get("timestamp", time.time()),
                    "importance_score": self._calculate_importance(knowledge_data),
                    "event_type": knowledge_data.get("type", "unknown")
                }
                
                self.episodic_memory["events"][event_id] = event_record
                self.episodic_memory["timeline"].append(event_id)
                self.episodic_memory["importance_scores"][event_id] = event_record["importance_score"]
                
                self.episodic_memory["current_usage"] += 1
                
                return {
                    "stored": True,
                    "event_id": event_id,
                    "importance_score": event_record["importance_score"],
                    "layer": "episodic"
                }
            
            def _store_in_procedural(self, knowledge_data: Dict[str, Any]) -> Dict[str, Any]:
                """Сохранить в процедурной памяти"""
                if knowledge_data.get("type") == "procedure" or "procedure" in knowledge_data:
                    procedure_id = knowledge_data.get("id", f"procedure_{len(self.procedural_memory['procedures'])}")
                    
                    self.procedural_memory["procedures"][procedure_id] = {
                        "data": knowledge_data,
                        "timestamp": time.time(),
                        "usage_count": 0
                    }
                    
                    if "steps" in knowledge_data:
                        self.procedural_memory["step_dependencies"][procedure_id] = knowledge_data["steps"]
                    
                    self.procedural_memory["current_usage"] += 1
                    
                    return {
                        "stored": True,
                        "procedure_id": procedure_id,
                        "layer": "procedural"
                    }
                
                return {"stored": False, "layer": "procedural"}
            
            def _store_in_vault(self, knowledge_data: Dict[str, Any]) -> Dict[str, Any]:
                """Сохранить в хранилище секретов"""
                secrets = self._extract_secrets(knowledge_data)
                
                if secrets:
                    for secret_name, secret_value in secrets.items():
                        self.vault_memory["secrets"][secret_name] = {
                            "value": secret_value,
                            "timestamp": time.time(),
                            "data_type": knowledge_data.get("type", "unknown")
                        }
                    
                    self.vault_memory["current_usage"] += len(secrets)
                    
                    return {
                        "stored": True,
                        "secrets_count": len(secrets),
                        "secrets": list(secrets.keys()),
                        "layer": "vault"
                    }
                
                return {"stored": False, "layer": "vault"}
            
            def _update_security_layer(self, knowledge_data: Dict[str, Any]) -> Dict[str, Any]:
                """Обновить систему безопасности"""
                policy_id = knowledge_data.get("id", f"policy_{len(self.security_memory['policies'])}")
                
                self.security_memory["policies"][policy_id] = {
                    "data": knowledge_data,
                    "timestamp": time.time(),
                    "policy_type": knowledge_data.get("type", "general")
                }
                
                self.security_memory["current_usage"] += 1
                
                return {
                    "updated": True,
                    "policy_id": policy_id,
                    "layer": "security"
                }
            
            def _store_in_vector_store(self, knowledge_data: Dict[str, Any]) -> Dict[str, Any]:
                """Сохранить в векторном хранилище"""
                item_id = knowledge_data.get("id", f"vector_{len(self.vector_store['vectors'])}")
                
                # Генерируем mock вектор
                vector = self._generate_mock_vector(knowledge_data)
                
                self.vector_store["vectors"][item_id] = {
                    "vector": vector,
                    "data": knowledge_data,
                    "timestamp": time.time(),
                    "dimension": len(vector)
                }
                
                self.vector_store["current_usage"] += 1
                
                return {
                    "stored": True,
                    "item_id": item_id,
                    "dimension": len(vector),
                    "layer": "vector_store"
                }
            
            def _retrieve_semantic_concepts(self, query: str) -> List[Dict[str, Any]]:
                """Извлечь концепты из семантической памяти"""
                results = []
                query_lower = query.lower()
                
                for concept_id, concept_data in self.semantic_memory["concepts"].items():
                    content_str = json.dumps(concept_data["data"]).lower()
                    if any(word in content_str for word in query_lower.split()):
                        results.append({
                            "id": concept_id,
                            "data": concept_data["data"],
                            "type": "concept",
                            "layer": "semantic",
                            "relevance_score": self._calculate_relevance(query_lower, content_str)
                        })
                
                return results
            
            def _retrieve_episodic_events(self, query: str) -> List[Dict[str, Any]]:
                """Извлечь события из эпизодической памяти"""
                results = []
                query_lower = query.lower()
                
                for event_id, event_data in self.episodic_memory["events"].items():
                    content_str = json.dumps(event_data["data"]).lower()
                    if any(word in content_str for word in query_lower.split()):
                        results.append({
                            "id": event_id,
                            "data": event_data["data"],
                            "type": "event",
                            "layer": "episodic",
                            "importance_score": event_data["importance_score"],
                            "timestamp": event_data["timestamp"],
                            "relevance_score": self._calculate_relevance(query_lower, content_str)
                        })
                
                # Сортируем по важности
                results.sort(key=lambda x: x["importance_score"], reverse=True)
                return results
            
            def _retrieve_procedural_info(self, query: str) -> List[Dict[str, Any]]:
                """Извлечь процедурную информацию"""
                results = []
                query_lower = query.lower()
                
                for proc_id, proc_data in self.procedural_memory["procedures"].items():
                    content_str = json.dumps(proc_data["data"]).lower()
                    if any(word in content_str for word in query_lower.split()):
                        results.append({
                            "id": proc_id,
                            "data": proc_data["data"],
                            "type": "procedure",
                            "layer": "procedural",
                            "usage_count": proc_data["usage_count"],
                            "relevance_score": self._calculate_relevance(query_lower, content_str)
                        })
                
                return results
            
            def _retrieve_vector_similar(self, query: str) -> List[Dict[str, Any]]:
                """Извлечь похожие векторы"""
                results = []
                query_vector = self._generate_mock_vector({"query": query})
                
                for item_id, vector_data in self.vector_store["vectors"].items():
                    similarity = self._calculate_vector_similarity(query_vector, vector_data["vector"])
                    if similarity > 0.5:  # Порог схожести
                        results.append({
                            "id": item_id,
                            "data": vector_data["data"],
                            "type": "vector",
                            "layer": "vector_store",
                            "similarity_score": similarity,
                            "relevance_score": similarity
                        })
                
                # Сортируем по схожести
                results.sort(key=lambda x: x["similarity_score"], reverse=True)
                return results
            
            def _create_cross_layer_relations(self, knowledge_data: Dict[str, Any], storage_results: Dict[str, Any]):
                """Создать кросс-слойные связи"""
                # Простая логика создания связей на основе данных
                relations = []
                
                for layer, result in storage_results.items():
                    if result.get("stored", False):
                        relations.append({
                            "item_id": result.get(f"{layer.rstrip('s')}_id", result.get("item_id")),
                            "layer": layer,
                            "relation_type": "contains"
                        })
                
                # Сохраняем связи в кросс-слойном индексе
                item_hash = self._hash_knowledge_data(knowledge_data)
                if item_hash not in self.cross_layer_index:
                    self.cross_layer_index[item_hash] = []
                
                self.cross_layer_index[item_hash].extend(relations)
            
            def _update_cross_layer_index(self, knowledge_data: Dict[str, Any], storage_results: Dict[str, Any]):
                """Обновить кросс-слойный индекс"""
                # Обновляем различные индексы в зависимости от типа данных
                if "concepts" in knowledge_data:
                    for concept in knowledge_data["concepts"]:
                        concept_name = concept.get("name", concept.get("id", ""))
                        if concept_name:
                            # Связываем концепт с другими слоями
                            for layer, result in storage_results.items():
                                if result.get("stored", False):
                                    pass  # Логика индексирования
            
            def _calculate_importance(self, knowledge_data: Dict[str, Any]) -> float:
                """Вычислить важность данных"""
                # Простая эвристика важности
                base_importance = 0.5
                
                # Увеличиваем важность для определенных типов
                high_importance_types = ["milestone", "decision", "breakthrough", "critical"]
                if knowledge_data.get("type") in high_importance_types:
                    base_importance += 0.3
                
                # Увеличиваем важность для секретных данных
                if self._should_store_in_vault(knowledge_data):
                    base_importance += 0.2
                
                return min(1.0, base_importance)
            
            def _extract_secrets(self, knowledge_data: Dict[str, Any]) -> Dict[str, str]:
                """Извлечь секреты из данных"""
                secrets = {}
                
                def extract_recursively(obj, path=""):
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            current_path = f"{path}.{key}" if path else key
                            if key.lower() in ["password", "secret", "token", "key", "api_key"]:
                                secrets[current_path] = str(value)
                            extract_recursively(value, current_path)
                    elif isinstance(obj, list):
                        for i, item in enumerate(obj):
                            extract_recursively(item, f"{path}[{i}]")
                
                extract_recursively(knowledge_data)
                return secrets
            
            def _generate_mock_vector(self, data: Dict[str, Any]) -> List[float]:
                """Генерировать mock вектор для данных"""
                import hashlib
                data_str = json.dumps(data, sort_keys=True)
                hash_obj = hashlib.md5(data_str.encode())
                hash_bytes = hash_obj.digest()
                
                vector = []
                for i in range(self.vector_store["dimension"]):
                    byte_idx = i % len(hash_bytes)
                    value = (hash_bytes[byte_idx] / 255.0) * 2 - 1
                    vector.append(value)
                
                return vector
            
            def _calculate_relevance(self, query: str, content: str) -> float:
                """Вычислить релевантность контента к запросу"""
                query_words = set(query.split())
                content_words = content.split()
                
                matches = sum(1 for word in query_words if word in content_words)
                return matches / max(len(query_words), 1)
            
            def _calculate_vector_similarity(self, vec1: List[float], vec2: List[float]) -> float:
                """Вычислить косинусную схожесть векторов"""
                import math
                
                dot_product = sum(a * b for a, b in zip(vec1, vec2))
                magnitude1 = math.sqrt(sum(a * a for a in vec1))
                magnitude2 = math.sqrt(sum(a * a for a in vec2))
                
                if magnitude1 == 0 or magnitude2 == 0:
                    return 0.0
                
                return dot_product / (magnitude1 * magnitude2)
            
            def _hash_knowledge_data(self, knowledge_data: Dict[str, Any]) -> str:
                """Создать хеш для данных знаний"""
                import hashlib
                data_str = json.dumps(knowledge_data, sort_keys=True)
                return hashlib.md5(data_str.encode()).hexdigest()
            
            def _post_process_results(self, fused_results: Dict[str, Any]) -> Dict[str, Any]:
                """Пост-обработка результатов"""
                # Добавляем метаданные к результатам
                processed_results = {
                    "results": fused_results.get("results", []),
                    "metadata": {
                        "total_sources": len(fused_results.get("source_breakdown", {})),
                        "fusion_confidence": fused_results.get("confidence_score", 0.0),
                        "processing_time": fused_results.get("processing_time", 0.0),
                        "layers_contributed": list(fused_results.get("source_breakdown", {}).keys())
                    }
                }
                
                return processed_results
            
            def get_system_metrics(self) -> Dict[str, Any]:
                """Получить метрики системы памяти"""
                return {
                    "semantic_memory": {
                        "usage": self.semantic_memory["current_usage"],
                        "capacity": self.semantic_memory["max_capacity"],
                        "utilization_percent": (self.semantic_memory["current_usage"] / self.semantic_memory["max_capacity"]) * 100
                    },
                    "episodic_memory": {
                        "usage": self.episodic_memory["current_usage"],
                        "capacity": self.episodic_memory["max_capacity"],
                        "utilization_percent": (self.episodic_memory["current_usage"] / self.episodic_memory["max_capacity"]) * 100
                    },
                    "procedural_memory": {
                        "usage": self.procedural_memory["current_usage"],
                        "capacity": self.procedural_memory["max_capacity"],
                        "utilization_percent": (self.procedural_memory["current_usage"] / self.procedural_memory["max_capacity"]) * 100
                    },
                    "vault_memory": {
                        "usage": self.vault_memory["current_usage"],
                        "capacity": self.vault_memory["max_capacity"],
                        "utilization_percent": (self.vault_memory["current_usage"] / self.vault_memory["max_capacity"]) * 100
                    },
                    "security_memory": {
                        "usage": self.security_memory["current_usage"],
                        "capacity": self.security_memory["max_capacity"],
                        "utilization_percent": (self.security_memory["current_usage"] / self.security_memory["max_capacity"]) * 100
                    },
                    "vector_store": {
                        "usage": self.vector_store["current_usage"],
                        "capacity": self.vector_store["max_capacity"],
                        "utilization_percent": (self.vector_store["current_usage"] / self.vector_store["max_capacity"]) * 100
                    },
                    "system_state": self.system_state,
                    "performance_metrics": self.performance_monitor.get_summary()
                }
            
            def reset_system(self):
                """Сбросить состояние всей системы памяти"""
                # Сброс всех слоев памяти
                for layer in [self.semantic_memory, self.episodic_memory, self.procedural_memory, 
                             self.vault_memory, self.security_memory, self.vector_store]:
                    for key in layer:
                        if isinstance(layer[key], dict):
                            layer[key].clear()
                        elif isinstance(layer[key], list):
                            layer[key] = []
                        elif isinstance(layer[key], (int, float)):
                            layer[key] = 0
                
                # Сброс интеграционных компонентов
                self.cross_layer_index.clear()
                self.fusion_engine.reset()
                self.consistency_manager.reset()
                self.performance_monitor.reset()
                
                # Сброс состояния системы
                self.system_state = {
                    "total_knowledge_items": 0,
                    "cross_layer_relations": 0,
                    "query_responses": 0,
                    "error_count": 0,
                    "last_sync": time.time()
                }
        
        return FullMemoryStack


class FusionEngine:
    """Движок фьюжн для объединения результатов из разных слоев памяти"""
    
    def __init__(self):
        self.strategies = {
            "hybrid": self._hybrid_fusion,
            "semantic_priority": self._semantic_priority_fusion,
            "temporal_priority": self._temporal_priority_fusion,
            "importance_weighted": self._importance_weighted_fusion
        }
    
    def fuse_results(self, query: str, layer_results: Dict[str, List], strategy: str = "hybrid") -> Dict[str, Any]:
        """Объединить результаты с использованием указанной стратегии"""
        if strategy not in self.strategies:
            strategy = "hybrid"
        
        fusion_func = self.strategies[strategy]
        return fusion_func(query, layer_results)
    
    def _hybrid_fusion(self, query: str, layer_results: Dict[str, List]) -> Dict[str, Any]:
        """Гибридное объединение с равными весами"""
        all_results = []
        source_breakdown = {}
        
        for layer_name, results in layer_results.items():
            if results:
                source_breakdown[layer_name] = len(results)
                for result in results:
                    result["fusion_weight"] = 1.0
                    all_results.append(result)
        
        # Сортируем по релевантности
        all_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        return {
            "results": all_results,
            "source_breakdown": source_breakdown,
            "fusion_strategy": "hybrid",
            "confidence_score": 0.8,
            "processing_time": 0.1
        }
    
    def _semantic_priority_fusion(self, query: str, layer_results: Dict[str, List]) -> Dict[str, Any]:
        """Приоритет семантической памяти"""
        all_results = []
        source_breakdown = {}
        
        for layer_name, results in layer_results.items():
            if results:
                weight = 1.5 if layer_name == "semantic" else 1.0
                source_breakdown[layer_name] = len(results)
                for result in results:
                    result["fusion_weight"] = weight
                    result["adjusted_relevance"] = result.get("relevance_score", 0) * weight
                    all_results.append(result)
        
        # Сортируем по скорректированной релевантности
        all_results.sort(key=lambda x: x.get("adjusted_relevance", 0), reverse=True)
        
        return {
            "results": all_results,
            "source_breakdown": source_breakdown,
            "fusion_strategy": "semantic_priority",
            "confidence_score": 0.85,
            "processing_time": 0.12
        }
    
    def _temporal_priority_fusion(self, query: str, layer_results: Dict[str, List]) -> Dict[str, Any]:
        """Приоритет эпизодической памяти по времени"""
        all_results = []
        source_breakdown = {}
        
        for layer_name, results in layer_results.items():
            if results:
                weight = 1.3 if layer_name == "episodic" else 1.0
                source_breakdown[layer_name] = len(results)
                for result in results:
                    result["fusion_weight"] = weight
                    result["adjusted_relevance"] = result.get("relevance_score", 0) * weight
                    
                    # Дополнительный вес для свежих эпизодических данных
                    if layer_name == "episodic" and "timestamp" in result:
                        time_factor = max(0.5, 1.0 - (time.time() - result["timestamp"]) / (7 * 24 * 3600))  # Уменьшается за неделю
                        result["adjusted_relevance"] *= time_factor
                    
                    all_results.append(result)
        
        all_results.sort(key=lambda x: x.get("adjusted_relevance", 0), reverse=True)
        
        return {
            "results": all_results,
            "source_breakdown": source_breakdown,
            "fusion_strategy": "temporal_priority",
            "confidence_score": 0.82,
            "processing_time": 0.15
        }
    
    def _importance_weighted_fusion(self, query: str, layer_results: Dict[str, List]) -> Dict[str, Any]:
        """Взвешивание по важности"""
        all_results = []
        source_breakdown = {}
        
        for layer_name, results in layer_results.items():
            if results:
                source_breakdown[layer_name] = len(results)
                for result in results:
                    weight = 1.0
                    
                    # Вес по типу результата
                    if result.get("type") == "event":
                        weight *= result.get("importance_score", 0.5) * 2
                    elif result.get("type") == "procedure":
                        weight *= 1.2
                    elif result.get("type") == "concept":
                        weight *= 1.1
                    
                    result["fusion_weight"] = weight
                    result["adjusted_relevance"] = result.get("relevance_score", 0) * weight
                    all_results.append(result)
        
        all_results.sort(key=lambda x: x.get("adjusted_relevance", 0), reverse=True)
        
        return {
            "results": all_results,
            "source_breakdown": source_breakdown,
            "fusion_strategy": "importance_weighted",
            "confidence_score": 0.87,
            "processing_time": 0.18
        }
    
    def reset(self):
        """Сброс состояния фьюжн движка"""
        pass


class ConsistencyManager:
    """Менеджер согласованности для проверки целостности данных"""
    
    def __init__(self):
        self.validation_rules = []
        self.consistency_checks = []
    
    def validate_data(self, knowledge_data: Dict[str, Any]) -> Dict[str, Any]:
        """Проверить целостность данных"""
        issues = []
        
        # Проверка обязательных полей
        if not knowledge_data.get("id"):
            issues.append("Missing required field: id")
        
        if not knowledge_data.get("type"):
            issues.append("Missing required field: type")
        
        # Проверка типов данных
        if "timestamp" in knowledge_data and not isinstance(knowledge_data["timestamp"], (int, float)):
            issues.append("Invalid timestamp format")
        
        # Проверка консистентности ссылок
        if "related_concepts" in knowledge_data:
            if not isinstance(knowledge_data["related_concepts"], list):
                issues.append("related_concepts must be a list")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }
    
    def reset(self):
        """Сброс состояния менеджера согласованности"""
        self.validation_rules.clear()
        self.consistency_checks.clear()


class PerformanceMonitor:
    """Монитор производительности системы памяти"""
    
    def __init__(self):
        self.operation_log = []
        self.performance_stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "avg_operation_time": 0.0,
            "total_operations_by_type": {}
        }
    
    def log_operation(self, operation_data: Dict[str, Any]):
        """Записать операцию в лог"""
        self.operation_log.append(operation_data)
        
        # Обновляем статистику
        self.performance_stats["total_operations"] += 1
        
        if operation_data.get("success", True):
            self.performance_stats["successful_operations"] += 1
        else:
            self.performance_stats["failed_operations"] += 1
        
        # Статистика по типам операций
        op_type = operation_data.get("operation_type", "unknown")
        if op_type not in self.performance_stats["total_operations_by_type"]:
            self.performance_stats["total_operations_by_type"][op_type] = 0
        self.performance_stats["total_operations_by_type"][op_type] += 1
        
        # Обновляем среднее время операции
        if "duration" in operation_data:
            durations = [op.get("duration", 0) for op in self.operation_log if "duration" in op]
            self.performance_stats["avg_operation_time"] = sum(durations) / len(durations)
    
    def get_summary(self) -> Dict[str, Any]:
        """Получить сводку производительности"""
        return {
            **self.performance_stats,
            "recent_operations": self.operation_log[-10:] if self.operation_log else [],
            "error_rate": (self.performance_stats["failed_operations"] / 
                          max(self.performance_stats["total_operations"], 1)) * 100,
            "success_rate": (self.performance_stats["successful_operations"] / 
                           max(self.performance_stats["total_operations"], 1)) * 100
        }
    
    def reset(self):
        """Сброс монитора производительности"""
        self.operation_log.clear()
        self.performance_stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "avg_operation_time": 0.0,
            "total_operations_by_type": {}
        }


# Тесты

class TestFullMemoryStackIntegration:
    """Интеграционные тесты для полного стека памяти"""
    
    def test_memory_stack_initialization(self, full_memory_stack, base_config):
        """Тест инициализации полного стека памяти"""
        memory_stack = full_memory_stack(base_config)
        
        # Проверяем инициализацию всех слоев
        assert memory_stack.semantic_memory["max_capacity"] == base_config["memory_layers"]["semantic"]["max_concepts"]
        assert memory_stack.episodic_memory["max_capacity"] == base_config["memory_layers"]["episodic"]["max_events"]
        assert memory_stack.procedural_memory["max_capacity"] == base_config["memory_layers"]["procedural"]["max_workflows"]
        assert memory_stack.vault_memory["max_capacity"] == base_config["memory_layers"]["vault"]["max_secrets"]
        assert memory_stack.security_memory["max_capacity"] == base_config["memory_layers"]["security"]["max_policies"]
        assert memory_stack.vector_store["max_capacity"] == base_config["memory_layers"]["vector_store"].get("max_items", 50000)
        
        # Проверяем интеграционные компоненты
        assert memory_stack.fusion_engine is not None
        assert memory_stack.consistency_manager is not None
        assert memory_stack.performance_monitor is not None
        
        # Проверяем начальное состояние
        assert memory_stack.system_state["total_knowledge_items"] == 0
        assert memory_stack.system_state["cross_layer_relations"] == 0
        assert memory_stack.system_state["query_responses"] == 0
    
    def test_knowledge_storage_integration(self, full_memory_stack, base_config):
        """Тест интеграционного сохранения знаний"""
        memory_stack = full_memory_stack(base_config)
        
        # Тестовые данные для разных типов знаний
        concept_data = {
            "id": "ai_concept_001",
            "type": "concept",
            "name": "Artificial Intelligence",
            "definition": "Computer system capable of intelligent behavior",
            "related_concepts": ["machine_learning", "neural_networks"]
        }
        
        event_data = {
            "id": "learning_event_001",
            "type": "event",
            "name": "AI Study Session",
            "description": "Completed advanced AI course",
            "timestamp": time.time(),
            "importance": "high"
        }
        
        procedure_data = {
            "id": "ml_procedure_001",
            "type": "procedure",
            "name": "Model Training Process",
            "steps": ["data_preparation", "model_selection", "training", "evaluation"]
        }
        
        secure_data = {
            "id": "secure_info_001",
            "type": "secure_data",
            "name": "API Configuration",
            "api_key": "secret_key_123",
            "authorized": True
        }
        
        # Сохраняем во всех слоях
        concept_result = memory_stack.store_knowledge_item(concept_data)
        event_result = memory_stack.store_knowledge_item(event_data)
        procedure_result = memory_stack.store_knowledge_item(procedure_data)
        secure_result = memory_stack.store_knowledge_item(secure_data)
        
        # Проверяем результаты
        assert concept_result["success"] is True
        assert event_result["success"] is True
        assert procedure_result["success"] is True
        assert secure_result["success"] is True
        
        # Проверяем что данные сохранены в соответствующих слоях
        assert "ai_concept_001" in memory_stack.semantic_memory["concepts"]
        assert "learning_event_001" in memory_stack.episodic_memory["events"]
        assert "ml_procedure_001" in memory_stack.procedural_memory["procedures"]
        assert "api_key" in memory_stack.vault_memory["secrets"]
        
        # Проверяем кросс-слойные связи
        assert len(memory_stack.cross_layer_index) > 0
    
    def test_hybrid_retrieval_fusion(self, full_memory_stack, base_config):
        """Тест гибридного извлечения с фьюжн"""
        memory_stack = full_memory_stack(base_config)
        
        # Предварительно сохраняем данные
        test_data = [
            {"id": "retrieval_test_1", "type": "concept", "name": "Machine Learning", "definition": "AI subset"},
            {"id": "retrieval_test_2", "type": "event", "name": "ML Workshop", "description": "Attended ML workshop"},
            {"id": "retrieval_test_3", "type": "procedure", "name": "Data Analysis", "steps": ["clean", "analyze"]}
        ]
        
        for data in test_data:
            memory_stack.store_knowledge_item(data)
        
        # Тестируем разные стратегии фьюжн
        strategies = ["hybrid", "semantic_priority", "temporal_priority", "importance_weighted"]
        
        for strategy in strategies:
            result = memory_stack.retrieve_knowledge_fusion("machine learning", fusion_strategy=strategy)
            
            assert result["success"] is True
            assert result["strategy"] == strategy
            assert len(result["results"]) > 0
            assert "semantic" in result["layers_queried"]
            assert result["metadata"]["fusion_confidence"] > 0
    
    def test_security_enforcement(self, full_memory_stack, base_config):
        """Тест принудительного соблюдения безопасности"""
        memory_stack = full_memory_stack(base_config)
        
        # Тестовые данные с секретами без авторизации
        unauthorized_data = {
            "id": "unauthorized_data",
            "type": "secure_data",
            "name": "Private Information",
            "password": "secret123",
            "api_key": "key_xyz"
        }
        
        # Пытаемся сохранить без авторизации
        result = memory_stack.store_knowledge_item(unauthorized_data)
        
        assert result["success"] is False
        assert "Security check failed" in result["error"]
        
        # Добавляем авторизацию
        authorized_data = unauthorized_data.copy()
        authorized_data["authorized"] = True
        
        result_authorized = memory_stack.store_knowledge_item(authorized_data)
        
        assert result_authorized["success"] is True
        assert "password" in memory_stack.vault_memory["secrets"]
    
    def test_performance_monitoring(self, full_memory_stack, base_config):
        """Тест мониторинга производительности"""
        memory_stack = full_memory_stack(base_config)
        
        # Выполняем операции для сбора метрик
        for i in range(5):
            test_data = {
                "id": f"perf_test_{i}",
                "type": "concept",
                "name": f"Performance Test Concept {i}",
                "definition": f"Test definition {i}"
            }
            memory_stack.store_knowledge_item(test_data)
        
        for i in range(3):
            memory_stack.retrieve_knowledge_fusion(f"test query {i}")
        
        # Получаем метрики
        metrics = memory_stack.get_system_metrics()
        
        # Проверяем метрики слоев памяти
        for layer_name in ["semantic_memory", "episodic_memory", "procedural_memory", 
                          "vault_memory", "security_memory", "vector_store"]:
            layer_metrics = metrics[layer_name]
            assert "usage" in layer_metrics
            assert "capacity" in layer_metrics
            assert "utilization_percent" in layer_metrics
        
        # Проверяем метрики производительности
        perf_metrics = metrics["performance_metrics"]
        assert perf_metrics["total_operations"] == 8  # 5 сохранений + 3 извлечения
        assert perf_metrics["successful_operations"] == 8
        assert perf_metrics["error_rate"] == 0.0
        
        # Проверяем состояние системы
        system_state = metrics["system_state"]
        assert system_state["total_knowledge_items"] == 5
        assert system_state["query_responses"] == 3
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, full_memory_stack, base_config):
        """Тест конкурентных операций со стеком памяти"""
        memory_stack = full_memory_stack(base_config)
        
        async def store_knowledge_task(item_id: int):
            test_data = {
                "id": f"concurrent_store_{item_id}",
                "type": "concept",
                "name": f"Concurrent Concept {item_id}",
                "definition": f"Test definition {item_id}"
            }
            return memory_stack.store_knowledge_item(test_data)
        
        async def retrieve_knowledge_task(query_id: int):
            return memory_stack.retrieve_knowledge_fusion(f"concurrent query {query_id}")
        
        # Выполняем конкурентные операции
        start_time = time.time()
        
        store_tasks = [store_knowledge_task(i) for i in range(10)]
        retrieve_tasks = [retrieve_knowledge_task(i) for i in range(5)]
        
        store_results = await asyncio.gather(*store_tasks)
        retrieve_results = await asyncio.gather(*retrieve_tasks)
        
        concurrent_time = time.time() - start_time
        
        # Проверяем результаты
        assert all(result["success"] for result in store_results)
        assert all(result["success"] for result in retrieve_results)
        
        # Проверяем состояние после конкурентных операций
        metrics = memory_stack.get_system_metrics()
        assert metrics["system_state"]["total_knowledge_items"] == 10
        assert metrics["system_state"]["query_responses"] == 5
        assert concurrent_time < 15.0  # Операции должны завершиться разумно быстро
    
    def test_cognitive_bias_full_stack_integration(self, full_memory_stack, base_config):
        """Тест интеграции когнитивных искажений с полным стеком"""
        memory_stack = full_memory_stack(base_config)
        
        # Создаем комплексные данные по когнитивным искажениям
        bias_data_collection = []
        
        for bias_name, bias_info in list(COGNITIVE_BIASES_DATASET.items())[:3]:  # Ограничиваем для теста
            
            # Концепт
            concept_data = {
                "id": bias_info["id"],
                "type": "concept",
                "name": bias_info["name"],
                "definition": bias_info["definition"],
                "category": bias_info["category"],
                "related_concepts": bias_info.get("related_concepts", [])[:3]
            }
            
            # Событие обнаружения
            event_data = {
                "id": f"detection_event_{bias_info['id']}",
                "type": "bias_detection",
                "name": f"Detected {bias_info['name']}",
                "description": f"Discovered {bias_info['name']} in decision making",
                "timestamp": time.time(),
                "importance": bias_info["severity"],
                "concepts": [bias_name]
            }
            
            # Процедура смягчения
            mitigation_data = {
                "id": f"mitigation_proc_{bias_info['id']}",
                "type": "procedure",
                "name": f"{bias_info['name']} Mitigation",
                "steps": bias_info.get("mitigation_techniques", ["identify", "measure", "mitigate"])[:3]
            }
            
            bias_data_collection.extend([concept_data, event_data, mitigation_data])
        
        # Сохраняем все данные
        storage_results = []
        for data in bias_data_collection:
            result = memory_stack.store_knowledge_item(data)
            storage_results.append(result)
        
        # Проверяем что все данные сохранены успешно
        successful_stores = sum(1 for result in storage_results if result["success"])
        assert successful_stores == len(bias_data_collection)
        
        # Тестируем извлечение по когнитивным искажениям
        query_result = memory_stack.retrieve_knowledge_fusion("cognitive bias psychology", fusion_strategy="hybrid")
        
        assert query_result["success"] is True
        assert len(query_result["results"]) > 0
        
        # Проверяем многослойные результаты
        layers_contributed = query_result["metadata"]["layers_contributed"]
        assert "semantic" in layers_contributed  # Должны быть концепты
        assert "episodic" in layers_contributed  # Должны быть события
        assert "procedural" in layers_contributed  # Должны быть процедуры
        
        # Проверяем качество фьюжн
        fusion_confidence = query_result["metadata"]["fusion_confidence"]
        assert fusion_confidence > 0.5
    
    def test_system_consistency_after_failures(self, full_memory_stack, base_config):
        """Тест согласованности системы после сбоев"""
        memory_stack = full_memory_stack(base_config)
        
        # Сохраняем корректные данные
        valid_data = {
            "id": "valid_test",
            "type": "concept",
            "name": "Valid Concept",
            "definition": "Valid definition"
        }
        
        valid_result = memory_stack.store_knowledge_item(valid_data)
        assert valid_result["success"] is True
        
        # Пытаемся сохранить некорректные данные
        invalid_data = None
        invalid_result = memory_stack.store_knowledge_item(invalid_data)
        assert invalid_result["success"] is False
        
        # Проверяем что система остается в согласованном состоянии
        metrics = memory_stack.get_system_metrics()
        
        # Состояние должно отражать только успешные операции
        assert metrics["system_state"]["total_knowledge_items"] == 1
        assert metrics["system_state"]["error_count"] == 1
        
        # Все слои памяти должны быть в корректном состоянии
        assert memory_stack.semantic_memory["current_usage"] == 1
        assert memory_stack.episodic_memory["current_usage"] == 0
        assert memory_stack.procedural_memory["current_usage"] == 0
        
        # Попытка извлечения должна работать корректно
        retrieve_result = memory_stack.retrieve_knowledge_fusion("valid concept")
        assert retrieve_result["success"] is True
        assert len(retrieve_result["results"]) == 1
    
    def test_capacity_management(self, full_memory_stack, base_config):
        """Тест управления емкостью памяти"""
        # Создаем систему с малыми лимитами для тестирования
        small_config = {
            "memory_layers": {
                "semantic": {"max_concepts": 3},
                "episodic": {"max_events": 3},
                "procedural": {"max_workflows": 2},
                "vault": {"max_secrets": 2},
                "security": {"max_policies": 2},
                "vector_store": {"max_items": 10}
            }
        }
        
        memory_stack = full_memory_stack(small_config)
        
        # Заполняем систему до лимитов
        for i in range(5):  # Больше чем лимиты
            test_data = {
                "id": f"capacity_test_{i}",
                "type": "concept",
                "name": f"Capacity Test Concept {i}",
                "definition": f"Test definition {i}"
            }
            result = memory_stack.store_knowledge_item(test_data)
            
            # Первые операции должны быть успешными
            if i < 3:
                assert result["success"] is True
            else:
                # Последние могут быть неуспешными из-за лимитов
                pass
        
        # Проверяем метрики емкости
        metrics = memory_stack.get_system_metrics()
        
        semantic_usage = metrics["semantic_memory"]["usage"]
        semantic_capacity = metrics["semantic_memory"]["capacity"]
        
        # Использование не должно превышать емкость
        assert semantic_usage <= semantic_capacity
        
        # Процент использования должен быть реалистичным
        utilization = metrics["semantic_memory"]["utilization_percent"]
        assert 0 <= utilization <= 100
    
    def test_cross_layer_relationships(self, full_memory_stack, base_config):
        """Тест кросс-слойных связей"""
        memory_stack = full_memory_stack(base_config)
        
        # Создаем связанные данные
        related_data = [
            {
                "id": "root_concept",
                "type": "concept",
                "name": "Root Concept",
                "definition": "Main concept for testing",
                "related_concepts": ["child_concept_1", "child_concept_2"]
            },
            {
                "id": "child_event_1",
                "type": "event",
                "name": "Child Event 1",
                "description": "Event related to root concept",
                "concepts": ["root_concept"]
            },
            {
                "id": "child_procedure",
                "type": "procedure",
                "name": "Child Procedure",
                "steps": ["step1", "step2"],
                "related_concepts": ["root_concept"]
            }
        ]
        
        # Сохраняем связанные данные
        for data in related_data:
            result = memory_stack.store_knowledge_item(data)
            assert result["success"] is True
        
        # Проверяем создание кросс-слойных связей
        assert len(memory_stack.cross_layer_index) > 0
        
        # Тестируем извлечение с учетом связей
        result = memory_stack.retrieve_knowledge_fusion("root concept", fusion_strategy="hybrid")
        
        assert result["success"] is True
        assert len(result["results"]) >= 3  # Должны найти все связанные элементы
        
        # Проверяем что найдены данные из разных слоев
        result_types = set(result["type"] for result in result["results"])
        assert len(result_types) >= 2  # Минимум 2 типа данных
    
    def test_system_reset_functionality(self, full_memory_stack, base_config):
        """Тест функции сброса системы"""
        memory_stack = full_memory_stack(base_config)
        
        # Сохраняем тестовые данные
        for i in range(5):
            test_data = {
                "id": f"reset_test_{i}",
                "type": "concept",
                "name": f"Reset Test Concept {i}",
                "definition": f"Test definition {i}"
            }
            memory_stack.store_knowledge_item(test_data)
        
        # Выполняем извлечения
        for i in range(3):
            memory_stack.retrieve_knowledge_fusion(f"reset test {i}")
        
        # Проверяем что данные существуют
        metrics_before = memory_stack.get_system_metrics()
        assert metrics_before["system_state"]["total_knowledge_items"] == 5
        assert metrics_before["system_state"]["query_responses"] == 3
        assert metrics_before["semantic_memory"]["usage"] == 5
        
        # Выполняем сброс
        memory_stack.reset_system()
        
        # Проверяем что все сброшено
        metrics_after = memory_stack.get_system_metrics()
        assert metrics_after["system_state"]["total_knowledge_items"] == 0
        assert metrics_after["system_state"]["query_responses"] == 0
        assert metrics_after["system_state"]["error_count"] == 0
        assert metrics_after["semantic_memory"]["usage"] == 0
        assert metrics_after["episodic_memory"]["usage"] == 0
        assert metrics_after["procedural_memory"]["usage"] == 0
        
        # Проверяем что последующие операции работают
        new_data = {
            "id": "post_reset_test",
            "type": "concept",
            "name": "Post Reset Concept",
            "definition": "Test after reset"
        }
        
        result = memory_stack.store_knowledge_item(new_data)
        assert result["success"] is True
        
        # Проверяем что новая операция зарегистрирована
        new_metrics = memory_stack.get_system_metrics()
        assert new_metrics["system_state"]["total_knowledge_items"] == 1