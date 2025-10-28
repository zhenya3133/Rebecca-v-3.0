"""
Integration тесты для интеграции KAG системы с эпизодической памятью
"""

import pytest
import asyncio
import time
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional

from ..fixtures.conftest import base_config, memory_manager_mock
from ..test_data.cognitive_biases import COGNITIVE_BIASES_DATASET


class EpisodicMemoryIntegrationTestSuite:
    """Комплексная тестовая подсистема для эпизодической памяти"""
    
    @pytest.fixture
    def episodic_memory_engine(self):
        """Mock движок эпизодической памяти для тестирования"""
        class EpisodicMemoryEngine:
            def __init__(self, config: Dict[str, Any]):
                self.config = config
                self.max_events = config.get("max_events", 5000)
                self.retention_days = config.get("retention_days", 30)
                self.events = {}
                self.event_timeline = []
                self.temporal_index = {}
                self.contextual_links = {}
                self.episodic_clusters = {}
                self.importance_weights = {
                    "novelty": 0.3,
                    "emotional_intensity": 0.4,
                    "consequence_magnitude": 0.3
                }
                self.performance_metrics = {
                    "store_time": [],
                    "retrieval_time": [],
                    "temporal_query_time": []
                }
                self.access_log = []
            
            def store_event(self, event_data: Dict[str, Any]) -> bool:
                """Сохранить событие в эпизодической памяти"""
                start_time = time.time()
                
                try:
                    event_id = event_data.get("id")
                    if not event_id:
                        event_id = f"event_{int(time.time() * 1000)}"
                    
                    # Проверяем емкость
                    if len(self.events) >= self.max_events:
                        # Применяем стратегию замещения на основе важности
                        self._apply_retention_policy()
                    
                    # Извлекаем временную метку
                    timestamp = event_data.get("timestamp", time.time())
                    if isinstance(timestamp, str):
                        timestamp = self._parse_timestamp(timestamp)
                    
                    # Вычисляем важность события
                    importance_score = self._calculate_importance(event_data)
                    
                    # Создаем событие
                    event_record = {
                        "id": event_id,
                        "data": event_data,
                        "timestamp": timestamp,
                        "datetime": datetime.fromtimestamp(timestamp),
                        "importance_score": importance_score,
                        "contextual_tags": self._extract_contextual_tags(event_data),
                        "linked_events": [],
                        "access_count": 0,
                        "cluster_id": None,
                        "stored_at": time.time()
                    }
                    
                    # Сохраняем событие
                    self.events[event_id] = event_record
                    
                    # Добавляем в временную линию
                    self.event_timeline.append(event_id)
                    self.event_timeline.sort(key=lambda x: self.events[x]["timestamp"])
                    
                    # Обновляем временной индекс
                    self._update_temporal_index(event_id, timestamp)
                    
                    # Создаем контекстуальные связи
                    self._create_contextual_links(event_id, event_data)
                    
                    # Присваиваем кластер
                    self._assign_to_cluster(event_id)
                    
                    # Записываем в лог
                    self.access_log.append({
                        "timestamp": time.time(),
                        "operation": "store_event",
                        "event_id": event_id,
                        "importance_score": importance_score
                    })
                    
                    elapsed_time = time.time() - start_time
                    self.performance_metrics["store_time"].append(elapsed_time)
                    
                    return True
                    
                except Exception as e:
                    self.access_log.append({
                        "timestamp": time.time(),
                        "operation": "store_event_failed",
                        "error": str(e),
                        "event_id": event_id if 'event_id' in locals() else "unknown"
                    })
                    return False
            
            def retrieve_events_by_timeframe(self, 
                                           start_time: float, 
                                           end_time: float,
                                           min_importance: float = 0.0) -> List[Dict[str, Any]]:
                """Извлечь события за временной период"""
                start_time_total = time.time()
                
                try:
                    matching_events = []
                    
                    for event_id, event_record in self.events.items():
                        event_timestamp = event_record["timestamp"]
                        if start_time <= event_timestamp <= end_time and event_record["importance_score"] >= min_importance:
                            matching_events.append({
                                "event_id": event_id,
                                "event_data": event_record["data"],
                                "timestamp": event_timestamp,
                                "datetime": event_record["datetime"].isoformat(),
                                "importance_score": event_record["importance_score"],
                                "contextual_tags": event_record["contextual_tags"],
                                "cluster_id": event_record["cluster_id"]
                            })
                            
                            # Обновляем счетчик доступа
                            event_record["access_count"] += 1
                    
                    # Сортируем по важности и времени
                    matching_events.sort(key=lambda x: (x["importance_score"], x["timestamp"]), reverse=True)
                    
                    # Записываем в лог
                    self.access_log.append({
                        "timestamp": time.time(),
                        "operation": "retrieve_by_timeframe",
                        "start_time": start_time,
                        "end_time": end_time,
                        "results_count": len(matching_events),
                        "min_importance": min_importance
                    })
                    
                    elapsed_time = time.time() - start_time_total
                    self.performance_metrics["temporal_query_time"].append(elapsed_time)
                    
                    return matching_events
                    
                except Exception as e:
                    self.access_log.append({
                        "timestamp": time.time(),
                        "operation": "retrieve_by_timeframe_failed",
                        "error": str(e)
                    })
                    return []
            
            def retrieve_events_by_context(self, context_tags: List[str], max_results: int = 50) -> List[Dict[str, Any]]:
                """Извлечь события по контекстуальным тегам"""
                start_time = time.time()
                
                try:
                    matching_events = []
                    tag_set = set(tag.lower() for tag in context_tags)
                    
                    for event_id, event_record in self.events.items():
                        event_tags = set(tag.lower() for tag in event_record["contextual_tags"])
                        if tag_set.intersection(event_tags):
                            matching_events.append({
                                "event_id": event_id,
                                "event_data": event_record["data"],
                                "timestamp": event_record["timestamp"],
                                "importance_score": event_record["importance_score"],
                                "matched_tags": list(tag_set.intersection(event_tags)),
                                "cluster_id": event_record["cluster_id"]
                            })
                            
                            event_record["access_count"] += 1
                    
                    # Сортируем по важности
                    matching_events.sort(key=lambda x: x["importance_score"], reverse=True)
                    
                    # Ограничиваем количество результатов
                    results = matching_events[:max_results]
                    
                    self.access_log.append({
                        "timestamp": time.time(),
                        "operation": "retrieve_by_context",
                        "context_tags": context_tags,
                        "results_count": len(results)
                    })
                    
                    elapsed_time = time.time() - start_time
                    self.performance_metrics["retrieval_time"].append(elapsed_time)
                    
                    return results
                    
                except Exception as e:
                    self.access_log.append({
                        "timestamp": time.time(),
                        "operation": "retrieve_by_context_failed",
                        "error": str(e)
                    })
                    return []
            
            def find_similar_events(self, reference_event_id: str, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
                """Найти похожие события"""
                if reference_event_id not in self.events:
                    return []
                
                reference_event = self.events[reference_event_id]
                similar_events = []
                
                for event_id, event_record in self.events.items():
                    if event_id == reference_event_id:
                        continue
                    
                    similarity_score = self._calculate_event_similarity(reference_event, event_record)
                    
                    if similarity_score >= similarity_threshold:
                        similar_events.append({
                            "event_id": event_id,
                            "event_data": event_record["data"],
                            "similarity_score": similarity_score,
                            "timestamp": event_record["timestamp"],
                            "importance_score": event_record["importance_score"]
                        })
                
                # Сортируем по схожести
                similar_events.sort(key=lambda x: x["similarity_score"], reverse=True)
                return similar_events
            
            def get_episodic_summary(self, time_period_days: int = 7) -> Dict[str, Any]:
                """Получить сводку эпизодических событий за период"""
                end_time = time.time()
                start_time = end_time - (time_period_days * 24 * 3600)
                
                # Фильтруем события за период
                period_events = self.retrieve_events_by_timeframe(start_time, end_time)
                
                if not period_events:
                    return {
                        "period_days": time_period_days,
                        "total_events": 0,
                        "avg_importance": 0.0,
                        "top_contexts": [],
                        "temporal_patterns": [],
                        "clusters": []
                    }
                
                # Вычисляем статистики
                total_events = len(period_events)
                importance_scores = [event["importance_score"] for event in period_events]
                avg_importance = sum(importance_scores) / len(importance_scores)
                
                # Анализируем контексты
                all_tags = []
                for event in period_events:
                    all_tags.extend(event["contextual_tags"])
                
                tag_frequency = {}
                for tag in all_tags:
                    tag_frequency[tag] = tag_frequency.get(tag, 0) + 1
                
                top_contexts = sorted(tag_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
                
                # Анализируем кластеры
                cluster_distribution = {}
                for event in period_events:
                    cluster_id = event.get("cluster_id")
                    if cluster_id:
                        cluster_distribution[cluster_id] = cluster_distribution.get(cluster_id, 0) + 1
                
                clusters = sorted(cluster_distribution.items(), key=lambda x: x[1], reverse=True)
                
                return {
                    "period_days": time_period_days,
                    "total_events": total_events,
                    "avg_importance": avg_importance,
                    "importance_std": self._calculate_std(importance_scores),
                    "top_contexts": top_contexts,
                    "clusters": clusters,
                    "temporal_density": total_events / time_period_days,  # Событий в день
                    "coverage_percent": (total_events / len(self.events)) * 100 if self.events else 0
                }
            
            def update_event_importance(self, event_id: str, new_importance: float) -> bool:
                """Обновить важность события"""
                if event_id not in self.events:
                    return False
                
                try:
                    old_importance = self.events[event_id]["importance_score"]
                    self.events[event_id]["importance_score"] = max(0.0, min(1.0, new_importance))
                    
                    self.access_log.append({
                        "timestamp": time.time(),
                        "operation": "update_importance",
                        "event_id": event_id,
                        "old_importance": old_importance,
                        "new_importance": new_importance
                    })
                    
                    return True
                    
                except Exception as e:
                    self.access_log.append({
                        "timestamp": time.time(),
                        "operation": "update_importance_failed",
                        "event_id": event_id,
                        "error": str(e)
                    })
                    return False
            
            def _calculate_importance(self, event_data: Dict[str, Any]) -> float:
                """Вычислить важность события"""
                # Новизна (на основе уникальности)
                novelty_score = self._calculate_novelty(event_data)
                
                # Эмоциональная интенсивность
                emotional_score = self._calculate_emotional_intensity(event_data)
                
                # Масштаб последствий
                consequence_score = self._calculate_consequence_magnitude(event_data)
                
                # Взвешенная сумма
                importance = (
                    novelty_score * self.importance_weights["novelty"] +
                    emotional_score * self.importance_weights["emotional_intensity"] +
                    consequence_score * self.importance_weights["consequence_magnitude"]
                )
                
                return max(0.0, min(1.0, importance))
            
            def _calculate_novelty(self, event_data: Dict[str, Any]) -> float:
                """Вычислить новизну события"""
                # Простая эвристика на основе уникальности ключевых атрибутов
                unique_attributes = ["name", "type", "category", "description"]
                uniqueness_score = 0.0
                
                for attr in unique_attributes:
                    if attr in event_data:
                        attr_value = str(event_data[attr]).lower()
                        # Подсчитываем сколько раз встречалось похожее значение
                        similar_count = 0
                        for existing_event in self.events.values():
                            existing_value = str(existing_event["data"].get(attr, "")).lower()
                            if self._string_similarity(attr_value, existing_value) > 0.8:
                                similar_count += 1
                        
                        # Чем меньше похожих событий, тем выше новизна
                        if similar_count == 0:
                            uniqueness_score += 1.0
                        else:
                            uniqueness_score += 1.0 / (1.0 + similar_count)
                
                return min(1.0, uniqueness_score / len(unique_attributes))
            
            def _calculate_emotional_intensity(self, event_data: Dict[str, Any]) -> float:
                """Вычислить эмоциональную интенсивность"""
                emotional_keywords = {
                    "critical": 1.0, "urgent": 0.9, "important": 0.8, "significant": 0.7,
                    "positive": 0.6, "negative": 0.6, "neutral": 0.3,
                    "error": 0.9, "failure": 0.8, "success": 0.7, "breakthrough": 0.9,
                    "crisis": 1.0, "opportunity": 0.8, "threat": 0.9
                }
                
                # Анализируем все текстовые поля
                text_content = json.dumps(event_data).lower()
                max_intensity = 0.0
                
                for keyword, intensity in emotional_keywords.items():
                    if keyword in text_content:
                        max_intensity = max(max_intensity, intensity)
                
                # Проверяем наличие эмоциональных полей
                if "emotional_impact" in event_data:
                    return min(1.0, event_data["emotional_impact"])
                
                return max_intensity
            
            def _calculate_consequence_magnitude(self, event_data: Dict[str, Any]) -> float:
                """Вычислить масштаб последствий"""
                # Проверяем наличие прямых индикаторов масштаба
                if "impact_level" in event_data:
                    impact_map = {"low": 0.3, "medium": 0.6, "high": 0.8, "critical": 1.0}
                    return impact_map.get(event_data["impact_level"], 0.5)
                
                # Анализируем тип события
                event_type = event_data.get("type", "").lower()
                type_magnitude = {
                    "decision": 0.8, "milestone": 0.9, "achievement": 0.7,
                    "learning": 0.6, "interaction": 0.4, "observation": 0.3,
                    "system": 0.5, "user": 0.6, "external": 0.7
                }
                
                return type_magnitude.get(event_type, 0.5)
            
            def _create_contextual_links(self, event_id: str, event_data: Dict[str, Any]):
                """Создать контекстуальные связи между событиями"""
                event_tags = self._extract_contextual_tags(event_data)
                
                # Находим похожие события по контексту
                for existing_id, existing_record in self.events.items():
                    if existing_id == event_id:
                        continue
                    
                    common_tags = set(event_tags).intersection(set(existing_record["contextual_tags"]))
                    
                    if len(common_tags) >= 2:  # Минимум 2 общих тега
                        if event_id not in self.contextual_links:
                            self.contextual_links[event_id] = []
                        if existing_id not in self.contextual_links:
                            self.contextual_links[existing_id] = []
                        
                        self.contextual_links[event_id].append(existing_id)
                        self.contextual_links[existing_id].append(event_id)
                        
                        # Обновляем связанные события
                        self.events[event_id]["linked_events"].append(existing_id)
                        self.events[existing_id]["linked_events"].append(event_id)
            
            def _assign_to_cluster(self, event_id: str):
                """Присвоить событие к кластеру"""
                event_record = self.events[event_id]
                event_tags = set(event_record["contextual_tags"])
                
                best_cluster = None
                best_overlap = 0.0
                
                # Находим наиболее подходящий кластер
                for cluster_id, cluster_events in self.episodic_clusters.items():
                    # Вычисляем пересечение тегов с кластером
                    cluster_tags = set()
                    for existing_event_id in cluster_events:
                        if existing_event_id in self.events:
                            cluster_tags.update(self.events[existing_event_id]["contextual_tags"])
                    
                    overlap = len(event_tags.intersection(cluster_tags)) / max(len(event_tags), 1)
                    
                    if overlap > best_overlap and overlap > 0.3:  # Минимальный порог
                        best_cluster = cluster_id
                        best_overlap = overlap
                
                # Создаем новый кластер или добавляем к существующему
                if best_cluster is None:
                    cluster_id = f"cluster_{len(self.episodic_clusters)}"
                    self.episodic_clusters[cluster_id] = []
                else:
                    cluster_id = best_cluster
                
                self.episodic_clusters[cluster_id].append(event_id)
                event_record["cluster_id"] = cluster_id
            
            def _extract_contextual_tags(self, event_data: Dict[str, Any]) -> List[str]:
                """Извлечь контекстуальные теги из события"""
                tags = []
                
                # Извлекаем из различных полей
                tag_fields = ["type", "category", "domain", "context", "environment"]
                for field in tag_fields:
                    if field in event_data:
                        value = event_data[field]
                        if isinstance(value, str):
                            tags.append(value.lower())
                        elif isinstance(value, list):
                            tags.extend([str(v).lower() for v in value])
                
                # Извлекаем из связанных концептов
                if "concepts" in event_data:
                    if isinstance(event_data["concepts"], list):
                        tags.extend([str(c).lower() for c in event_data["concepts"]])
                
                # Извлекаем из названия и описания (простые ключевые слова)
                text_fields = ["name", "title", "description", "summary"]
                for field in text_fields:
                    if field in event_data:
                        text = str(event_data[field]).lower()
                        words = text.split()
                        # Добавляем значимые слова (длиннее 3 символов, не стоп-слова)
                        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
                        significant_words = [w for w in words if len(w) > 3 and w not in stop_words]
                        tags.extend(significant_words[:3])  # Максимум 3 слова
            
                return list(set(tags))  # Убираем дубликаты
            
            def _update_temporal_index(self, event_id: str, timestamp: float):
                """Обновить временной индекс"""
                date_key = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
                
                if date_key not in self.temporal_index:
                    self.temporal_index[date_key] = []
                
                self.temporal_index[date_key].append(event_id)
                self.temporal_index[date_key].sort(key=lambda x: self.events[x]["timestamp"])
            
            def _apply_retention_policy(self):
                """Применить политику задержки (удаление наименее важных событий)"""
                # Сортируем события по важности
                sorted_events = sorted(self.events.items(), key=lambda x: x[1]["importance_score"])
                
                # Удаляем 10% наименее важных событий
                deletion_count = max(1, len(self.events) // 10)
                
                for i in range(deletion_count):
                    event_id, event_record = sorted_events[i]
                    self._remove_event(event_id)
            
            def _remove_event(self, event_id: str):
                """Удалить событие и связанные данные"""
                if event_id not in self.events:
                    return
                
                event_record = self.events[event_id]
                
                # Удаляем из временной линии
                if event_id in self.event_timeline:
                    self.event_timeline.remove(event_id)
                
                # Удаляем из временного индекса
                for date_key, event_list in self.temporal_index.items():
                    if event_id in event_list:
                        event_list.remove(event_id)
                        if not event_list:  # Если список пустой, удаляем дату
                            del self.temporal_index[date_key]
                
                # Удаляем контекстуальные связи
                if event_id in self.contextual_links:
                    for linked_id in self.contextual_links[event_id]:
                        if linked_id in self.contextual_links:
                            if event_id in self.contextual_links[linked_id]:
                                self.contextual_links[linked_id].remove(event_id)
                    del self.contextual_links[event_id]
                
                # Удаляем из кластеров
                cluster_id = event_record.get("cluster_id")
                if cluster_id and cluster_id in self.episodic_clusters:
                    if event_id in self.episodic_clusters[cluster_id]:
                        self.episodic_clusters[cluster_id].remove(event_id)
                    if not self.episodic_clusters[cluster_id]:  # Если кластер пустой
                        del self.episodic_clusters[cluster_id]
                
                # Удаляем событие
                del self.events[event_id]
            
            def _calculate_event_similarity(self, event1: Dict[str, Any], event2: Dict[str, Any]) -> float:
                """Вычислить схожесть между событиями"""
                # Схожесть по контекстуальным тегам
                tags1 = set(event1["contextual_tags"])
                tags2 = set(event2["contextual_tags"])
                
                if not tags1 and not tags2:
                    tag_similarity = 1.0
                elif not tags1 or not tags2:
                    tag_similarity = 0.0
                else:
                    intersection = tags1.intersection(tags2)
                    union = tags1.union(tags2)
                    tag_similarity = len(intersection) / len(union)
                
                # Схожесть по типу события
                type1 = event1["data"].get("type", "")
                type2 = event2["data"].get("type", "")
                type_similarity = 1.0 if type1 == type2 else 0.0
                
                # Временная близость (обратно пропорциональна разнице во времени)
                time_diff = abs(event1["timestamp"] - event2["timestamp"])
                time_similarity = max(0.0, 1.0 - (time_diff / (24 * 3600)))  # Уменьшается за 24 часа
                
                # Взвешенная схожесть
                total_similarity = (
                    tag_similarity * 0.5 +
                    type_similarity * 0.3 +
                    time_similarity * 0.2
                )
                
                return total_similarity
            
            def _string_similarity(self, str1: str, str2: str) -> float:
                """Простое вычисление схожести строк"""
                if str1 == str2:
                    return 1.0
                
                # Простая схожесть на основе общих символов
                set1 = set(str1)
                set2 = set(str2)
                
                if not set1 and not set2:
                    return 1.0
                if not set1 or not set2:
                    return 0.0
                
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                
                return intersection / union if union > 0 else 0.0
            
            def _parse_timestamp(self, timestamp_str: str) -> float:
                """Парсить временную метку из строки"""
                try:
                    # Пытаемся различные форматы
                    for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                        try:
                            return datetime.strptime(timestamp_str, fmt).timestamp()
                        except ValueError:
                            continue
                    
                    # Если ничего не получилось, используем текущее время
                    return time.time()
                    
                except Exception:
                    return time.time()
            
            def _calculate_std(self, values: List[float]) -> float:
                """Вычислить стандартное отклонение"""
                if len(values) <= 1:
                    return 0.0
                
                mean = sum(values) / len(values)
                variance = sum((x - mean) ** 2 for x in values) / len(values)
                return variance ** 0.5
            
            def get_performance_metrics(self) -> Dict[str, Any]:
                """Получить метрики производительности"""
                return {
                    "total_events": len(self.events),
                    "total_clusters": len(self.episodic_clusters),
                    "total_contextual_links": len(self.contextual_links),
                    "avg_importance": sum(e["importance_score"] for e in self.events.values()) / max(len(self.events), 1),
                    "avg_store_time": sum(self.performance_metrics["store_time"]) / max(len(self.performance_metrics["store_time"]), 1),
                    "avg_retrieval_time": sum(self.performance_metrics["retrieval_time"]) / max(len(self.performance_metrics["retrieval_time"]), 1),
                    "capacity_usage": (len(self.events) / self.max_events) * 100,
                    "temporal_coverage_days": len(self.temporal_index),
                    "access_count_total": sum(e["access_count"] for e in self.events.values())
                }
            
            def reset(self):
                """Сбросить состояние эпизодической памяти"""
                self.events.clear()
                self.event_timeline.clear()
                self.temporal_index.clear()
                self.contextual_links.clear()
                self.episodic_clusters.clear()
                self.performance_metrics = {
                    "store_time": [],
                    "retrieval_time": [],
                    "temporal_query_time": []
                }
                self.access_log.clear()
        
        return EpisodicMemoryEngine
    
    def test_episodic_memory_initialization(self, episodic_memory_engine, base_config):
        """Тест инициализации эпизодической памяти"""
        episodic_config = base_config["memory_layers"]["episodic"]
        memory_engine = episodic_memory_engine(episodic_config)
        
        assert memory_engine.max_events == episodic_config["max_events"]
        assert memory_engine.retention_days == episodic_config["retention_days"]
        assert len(memory_engine.events) == 0
        assert len(memory_engine.event_timeline) == 0
        assert len(memory_engine.temporal_index) == 0
    
    def test_event_storage_basic(self, episodic_memory_engine, base_config):
        """Тест базового сохранения событий"""
        episodic_config = base_config["memory_layers"]["episodic"]
        memory_engine = episodic_memory_engine(episodic_config)
        
        test_event = {
            "id": "event_001",
            "type": "knowledge_acquisition",
            "name": "Learning About AI",
            "description": "Acquired knowledge about artificial intelligence concepts",
            "category": "learning",
            "timestamp": time.time(),
            "importance_level": "high"
        }
        
        result = memory_engine.store_event(test_event)
        
        assert result is True
        assert "event_001" in memory_engine.events
        assert memory_engine.events["event_001"]["data"]["name"] == "Learning About AI"
        assert memory_engine.events["event_001"]["importance_score"] > 0
        assert "event_001" in memory_engine.event_timeline
    
    def test_temporal_event_retrieval(self, episodic_memory_engine, base_config):
        """Тест извлечения событий по времени"""
        episodic_config = base_config["memory_layers"]["episodic"]
        memory_engine = episodic_memory_engine(episodic_config)
        
        # Создаем события с разными временными метками
        base_time = time.time()
        events_data = [
            {"id": "event_early", "type": "learning", "name": "Early Event", "timestamp": base_time - 86400},  # 1 день назад
            {"id": "event_recent", "type": "application", "name": "Recent Event", "timestamp": base_time},  # сейчас
            {"id": "event_future", "type": "planning", "name": "Future Event", "timestamp": base_time + 86400}  # 1 день в будущем
        ]
        
        for event_data in events_data:
            memory_engine.store_event(event_data)
        
        # Извлекаем события за последние 2 дня
        start_time = base_time - 86400
        end_time = base_time + 86400
        
        retrieved_events = memory_engine.retrieve_events_by_timeframe(start_time, end_time)
        
        assert len(retrieved_events) == 2
        
        event_ids = [event["event_id"] for event in retrieved_events]
        assert "event_early" in event_ids
        assert "event_recent" in event_ids
        assert "event_future" not in event_ids
    
    def test_contextual_event_retrieval(self, episodic_memory_engine, base_config):
        """Тест извлечения событий по контексту"""
        episodic_config = base_config["memory_layers"]["episodic"]
        memory_engine = episodic_memory_engine(episodic_config)
        
        # Создаем события с разными контекстами
        events_data = [
            {
                "id": "ai_event_1",
                "type": "learning",
                "name": "AI Study Session",
                "description": "Studied machine learning algorithms",
                "category": "education",
                "concepts": ["artificial_intelligence", "machine_learning"]
            },
            {
                "id": "ai_event_2",
                "type": "application",
                "name": "AI Project Implementation",
                "description": "Implemented neural network model",
                "category": "project",
                "concepts": ["artificial_intelligence", "neural_networks"]
            },
            {
                "id": "bias_event",
                "type": "research",
                "name": "Cognitive Bias Study",
                "description": "Researched confirmation bias effects",
                "category": "research",
                "concepts": ["cognitive_bias", "confirmation_bias"]
            }
        ]
        
        for event_data in events_data:
            memory_engine.store_event(event_data)
        
        # Ищем события связанные с AI
        ai_events = memory_engine.retrieve_events_by_context(["artificial_intelligence"])
        
        assert len(ai_events) == 2
        
        event_ids = [event["event_id"] for event in ai_events]
        assert "ai_event_1" in event_ids
        assert "ai_event_2" in event_ids
        assert "bias_event" not in event_ids
        
        # Ищем события по типу
        learning_events = memory_engine.retrieve_events_by_context(["learning"])
        assert len(learning_events) >= 1
    
    def test_event_importance_calculation(self, episodic_memory_engine, base_config):
        """Тест вычисления важности событий"""
        episodic_config = base_config["memory_layers"]["episodic"]
        memory_engine = episodic_memory_engine(episodic_config)
        
        # Создаем события разных типов важности
        high_importance_event = {
            "id": "critical_event",
            "type": "milestone",
            "name": "Critical System Failure",
            "description": "System experienced critical failure requiring immediate attention",
            "impact_level": "critical",
            "emotional_impact": 1.0
        }
        
        low_importance_event = {
            "id": "minor_event",
            "type": "observation",
            "name": "Minor Observation",
            "description": "Noticed small UI improvement opportunity",
            "impact_level": "low"
        }
        
        memory_engine.store_event(high_importance_event)
        memory_engine.store_event(low_importance_event)
        
        high_importance = memory_engine.events["critical_event"]["importance_score"]
        low_importance = memory_engine.events["minor_event"]["importance_score"]
        
        assert high_importance > low_importance
        assert 0.0 <= high_importance <= 1.0
        assert 0.0 <= low_importance <= 1.0
        assert high_importance >= 0.7  # Критическое событие должно иметь высокую важность
    
    def test_event_clustering(self, episodic_memory_engine, base_config):
        """Тест кластеризации событий"""
        episodic_config = base_config["memory_layers"]["episodic"]
        memory_engine = episodic_memory_engine(episodic_config)
        
        # Создаем связанные события для одного кластера
        cluster_events = [
            {
                "id": "cluster_1_1",
                "type": "learning",
                "name": "AI Learning Session 1",
                "category": "education",
                "concepts": ["artificial_intelligence"]
            },
            {
                "id": "cluster_1_2", 
                "type": "application",
                "name": "AI Project Work",
                "category": "project",
                "concepts": ["artificial_intelligence", "machine_learning"]
            },
            {
                "id": "cluster_1_3",
                "type": "review",
                "name": "AI Knowledge Review",
                "category": "review",
                "concepts": ["artificial_intelligence", "neural_networks"]
            }
        ]
        
        # Создаем событие для другого кластера
        different_cluster_event = {
            "id": "cluster_2_1",
            "type": "learning",
            "name": "Psychology Study",
            "category": "education", 
            "concepts": ["cognitive_psychology"]
        }
        
        for event_data in cluster_events:
            memory_engine.store_event(event_data)
        
        memory_engine.store_event(different_cluster_event)
        
        # Проверяем кластеризацию
        assert len(memory_engine.episodic_clusters) >= 1
        
        # События из AI домена должны попасть в один кластер
        cluster_1_events = []
        for cluster_id, event_ids in memory_engine.episodic_clusters.items():
            if "cluster_1" in event_ids[0] if event_ids else False:
                cluster_1_events.extend(event_ids)
        
        # Проверяем что AI события попали в один кластер
        ai_cluster_count = 0
        for cluster_id, event_ids in memory_engine.episodic_clusters.items():
            ai_related = [eid for eid in event_ids if any(memory_engine.events[eid]["data"].get("concepts", []))]
            if len(ai_related) >= 2:  # Кластер с AI событиями
                ai_cluster_count = len(ai_related)
                break
        
        assert ai_cluster_count >= 2
    
    def test_similar_event_discovery(self, episodic_memory_engine, base_config):
        """Тест поиска похожих событий"""
        episodic_config = base_config["memory_layers"]["episodic"]
        memory_engine = episodic_memory_engine(episodic_config)
        
        # Создаем похожие события
        base_event = {
            "id": "base_event",
            "type": "learning",
            "name": "AI Fundamentals Study",
            "category": "education",
            "concepts": ["artificial_intelligence", "machine_learning"],
            "timestamp": time.time()
        }
        
        similar_event = {
            "id": "similar_event",
            "type": "learning",
            "name": "Machine Learning Basics", 
            "category": "education",
            "concepts": ["artificial_intelligence", "machine_learning"],
            "timestamp": time.time() + 3600  # Через час
        }
        
        different_event = {
            "id": "different_event",
            "type": "application",
            "name": "Web Development Project",
            "category": "project",
            "concepts": ["web_development", "javascript"],
            "timestamp": time.time()
        }
        
        memory_engine.store_event(base_event)
        memory_engine.store_event(similar_event)
        memory_engine.store_event(different_event)
        
        # Ищем похожие события
        similar_events = memory_engine.find_similar_events("base_event", similarity_threshold=0.5)
        
        assert len(similar_events) > 0
        
        # Найденное похожее событие должно иметь высокую оценку схожести
        if similar_events:
            most_similar = similar_events[0]
            assert most_similar["event_id"] == "similar_event"
            assert most_similar["similarity_score"] > 0.5
    
    def test_episodic_summary_generation(self, episodic_memory_engine, base_config):
        """Тест генерации сводки эпизодической памяти"""
        episodic_config = base_config["memory_layers"]["episodic"]
        memory_engine = episodic_memory_engine(episodic_config)
        
        # Создаем разнообразные события за последнюю неделю
        current_time = time.time()
        
        events_data = [
            {
                "id": "summary_event_1",
                "type": "learning",
                "name": "AI Study",
                "category": "education",
                "concepts": ["artificial_intelligence"],
                "timestamp": current_time - 86400 * 2
            },
            {
                "id": "summary_event_2",
                "type": "application", 
                "name": "ML Project",
                "category": "project",
                "concepts": ["machine_learning"],
                "timestamp": current_time - 86400
            },
            {
                "id": "summary_event_3",
                "type": "research",
                "name": "Bias Research",
                "category": "research", 
                "concepts": ["cognitive_bias"],
                "timestamp": current_time - 3600
            }
        ]
        
        for event_data in events_data:
            memory_engine.store_event(event_data)
        
        # Генерируем сводку за 7 дней
        summary = memory_engine.get_episodic_summary(time_period_days=7)
        
        assert summary["period_days"] == 7
        assert summary["total_events"] == 3
        assert summary["avg_importance"] > 0
        assert len(summary["top_contexts"]) > 0
        assert len(summary["clusters"]) >= 0
        assert summary["temporal_density"] > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_episodic_operations(self, episodic_memory_engine, base_config):
        """Тест конкурентных операций с эпизодической памятью"""
        episodic_config = base_config["memory_layers"]["episodic"]
        memory_engine = episodic_memory_engine(episodic_config)
        
        async def store_event_task(event_id: int):
            event_data = {
                "id": f"concurrent_event_{event_id}",
                "type": "test_operation",
                "name": f"Concurrent Event {event_id}",
                "description": f"Event {event_id} for concurrent testing",
                "category": "concurrent_test",
                "timestamp": time.time()
            }
            return memory_engine.store_event(event_data)
        
        async def retrieve_event_task(query_tags: List[str]):
            return memory_engine.retrieve_events_by_context(query_tags, max_results=10)
        
        # Выполняем конкурентные операции
        start_time = time.time()
        
        store_tasks = [store_event_task(i) for i in range(15)]
        retrieve_tasks = [retrieve_event_task([f"tag_{i}"]) for i in range(5)]
        
        store_results = await asyncio.gather(*store_tasks)
        retrieve_results = await asyncio.gather(*retrieve_tasks)
        
        concurrent_time = time.time() - start_time
        
        # Проверяем результаты
        assert all(result is True for result in store_results)
        assert all(isinstance(result, list) for result in retrieve_results)
        
        # Проверяем итоговое состояние
        metrics = memory_engine.get_performance_metrics()
        assert metrics["total_events"] == 15
        assert concurrent_time < 10.0
    
    def test_cognitive_bias_episodic_integration(self, episodic_memory_engine, base_config):
        """Тест интеграции с эпизодической памятью для когнитивных искажений"""
        episodic_config = base_config["memory_layers"]["episodic"]
        memory_engine = episodic_memory_engine(episodic_config)
        
        # Создаем эпизоды связанные с когнитивными искажениями
        bias_events = []
        for bias_name, bias_data in list(COGNITIVE_BIASES_DATASET.items())[:5]:  # Ограничиваем для теста
            event_data = {
                "id": f"bias_event_{bias_data['id']}",
                "type": "bias_detection",
                "name": f"Detected {bias_data['name']}",
                "description": bias_data["definition"],
                "category": "cognitive_analysis",
                "concepts": [bias_name] + bias_data.get("related_concepts", [])[:3],  # Ограничиваем количество
                "severity": bias_data["severity"],
                "examples": bias_data.get("examples", {}),
                "detection_context": bias_data.get("contexts", {}),
                "timestamp": time.time() - len(bias_events) * 3600  # Разные времена
            }
            
            result = memory_engine.store_event(event_data)
            assert result is True
            bias_events.append(event_data)
        
        # Тестируем извлечение событий по когнитивным искажениям
        bias_related_events = memory_engine.retrieve_events_by_context(["cognitive_bias"])
        
        assert len(bias_related_events) > 0
        
        # Проверяем что найдены события по разным искажениям
        found_bias_types = set()
        for event in bias_related_events:
            concepts = event["event_data"].get("concepts", [])
            for concept in concepts:
                if "bias" in concept:
                    found_bias_types.add(concept)
        
        assert len(found_bias_types) > 1  # Должно быть найдено несколько типов искажений
    
    def test_episodic_memory_error_handling(self, episodic_memory_engine, base_config):
        """Тест обработки ошибок в эпизодической памяти"""
        episodic_config = base_config["memory_layers"]["episodic"]
        memory_engine = episodic_memory_engine(episodic_config)
        
        # Тест с некорректными данными
        invalid_event = None
        result = memory_engine.store_event(invalid_event)
        assert result is False
        
        # Тест с временными рамками
        invalid_timeframe = memory_engine.retrieve_events_by_timeframe(0, -1)
        assert isinstance(invalid_timeframe, list)
        
        # Тест поиска похожих событий с несуществующим ID
        similar = memory_engine.find_similar_events("nonexistent_event")
        assert similar == []
        
        # Тест обновления важности несуществующего события
        result = memory_engine.update_event_importance("nonexistent_event", 0.5)
        assert result is False
        
        # Проверяем логи ошибок
        error_logs = [log for log in memory_engine.access_log if "failed" in log["operation"]]
        assert len(error_logs) > 0
    
    def test_event_update_functionality(self, episodic_memory_engine, base_config):
        """Тест обновления событий"""
        episodic_config = base_config["memory_layers"]["episodic"]
        memory_engine = episodic_memory_engine(episodic_config)
        
        # Создаем событие
        original_event = {
            "id": "update_test_event",
            "type": "initial",
            "name": "Original Event",
            "description": "Original description",
            "timestamp": time.time()
        }
        
        memory_engine.store_event(original_event)
        original_importance = memory_engine.events["update_test_event"]["importance_score"]
        
        # Обновляем важность
        new_importance = min(1.0, original_importance + 0.3)
        result = memory_engine.update_event_importance("update_test_event", new_importance)
        
        assert result is True
        
        updated_importance = memory_engine.events["update_test_event"]["importance_score"]
        assert updated_importance == new_importance
        
        # Проверяем что важность остается в допустимых пределах
        memory_engine.update_event_importance("update_test_event", 2.0)  # Попытка установить значение > 1
        assert memory_engine.events["update_test_event"]["importance_score"] <= 1.0
        
        memory_engine.update_event_importance("update_test_event", -0.5)  # Попытка установить отрицательное значение
        assert memory_engine.events["update_test_event"]["importance_score"] >= 0.0
    
    def test_temporal_consistency(self, episodic_memory_engine, base_config):
        """Тест временной согласованности событий"""
        episodic_config = base_config["memory_layers"]["episodic"]
        memory_engine = episodic_memory_engine(episodic_config)
        
        # Создаем события в правильном временном порядке
        events_data = [
            {"id": "temporal_1", "type": "step1", "name": "First Step", "timestamp": time.time() - 100},
            {"id": "temporal_2", "type": "step2", "name": "Second Step", "timestamp": time.time() - 50},
            {"id": "temporal_3", "type": "step3", "name": "Third Step", "timestamp": time.time()}
        ]
        
        for event_data in events_data:
            memory_engine.store_event(event_data)
        
        # Проверяем временную линию
        assert len(memory_engine.event_timeline) == 3
        
        # Проверяем порядок событий в временной линии
        timeline_timestamps = [memory_engine.events[event_id]["timestamp"] for event_id in memory_engine.event_timeline]
        
        for i in range(len(timeline_timestamps) - 1):
            assert timeline_timestamps[i] <= timeline_timestamps[i + 1]  # Хронологический порядок
        
        # Проверяем временной индекс
        assert len(memory_engine.temporal_index) > 0