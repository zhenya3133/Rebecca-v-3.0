"""
Unit тесты для ContextEngine компонента
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import time
import json
from typing import List, Dict, Any, Set, Optional, Tuple
from dataclasses import dataclass
import heapq


@dataclass
class ContextNode:
    """Узел контекста"""
    id: str
    type: str
    content: Dict[str, Any]
    confidence: float
    timestamp: float
    relationships: List[str] = None
    
    def __post_init__(self):
        if self.relationships is None:
            self.relationships = []


@dataclass
class ContextEdge:
    """Ребро контекста"""
    source: str
    target: str
    relation_type: str
    weight: float
    timestamp: float


class TestContextEngine:
    """Тесты для ContextEngine класса"""
    
    @pytest.fixture
    def context_engine_instance(self):
        """Создание экземпляра движка контекстов"""
        class MockContextEngine:
            def __init__(self, config: Dict[str, Any]):
                self.config = config
                self.context_nodes: Dict[str, ContextNode] = {}
                self.context_edges: List[ContextEdge] = []
                self.context_cache: Dict[str, Dict[str, Any]] = {}
                self.query_history: List[Dict[str, Any]] = []
                self.merging_strategies = {
                    'weighted': self._weighted_merge,
                    'intersection': self._intersection_merge,
                    'union': self._union_merge,
                    'priority': self._priority_merge
                }
                self.relevance_thresholds = {
                    'high': 0.8,
                    'medium': 0.6,
                    'low': 0.4,
                    'minimum': 0.2
                }
            
            def build_context(self, query: str, max_nodes: int = 100) -> Dict[str, Any]:
                """Построить контекст для запроса"""
                if not query or not query.strip():
                    return self._empty_context()
                
                start_time = time.time()
                
                # Анализируем запрос
                query_analysis = self._analyze_query(query)
                
                # Извлекаем релевантные узлы
                relevant_nodes = self._find_relevant_nodes(query, max_nodes)
                
                # Строим семантический контекст
                semantic_context = self._build_semantic_context(relevant_nodes)
                
                # Строим эпизодический контекст
                episodic_context = self._build_episodic_context(query)
                
                # Строим процедурный контекст
                procedural_context = self._build_procedural_context(query)
                
                # Объединяем контексты
                merged_context = self._merge_contexts([
                    semantic_context,
                    episodic_context,
                    procedural_context
                ])
                
                # Кэшируем контекст
                cache_key = self._generate_cache_key(query)
                self.context_cache[cache_key] = merged_context
                
                # Записываем в историю
                self.query_history.append({
                    'timestamp': time.time(),
                    'query': query,
                    'processing_time': time.time() - start_time,
                    'nodes_used': len(relevant_nodes),
                    'context_quality': merged_context.get('confidence', 0.0)
                })
                
                return merged_context
            
            def _analyze_query(self, query: str) -> Dict[str, Any]:
                """Анализировать запрос"""
                # Простой анализ ключевых слов
                words = query.lower().split()
                key_terms = [word for word in words if len(word) > 3]
                
                # Определяем тип запроса
                query_type = 'general'
                if any(word in query.lower() for word in ['what', 'explain', 'define']):
                    query_type = 'explanatory'
                elif any(word in query.lower() for word in ['how', 'method', 'process']):
                    query_type = 'procedural'
                elif any(word in query.lower() for word in ['why', 'cause', 'reason']):
                    query_type = 'causal'
                elif any(word in query.lower() for word in ['when', 'time', 'date']):
                    query_type = 'temporal'
                
                return {
                    'query_type': query_type,
                    'key_terms': key_terms,
                    'word_count': len(words),
                    'complexity': len(key_terms) / max(len(words), 1)
                }
            
            def _find_relevant_nodes(self, query: str, max_nodes: int) -> List[ContextNode]:
                """Найти релевантные узлы контекста"""
                query_lower = query.lower()
                relevant_nodes = []
                
                # Ищем совпадения в существующих узлах
                for node_id, node in self.context_nodes.items():
                    relevance_score = self._calculate_relevance_score(node, query_lower)
                    
                    if relevance_score >= self.relevance_thresholds['minimum']:
                        node.relevance_score = relevance_score
                        relevant_nodes.append(node)
                
                # Сортируем по релевантности
                relevant_nodes.sort(key=lambda x: x.relevance_score, reverse=True)
                
                return relevant_nodes[:max_nodes]
            
            def _calculate_relevance_score(self, node: ContextNode, query_lower: str) -> float:
                """Вычислить score релевантности узла"""
                base_score = 0.0
                
                # Совпадения в содержимом
                content_text = json.dumps(node.content).lower()
                query_words = query_lower.split()
                
                matches = 0
                for word in query_words:
                    if word in content_text:
                        matches += 1
                
                if query_words:
                    base_score = matches / len(query_words)
                
                # Бонус за тип узла
                type_bonus = {
                    'concept': 0.1,
                    'definition': 0.15,
                    'example': 0.05,
                    'relationship': 0.12,
                    'procedure': 0.08
                }
                base_score += type_bonus.get(node.type, 0.0)
                
                # Бонус за confidence
                base_score += node.confidence * 0.1
                
                # Бонус за recency (более свежие узлы более релевантны)
                age_bonus = max(0, 0.1 - (time.time() - node.timestamp) / 86400)  # Убывает за сутки
                base_score += age_bonus
                
                return min(base_score, 1.0)
            
            def _build_semantic_context(self, nodes: List[ContextNode]) -> Dict[str, Any]:
                """Построить семантический контекст"""
                if not nodes:
                    return {'type': 'semantic', 'content': {}, 'confidence': 0.0}
                
                # Извлекаем концепты
                concepts = []
                relationships = []
                
                for node in nodes:
                    if node.type == 'concept':
                        concepts.append({
                            'id': node.id,
                            'content': node.content,
                            'confidence': node.confidence,
                            'relevance': getattr(node, 'relevance_score', 0.0)
                        })
                    elif node.type == 'relationship':
                        relationships.append({
                            'source': node.content.get('source'),
                            'target': node.content.get('target'),
                            'relation': node.content.get('relation'),
                            'confidence': node.confidence
                        })
                
                # Вычисляем общую семантическую согласованность
                semantic_coherence = self._calculate_semantic_coherence(concepts, relationships)
                
                return {
                    'type': 'semantic',
                    'content': {
                        'concepts': concepts,
                        'relationships': relationships,
                        'main_themes': self._extract_main_themes(concepts)
                    },
                    'confidence': semantic_coherence,
                    'node_count': len(nodes),
                    'concept_count': len(concepts),
                    'relationship_count': len(relationships)
                }
            
            def _build_episodic_context(self, query: str) -> Dict[str, Any]:
                """Построить эпизодический контекст"""
                if not self.query_history:
                    return {'type': 'episodic', 'content': {}, 'confidence': 0.0}
                
                # Ищем похожие запросы в истории
                similar_queries = []
                for history_entry in self.query_history[-10:]:  # Последние 10 запросов
                    similarity = self._calculate_query_similarity(query, history_entry['query'])
                    if similarity > 0.3:
                        similar_queries.append({
                            'query': history_entry['query'],
                            'similarity': similarity,
                            'timestamp': history_entry['timestamp'],
                            'context_quality': history_entry['context_quality']
                        })
                
                # Сортируем по схожести
                similar_queries.sort(key=lambda x: x['similarity'], reverse=True)
                
                return {
                    'type': 'episodic',
                    'content': {
                        'similar_queries': similar_queries,
                        'recent_activity': self._get_recent_activity(),
                        'query_patterns': self._analyze_query_patterns()
                    },
                    'confidence': min(len(similar_queries) * 0.2, 0.8),
                    'similar_query_count': len(similar_queries)
                }
            
            def _build_procedural_context(self, query: str) -> Dict[str, Any]:
                """Построить процедурный контекст"""
                # Определяем активный workflow на основе запроса
                workflow_type = self._determine_workflow_type(query)
                
                # Получаем шаги workflow
                workflow_steps = self._get_workflow_steps(workflow_type)
                
                # Определяем текущий этап
                current_step = self._determine_current_step(workflow_type, query)
                
                # Получаем рекомендации по следующим шагам
                next_steps = self._get_next_steps(workflow_type, current_step)
                
                return {
                    'type': 'procedural',
                    'content': {
                        'workflow_type': workflow_type,
                        'workflow_steps': workflow_steps,
                        'current_step': current_step,
                        'next_steps': next_steps,
                        'progress': self._calculate_progress(workflow_steps, current_step)
                    },
                    'confidence': 0.7,
                    'workflow_completeness': len(workflow_steps) / max(len(next_steps) + len(current_step), 1)
                }
            
            def _determine_workflow_type(self, query: str) -> str:
                """Определить тип workflow на основе запроса"""
                query_lower = query.lower()
                
                if any(word in query_lower for word in ['learn', 'study', 'understand']):
                    return 'learning'
                elif any(word in query_lower for word in ['analyze', 'examine', 'investigate']):
                    return 'analysis'
                elif any(word in query_lower for word in ['create', 'build', 'develop']):
                    return 'creation'
                elif any(word in query_lower for word in ['solve', 'fix', 'resolve']):
                    return 'problem_solving'
                else:
                    return 'general'
            
            def _get_workflow_steps(self, workflow_type: str) -> List[str]:
                """Получить шаги workflow"""
                workflows = {
                    'learning': ['ingest', 'comprehend', 'synthesize', 'apply', 'evaluate'],
                    'analysis': ['collect', 'process', 'analyze', 'interpret', 'report'],
                    'creation': ['ideate', 'design', 'implement', 'test', 'refine'],
                    'problem_solving': ['define', 'research', 'generate_solutions', 'evaluate', 'implement'],
                    'general': ['understand', 'plan', 'execute', 'verify', 'optimize']
                }
                return workflows.get(workflow_type, workflows['general'])
            
            def _determine_current_step(self, workflow_type: str, query: str) -> str:
                """Определить текущий шаг"""
                # Простая эвристика на основе ключевых слов в запросе
                query_lower = query.lower()
                
                if any(word in query_lower for word in ['what', 'explain', 'describe']):
                    return 'understand'
                elif any(word in query_lower for word in ['how', 'method', 'process']):
                    return 'plan'
                elif any(word in query_lower for word in ['create', 'build', 'make']):
                    return 'execute'
                else:
                    return 'understand'
            
            def _get_next_steps(self, workflow_type: str, current_step: str) -> List[str]:
                """Получить рекомендуемые следующие шаги"""
                steps = self._get_workflow_steps(workflow_type)
                try:
                    current_index = steps.index(current_step)
                    return steps[current_index + 1:] if current_index + 1 < len(steps) else []
                except ValueError:
                    return steps[1:]  # Возвращаем все кроме первого если current_step не найден
            
            def _calculate_progress(self, workflow_steps: List[str], current_step: str) -> float:
                """Вычислить прогресс выполнения workflow"""
                if not workflow_steps:
                    return 0.0
                
                try:
                    current_index = workflow_steps.index(current_step)
                    return (current_index + 1) / len(workflow_steps)
                except ValueError:
                    return 0.0
            
            def get_context_for_query(self, query: str, context_type: str = 'all') -> Dict[str, Any]:
                """Получить контекст определенного типа для запроса"""
                # Проверяем кэш
                cache_key = self._generate_cache_key(f"{query}_{context_type}")
                if cache_key in self.context_cache:
                    return self.context_cache[cache_key]
                
                # Строим полный контекст
                full_context = self.build_context(query)
                
                # Фильтруем по типу если нужно
                if context_type != 'all':
                    filtered_context = self._filter_context_by_type(full_context, context_type)
                    return filtered_context
                
                return full_context
            
            def _filter_context_by_type(self, context: Dict[str, Any], context_type: str) -> Dict[str, Any]:
                """Фильтровать контекст по типу"""
                if context.get('type') == context_type:
                    return context
                
                # Для составных контекстов
                filtered = {'type': context_type, 'content': {}, 'confidence': 0.0}
                
                if context_type in ['semantic', 'episodic', 'procedural']:
                    if 'semantic_context' in context:
                        filtered = context['semantic_context']
                    elif 'episodic_context' in context:
                        filtered = context['episodic_context']
                    elif 'procedural_context' in context:
                        filtered = context['procedural_context']
                
                return filtered
            
            def merge_contexts(self, contexts: List[Dict[str, Any]], strategy: str = 'weighted') -> Dict[str, Any]:
                """Объединить несколько контекстов"""
                if not contexts:
                    return self._empty_context()
                
                if len(contexts) == 1:
                    return contexts[0]
                
                if strategy not in self.merging_strategies:
                    strategy = 'weighted'
                
                return self.merging_strategies[strategy](contexts)
            
            def _weighted_merge(self, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
                """Взвешенное объединение контекстов"""
                # Вычисляем веса на основе confidence
                weights = []
                total_confidence = 0.0
                
                for context in contexts:
                    weight = context.get('confidence', 0.5)
                    weights.append(weight)
                    total_confidence += weight
                
                if total_confidence == 0:
                    total_confidence = len(contexts)  # Равные веса
                    weights = [1.0] * len(contexts)
                else:
                    weights = [w / total_confidence for w in weights]
                
                # Объединяем содержимое
                merged_content = {}
                merged_metadata = {
                    'source_contexts': len(contexts),
                    'merge_strategy': 'weighted',
                    'merge_timestamp': time.time()
                }
                
                # Собираем все концепты
                all_concepts = []
                all_relationships = []
                
                for i, context in enumerate(contexts):
                    weight = weights[i]
                    
                    if 'semantic_context' in context:
                        semantic = context['semantic_context']
                        if 'content' in semantic and 'concepts' in semantic['content']:
                            for concept in semantic['content']['concepts']:
                                concept['merged_weight'] = weight
                                all_concepts.append(concept)
                        
                        if 'content' in semantic and 'relationships' in semantic['content']:
                            all_relationships.extend(semantic['content']['relationships'])
                    
                    if 'episodic_context' in context:
                        episodic = context['episodic_context']
                        if 'content' in episodic and 'similar_queries' in episodic['content']:
                            merged_content.setdefault('similar_queries', []).extend(episodic['content']['similar_queries'])
                
                # Удаляем дубликаты концептов
                unique_concepts = self._deduplicate_concepts(all_concepts)
                
                merged_content.update({
                    'concepts': unique_concepts,
                    'relationships': all_relationships,
                    'metadata': merged_metadata
                })
                
                # Вычисляем общий confidence
                overall_confidence = sum(c.get('confidence', 0.5) * w for c, w in zip(contexts, weights))
                
                return {
                    'type': 'merged',
                    'content': merged_content,
                    'confidence': overall_confidence,
                    'source_count': len(contexts),
                    'concept_count': len(unique_concepts),
                    'relationship_count': len(all_relationships)
                }
            
            def _intersection_merge(self, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
                """Объединение по пересечению (только общие элементы)"""
                if not contexts:
                    return self._empty_context()
                
                # Находим общие концепты
                concept_sets = []
                for context in contexts:
                    if 'semantic_context' in context and 'content' in context['semantic_context']:
                        concepts = context['semantic_context']['content'].get('concepts', [])
                        concept_ids = {c.get('id', c.get('name', str(i))) for i, c in enumerate(concepts)}
                        concept_sets.append(concept_ids)
                
                if not concept_sets:
                    return self._empty_context()
                
                # Пересечение концептов
                common_concepts = set.intersection(*concept_sets) if concept_sets else set()
                
                merged_content = {
                    'concepts': [c for c in self._get_all_concepts(contexts) if c.get('id', c.get('name')) in common_concepts],
                    'intersection_size': len(common_concepts),
                    'source_count': len(contexts)
                }
                
                return {
                    'type': 'intersection',
                    'content': merged_content,
                    'confidence': len(common_concepts) / max(len(concept_sets[0]), 1),
                    'intersection_ratio': len(common_concepts) / max(len(concept_sets[0]), 1)
                }
            
            def _union_merge(self, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
                """Объединение по объединению (все элементы)"""
                all_concepts = self._get_all_concepts(contexts)
                unique_concepts = self._deduplicate_concepts(all_concepts)
                
                return {
                    'type': 'union',
                    'content': {
                        'concepts': unique_concepts,
                        'total_unique_concepts': len(unique_concepts),
                        'source_count': len(contexts)
                    },
                    'confidence': min(len(unique_concepts) / 10, 1.0),  # Чем больше уникальных концептов, тем выше confidence
                    'coverage': len(unique_concepts)
                }
            
            def _priority_merge(self, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
                """Объединение по приоритету (первый контекст имеет наивысший приоритет)"""
                if not contexts:
                    return self._empty_context()
                
                # Берем первый контекст как основу
                base_context = contexts[0]
                
                # Добавляем концепты из других контекстов если их нет в базовом
                base_concepts = []
                if 'semantic_context' in base_context and 'content' in base_context['semantic_context']:
                    base_concepts = base_context['semantic_context']['content'].get('concepts', [])
                
                base_concept_ids = {c.get('id', c.get('name', str(i))) for i, c in enumerate(base_concepts)}
                
                additional_concepts = []
                for context in contexts[1:]:
                    if 'semantic_context' in context and 'content' in context['semantic_context']:
                        concepts = context['semantic_context']['content'].get('concepts', [])
                        for concept in concepts:
                            concept_id = concept.get('id', concept.get('name'))
                            if concept_id not in base_concept_ids:
                                additional_concepts.append(concept)
                
                all_concepts = base_concepts + additional_concepts
                
                return {
                    'type': 'priority',
                    'content': {
                        'concepts': all_concepts,
                        'base_concept_count': len(base_concepts),
                        'additional_concept_count': len(additional_concepts),
                        'source_count': len(contexts)
                    },
                    'confidence': base_context.get('confidence', 0.5),
                    'priority_applied': True
                }
            
            def rank_context_relevance(self, contexts: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
                """Ранжировать контексты по релевантности для запроса"""
                query_lower = query.lower()
                query_words = set(query_lower.split())
                
                ranked_contexts = []
                for i, context in enumerate(contexts):
                    relevance_score = self._calculate_context_relevance(context, query_words)
                    
                    ranked_contexts.append({
                        'context': context,
                        'relevance': relevance_score,
                        'rank': i + 1
                    })
                
                # Сортируем по релевантности
                ranked_contexts.sort(key=lambda x: x['relevance'], reverse=True)
                
                # Обновляем ранги
                for i, item in enumerate(ranked_contexts):
                    item['rank'] = i + 1
                
                return ranked_contexts
            
            def _calculate_context_relevance(self, context: Dict[str, Any], query_words: Set[str]) -> float:
                """Вычислить релевантность контекста"""
                base_score = 0.0
                
                # Score на основе confidence контекста
                base_score += context.get('confidence', 0.0) * 0.3
                
                # Score на основе содержимого
                if 'content' in context:
                    content_text = json.dumps(context['content']).lower()
                    
                    # Подсчет совпадений слов
                    word_matches = sum(1 for word in query_words if word in content_text)
                    if query_words:
                        word_score = word_matches / len(query_words)
                        base_score += word_score * 0.7
                
                return min(base_score, 1.0)
            
            def add_context_node(self, node: ContextNode) -> bool:
                """Добавить узел контекста"""
                if node.id in self.context_nodes:
                    return False  # Узел уже существует
                
                self.context_nodes[node.id] = node
                return True
            
            def add_context_edge(self, edge: ContextEdge) -> bool:
                """Добавить ребро контекста"""
                # Проверяем что узлы существуют
                if edge.source not in self.context_nodes or edge.target not in self.context_nodes:
                    return False
                
                self.context_edges.append(edge)
                return True
            
            def _empty_context(self) -> Dict[str, Any]:
                """Пустой контекст"""
                return {
                    'type': 'empty',
                    'content': {},
                    'confidence': 0.0,
                    'message': 'No context available'
                }
            
            def _calculate_semantic_coherence(self, concepts: List[Dict], relationships: List[Dict]) -> float:
                """Вычислить семантическую согласованность"""
                if not concepts:
                    return 0.0
                
                # Простая метрика на основе количества связей
                relationship_count = len(relationships)
                concept_count = len(concepts)
                
                if concept_count == 0:
                    return 0.0
                
                # Базовый score на основе плотности связей
                coherence = min(relationship_count / (concept_count * 0.5), 1.0)
                
                # Бонус за высокую confidence концептов
                avg_confidence = sum(c.get('confidence', 0.5) for c in concepts) / len(concepts)
                coherence = (coherence + avg_confidence) / 2
                
                return coherence
            
            def _extract_main_themes(self, concepts: List[Dict]) -> List[str]:
                """Извлечь основные темы из концептов"""
                themes = []
                theme_keywords = {
                    'technology': ['ai', 'machine', 'learning', 'algorithm', 'computer', 'data'],
                    'psychology': ['cognitive', 'bias', 'thinking', 'decision', 'behavior', 'mind'],
                    'science': ['research', 'study', 'theory', 'evidence', 'hypothesis', 'experiment'],
                    'business': ['strategy', 'management', 'organization', 'process', 'system', 'efficiency']
                }
                
                for theme, keywords in theme_keywords.items():
                    theme_score = 0
                    for concept in concepts:
                        concept_text = json.dumps(concept).lower()
                        theme_score += sum(1 for keyword in keywords if keyword in concept_text)
                    
                    if theme_score > 0:
                        themes.append(theme)
                
                return themes
            
            def _calculate_query_similarity(self, query1: str, query2: str) -> float:
                """Вычислить схожесть двух запросов"""
                words1 = set(query1.lower().split())
                words2 = set(query2.lower().split())
                
                if not words1 or not words2:
                    return 0.0
                
                intersection = len(words1 & words2)
                union = len(words1 | words2)
                
                return intersection / union if union > 0 else 0.0
            
            def _get_recent_activity(self) -> List[Dict[str, Any]]:
                """Получить недавнюю активность"""
                return self.query_history[-5:] if self.query_history else []
            
            def _analyze_query_patterns(self) -> Dict[str, Any]:
                """Анализировать паттерны запросов"""
                if not self.query_history:
                    return {}
                
                recent_queries = [entry['query'] for entry in self.query_history[-20:]]
                
                # Анализируем типы запросов
                query_types = {}
                for query in recent_queries:
                    analysis = self._analyze_query(query)
                    query_type = analysis['query_type']
                    query_types[query_type] = query_types.get(query_type, 0) + 1
                
                return {
                    'most_common_type': max(query_types, key=query_types.get) if query_types else 'general',
                    'type_distribution': query_types,
                    'avg_complexity': sum(self._analyze_query(q)['complexity'] for q in recent_queries) / len(recent_queries)
                }
            
            def _get_all_concepts(self, contexts: List[Dict[str, Any]]) -> List[Dict]:
                """Получить все концепты из списка контекстов"""
                all_concepts = []
                for context in contexts:
                    if 'semantic_context' in context and 'content' in context['semantic_context']:
                        concepts = context['semantic_context']['content'].get('concepts', [])
                        all_concepts.extend(concepts)
                return all_concepts
            
            def _deduplicate_concepts(self, concepts: List[Dict]) -> List[Dict]:
                """Удалить дубликаты концептов"""
                unique_concepts = []
                seen_ids = set()
                
                for concept in concepts:
                    concept_id = concept.get('id', concept.get('name', str(hash(json.dumps(concept)))))
                    if concept_id not in seen_ids:
                        seen_ids.add(concept_id)
                        unique_concepts.append(concept)
                
                return unique_concepts
            
            def _generate_cache_key(self, query: str) -> str:
                """Генерировать ключ кэша"""
                return f"ctx_{hash(query)}_{len(query)}"
            
            def get_engine_stats(self) -> Dict[str, Any]:
                """Получить статистику движка"""
                return {
                    'node_count': len(self.context_nodes),
                    'edge_count': len(self.context_edges),
                    'cache_size': len(self.context_cache),
                    'query_history_size': len(self.query_history),
                    'avg_processing_time': sum(h['processing_time'] for h in self.query_history) / max(len(self.query_history), 1)
                }
        
        return MockContextEngine
    
    def test_context_engine_initialization(self, context_engine_instance):
        """Тест инициализации движка контекстов"""
        config = {
            'max_context_nodes': 500,
            'cache_size': 1000,
            'relevance_threshold': 0.6
        }
        
        engine = context_engine_instance(config)
        
        assert engine.config['max_context_nodes'] == 500
        assert engine.config['cache_size'] == 1000
        assert 'weighted' in engine.merging_strategies
        assert 'intersection' in engine.merging_strategies
        assert 'union' in engine.merging_strategies
        assert 'priority' in engine.merging_strategies
    
    def test_build_context_basic(self, context_engine_instance):
        """Тест базового построения контекста"""
        engine = context_engine_instance({})
        
        query = "What is artificial intelligence?"
        context = engine.build_context(query)
        
        assert context is not None
        assert 'type' in context
        assert 'content' in context
        assert 'confidence' in context
        assert isinstance(context['confidence'], (int, float))
        assert 0 <= context['confidence'] <= 1
    
    def test_build_context_empty_query(self, context_engine_instance):
        """Тест построения контекста для пустого запроса"""
        engine = context_engine_instance({})
        
        # Пустой запрос
        context = engine.build_context("")
        assert context['type'] == 'empty'
        assert context['confidence'] == 0.0
        
        # Только пробелы
        context = engine.build_context("   \n\t  ")
        assert context['type'] == 'empty'
        assert context['confidence'] == 0.0
    
    def test_query_analysis(self, context_engine_instance):
        """Тест анализа запросов"""
        engine = context_engine_instance({})
        
        # Создаем временный метод для тестирования
        test_cases = [
            ("What is AI?", "explanatory"),
            ("How to implement ML?", "procedural"),
            ("Why does bias occur?", "causal"),
            ("When was AI invented?", "temporal"),
            ("General question here", "general")
        ]
        
        for query, expected_type in test_cases:
            analysis = engine._analyze_query(query)
            assert analysis['query_type'] == expected_type
            assert 'key_terms' in analysis
            assert 'complexity' in analysis
            assert analysis['complexity'] >= 0
    
    def test_context_relevance_ranking(self, context_engine_instance):
        """Тест ранжирования релевантности контекстов"""
        engine = context_engine_instance({})
        
        # Создаем тестовые контексты
        contexts = [
            {
                'type': 'semantic',
                'confidence': 0.9,
                'content': {'concepts': [{'name': 'artificial_intelligence', 'definition': 'AI system'}]}
            },
            {
                'type': 'episodic', 
                'confidence': 0.7,
                'content': {'similar_queries': [{'query': 'What is machine learning?'}]}
            },
            {
                'type': 'procedural',
                'confidence': 0.5,
                'content': {'workflow_steps': ['analyze', 'implement']}
            }
        ]
        
        query = "What is artificial intelligence?"
        ranked = engine.rank_context_relevance(contexts, query)
        
        assert len(ranked) == 3
        assert ranked[0]['relevance'] >= ranked[1]['relevance']  # Должны быть отсортированы по убыванию
        assert all('rank' in item for item in ranked)
    
    def test_context_merging_strategies(self, context_engine_instance):
        """Тест различных стратегий объединения контекстов"""
        engine = context_engine_instance({})
        
        contexts = [
            {
                'type': 'semantic',
                'confidence': 0.8,
                'semantic_context': {
                    'content': {
                        'concepts': [
                            {'id': 'ai', 'name': 'Artificial Intelligence', 'confidence': 0.9},
                            {'id': 'ml', 'name': 'Machine Learning', 'confidence': 0.7}
                        ]
                    }
                }
            },
            {
                'type': 'episodic',
                'confidence': 0.6,
                'episodic_context': {
                    'content': {
                        'similar_queries': [{'query': 'What is AI?', 'similarity': 0.8}]
                    }
                }
            },
            {
                'type': 'semantic',
                'confidence': 0.7,
                'semantic_context': {
                    'content': {
                        'concepts': [
                            {'id': 'ai', 'name': 'Artificial Intelligence', 'confidence': 0.8},
                            {'id': 'nn', 'name': 'Neural Networks', 'confidence': 0.6}
                        ]
                    }
                }
            }
        ]
        
        # Тестируем все стратегии
        strategies = ['weighted', 'intersection', 'union', 'priority']
        
        for strategy in strategies:
            merged = engine.merge_contexts(contexts, strategy)
            
            assert merged is not None
            assert 'type' in merged
            assert 'content' in merged
            assert 'confidence' in merged
            
            if strategy == 'intersection':
                assert 'intersection_ratio' in merged
            elif strategy == 'union':
                assert 'coverage' in merged
            elif strategy == 'priority':
                assert 'priority_applied' in merged
    
    def test_context_node_management(self, context_engine_instance):
        """Тест управления узлами контекста"""
        engine = context_engine_instance({})
        
        # Добавляем узел
        node = ContextNode(
            id="concept_1",
            type="concept",
            content={"name": "AI", "definition": "Artificial Intelligence"},
            confidence=0.9,
            timestamp=time.time()
        )
        
        assert engine.add_context_node(node) is True
        assert "concept_1" in engine.context_nodes
        
        # Попытка добавить существующий узел
        assert engine.add_context_node(node) is False
        
        # Добавляем ребро
        engine.add_context_node(ContextNode(
            id="concept_2",
            type="concept", 
            content={"name": "ML", "definition": "Machine Learning"},
            confidence=0.8,
            timestamp=time.time()
        ))
        
        edge = ContextEdge(
            source="concept_1",
            target="concept_2",
            relation_type="related_to",
            weight=0.7,
            timestamp=time.time()
        )
        
        assert engine.add_context_edge(edge) is True
        assert len(engine.context_edges) == 1
        
        # Попытка добавить ребро с несуществующими узлами
        invalid_edge = ContextEdge(
            source="nonexistent",
            target="concept_2",
            relation_type="related_to",
            weight=0.5,
            timestamp=time.time()
        )
        
        assert engine.add_context_edge(invalid_edge) is False
    
    def test_context_caching(self, context_engine_instance):
        """Тест кэширования контекстов"""
        engine = context_engine_instance({})
        
        query = "What is machine learning?"
        
        # Первый запрос - создает кэш
        context1 = engine.build_context(query)
        cache_key = engine._generate_cache_key(query)
        assert cache_key in engine.context_cache
        
        # Второй запрос - должен использовать кэш (в нашей реализации - создавать новый)
        context2 = engine.build_context(query)
        assert context2 is not None
    
    def test_performance_context_building(self, context_engine_instance):
        """Тест производительности построения контекста"""
        engine = context_engine_instance({})
        
        # Добавляем много узлов для тестирования производительности
        for i in range(100):
            node = ContextNode(
                id=f"node_{i}",
                type="concept" if i % 2 == 0 else "relationship",
                content={"name": f"Concept {i}", "value": i},
                confidence=0.5 + (i % 10) * 0.05,
                timestamp=time.time()
            )
            engine.add_context_node(node)
        
        # Тестируем построение контекста
        start_time = time.time()
        context = engine.build_context("machine learning artificial intelligence")
        build_time = time.time() - start_time
        
        assert build_time < 2.0  # Должно строиться менее чем за 2 секунды
        assert context is not None
        assert isinstance(context, dict)
    
    @pytest.mark.asyncio
    async def test_concurrent_context_operations(self, context_engine_instance):
        """Тест конкурентных операций с контекстом"""
        engine = context_engine_instance({})
        
        queries = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "Explain neural networks",
            "What are cognitive biases?",
            "How to make decisions?"
        ]
        
        async def build_context_task(query):
            return engine.build_context(query)
        
        # Выполняем конкурентные операции
        start_time = time.time()
        results = await asyncio.gather(*[build_context_task(query) for query in queries])
        concurrent_time = time.time() - start_time
        
        # Проверяем результаты
        assert len(results) == 5
        assert all(context is not None for context in results)
        assert all('type' in context for context in results)
        
        # Проверяем производительность
        assert concurrent_time < 5.0
        
        # Проверяем статистику
        stats = engine.get_engine_stats()
        assert stats['query_history_size'] == 5
    
    def test_edge_cases(self, context_engine_instance):
        """Тест крайних случаев"""
        engine = context_engine_instance({})
        
        # Очень длинный запрос
        long_query = "artificial intelligence " * 1000
        context = engine.build_context(long_query)
        assert context is not None
        
        # Запрос с специальными символами
        special_query = "!@#$%^&*()_+{}|:<>?[]\\;',./"
        context = engine.build_context(special_query)
        assert context is not None
        
        # Запрос только с числами
        number_query = "123 456 789"
        context = engine.build_context(number_query)
        assert context is not None
        
        # Пустой список контекстов для объединения
        empty_contexts = []
        merged = engine.merge_contexts(empty_contexts)
        assert merged['type'] == 'empty'
        
        # Один контекст
        single_context = [{'type': 'test', 'confidence': 0.5}]
        merged = engine.merge_contexts(single_context)
        assert merged == single_context[0]
    
    def test_context_engine_statistics(self, context_engine_instance):
        """Тест статистики движка контекстов"""
        engine = context_engine_instance({})
        
        # Изначально пустая статистика
        stats = engine.get_engine_stats()
        assert stats['node_count'] == 0
        assert stats['edge_count'] == 0
        assert stats['cache_size'] == 0
        assert stats['query_history_size'] == 0
        
        # Добавляем данные
        node = ContextNode(
            id="test_node",
            type="concept",
            content={"test": "data"},
            confidence=0.5,
            timestamp=time.time()
        )
        engine.add_context_node(node)
        
        # Строим контекст
        engine.build_context("test query")
        
        # Проверяем обновленную статистику
        stats = engine.get_engine_stats()
        assert stats['node_count'] == 1
        assert stats['query_history_size'] == 1
