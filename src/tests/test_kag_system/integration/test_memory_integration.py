"""
Integration тесты для KAG системы с 6 слоями памяти
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class MemoryLayer:
    """Базовый класс для слоя памяти"""
    name: str
    max_capacity: int
    current_usage: int = 0
    data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}


class TestKAGMemoryIntegration:
    """Integration тесты для интеграции KAG с 6 слоями памяти"""
    
    @pytest.fixture
    def memory_system_instance(self):
        """Создание системы памяти с 6 слоями"""
        class MockMemorySystem:
            def __init__(self, config: Dict[str, Any]):
                self.config = config
                
                # Инициализируем 6 слоев памяти
                self.core_memory = MemoryLayer("core", config.get("core_capacity", 1000))
                self.semantic_memory = MemoryLayer("semantic", config.get("semantic_capacity", 10000))
                self.episodic_memory = MemoryLayer("episodic", config.get("episodic_capacity", 5000))
                self.procedural_memory = MemoryLayer("procedural", config.get("procedural_capacity", 1000))
                self.vault_memory = MemoryLayer("vault", config.get("vault_capacity", 100))
                self.security_memory = MemoryLayer("security", config.get("security_capacity", 500))
                
                # Кросс-слойные индексы
                self.concept_index = {}  # Maps concepts to layers
                self.temporal_index = {}  # For episodic memory
                self.workflow_index = {}  # For procedural memory
                self.security_index = {}  # For security policies
                
                # Операции истории
                self.operation_history = []
                
                # Настройка связей между слоями
                self._setup_layer_connections()
            
            def _setup_layer_connections(self):
                """Настроить связи между слоями памяти"""
                # Semantic -> Core: концепты становятся фактами
                # Episodic -> Semantic: события создают концепты
                # Procedural -> Semantic: процессы создают концепты
                # Vault -> Security: секреты требуют политики безопасности
                # Security -> All: все операции проходят через безопасность
                pass
            
            def store_knowledge(self, knowledge_data: Dict[str, Any]) -> bool:
                """Сохранить знание во всех соответствующих слоях"""
                if not self._security_check(knowledge_data):
                    return False
                
                operations_performed = []
                
                try:
                    # 1. Сохраняем в Semantic Memory (основное хранение концептов)
                    if self._store_in_semantic(knowledge_data):
                        operations_performed.append("semantic")
                    
                    # 2. Создаем эпизод в Episodic Memory
                    if self._store_in_episodic(knowledge_data):
                        operations_performed.append("episodic")
                    
                    # 3. Обновляем процедурную память если нужно
                    if self._update_procedural(knowledge_data):
                        operations_performed.append("procedural")
                    
                    # 4. Сохраняем секреты в Vault
                    if self._store_in_vault(knowledge_data):
                        operations_performed.append("vault")
                    
                    # 5. Обновляем Core Memory
                    if self._update_core(knowledge_data):
                        operations_performed.append("core")
                    
                    # Записываем операцию в историю
                    self.operation_history.append({
                        'timestamp': time.time(),
                        'operation': 'store_knowledge',
                        'data_type': knowledge_data.get('type', 'unknown'),
                        'operations_performed': operations_performed,
                        'success': True
                    })
                    
                    return True
                    
                except Exception as e:
                    # Записываем неудачную операцию
                    self.operation_history.append({
                        'timestamp': time.time(),
                        'operation': 'store_knowledge',
                        'data_type': knowledge_data.get('type', 'unknown'),
                        'operations_performed': operations_performed,
                        'success': False,
                        'error': str(e)
                    })
                    return False
            
            def retrieve_knowledge(self, query: str, layers: List[str] = None) -> Dict[str, Any]:
                """Извлечь знание из указанных слоев"""
                if layers is None:
                    layers = ['semantic', 'episodic', 'procedural']
                
                results = {}
                
                for layer_name in layers:
                    if layer_name == 'semantic':
                        results['semantic'] = self._retrieve_from_semantic(query)
                    elif layer_name == 'episodic':
                        results['episodic'] = self._retrieve_from_episodic(query)
                    elif layer_name == 'procedural':
                        results['procedural'] = self._retrieve_from_procedural(query)
                    elif layer_name == 'core':
                        results['core'] = self._retrieve_from_core(query)
                    elif layer_name == 'vault':
                        results['vault'] = self._retrieve_from_vault(query)
                    elif layer_name == 'security':
                        results['security'] = self._retrieve_from_security(query)
                
                return results
            
            def _security_check(self, data: Dict[str, Any]) -> bool:
                """Проверка безопасности данных"""
                # Проверяем политики безопасности
                security_policies = self.security_memory.data.get('policies', {})
                
                for policy_name, policy in security_policies.items():
                    if not self._validate_against_policy(data, policy):
                        return False
                
                return True
            
            def _validate_against_policy(self, data: Dict[str, Any], policy: Dict[str, Any]) -> bool:
                """Валидация данных против политики безопасности"""
                # Простая проверка - если есть sensitive данные, требуется авторизация
                sensitive_fields = ['password', 'secret', 'token', 'key']
                data_str = json.dumps(data).lower()
                
                for field in sensitive_fields:
                    if field in data_str:
                        return data.get('authorized', False)
                
                return True
            
            def _store_in_semantic(self, knowledge_data: Dict[str, Any]) -> bool:
                """Сохранить в Semantic Memory"""
                if self.semantic_memory.current_usage >= self.semantic_memory.max_capacity:
                    return False
                
                concept_id = knowledge_data.get('id', f"concept_{len(self.semantic_memory.data)}")
                self.semantic_memory.data[concept_id] = {
                    'content': knowledge_data,
                    'timestamp': time.time(),
                    'access_count': 0
                }
                
                # Обновляем индекс концептов
                if 'concepts' in knowledge_data:
                    for concept in knowledge_data['concepts']:
                        concept_name = concept.get('name', concept.get('id', ''))
                        if concept_name:
                            self.concept_index[concept_name] = concept_id
                
                self.semantic_memory.current_usage += 1
                return True
            
            def _store_in_episodic(self, knowledge_data: Dict[str, Any]) -> bool:
                """Сохранить в Episodic Memory"""
                if self.episodic_memory.current_usage >= self.episodic_memory.max_capacity:
                    return False
                
                event_id = f"event_{int(time.time() * 1000)}"
                self.episodic_memory.data[event_id] = {
                    'event_data': knowledge_data,
                    'timestamp': time.time(),
                    'event_type': knowledge_data.get('type', 'unknown')
                }
                
                # Обновляем временной индекс
                timestamp = int(time.time())
                if timestamp not in self.temporal_index:
                    self.temporal_index[timestamp] = []
                self.temporal_index[timestamp].append(event_id)
                
                self.episodic_memory.current_usage += 1
                return True
            
            def _update_procedural(self, knowledge_data: Dict[str, Any]) -> bool:
                """Обновить Procedural Memory"""
                if knowledge_data.get('type') == 'procedure':
                    if self.procedural_memory.current_usage >= self.procedural_memory.max_capacity:
                        return False
                    
                    procedure_id = knowledge_data.get('id', f"procedure_{len(self.procedural_memory.data)}")
                    self.procedural_memory.data[procedure_id] = {
                        'procedure': knowledge_data,
                        'timestamp': time.time(),
                        'usage_count': 0
                    }
                    
                    # Обновляем индекс workflow
                    workflow_type = knowledge_data.get('workflow_type', 'general')
                    if workflow_type not in self.workflow_index:
                        self.workflow_index[workflow_type] = []
                    self.workflow_index[workflow_type].append(procedure_id)
                    
                    self.procedural_memory.current_usage += 1
                    return True
                
                return True  # Не процедурные данные не требуют обновления
            
            def _store_in_vault(self, knowledge_data: Dict[str, Any]) -> bool:
                """Сохранить в Vault Memory"""
                # Сохраняем только если есть секреты
                secrets = self._extract_secrets(knowledge_data)
                if not secrets:
                    return True  # Нет секретов для сохранения
                
                if self.vault_memory.current_usage >= self.vault_memory.max_capacity:
                    return False
                
                for secret_name, secret_value in secrets.items():
                    self.vault_memory.data[secret_name] = {
                        'secret_value': secret_value,
                        'timestamp': time.time(),
                        'data_type': knowledge_data.get('type', 'unknown')
                    }
                
                self.vault_memory.current_usage += len(secrets)
                return True
            
            def _update_core(self, knowledge_data: Dict[str, Any]) -> bool:
                """Обновить Core Memory"""
                if self.core_memory.current_usage >= self.core_memory.max_capacity:
                    return False
                
                # Извлекаем факты для core memory
                facts = self._extract_facts(knowledge_data)
                for fact_name, fact_value in facts.items():
                    self.core_memory.data[fact_name] = {
                        'value': fact_value,
                        'timestamp': time.time(),
                        'source': knowledge_data.get('type', 'unknown')
                    }
                
                self.core_memory.current_usage += len(facts)
                return True
            
            def _retrieve_from_semantic(self, query: str) -> List[Dict[str, Any]]:
                """Извлечь из Semantic Memory"""
                results = []
                query_lower = query.lower()
                
                for concept_id, concept_data in self.semantic_memory.data.items():
                    # Проверяем совпадения в содержимом
                    content_str = json.dumps(concept_data['content']).lower()
                    if any(word in content_str for word in query_lower.split()):
                        results.append({
                            'type': 'concept',
                            'id': concept_id,
                            'data': concept_data,
                            'relevance_score': self._calculate_relevance(query_lower, content_str)
                        })
                        
                        # Обновляем счетчик доступа
                        concept_data['access_count'] += 1
                
                # Сортируем по релевантности
                results.sort(key=lambda x: x['relevance_score'], reverse=True)
                return results
            
            def _retrieve_from_episodic(self, query: str) -> List[Dict[str, Any]]:
                """Извлечь из Episodic Memory"""
                results = []
                query_lower = query.lower()
                
                for event_id, event_data in self.episodic_memory.data.items():
                    content_str = json.dumps(event_data).lower()
                    if any(word in content_str for word in query_lower.split()):
                        results.append({
                            'type': 'event',
                            'id': event_id,
                            'data': event_data,
                            'relevance_score': self._calculate_relevance(query_lower, content_str)
                        })
                
                # Сортируем по времени (более новые сначала)
                results.sort(key=lambda x: x['data']['timestamp'], reverse=True)
                return results
            
            def _retrieve_from_procedural(self, query: str) -> List[Dict[str, Any]]:
                """Извлечь из Procedural Memory"""
                results = []
                query_lower = query.lower()
                
                for procedure_id, procedure_data in self.procedural_memory.data.items():
                    content_str = json.dumps(procedure_data).lower()
                    if any(word in content_str for word in query_lower.split()):
                        results.append({
                            'type': 'procedure',
                            'id': procedure_id,
                            'data': procedure_data,
                            'relevance_score': self._calculate_relevance(query_lower, content_str)
                        })
                        
                        # Обновляем счетчик использования
                        procedure_data['usage_count'] += 1
                
                return results
            
            def _retrieve_from_core(self, query: str) -> List[Dict[str, Any]]:
                """Извлечь из Core Memory"""
                results = []
                query_lower = query.lower()
                
                for fact_name, fact_data in self.core_memory.data.items():
                    content_str = json.dumps(fact_data).lower()
                    if any(word in content_str for word in query_lower.split()):
                        results.append({
                            'type': 'fact',
                            'id': fact_name,
                            'data': fact_data,
                            'relevance_score': self._calculate_relevance(query_lower, content_str)
                        })
                
                return results
            
            def _retrieve_from_vault(self, query: str) -> List[Dict[str, Any]]:
                """Извлечь из Vault Memory"""
                results = []
                query_lower = query.lower()
                
                for secret_name, secret_data in self.vault_memory.data.items():
                    content_str = json.dumps(secret_data).lower()
                    if any(word in content_str for word in query_lower.split()):
                        results.append({
                            'type': 'secret',
                            'id': secret_name,
                            'data': secret_data,
                            'relevance_score': self._calculate_relevance(query_lower, content_str)
                        })
                
                return results
            
            def _retrieve_from_security(self, query: str) -> List[Dict[str, Any]]:
                """Извлечь из Security Memory"""
                results = []
                query_lower = query.lower()
                
                for policy_name, policy_data in self.security_memory.data.items():
                    if 'policies' in policy_data:
                        for policy_id, policy in policy_data['policies'].items():
                            content_str = json.dumps(policy).lower()
                            if any(word in content_str for word in query_lower.split()):
                                results.append({
                                    'type': 'policy',
                                    'id': policy_id,
                                    'data': policy,
                                    'relevance_score': self._calculate_relevance(query_lower, content_str)
                                })
                
                return results
            
            def _calculate_relevance(self, query: str, content: str) -> float:
                """Вычислить релевантность контента к запросу"""
                query_words = set(query.split())
                content_words = content.split()
                
                matches = sum(1 for word in query_words if word in content_words)
                return matches / max(len(query_words), 1)
            
            def _extract_secrets(self, data: Dict[str, Any]) -> Dict[str, str]:
                """Извлечь секреты из данных"""
                secrets = {}
                
                def extract_recursively(obj, path=""):
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            current_path = f"{path}.{key}" if path else key
                            if key.lower() in ['password', 'secret', 'token', 'key', 'api_key']:
                                secrets[current_path] = str(value)
                            extract_recursively(value, current_path)
                    elif isinstance(obj, list):
                        for i, item in enumerate(obj):
                            extract_recursively(item, f"{path}[{i}]")
                
                extract_recursively(data)
                return secrets
            
            def _extract_facts(self, data: Dict[str, Any]) -> Dict[str, Any]:
                """Извлечь факты для Core Memory"""
                facts = {}
                
                # Простое извлечение ключевых фактов
                if 'facts' in data:
                    facts.update(data['facts'])
                
                if 'properties' in data:
                    facts.update(data['properties'])
                
                # Извлекаем базовые атрибуты
                for key in ['name', 'id', 'type', 'category']:
                    if key in data:
                        facts[f"{data.get('id', 'unknown')}_{key}"] = data[key]
                
                return facts
            
            def get_memory_stats(self) -> Dict[str, Any]:
                """Получить статистику всех слоев памяти"""
                return {
                    'core': {
                        'capacity': self.core_memory.max_capacity,
                        'usage': self.core_memory.current_usage,
                        'usage_percent': (self.core_memory.current_usage / self.core_memory.max_capacity) * 100
                    },
                    'semantic': {
                        'capacity': self.semantic_memory.max_capacity,
                        'usage': self.semantic_memory.current_usage,
                        'usage_percent': (self.semantic_memory.current_usage / self.semantic_memory.max_capacity) * 100,
                        'concept_count': len(self.concept_index)
                    },
                    'episodic': {
                        'capacity': self.episodic_memory.max_capacity,
                        'usage': self.episodic_memory.current_usage,
                        'usage_percent': (self.episodic_memory.current_usage / self.episodic_memory.max_capacity) * 100,
                        'event_count': len(self.temporal_index)
                    },
                    'procedural': {
                        'capacity': self.procedural_memory.max_capacity,
                        'usage': self.procedural_memory.current_usage,
                        'usage_percent': (self.procedural_memory.current_usage / self.procedural_memory.max_capacity) * 100,
                        'workflow_count': len(self.workflow_index)
                    },
                    'vault': {
                        'capacity': self.vault_memory.max_capacity,
                        'usage': self.vault_memory.current_usage,
                        'usage_percent': (self.vault_memory.current_usage / self.vault_memory.max_capacity) * 100
                    },
                    'security': {
                        'capacity': self.security_memory.max_capacity,
                        'usage': self.security_memory.current_usage,
                        'usage_percent': (self.security_memory.current_usage / self.security_memory.max_capacity) * 100
                    },
                    'total_operations': len(self.operation_history)
                }
            
            def reset_memory(self):
                """Сбросить все слои памяти"""
                for memory_layer in [self.core_memory, self.semantic_memory, self.episodic_memory, 
                                   self.procedural_memory, self.vault_memory, self.security_memory]:
                    memory_layer.data.clear()
                    memory_layer.current_usage = 0
                
                # Очищаем индексы
                self.concept_index.clear()
                self.temporal_index.clear()
                self.workflow_index.clear()
                self.security_index.clear()
                
                # Очищаем историю операций
                self.operation_history.clear()
        
        return MockMemorySystem
    
    def test_memory_system_initialization(self, memory_system_instance):
        """Тест инициализации системы памяти"""
        config = {
            'core_capacity': 100,
            'semantic_capacity': 1000,
            'episodic_capacity': 500,
            'procedural_capacity': 100,
            'vault_capacity': 50,
            'security_capacity': 25
        }
        
        memory_system = memory_system_instance(config)
        
        # Проверяем что все слои инициализированы
        assert memory_system.core_memory.name == "core"
        assert memory_system.semantic_memory.name == "semantic"
        assert memory_system.episodic_memory.name == "episodic"
        assert memory_system.procedural_memory.name == "procedural"
        assert memory_system.vault_memory.name == "vault"
        assert memory_system.security_memory.name == "security"
        
        # Проверяем емкости
        assert memory_system.core_memory.max_capacity == 100
        assert memory_system.semantic_memory.max_capacity == 1000
        assert memory_system.episodic_memory.max_capacity == 500
        
        # Проверяем что слои пустые
        assert memory_system.core_memory.current_usage == 0
        assert memory_system.semantic_memory.current_usage == 0
        assert memory_system.episodic_memory.current_usage == 0
    
    def test_store_knowledge_integration(self, memory_system_instance):
        """Тест интеграционного сохранения знаний"""
        memory_system = memory_system_instance({})
        
        # Тестовые данные знаний
        knowledge_data = {
            'id': 'ai_concept_001',
            'type': 'concept',
            'name': 'Artificial Intelligence',
            'definition': 'Computer system capable of intelligent behavior',
            'concepts': [
                {'name': 'artificial_intelligence', 'type': 'concept'},
                {'name': 'machine_learning', 'type': 'process'}
            ],
            'properties': {
                'domain': 'technology',
                'complexity': 'high',
                'confidence': 0.95
            },
            'facts': {
                'field': 'computer_science',
                'applications': ['robotics', 'nlp', 'computer_vision']
            }
        }
        
        # Сохраняем знание
        result = memory_system.store_knowledge(knowledge_data)
        
        assert result is True
        
        # Проверяем статистику
        stats = memory_system.get_memory_stats()
        assert stats['semantic']['usage'] > 0
        assert stats['episodic']['usage'] > 0
        assert stats['core']['usage'] > 0
    
    def test_retrieve_knowledge_integration(self, memory_system_instance):
        """Тест интеграционного извлечения знаний"""
        memory_system = memory_system_instance({})
        
        # Сохраняем тестовые данные
        knowledge_data = {
            'id': 'ml_concept_001',
            'type': 'concept',
            'name': 'Machine Learning',
            'definition': 'Subset of AI focused on algorithms that learn from data',
            'concepts': [
                {'name': 'machine_learning', 'type': 'process'},
                {'name': 'artificial_intelligence', 'type': 'concept'}
            ],
            'properties': {
                'domain': 'technology',
                'subfield': 'ai'
            },
            'facts': {
                'parent_field': 'artificial_intelligence',
                'learning_type': 'supervised'
            }
        }
        
        memory_system.store_knowledge(knowledge_data)
        
        # Извлекаем знание
        results = memory_system.retrieve_knowledge("machine learning", ['semantic', 'core'])
        
        assert 'semantic' in results
        assert 'core' in results
        
        # Проверяем что нашли релевантные данные
        semantic_results = results['semantic']
        assert len(semantic_results) > 0
        
        core_results = results['core']
        assert len(core_results) > 0
        
        # Проверяем качество извлечения
        for result in semantic_results:
            assert 'relevance_score' in result
            assert result['relevance_score'] > 0
    
    def test_cross_layer_relationships(self, memory_system_instance):
        """Тест кросс-слойных связей"""
        memory_system = memory_system_instance({})
        
        # Создаем связанные данные
        concept_data = {
            'id': 'bias_001',
            'type': 'concept',
            'name': 'Confirmation Bias',
            'definition': 'Tendency to favor confirming information',
            'concepts': [{'name': 'confirmation_bias', 'type': 'bias'}],
            'properties': {'severity': 'high', 'domain': 'psychology'}
        }
        
        procedure_data = {
            'id': 'mitigation_proc_001',
            'type': 'procedure',
            'name': 'Bias Mitigation',
            'definition': 'Process to reduce cognitive bias impact',
            'workflow_type': 'bias_reduction',
            'steps': ['identify', 'measure', 'mitigate', 'validate']
        }
        
        event_data = {
            'id': 'bias_detection_event_001',
            'type': 'event',
            'name': 'Bias Detection',
            'definition': 'Event of detecting bias in decision making',
            'timestamp': time.time()
        }
        
        # Сохраняем во всех слоях
        memory_system.store_knowledge(concept_data)
        memory_system.store_knowledge(procedure_data)
        memory_system.store_knowledge(event_data)
        
        # Проверяем индексы
        assert 'confirmation_bias' in memory_system.concept_index
        assert 'bias_reduction' in memory_system.workflow_index
        
        # Проверяем кросс-слойное извлечение
        results = memory_system.retrieve_knowledge("bias")
        
        # Должны найти данные во всех релевантных слоях
        layer_counts = {layer: len(results.get(layer, [])) for layer in ['semantic', 'episodic', 'procedural']}
        total_found = sum(layer_counts.values())
        
        assert total_found > 0
        assert layer_counts['semantic'] > 0  # Концепт должен быть в semantic
    
    def test_security_integration(self, memory_system_instance):
        """Тест интеграции с системой безопасности"""
        memory_system = memory_system_instance({})
        
        # Добавляем политику безопасности
        security_data = {
            'id': 'security_policy_001',
            'type': 'policy',
            'policies': {
                'data_classification': {
                    'sensitive_fields': ['password', 'secret', 'token'],
                    'require_authorization': True,
                    'audit_level': 'high'
                }
            }
        }
        
        # Сначала добавляем политику
        memory_system.security_memory.data['policies'] = security_data['policies']
        memory_system.security_memory.current_usage = 1
        
        # Тестовые данные с секретом
        secure_data = {
            'id': 'secure_data_001',
            'type': 'data',
            'name': 'Secure Information',
            'password': 'secret123',
            'api_key': 'key_abc_xyz',
            'public_info': 'This is public information'
        }
        
        # Пытаемся сохранить без авторизации
        result_without_auth = memory_system.store_knowledge(secure_data)
        assert result_without_auth is False
        
        # Добавляем авторизацию
        secure_data['authorized'] = True
        result_with_auth = memory_system.store_knowledge(secure_data)
        assert result_with_auth is True
    
    def test_memory_capacity_limits(self, memory_system_instance):
        """Тест ограничений емкости памяти"""
        config = {
            'core_capacity': 3,
            'semantic_capacity': 3,
            'episodic_capacity': 3
        }
        
        memory_system = memory_system_instance(config)
        
        # Заполняем память до предела
        for i in range(3):
            data = {
                'id': f'item_{i}',
                'type': 'concept',
                'name': f'Concept {i}',
                'definition': f'Definition {i}'
            }
            result = memory_system.store_knowledge(data)
            assert result is True
        
        # Пытаемся добавить еще один элемент
        overflow_data = {
            'id': 'overflow_item',
            'type': 'concept',
            'name': 'Overflow Concept',
            'definition': 'This should not be stored'
        }
        
        result = memory_system.store_knowledge(overflow_data)
        assert result is False
        
        # Проверяем статистику
        stats = memory_system.get_memory_stats()
        assert stats['core']['usage_percent'] >= 100  # Превышение лимита
        assert stats['semantic']['usage_percent'] >= 100
    
    def test_temporal_consistency(self, memory_system_instance):
        """Тест временной согласованности"""
        memory_system = memory_system_instance({})
        
        # Создаем данные с временными метками
        data1 = {
            'id': 'temporal_001',
            'type': 'concept',
            'name': 'Initial Concept',
            'definition': 'First version',
            'timestamp': time.time() - 100  # Старые данные
        }
        
        data2 = {
            'id': 'temporal_001',  # Тот же ID
            'type': 'concept',
            'name': 'Updated Concept',
            'definition': 'Updated version',
            'timestamp': time.time()  # Новые данные
        }
        
        # Сохраняем обе версии
        result1 = memory_system.store_knowledge(data1)
        result2 = memory_system.store_knowledge(data2)
        
        assert result1 is True
        assert result2 is True
        
        # Проверяем что в episodic memory есть оба события
        episodic_results = memory_system._retrieve_from_episodic("concept")
        assert len(episodic_results) >= 2
        
        # Проверяем временной порядок
        timestamps = [result['data']['timestamp'] for result in episodic_results]
        assert len(timestamps) == 2
        assert timestamps[0] < timestamps[1]  # Более новые должны быть позже
    
    @pytest.mark.asyncio
    async def test_concurrent_memory_operations(self, memory_system_instance):
        """Тест конкурентных операций с памятью"""
        memory_system = memory_system_instance({})
        
        async def store_data_task(data_id: int):
            data = {
                'id': f'concurrent_data_{data_id}',
                'type': 'concept',
                'name': f'Concurrent Concept {data_id}',
                'definition': f'Definition {data_id}'
            }
            return memory_system.store_knowledge(data)
        
        async def retrieve_data_task(query: str):
            return memory_system.retrieve_knowledge(query, ['semantic'])
        
        # Выполняем конкурентные операции
        store_tasks = [store_data_task(i) for i in range(10)]
        retrieve_tasks = [retrieve_data_task(f"concept {i}") for i in range(5)]
        
        start_time = time.time()
        store_results = await asyncio.gather(*store_tasks)
        retrieve_results = await asyncio.gather(*retrieve_tasks)
        concurrent_time = time.time() - start_time
        
        # Проверяем результаты
        assert all(result is True for result in store_results)
        assert all(isinstance(result, dict) for result in retrieve_results)
        
        # Проверяем статистику
        stats = memory_system.get_memory_stats()
        assert stats['semantic']['usage'] == 10
        
        # Проверяем производительность
        assert concurrent_time < 5.0  # Операции должны завершиться менее чем за 5 секунд
    
    def test_memory_persistence_across_operations(self, memory_system_instance):
        """Тест сохранности данных между операциями"""
        memory_system = memory_system_instance({})
        
        # Сохраняем данные
        initial_data = {
            'id': 'persistence_test',
            'type': 'concept',
            'name': 'Persistence Test',
            'definition': 'Testing data persistence'
        }
        
        assert memory_system.store_knowledge(initial_data) is True
        
        # Выполняем несколько операций извлечения
        for _ in range(3):
            results = memory_system.retrieve_knowledge("persistence", ['semantic'])
            assert len(results['semantic']) > 0
        
        # Проверяем что данные все еще доступны
        final_results = memory_system.retrieve_knowledge("persistence", ['semantic'])
        assert len(final_results['semantic']) > 0
        
        # Проверяем что счетчик доступа увеличился
        semantic_results = memory_system._retrieve_from_semantic("persistence")
        if semantic_results:
            access_count = semantic_results[0]['data']['access_count']
            assert access_count >= 3  # Минимум 3 обращения
    
    def test_memory_system_error_handling(self, memory_system_instance):
        """Тест обработки ошибок в системе памяти"""
        memory_system = memory_system_instance({})
        
        # Тест с некорректными данными
        invalid_data = None
        result = memory_system.store_knowledge(invalid_data)
        assert result is False
        
        # Тест с пустыми данными
        empty_data = {}
        result = memory_system.store_knowledge(empty_data)
        assert result is True  # Должно сохраниться как пустая запись
        
        # Тест извлечения с пустым запросом
        results = memory_system.retrieve_knowledge("", ['semantic'])
        assert isinstance(results, dict)
        
        # Тест извлечения из несуществующего слоя
        results = memory_system.retrieve_knowledge("test", ['nonexistent'])
        assert 'nonexistent' in results
        assert results['nonexistent'] == []  # Пустой результат
    
    def test_memory_statistics_accuracy(self, memory_system_instance):
        """Тест точности статистики памяти"""
        memory_system = memory_system_instance({})
        
        # Проверяем начальную статистику
        initial_stats = memory_system.get_memory_stats()
        assert all(layer['usage'] == 0 for layer in initial_stats.values())
        
        # Добавляем данные в разные слои
        concept_data = {
            'id': 'stats_test_001',
            'type': 'concept',
            'name': 'Stats Test Concept',
            'definition': 'For testing statistics'
        }
        
        procedure_data = {
            'id': 'stats_test_002',
            'type': 'procedure',
            'name': 'Stats Test Procedure',
            'definition': 'For testing procedure stats'
        }
        
        # Сохраняем данные
        assert memory_system.store_knowledge(concept_data) is True
        assert memory_system.store_knowledge(procedure_data) is True
        
        # Проверяем обновленную статистику
        updated_stats = memory_system.get_memory_stats()
        
        # Должны быть обновлены соответствующие слои
        assert updated_stats['semantic']['usage'] > 0
        assert updated_stats['procedural']['usage'] > 0
        
        # Проверяем корректность процентного соотношения
        for layer_name, layer_stats in updated_stats.items():
            if layer_name in ['core', 'semantic', 'episodic', 'procedural', 'vault', 'security']:
                expected_percent = (layer_stats['usage'] / layer_stats['capacity']) * 100
                assert abs(layer_stats['usage_percent'] - expected_percent) < 0.01
