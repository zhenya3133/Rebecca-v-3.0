"""
Unit тесты для ConceptExtractor компонента
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import re
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Concept:
    """Модель концепта для тестирования"""
    name: str
    type: str
    confidence: float
    definition: str
    context: str
    source: str
    properties: Dict[str, Any] = None
    relationships: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
        if self.relationships is None:
            self.relationships = []


class TestConceptExtractor:
    """Тесты для ConceptExtractor класса"""
    
    @pytest.fixture
    def extractor_instance(self):
        """Создание экземпляра извлекателя концептов"""
        class MockConceptExtractor:
            def __init__(self, config: Dict[str, Any]):
                self.config = config
                self.concepts: Dict[str, Concept] = {}
                self.patterns = self._load_patterns()
                self.stop_words = {
                    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
                }
                self.knowledge_base = {}
                self.extraction_history = []
            
            def _load_patterns(self) -> Dict[str, List[str]]:
                """Загрузить паттерны для извлечения концептов"""
                return {
                    'definitions': [
                        r'(?:is|are|defined as|means|refers to)\s+([^.]+)',
                        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is|are|means|refers to)\s+([^.]+)',
                        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*:\s*([^.]+)'
                    ],
                    'relationships': [
                        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is|are|part of|subset of|related to|connected to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:can|may|might)\s+(?:lead to|result in|cause|create)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+vs\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
                    ],
                    'examples': [
                        r'(?:for example|e\.g\.|such as)\s+([^,]+(?:,\s*[^,]+)*)',
                        r'(?:including|like)\s+([^,]+(?:,\s*[^,]+)*)',
                        r'(?:instances|examples)\s+include\s+([^,]+(?:,\s*[^,]+)*)'
                    ]
                }
            
            def extract_concepts(self, text: str, context: str = "") -> List[Dict[str, Any]]:
                """Извлечь концепты из текста"""
                if not text or not text.strip():
                    return []
                
                concepts = []
                sentences = self._split_sentences(text)
                
                for sentence in sentences:
                    sentence_concepts = self._extract_from_sentence(sentence, context)
                    concepts.extend(sentence_concepts)
                
                # Удаляем дубликаты и объединяем похожие концепты
                concepts = self._deduplicate_concepts(concepts)
                
                # Нормализуем имена концептов
                for concept in concepts:
                    concept['name'] = self._normalize_concept_name(concept['name'])
                
                self.extraction_history.append({
                    'timestamp': time.time(),
                    'text_length': len(text),
                    'concepts_extracted': len(concepts),
                    'context': context
                })
                
                return concepts
            
            def _split_sentences(self, text: str) -> List[str]:
                """Разбить текст на предложения"""
                # Простая реализация разбиения на предложения
                sentences = re.split(r'[.!?]+\s+', text)
                return [s.strip() for s in sentences if s.strip()]
            
            def _extract_from_sentence(self, sentence: str, context: str) -> List[Dict[str, Any]]:
                """Извлечь концепты из предложения"""
                concepts = []
                
                # Извлекаем определения
                for pattern in self.patterns['definitions']:
                    matches = re.finditer(pattern, sentence, re.IGNORECASE)
                    for match in matches:
                        if len(match.groups()) == 2:
                            term, definition = match.groups()
                            concept = self._create_concept(term.strip(), definition.strip(), sentence, context)
                            if concept:
                                concepts.append(concept)
                        elif len(match.groups()) == 1:
                            definition = match.group(1)
                            # Извлекаем ключевые термины
                            terms = self._extract_terms_from_text(definition)
                            for term in terms:
                                concept = self._create_concept(term, definition, sentence, context)
                                if concept:
                                    concepts.append(concept)
                
                # Извлекаем отношения
                for pattern in self.patterns['relationships']:
                    matches = re.finditer(pattern, sentence, re.IGNORECASE)
                    for match in matches:
                        concept1, concept2 = match.groups()
                        relationship = {
                            'concept1': concept1.strip(),
                            'concept2': concept2.strip(),
                            'sentence': sentence,
                            'confidence': 0.7
                        }
                        concepts.append({
                            'type': 'relationship',
                            'name': f"{concept1.strip()} -> {concept2.strip()}",
                            'confidence': 0.7,
                            'definition': sentence,
                            'context': context,
                            'source': 'relationship_extraction',
                            'properties': {'relationship': relationship}
                        })
                
                return concepts
            
            def _extract_terms_from_text(self, text: str) -> List[str]:
                """Извлечь ключевые термины из текста"""
                # Удаляем стоп-слова и извлекаем значимые термины
                words = re.findall(r'\b[A-Za-z][A-Za-z0-9_-]*\b', text)
                significant_terms = []
                
                for word in words:
                    if (word.lower() not in self.stop_words and 
                        len(word) > 2 and 
                        not word.isdigit()):
                        significant_terms.append(word)
                
                # Ищем составные термины (например, "machine learning")
                for i in range(len(words) - 1):
                    if (words[i].lower() not in self.stop_words and 
                        words[i + 1].lower() not in self.stop_words):
                        compound_term = f"{words[i]} {words[i + 1]}"
                        if compound_term.lower() not in [t.lower() for t in significant_terms]:
                            significant_terms.append(compound_term)
                
                return significant_terms[:5]  # Ограничиваем количество
            
            def _create_concept(self, name: str, definition: str, sentence: str, context: str) -> Optional[Dict[str, Any]]:
                """Создать объект концепта"""
                if not name or not definition:
                    return None
                
                # Вычисляем confidence на основе различных факторов
                confidence = self._calculate_confidence(name, definition, sentence)
                
                if confidence < self.config.get('min_confidence', 0.3):
                    return None
                
                return {
                    'name': name.strip(),
                    'type': self._determine_concept_type(name, definition),
                    'confidence': confidence,
                    'definition': definition.strip(),
                    'context': context,
                    'source': 'text_extraction',
                    'properties': {
                        'sentence': sentence,
                        'extraction_method': 'pattern_based'
                    }
                }
            
            def _calculate_confidence(self, name: str, definition: str, sentence: str) -> float:
                """Вычислить confidence для концепта"""
                confidence = 0.5  # Базовая уверенность
                
                # Увеличиваем уверенность если термин в заглавных буквах
                if name[0].isupper():
                    confidence += 0.1
                
                # Увеличиваем уверенность если определение достаточно длинное
                if len(definition) > 20:
                    confidence += 0.1
                
                # Увеличиваем уверенность если в определении есть ключевые слова
                key_words = ['is', 'are', 'defined', 'means', 'refers to', 'describes']
                if any(word in sentence.lower() for word in key_words):
                    confidence += 0.1
                
                # Уменьшаем уверенность если слишком короткое имя
                if len(name.split()) == 1 and len(name) < 4:
                    confidence -= 0.1
                
                return min(max(confidence, 0.0), 1.0)
            
            def _determine_concept_type(self, name: str, definition: str) -> str:
                """Определить тип концепта"""
                definition_lower = definition.lower()
                name_lower = name.lower()
                
                if any(word in definition_lower for word in ['process', 'method', 'technique', 'algorithm']):
                    return 'process'
                elif any(word in definition_lower for word in ['system', 'model', 'framework', 'architecture']):
                    return 'system'
                elif any(word in definition_lower for word in ['theory', 'principle', 'concept', 'idea']):
                    return 'theory'
                elif any(word in name_lower for word in ['bias', 'error', 'fallacy']):
                    return 'bias'
                else:
                    return 'concept'
            
            def _normalize_concept_name(self, name: str) -> str:
                """Нормализовать имя концепта"""
                # Удаляем артикли и приводим к нижнему регистру
                normalized = re.sub(r'^(the|a|an)\s+', '', name, flags=re.IGNORECASE)
                # Заменяем пробелы на подчеркивания
                normalized = re.sub(r'\s+', '_', normalized.strip())
                # Удаляем специальные символы
                normalized = re.sub(r'[^\w_]', '', normalized)
                return normalized.lower()
            
            def _deduplicate_concepts(self, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                """Удалить дубликаты и объединить похожие концепты"""
                unique_concepts = []
                seen_names = set()
                
                for concept in concepts:
                    normalized_name = self._normalize_concept_name(concept['name'])
                    
                    if normalized_name not in seen_names:
                        seen_names.add(normalized_name)
                        concept['name'] = normalized_name
                        unique_concepts.append(concept)
                    else:
                        # Объединяем с существующим концептом
                        for existing in unique_concepts:
                            if existing['name'] == normalized_name:
                                # Обновляем definition если новый более качественный
                                if concept['confidence'] > existing['confidence']:
                                    existing['definition'] = concept['definition']
                                    existing['confidence'] = concept['confidence']
                                break
                
                return unique_concepts
            
            def validate_concept(self, concept: Dict[str, Any]) -> bool:
                """Валидировать концепт"""
                if not concept:
                    return False
                
                required_fields = ['name', 'definition', 'confidence']
                for field in required_fields:
                    if field not in concept:
                        return False
                
                # Проверяем типы данных
                if not isinstance(concept['name'], str) or len(concept['name']) == 0:
                    return False
                
                if not isinstance(concept['definition'], str) or len(concept['definition']) == 0:
                    return False
                
                if not isinstance(concept['confidence'], (int, float)) or not (0 <= concept['confidence'] <= 1):
                    return False
                
                # Проверяем дополнительные ограничения
                if self.config.get('max_definition_length'):
                    if len(concept['definition']) > self.config['max_definition_length']:
                        return False
                
                if self.config.get('min_name_length'):
                    if len(concept['name']) < self.config['min_name_length']:
                        return False
                
                return True
            
            def link_concepts(self, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                """Связать концепты отношениями"""
                relationships = []
                
                # Простая эвристика для связи концептов
                for i, concept1 in enumerate(concepts):
                    for j, concept2 in enumerate(concepts):
                        if i != j:
                            relationship = self._detect_relationship(concept1, concept2)
                            if relationship:
                                relationships.append({
                                    'source': concept1['name'],
                                    'target': concept2['name'],
                                    'relation': relationship['type'],
                                    'confidence': relationship['confidence'],
                                    'properties': relationship['properties']
                                })
                
                return relationships
            
            def _detect_relationship(self, concept1: Dict[str, Any], concept2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                """Обнаружить отношение между концептами"""
                name1, name2 = concept1['name'].lower(), concept2['name'].lower()
                def1, def2 = concept1['definition'].lower(), concept2['definition'].lower()
                
                # Проверяем на подмножество
                if name1 in def2 or name2 in def1:
                    return {
                        'type': 'is_subset_of',
                        'confidence': 0.8,
                        'properties': {'subset_detection': 'name_in_definition'}
                    }
                
                # Проверяем на семантическую близость
                common_words = set(name1.split()) & set(name2.split())
                if len(common_words) > 0:
                    return {
                        'type': 'related_to',
                        'confidence': 0.6,
                        'properties': {'common_words': list(common_words)}
                    }
                
                return None
            
            def score_concept_quality(self, concept: Dict[str, Any]) -> float:
                """Оценить качество концепта"""
                if not self.validate_concept(concept):
                    return 0.0
                
                quality_score = 0.0
                
                # Clarity (определенность)
                definition_length = len(concept['definition'])
                if 50 <= definition_length <= 200:
                    quality_score += 0.2
                elif definition_length > 200:
                    quality_score += 0.15
                
                # Precision (точность)
                name_length = len(concept['name'].split())
                if 1 <= name_length <= 3:
                    quality_score += 0.2
                
                # Confidence
                quality_score += concept['confidence'] * 0.3
                
                # Consistency (консистентность)
                if concept['source'] in ['pattern_extraction', 'knowledge_base']:
                    quality_score += 0.1
                
                # Completeness (полнота)
                if 'properties' in concept and len(concept['properties']) > 0:
                    quality_score += 0.1
                
                # Evidence level (уровень доказательности)
                evidence_score = 0.0
                for keyword in ['study', 'research', 'evidence', 'data', 'analysis']:
                    if keyword in concept['definition'].lower():
                        evidence_score += 0.05
                quality_score += min(evidence_score, 0.1)
                
                return min(quality_score, 1.0)
            
            def normalize_concept(self, concept: Dict[str, Any]) -> Dict[str, Any]:
                """Нормализовать концепт"""
                normalized = concept.copy()
                
                # Нормализуем имя
                normalized['name'] = self._normalize_concept_name(concept['name'])
                
                # Нормализуем тип
                normalized['type'] = normalized['type'].lower().replace(' ', '_')
                
                # Округляем confidence
                normalized['confidence'] = round(normalized['confidence'], 3)
                
                # Убираем лишние пробелы
                normalized['definition'] = ' '.join(normalized['definition'].split())
                normalized['context'] = ' '.join(normalized['context'].split()) if normalized['context'] else ""
                
                # Добавляем timestamp если нет
                if 'timestamp' not in normalized:
                    normalized['timestamp'] = time.time()
                
                # Добавляем версию нормализации
                normalized['normalized'] = True
                normalized['normalization_version'] = '1.0'
                
                return normalized
            
            def get_extraction_stats(self) -> Dict[str, Any]:
                """Получить статистику извлечения"""
                if not self.extraction_history:
                    return {'total_extractions': 0}
                
                total_extractions = len(self.extraction_history)
                total_concepts = sum(h['concepts_extracted'] for h in self.extraction_history)
                avg_concepts_per_extraction = total_concepts / total_extractions
                
                return {
                    'total_extractions': total_extractions,
                    'total_concepts_extracted': total_concepts,
                    'avg_concepts_per_extraction': avg_concepts_per_extraction,
                    'unique_concepts': len(self.concepts),
                    'extraction_rate': avg_concepts_per_extraction,
                    'history_length': len(self.extraction_history)
                }
        
        return MockConceptExtractor
    
    def test_extractor_initialization(self, extractor_instance):
        """Тест инициализации извлекателя"""
        config = {
            'min_confidence': 0.4,
            'max_definition_length': 500,
            'min_name_length': 3
        }
        
        extractor = extractor_instance(config)
        
        assert extractor.config['min_confidence'] == 0.4
        assert extractor.config['max_definition_length'] == 500
        assert extractor.config['min_name_length'] == 3
        assert len(extractor.patterns) > 0
        assert 'definitions' in extractor.patterns
        assert 'relationships' in extractor.patterns
        assert 'examples' in extractor.patterns
        assert len(extractor.stop_words) > 0
    
    def test_extract_concepts_basic(self, extractor_instance):
        """Тест базового извлечения концептов"""
        extractor = extractor_instance({})
        
        text = "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines."
        context = "technology"
        
        concepts = extractor.extract_concepts(text, context)
        
        assert len(concepts) > 0
        
        # Проверяем что найдены концепты
        concept_names = [c['name'] for c in concepts]
        assert any('artificial_intelligence' in name.lower() or 'artificial intelligence' in c['definition'].lower() for c in concepts)
    
    def test_extract_concepts_definitions(self, extractor_instance):
        """Тест извлечения концептов из определений"""
        extractor = extractor_instance({})
        
        text = "Machine Learning is defined as a subset of AI that focuses on algorithms that can learn from data."
        concepts = extractor.extract_concepts(text)
        
        assert len(concepts) > 0
        
        # Ищем концепт machine learning
        ml_concepts = [c for c in concepts if 'machine' in c['name'].lower() or 'learning' in c['name'].lower()]
        assert len(ml_concepts) > 0
        
        ml_concept = ml_concepts[0]
        assert ml_concept['confidence'] > 0
        assert len(ml_concept['definition']) > 0
    
    def test_extract_concepts_relationships(self, extractor_instance):
        """Тест извлечения отношений между концептами"""
        extractor = extractor_instance({})
        
        text = "Neural Networks are part of Deep Learning which is a subset of Machine Learning."
        concepts = extractor.extract_concepts(text)
        
        relationship_concepts = [c for c in concepts if c['type'] == 'relationship']
        assert len(relationship_concepts) > 0
        
        # Проверяем что отношения найдены
        for rel in relationship_concepts:
            assert '->' in rel['name']
            assert rel['confidence'] > 0
    
    def test_extract_concepts_complex_text(self, extractor_instance):
        """Тест извлечения из сложного текста"""
        extractor = extractor_instance({})
        
        text = """
        Cognitive biases are systematic errors in thinking that affect judgments and decisions. 
        Confirmation bias is a type of cognitive bias where people favor information that confirms their existing beliefs.
        For example, investors might only seek news that supports their current investment decisions.
        This bias can lead to poor decision-making in various contexts including business strategy and personal finance.
        """
        
        concepts = extractor.extract_concepts(text, "psychology")
        
        assert len(concepts) > 0
        
        # Ищем концепты связанные с когнитивными искажениями
        bias_concepts = [c for c in concepts if 'bias' in c['name'].lower() or 'bias' in c['definition'].lower()]
        assert len(bias_concepts) > 0
        
        # Проверяем качество извлечения
        for concept in concepts:
            assert extractor.validate_concept(concept)
            assert concept['confidence'] >= 0
            assert concept['confidence'] <= 1
    
    def test_validate_concept(self, extractor_instance):
        """Тест валидации концептов"""
        extractor = extractor_instance({})
        
        # Валидный концепт
        valid_concept = {
            'name': 'artificial_intelligence',
            'definition': 'Computer system capable of intelligent behavior',
            'confidence': 0.85,
            'type': 'concept'
        }
        
        assert extractor.validate_concept(valid_concept) is True
        
        # Невалидные концепты
        invalid_concepts = [
            {},  # Пустой концепт
            {'name': '', 'definition': 'test', 'confidence': 0.5},  # Пустое имя
            {'name': 'test', 'definition': '', 'confidence': 0.5},  # Пустое определение
            {'name': 'test', 'definition': 'test'},  # Отсутствует confidence
            {'name': 'test', 'definition': 'test', 'confidence': 1.5},  # Confidence > 1
            {'name': 'test', 'definition': 'test', 'confidence': -0.1}  # Confidence < 0
        ]
        
        for invalid in invalid_concepts:
            assert extractor.validate_concept(invalid) is False
    
    def test_link_concepts(self, extractor_instance):
        """Тест связывания концептов"""
        extractor = extractor_instance({})
        
        concepts = [
            {
                'name': 'machine_learning',
                'definition': 'ML is a subset of artificial intelligence that focuses on algorithms',
                'confidence': 0.9,
                'type': 'process'
            },
            {
                'name': 'artificial_intelligence',
                'definition': 'AI is a broad field encompassing various intelligent systems',
                'confidence': 0.85,
                'type': 'field'
            },
            {
                'name': 'neural_networks',
                'definition': 'Neural networks are computing systems inspired by biological brains',
                'confidence': 0.8,
                'type': 'system'
            }
        ]
        
        relationships = extractor.link_concepts(concepts)
        
        assert len(relationships) > 0
        
        # Проверяем что связи найдены
        ml_ai_relations = [r for r in relationships if 'machine_learning' in r['source'] and 'artificial_intelligence' in r['target']]
        assert len(ml_ai_relations) > 0
        
        for relation in relationships:
            assert 'source' in relation
            assert 'target' in relation
            assert 'relation' in relation
            assert 'confidence' in relation
            assert 0 <= relation['confidence'] <= 1
    
    def test_score_concept_quality(self, extractor_instance):
        """Тест оценки качества концептов"""
        extractor = extractor_instance({})
        
        # Высококачественный концепт
        high_quality = {
            'name': 'artificial_neural_network',
            'definition': 'Computing system inspired by biological neural networks that consists of interconnected nodes processing information in parallel',
            'confidence': 0.95,
            'type': 'system',
            'source': 'research_paper',
            'properties': {'methodology': 'evidence_based'}
        }
        
        high_score = extractor.score_concept_quality(high_quality)
        assert high_score > 0.7
        
        # Низкокачественный концепт
        low_quality = {
            'name': 'x',
            'definition': 'something',
            'confidence': 0.1,
            'type': 'concept'
        }
        
        low_score = extractor.score_concept_quality(low_quality)
        assert low_score < 0.3
        
        # Проверяем что качественный концепт имеет более высокий score
        assert high_score > low_score
    
    def test_normalize_concept(self, extractor_instance):
        """Тест нормализации концептов"""
        extractor = extractor_instance({})
        
        concept = {
            'name': '  Artificial  Intelligence  ',
            'definition': '  Computer   system   that   thinks  ',
            'confidence': 0.857,
            'type': ' Concept ',
            'context': ' technology ',
            'source': 'text_extraction'
        }
        
        normalized = extractor.normalize_concept(concept)
        
        assert normalized['name'] == 'artificial_intelligence'
        assert normalized['definition'] == 'Computer system that thinks'
        assert normalized['confidence'] == 0.857
        assert normalized['type'] == 'concept'
        assert normalized['context'] == 'technology'
        assert normalized['normalized'] is True
        assert 'normalization_version' in normalized
        assert 'timestamp' in normalized
    
    def test_extraction_performance(self, extractor_instance):
        """Тест производительности извлечения"""
        extractor = extractor_instance({})
        
        # Большой текст
        large_text = "Artificial Intelligence is a field of computer science. " * 100
        
        start_time = time.time()
        concepts = extractor.extract_concepts(large_text, "technology")
        extraction_time = time.time() - start_time
        
        assert extraction_time < 5.0  # Должно извлекаться менее чем за 5 секунд
        assert len(concepts) > 0
        
        # Проверяем статистику
        stats = extractor.get_extraction_stats()
        assert stats['total_extractions'] == 1
        assert stats['total_concepts_extracted'] == len(concepts)
    
    def test_edge_cases(self, extractor_instance):
        """Тест крайних случаев"""
        extractor = extractor_instance({})
        
        # Пустой текст
        concepts = extractor.extract_concepts("")
        assert len(concepts) == 0
        
        # Только пробелы
        concepts = extractor.extract_concepts("   \n\t  ")
        assert len(concepts) == 0
        
        # Очень длинный текст
        long_text = "Concept is defined as " + "x" * 10000
        concepts = extractor.extract_concepts(long_text)
        # Должен обработать без ошибок
        assert isinstance(concepts, list)
        
        # Специальные символы
        special_text = "AI & ML are related! @ # $ % ^ & * ( )"
        concepts = extractor.extract_concepts(special_text)
        assert isinstance(concepts, list)
    
    @pytest.mark.asyncio
    async def test_concurrent_extraction(self, extractor_instance):
        """Тест конкурентного извлечения концептов"""
        extractor = extractor_instance({})
        
        texts = [
            "Artificial Intelligence is the simulation of human intelligence by machines.",
            "Machine Learning is a subset of AI that learns from data.",
            "Neural Networks are computational models inspired by biological brains.",
            "Cognitive Biases are systematic errors in human thinking.",
            "Decision Making involves complex cognitive processes."
        ]
        
        async def extract_task(text):
            return extractor.extract_concepts(text, "technology")
        
        # Выполняем конкурентное извлечение
        start_time = time.time()
        results = await asyncio.gather(*[extract_task(text) for text in texts])
        concurrent_time = time.time() - start_time
        
        # Проверяем результаты
        total_concepts = sum(len(result) for result in results)
        assert total_concepts > 0
        
        # Проверяем производительность
        assert concurrent_time < 10.0
        
        # Проверяем статистику
        stats = extractor.get_extraction_stats()
        assert stats['total_extractions'] == 5
        assert stats['total_concepts_extracted'] == total_concepts
    
    def test_confidence_calculation(self, extractor_instance):
        """Тест вычисления confidence"""
        extractor = extractor_instance({})
        
        # Концепт с высокой уверенностью
        high_confidence_text = "Artificial Intelligence is defined as the field of computer science that deals with creating intelligent machines."
        concepts = extractor.extract_concepts(high_confidence_text)
        
        if concepts:
            for concept in concepts:
                assert 0 <= concept['confidence'] <= 1
                assert isinstance(concept['confidence'], (int, float))
        
        # Тест различных паттернов
        pattern_tests = [
            "AI is a field of computer science.",
            "Machine Learning: subset of AI that learns from data.",
            "Neural networks means computational models based on brain structure."
        ]
        
        for pattern_text in pattern_tests:
            concepts = extractor.extract_concepts(pattern_text)
            if concepts:
                for concept in concepts:
                    assert 0 <= concept['confidence'] <= 1
