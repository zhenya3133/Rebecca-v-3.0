"""
Модуль извлечения концептов и связей для Rebecca-Platform.

Реализует современную систему извлечения концептов из текстовых данных с использованием:
- Named Entity Recognition (NER)
- Dependency Parsing  
- Semantic Role Labeling
- Семантическое группирование
- Контекстное ранжирование
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path

# Configuration import
try:
    from configuration import is_offline_mode
except ImportError:
    import os
    def is_offline_mode() -> bool:
        return (
            os.environ.get("REBECCA_OFFLINE_MODE", "").lower() in ("1", "true", "yes", "on") or
            os.environ.get("REBECCA_TEST_MODE", "").lower() in ("1", "true", "yes", "on")
        )

# NLP библиотеки - ленивый импорт для offline mode
spacy = None
nltk = None
SentenceTransformer = None

if not is_offline_mode():
    try:
        import spacy
    except ImportError:
        logging.warning("spacy недоступен")
    
    try:
        import nltk
    except ImportError:
        logging.warning("nltk недоступен")
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logging.warning("sentence_transformers недоступен")

# sklearn импортируем всегда, так как это легковесная библиотека без моделей
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    logging.warning(f"sklearn недоступен: {e}")
    TfidfVectorizer = None
    cosine_similarity = None

# Импорты памяти (с fallback)
try:
    from memory_manager.memory_context import SEMANTIC
    from memory_manager.memory_manager import MemoryManager
    from memory_manager.semantic_memory import SemanticMemory
except ImportError as e:
    logging.warning(f"Не удалось импортировать модули памяти: {e}")
    # Mock значения для тестирования
    SEMANTIC = "SEMANTIC"
    MemoryManager = None
    SemanticMemory = None

# Настройка логгера
logger = logging.getLogger(__name__)


@dataclass
class Concept:
    """Структура концепта для извлечения."""
    id: str
    text: str
    label: str  # Тип сущности (PERSON, ORG, GPE, etc.)
    confidence: float  # Уверенность модели (0-1)
    start_pos: int  # Начальная позиция в тексте
    end_pos: int  # Конечная позиция в тексте
    context: str  # Контекст вокруг концепта
    metadata: Dict[str, Any] = field(default_factory=dict)
    frequency: int = 1  # Частота встречания
    importance_score: float = 0.0  # Важность концепта


@dataclass
class Relationship:
    """Структура связи между концептами."""
    id: str
    source_concept_id: str
    target_concept_id: str
    relationship_type: str  #Тип связи (CAUSE, EFFECT, PART_OF, etc.)
    confidence: float
    context: str  # Контекст, где была обнаружена связь
    evidence: List[str] = field(default_factory=list)  # Список доказательств
    strength: float = 0.0  # Сила связи (0-1)


@dataclass
class ExtractedKnowledge:
    """Результат извлечения знаний из текста."""
    concepts: List[Concept]
    relationships: List[Relationship]
    text_id: str
    source_text: str
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticGrouper:
    """Группировка концептов по семантическому сходству."""
    
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self.embeddings_model = None
        
        # В offline mode не пытаемся загружать модель
        if is_offline_mode():
            logger.info("Offline mode: пропускаем загрузку модели SentenceTransformer")
            return
        
        # Инициализация модели эмбеддингов
        if SentenceTransformer is not None:
            try:
                self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Загружена модель SentenceTransformer")
            except Exception as e:
                logger.warning(f"Не удалось загрузить модель эмбеддингов: {e}")
        else:
            logger.warning("SentenceTransformer не доступен")
    
    async def group_concepts(self, concepts: List[Concept]) -> List[List[Concept]]:
        """Группирует концепты по семантическому сходству."""
        if not concepts:
            return []
        
        # Если модель эмбеддингов недоступна, используем простое сравнение по тексту
        if self.embeddings_model is None:
            return self._group_by_exact_match(concepts)
        
        try:
            # Получаем эмбеддинги для текстов концептов
            texts = [concept.text.lower().strip() for concept in concepts]
            embeddings = self.embeddings_model.encode(texts)
            
            # Вычисляем матрицу сходства
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(embeddings)
            
            # Группируем концепты
            groups = []
            used_indices = set()
            
            for i, concept in enumerate(concepts):
                if i in used_indices:
                    continue
                    
                # Находим похожие концепты
                group = [concept]
                used_indices.add(i)
                
                for j in range(i + 1, len(concepts)):
                    if j not in used_indices:
                        similarity = similarity_matrix[i][j]
                        if similarity >= self.similarity_threshold:
                            group.append(concepts[j])
                            used_indices.add(j)
                
                groups.append(group)
            
            return groups
            
        except Exception as e:
            logger.error(f"Ошибка группировки концептов: {e}")
            return self._group_by_exact_match(concepts)
    
    def _group_by_exact_match(self, concepts: List[Concept]) -> List[List[Concept]]:
        """Простая группировка по точному совпадению текста."""
        groups_dict = defaultdict(list)
        
        for concept in concepts:
            key = concept.text.lower().strip()
            groups_dict[key].append(concept)
        
        return list(groups_dict.values())


class RelevanceScorer:
    """Контекстное ранжирование релевантности концептов и связей."""
    
    def __init__(self, tfidf_vectorizer: Optional[Any] = None):
        self.tfidf = None
        self.cosine_similarity = None
        self.available = False
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            self.tfidf = tfidf_vectorizer or TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.cosine_similarity = cosine_similarity
            self.available = True
        except ImportError:
            logger.warning("scikit-learn недоступен, используется базовое ранжирование")
            self.tfidf = tfidf_vectorizer  # Может быть None
        
        self.document_frequencies = {}
    
    async def score_concepts(self, concepts: List[Concept], 
                           document_text: str) -> List[Concept]:
        """Ранжирует концепты по релевантности."""
        if not concepts:
            return concepts
        
        try:
            if self.available and self.tfidf and self.cosine_similarity:
                # Используем TF-IDF если доступен
                all_texts = [concept.context for concept in concepts] + [document_text]
                tfidf_matrix = self.tfidf.fit_transform(all_texts)
                doc_vector = tfidf_matrix[-1]
                
                for i, concept in enumerate(concepts):
                    concept_vector = tfidf_matrix[i]
                    relevance = self.cosine_similarity(concept_vector, doc_vector)[0][0]
                    concept.importance_score = max(0.0, min(1.0, relevance))
            else:
                # Базовое ранжирование без TF-IDF
                for i, concept in enumerate(concepts):
                    # Простая оценка на основе длины и типа
                    base_score = len(concept.text) / 50.0  # Нормализуем по длине
                    concept.importance_score = min(1.0, base_score)
                
                # Учитываем частоту встречания
                for concept in concepts:
                    if concept.frequency > 1:
                        concept.importance_score *= min(2.0, 1.0 + (concept.frequency - 1) * 0.1)
            
            # Учитываем тип сущности
            entity_weights = {
                'PERSON': 1.2, 'ORG': 1.1, 'GPE': 1.1,
                'DATE': 0.8, 'MONEY': 0.9, 'EVENT': 1.0,
                'WORK_OF_ART': 0.7, 'LANGUAGE': 0.6,
                'NAMED_ENTITY': 0.9, 'NOUN_PHRASE': 0.8
            }
            for concept in concepts:
                weight = entity_weights.get(concept.label, 1.0)
                concept.importance_score *= weight
                
        except Exception as e:
            logger.error(f"Ошибка ранжирования концептов: {e}")
            # Fallback - простые оценки
            for concept in concepts:
                concept.importance_score = 0.5
        
        return concepts
    
    async def score_relationships(self, relationships: List[Relationship]) -> List[Relationship]:
        """Ранжирует связи по релевантности."""
        if not relationships:
            return relationships
        
        # Простое ранжирование на основе уверенности модели
        for relationship in relationships:
            # Нормализуем силу связи
            relationship.strength = max(0.0, min(1.0, relationship.confidence))
        
        # Сортируем по силе связи
        relationships.sort(key=lambda r: r.strength, reverse=True)
        
        return relationships


class ConceptExtractor:
    """Основной класс для извлечения концептов и связей из текста."""
    
    def __init__(self, 
                 memory_manager: Optional[MemoryManager] = None,
                 enable_semantic_grouping: bool = True,
                 similarity_threshold: float = 0.7):
        """Инициализирует экстрактор концептов.
        
        Args:
            memory_manager: Менеджер памяти для интеграции
            enable_semantic_grouping: Включить семантическое группирование
            similarity_threshold: Порог сходства для группировки
        """
        self.memory_manager = memory_manager
        self.nlp_model = None
        self.semantic_grouper = SemanticGrouper(similarity_threshold) if enable_semantic_grouping else None
        self.relevance_scorer = RelevanceScorer()
        
        # Статистика извлечения
        self.stats = {
            'total_concepts_extracted': 0,
            'total_relationships_extracted': 0,
            'processing_time_total': 0.0,
            'documents_processed': 0
        }
        
        # Инициализация NLP модели
        self._initialize_nlp_model()
        
        logger.info("ConceptExtractor инициализирован")
    
    def _initialize_nlp_model(self):
        """Инициализирует NLP модель для обработки текста."""
        # В offline mode пропускаем загрузку spaCy моделей
        if is_offline_mode():
            logger.info("Offline mode: пропускаем загрузку spaCy моделей, используем rule-based методы")
            self.nlp_model = None
            return
        
        # Если spaCy не импортирован, пропускаем
        if spacy is None:
            logger.warning("spaCy не доступен, будут использоваться только базовые методы извлечения")
            self.nlp_model = None
            return
        
        try:
            # Пытаемся загрузить модель spaCy
            self.nlp_model = spacy.load("ru_core_news_sm")
            logger.info("Загружена русская модель spaCy")
        except Exception as e:
            try:
                # Пытаемся загрузить английскую модель
                self.nlp_model = spacy.load("en_core_web_sm")
                logger.info("Загружена английская модель spaCy")
            except Exception as e2:
                logger.warning(f"Не удалось загрузить модель spaCy: {e2}")
                logger.warning("Будут использоваться только базовые методы извлечения")
                self.nlp_model = None
    
    async def extract_from_text(self, 
                              text: str, 
                              text_id: Optional[str] = None,
                              document_type: str = "general") -> ExtractedKnowledge:
        """Извлекает концепты и связи из текста.
        
        Args:
            text: Текст для анализа
            text_id: Уникальный идентификатор текста
            document_type: Тип документа (article, report, etc.)
            
        Returns:
            Результат извлечения знаний
        """
        start_time = datetime.now()
        text_id = text_id or str(uuid.uuid4())
        
        logger.info(f"Начало извлечения концептов из текста {text_id}")
        
        try:
            # Предварительная обработка текста
            processed_text = self._preprocess_text(text)
            
            # Извлечение концептов
            concepts = await self._extract_concepts(processed_text)
            
            # Извлечение связей
            relationships = await self._extract_relationships(concepts, processed_text)
            
            # Семантическое группирование (если включено)
            if self.semantic_grouper:
                await self._apply_semantic_grouping(concepts)
            
            # Контекстное ранжирование
            concepts = await self.relevance_scorer.score_concepts(concepts, processed_text)
            relationships = await self.relevance_scorer.score_relationships(relationships)
            
            # Сохранение в память (если доступен MemoryManager)
            if self.memory_manager:
                await self._save_to_memory(concepts, relationships, text_id, document_type)
            
            # Обновление статистики
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(len(concepts), len(relationships), processing_time)
            
            result = ExtractedKnowledge(
                concepts=concepts,
                relationships=relationships,
                text_id=text_id,
                source_text=text,
                processing_time=processing_time,
                metadata={
                    "document_type": document_type,
                    "extraction_timestamp": datetime.now().isoformat(),
                    "nlp_model": "spaCy" if self.nlp_model else "basic",
                    "semantic_grouping_enabled": self.semantic_grouper is not None
                }
            )
            
            logger.info(f"Извлечено {len(concepts)} концептов и {len(relationships)} связей за {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка извлечения концептов: {e}")
            raise RuntimeError(f"Не удалось извлечь концепты из текста: {e}")
    
    def _preprocess_text(self, text: str) -> str:
        """Предварительная обработка текста."""
        # Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Удаление специальных символов (но сохранение пунктуации)
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\'\/]', '', text)
        
        return text
    
    async def _extract_concepts(self, text: str) -> List[Concept]:
        """Извлекает концепты из текста."""
        concepts = []
        
        if self.nlp_model:
            concepts = await self._extract_with_spacy(text)
        else:
            concepts = await self._extract_with_basic_methods(text)
        
        # Объединяем концепты с одинаковым текстом
        concepts = self._merge_duplicate_concepts(concepts)
        
        return concepts
    
    async def _extract_with_spacy(self, text: str) -> List[Concept]:
        """Извлечение концептов с помощью spaCy."""
        concepts = []
        doc = self.nlp_model(text)
        
        for ent in doc.ents:
            # Определяем контекст
            start = max(0, ent.start_char - 50)
            end = min(len(text), ent.end_char + 50)
            context = text[start:end].strip()
            
            concept = Concept(
                id=str(uuid.uuid4()),
                text=ent.text,
                label=ent.label_,
                confidence=ent._.confidence if hasattr(ent._, 'confidence') else 0.8,
                start_pos=ent.start_char,
                end_pos=ent.end_char,
                context=context,
                metadata={
                    "pos_tags": [token.pos_ for token in ent],
                    "dep_tags": [token.dep_ for token in ent]
                }
            )
            concepts.append(concept)
        
        # Дополнительное извлечение с помощью dependency parsing
        for token in doc:
            # Извлекаем ключевые существительные и именованные группы
            if token.pos_ in ["NOUN", "PROPN"] and token.dep_ in ["nsubj", "dobj", "pobj"]:
                if len(token.text) > 2:  # Игнорируем очень короткие слова
                    # Находим полную именованную группу
                    span = doc[token.i].subtree
                    start_char = span.start_char
                    end_char = span.end_char
                    context = text[max(0, start_char - 30):min(len(text), end_char + 30)]
                    
                    concept = Concept(
                        id=str(uuid.uuid4()),
                        text=span.text,
                        label="NOUN_PHRASE",
                        confidence=0.6,
                        start_pos=start_char,
                        end_pos=end_char,
                        context=context.strip(),
                        metadata={
                            "extraction_method": "dependency_parsing",
                            "head_token": token.text,
                            "dependency": token.dep_
                        }
                    )
                    concepts.append(concept)
        
        return concepts
    
    async def _extract_with_basic_methods(self, text: str) -> List[Concept]:
        """Базисное извлечение концептов без NLP модели."""
        concepts = []
        
        # Простое извлечение существительных и именованных групп
        words = text.split()
        
        # Ищем последовательности заглавных букв (потенциальные именованные сущности)
        pattern = r'\b[A-ZА-Я][a-zа-я]*(?:\s+[A-ZА-Я][a-zа-я]*)*\b'
        matches = re.finditer(pattern, text)
        
        for match in matches:
            concept_text = match.group().strip()
            
            if len(concept_text) > 2:  # Игнорируем очень короткие концепты
                start_pos = match.start()
                end_pos = match.end()
                
                # Контекст вокруг концепта
                start = max(0, start_pos - 30)
                end = min(len(text), end_pos + 30)
                context = text[start:end]
                
                concept = Concept(
                    id=str(uuid.uuid4()),
                    text=concept_text,
                    label="NAMED_ENTITY",
                    confidence=0.5,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    context=context.strip(),
                    metadata={
                        "extraction_method": "basic_regex"
                    }
                )
                concepts.append(concept)
        
        return concepts
    
    def _merge_duplicate_concepts(self, concepts: List[Concept]) -> List[Concept]:
        """Объединяет концепты с одинаковым текстом."""
        concept_dict = {}
        
        for concept in concepts:
            key = concept.text.lower().strip()
            
            if key not in concept_dict:
                concept_dict[key] = concept
            else:
                # Объединяем информацию
                existing = concept_dict[key]
                existing.frequency += 1
                
                # Сохраняем концепт с максимальной уверенностью
                if concept.confidence > existing.confidence:
                    existing.confidence = concept.confidence
                    existing.metadata.update(concept.metadata)
        
        return list(concept_dict.values())
    
    async def _extract_relationships(self, concepts: List[Concept], text: str) -> List[Relationship]:
        """Извлекает связи между концептами."""
        relationships = []
        
        if self.nlp_model:
            relationships = await self._extract_with_dependency_parsing(concepts, text)
        else:
            relationships = await self._extract_with_patterns(concepts, text)
        
        return relationships
    
    async def _extract_with_dependency_parsing(self, concepts: List[Concept], text: str) -> List[Relationship]:
        """Извлечение связей с помощью dependency parsing."""
        relationships = []
        doc = self.nlp_model(text)
        
        for sent in doc.sents:
            for token in sent:
                # Поиск связей между концептами в предложении
                if token.pos_ == "VERB":
                    subject = None
                    object_ = None
                    
                    # Находим подлежащее
                    for child in token.children:
                        if child.dep_ in ["nsubj", "nsubj:pass"]:
                            subject = child
                            break
                    
                    # Находим прямое дополнение
                    for child in token.children:
                        if child.dep_ == "dobj":
                            object_ = child
                            break
                    
                    # Создаем связь, если найдены и подлежащее, и дополнение
                    if subject and object_:
                        # Находим соответствующие концепты
                        subject_concept = self._find_concept_for_token(subject, concepts)
                        object_concept = self._find_concept_for_token(object_, concepts)
                        
                        if subject_concept and object_concept:
                            relationship = Relationship(
                                id=str(uuid.uuid4()),
                                source_concept_id=subject_concept.id,
                                target_concept_id=object_concept.id,
                                relationship_type="ACTION",
                                confidence=0.7,
                                context=sent.text,
                                evidence=[f"Глагол '{token.text}' связывает {subject.text} и {object_.text}"],
                                strength=0.7
                            )
                            relationships.append(relationship)
        
        return relationships
    
    def _find_concept_for_token(self, token, concepts: List[Concept]) -> Optional[Concept]:
        """Находит концепт, соответствующий токену."""
        token_text = token.text.lower().strip()
        
        for concept in concepts:
            if concept.text.lower().strip() == token_text:
                return concept
        
        return None
    
    async def _extract_with_patterns(self, concepts: List[Concept], text: str) -> List[Relationship]:
        """Извлечение связей с помощью паттернов."""
        relationships = []
        
        # Простые паттерны для поиска связей
        patterns = [
            (r'(\w+)\s+(?:является|есть|represents|is)\s+(\w+)', 'IS_A'),
            (r'(\w+)\s+(?:часть|part\s+of)\s+(\w+)', 'PART_OF'),
            (r'(\w+)\s+(?:связан|correlated\s+with)\s+(\w+)', 'RELATED_TO'),
            (r'(\w+)\s+(?:вызывает|causes)\s+(\w+)', 'CAUSES'),
        ]
        
        for pattern, rel_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                concept1_text = match.group(1).lower()
                concept2_text = match.group(2).lower()
                
                # Находим соответствующие концепты
                concept1 = self._find_concept_by_text(concepts, concept1_text)
                concept2 = self._find_concept_by_text(concepts, concept2_text)
                
                if concept1 and concept2:
                    relationship = Relationship(
                        id=str(uuid.uuid4()),
                        source_concept_id=concept1.id,
                        target_concept_id=concept2.id,
                        relationship_type=rel_type,
                        confidence=0.6,
                        context=match.group(0),
                        evidence=[f"Паттерн: {match.group(0)}"],
                        strength=0.6
                    )
                    relationships.append(relationship)
        
        return relationships
    
    def _find_concept_by_text(self, concepts: List[Concept], text: str) -> Optional[Concept]:
        """Находит концепт по тексту (без учета регистра)."""
        text = text.lower().strip()
        
        for concept in concepts:
            if concept.text.lower().strip() == text:
                return concept
        
        return None
    
    async def _apply_semantic_grouping(self, concepts: List[Concept]):
        """Применяет семантическое группирование к концептам."""
        if not self.semantic_grouper:
            return
        
        try:
            groups = await self.semantic_grouper.group_concepts(concepts)
            
            # Обновляем метаданные концептов с информацией о группировке
            for group in groups:
                if len(group) > 1:
                    for concept in group:
                        concept.metadata['semantic_group'] = {
                            'group_size': len(group),
                            'group_concepts': [c.text for c in group],
                            'is_grouped': True
                        }
            
            logger.info(f"Семантическое группирование создало {len(groups)} групп из {len(concepts)} концептов")
            
        except Exception as e:
            logger.error(f"Ошибка семантического группирования: {e}")
    
    async def _save_to_memory(self, 
                            concepts: List[Concept], 
                            relationships: List[Relationship],
                            text_id: str,
                            document_type: str):
        """Сохраняет извлеченные данные в память."""
        if not self.memory_manager:
            return
        
        try:
            # Сохраняем концепты
            for concept in concepts:
                concept_data = {
                    'concept': {
                        'text': concept.text,
                        'label': concept.label,
                        'confidence': concept.confidence,
                        'importance_score': concept.importance_score,
                        'frequency': concept.frequency,
                        'metadata': concept.metadata
                    },
                    'relationships': []
                }
                
                # Добавляем связи для этого концепта
                for rel in relationships:
                    if rel.source_concept_id == concept.id or rel.target_concept_id == concept.id:
                        concept_data['relationships'].append({
                            'id': rel.id,
                            'type': rel.relationship_type,
                            'confidence': rel.confidence,
                            'strength': rel.strength
                        })
                
                await self.memory_manager.store(
                    layer=SEMANTIC,
                    data=concept_data,
                    metadata={
                        'extraction_source': 'concept_extractor',
                        'source_text_id': text_id,
                        'document_type': document_type,
                        'processing_timestamp': datetime.now().isoformat()
                    },
                    tags=['concept', document_type],
                    priority=int(concept.importance_score * 10)
                )
            
            logger.info(f"Сохранено {len(concepts)} концептов в семантическую память")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения в память: {e}")
    
    def _update_stats(self, concept_count: int, relationship_count: int, processing_time: float):
        """Обновляет статистику извлечения."""
        self.stats['total_concepts_extracted'] += concept_count
        self.stats['total_relationships_extracted'] += relationship_count
        self.stats['processing_time_total'] += processing_time
        self.stats['documents_processed'] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику работы экстрактора."""
        stats = self.stats.copy()
        
        if stats['documents_processed'] > 0:
            stats['average_concepts_per_document'] = (
                stats['total_concepts_extracted'] / stats['documents_processed']
            )
            stats['average_relationships_per_document'] = (
                stats['total_relationships_extracted'] / stats['documents_processed']
            )
            stats['average_processing_time'] = (
                stats['processing_time_total'] / stats['documents_processed']
            )
        
        return stats
    
    async def batch_extract_from_files(self, 
                                     file_paths: List[str],
                                     document_type: str = "document") -> List[ExtractedKnowledge]:
        """Пакетное извлечение концептов из файлов.
        
        Args:
            file_paths: Список путей к файлам
            document_type: Тип документов
            
        Returns:
            Список результатов извлечения
        """
        results = []
        
        for file_path in file_paths:
            try:
                # Читаем файл
                file_path_obj = Path(file_path)
                if not file_path_obj.exists():
                    logger.warning(f"Файл не найден: {file_path}")
                    continue
                
                text = file_path_obj.read_text(encoding='utf-8')
                
                # Извлекаем концепты
                result = await self.extract_from_text(
                    text=text,
                    text_id=f"file_{file_path_obj.stem}",
                    document_type=document_type
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Ошибка обработки файла {file_path}: {e}")
        
        logger.info(f"Завершена пакетная обработка {len(results)} файлов")
        return results


# Вспомогательные функции

async def create_concept_extractor(memory_manager: Optional[MemoryManager] = None) -> ConceptExtractor:
    """Создает и настраивает экстрактор концептов."""
    return ConceptExtractor(memory_manager=memory_manager)


async def quick_concept_test(text: str) -> ExtractedKnowledge:
    """Быстрый тест извлечения концептов."""
    extractor = await create_concept_extractor()
    return await extractor.extract_from_text(text)


# Экспорт основных классов
__all__ = [
    'ConceptExtractor',
    'Concept', 
    'Relationship',
    'ExtractedKnowledge',
    'SemanticGrouper',
    'RelevanceScorer',
    'create_concept_extractor',
    'quick_concept_test'
]