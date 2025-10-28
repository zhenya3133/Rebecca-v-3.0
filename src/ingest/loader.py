"""Полноценный IngestPipeline для обработки различных типов документов.

Поддерживает:
- PDF документы (извлечение текста)
- Markdown файлы (парсинг, метаданные)
- Git-репозитории (README, исходные файлы)
- Текстовые файлы (.txt, .docx, .html, .csv, .json)
- Интеграция с MemoryManager для слоев памяти
- Comprehensive error handling и логирование
"""

import asyncio
import hashlib
import io
import logging
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Iterable
from urllib.parse import urlparse
import mimetypes

import pandas as pd
import yaml
from bs4 import BeautifulSoup
from dateutil.parser import parse as parse_date
from docx import Document
from git import Repo
from markdown import markdown
from pdfplumber import PDF
from pydantic import BaseModel

from event_graph.event_graph import InMemoryEventGraph
from .ingestion_models import IngestRecord
from memory_manager.memory_manager import MemoryManager
from schema.nodes import Event, Fact
from storage.graph_view import InMemoryGraphView
from storage.object_store import InMemoryObjectStore
from storage.pg_dao import InMemoryDAO
from retrieval.indexes import InMemoryBM25Index, InMemoryVectorIndex, InMemoryGraphIndex

# Настройка логгера
logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Метаданные документа."""
    source_type: str
    source_path: str
    file_size: int
    file_hash: str
    title: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    tags: Set[str] = field(default_factory=set)
    language: Optional[str] = None
    summary: Optional[str] = None
    checksum: Optional[str] = None
    encoding: Optional[str] = None


@dataclass
class ChunkData:
    """Часть текста с метаданными."""
    content: str
    metadata: DocumentMetadata
    chunk_id: str
    start_pos: int
    end_pos: int
    tokens_count: int = 0


class TextExtractor:
    """Утилита для извлечения текста из различных форматов."""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> Tuple[str, DocumentMetadata]:
        """Извлекает текст и метаданные из PDF."""
        try:
            text_content = []
            metadata = DocumentMetadata(
                source_type="pdf",
                source_path=pdf_path,
                file_size=os.path.getsize(pdf_path),
                file_hash=TextExtractor._calculate_file_hash(pdf_path)
            )
            
            with PDF(pdf_path) as pdf:
                # Извлечение текста
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(f"--- Страница {page_num + 1} ---\n{page_text}")
                
                # Извлечение метаданных
                pdf_metadata = pdf.metadata
                if pdf_metadata:
                    metadata.title = pdf_metadata.get('Title')
                    metadata.author = pdf_metadata.get('Author')
                    metadata.created_at = pdf_metadata.get('CreationDate')
                    metadata.summary = pdf_metadata.get('Subject')
            
            full_text = "\n\n".join(text_content)
            metadata.checksum = hashlib.md5(full_text.encode()).hexdigest()
            
            return full_text, metadata
            
        except Exception as e:
            logger.error(f"Ошибка при извлечении текста из PDF {pdf_path}: {e}")
            raise
    
    @staticmethod
    def extract_text_from_markdown(md_path: str) -> Tuple[str, DocumentMetadata]:
        """Извлекает текст и метаданные из Markdown."""
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            metadata = DocumentMetadata(
                source_type="markdown",
                source_path=md_path,
                file_size=len(content),
                file_hash=TextExtractor._calculate_file_hash(md_path),
                encoding="utf-8"
            )
            
            # Извлечение YAML front matter если есть
            front_matter = {}
            if content.startswith('---'):
                try:
                    end_front_matter = content.find('---', 3)
                    if end_front_matter != -1:
                        yaml_content = content[3:end_front_matter].strip()
                        front_matter = yaml.safe_load(yaml_content) or {}
                        content = content[end_front_matter + 3:]
                except:
                    pass
            
            # Обновление метаданных из front matter
            if front_matter:
                metadata.title = front_matter.get('title', metadata.title)
                metadata.author = front_matter.get('author', metadata.author)
                metadata.created_at = front_matter.get('date', metadata.created_at)
                metadata.tags = set(front_matter.get('tags', []))
                metadata.summary = front_matter.get('description', metadata.summary)
            
            # Конвертация markdown в HTML и обратно в текст для лучшего форматирования
            html_content = markdown(content, extensions=['extra', 'codehilite'])
            soup = BeautifulSoup(html_content, 'html.parser')
            text_content = soup.get_text()
            
            metadata.checksum = hashlib.md5(text_content.encode()).hexdigest()
            
            return text_content, metadata
            
        except Exception as e:
            logger.error(f"Ошибка при обработке Markdown {md_path}: {e}")
            raise
    
    @staticmethod
    def extract_text_from_docx(docx_path: str) -> Tuple[str, DocumentMetadata]:
        """Извлекает текст из DOCX."""
        try:
            doc = Document(docx_path)
            text_content = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content.append(paragraph.text)
            
            full_text = "\n\n".join(text_content)
            
            metadata = DocumentMetadata(
                source_type="docx",
                source_path=docx_path,
                file_size=os.path.getsize(docx_path),
                file_hash=TextExtractor._calculate_file_hash(docx_path),
                encoding="utf-8"
            )
            metadata.checksum = hashlib.md5(full_text.encode()).hexdigest()
            
            return full_text, metadata
            
        except Exception as e:
            logger.error(f"Ошибка при обработке DOCX {docx_path}: {e}")
            raise
    
    @staticmethod
    def extract_text_from_html(html_path: str) -> Tuple[str, DocumentMetadata]:
        """Извлекает текст из HTML."""
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # Удаляем скрипты и стили
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Извлекаем текст
            text_content = soup.get_text()
            
            # Очищаем от лишних пробелов
            lines = (line.strip() for line in text_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text_content = '\n'.join(chunk for chunk in chunks if chunk)
            
            metadata = DocumentMetadata(
                source_type="html",
                source_path=html_path,
                file_size=len(content.encode('utf-8')),
                file_hash=TextExtractor._calculate_file_hash(html_path),
                encoding="utf-8"
            )
            metadata.checksum = hashlib.md5(text_content.encode()).hexdigest()
            
            return text_content, metadata
            
        except Exception as e:
            logger.error(f"Ошибка при обработке HTML {html_path}: {e}")
            raise
    
    @staticmethod
    def extract_text_from_csv(csv_path: str) -> Tuple[str, DocumentMetadata]:
        """Извлекает текст из CSV."""
        try:
            df = pd.read_csv(csv_path)
            
            # Создаем текстовое представление CSV
            text_lines = []
            text_lines.append(f"CSV Файл: {Path(csv_path).name}")
            text_lines.append(f"Размер: {df.shape[0]} строк, {df.shape[1]} столбцов")
            text_lines.append("\nСтолбцы:")
            text_lines.extend([f"- {col}: {df[col].dtype}" for col in df.columns])
            
            text_lines.append("\nПервые 10 строк данных:")
            text_lines.append(df.head(10).to_string())
            
            text_content = "\n".join(text_lines)
            
            metadata = DocumentMetadata(
                source_type="csv",
                source_path=csv_path,
                file_size=os.path.getsize(csv_path),
                file_hash=TextExtractor._calculate_file_hash(csv_path),
                encoding="utf-8"
            )
            metadata.checksum = hashlib.md5(text_content.encode()).hexdigest()
            
            return text_content, metadata
            
        except Exception as e:
            logger.error(f"Ошибка при обработке CSV {csv_path}: {e}")
            raise
    
    @staticmethod
    def extract_text_from_json(json_path: str) -> Tuple[str, DocumentMetadata]:
        """Извлекает текст из JSON."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Создаем читаемое представление JSON
            text_content = json.dumps(data, ensure_ascii=False, indent=2)
            
            metadata = DocumentMetadata(
                source_type="json",
                source_path=json_path,
                file_size=os.path.getsize(json_path),
                file_hash=TextExtractor._calculate_file_hash(json_path),
                encoding="utf-8"
            )
            metadata.checksum = hashlib.md5(text_content.encode()).hexdigest()
            
            return text_content, metadata
            
        except Exception as e:
            logger.error(f"Ошибка при обработке JSON {json_path}: {e}")
            raise
    
    @staticmethod
    def extract_text_from_txt(txt_path: str) -> Tuple[str, DocumentMetadata]:
        """Извлекает текст из текстового файла."""
        try:
            # Пытаемся определить кодировку
            encodings = ['utf-8', 'utf-8-sig', 'cp1251', 'cp1252', 'iso-8859-1']
            content = None
            
            for encoding in encodings:
                try:
                    with open(txt_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    metadata_encoding = encoding
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                raise ValueError(f"Не удалось определить кодировку файла {txt_path}")
            
            metadata = DocumentMetadata(
                source_type="text",
                source_path=txt_path,
                file_size=os.path.getsize(txt_path),
                file_hash=TextExtractor._calculate_file_hash(txt_path),
                encoding=metadata_encoding
            )
            metadata.checksum = hashlib.md5(content.encode()).hexdigest()
            
            return content, metadata
            
        except Exception as e:
            logger.error(f"Ошибка при обработке текстового файла {txt_path}: {e}")
            raise
    
    @staticmethod
    def _calculate_file_hash(file_path: str) -> str:
        """Вычисляет хеш файла."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()


class GitRepositoryProcessor:
    """Обработчик Git-репозиториев."""
    
    @staticmethod
    def clone_repository(repo_url: str, branch: str = "main", target_dir: Optional[str] = None) -> str:
        """Клонирует репозиторий."""
        try:
            if target_dir is None:
                target_dir = tempfile.mkdtemp()
            
            repo = Repo.clone_from(repo_url, target_dir, branch=branch)
            logger.info(f"Репозиторий {repo_url} успешно клонирован в {target_dir}")
            
            return target_dir
            
        except Exception as e:
            logger.error(f"Ошибка при клонировании репозитория {repo_url}: {e}")
            raise
    
    @staticmethod
    def extract_readme_content(repo_path: str) -> Optional[Tuple[str, DocumentMetadata]]:
        """Извлекает содержимое README файлов."""
        readme_files = []
        
        # Поиск README файлов
        for pattern in ['README*', 'readme*', 'ReadMe*']:
            readme_files.extend(Path(repo_path).glob(pattern))
        
        if not readme_files:
            return None
        
        # Берем первый найденный README
        readme_path = readme_files[0]
        
        if readme_path.suffix.lower() == '.md':
            return TextExtractor.extract_text_from_markdown(str(readme_path))
        elif readme_path.suffix.lower() == '.rst':
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            metadata = DocumentMetadata(
                source_type="readme_rst",
                source_path=str(readme_path),
                file_size=len(content),
                file_hash=TextExtractor._calculate_file_hash(str(readme_path)),
                encoding="utf-8"
            )
            metadata.checksum = hashlib.md5(content.encode()).hexdigest()
            
            return content, metadata
        else:
            return TextExtractor.extract_text_from_txt(str(readme_path))
    
    @staticmethod
    def extract_source_files(repo_path: str, extensions: Set[str] = None) -> List[Tuple[str, DocumentMetadata]]:
        """Извлекает содержимое исходных файлов."""
        if extensions is None:
            extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.cs', '.go', '.rs', '.php', '.rb'}
        
        source_files = []
        
        for ext in extensions:
            source_files.extend(Path(repo_path).rglob(f"*{ext}"))
        
        results = []
        for file_path in source_files[:50]:  # Ограничиваем количество файлов
            try:
                if file_path.suffix.lower() == '.md':
                    content, metadata = TextExtractor.extract_text_from_markdown(str(file_path))
                else:
                    content, metadata = TextExtractor.extract_text_from_txt(str(file_path))
                
                # Обновляем метаданные для исходных файлов
                metadata.source_type = "source_code"
                metadata.summary = f"Исходный код: {file_path.name}"
                
                results.append((content, metadata))
                
            except Exception as e:
                logger.warning(f"Не удалось обработать файл {file_path}: {e}")
                continue
        
        return results


class TextChunker:
    """Утилита для разбиения текста на части."""
    
    @staticmethod
    def chunk_text(text: str, metadata: DocumentMetadata, chunk_size: int = 1000, overlap: int = 200) -> List[ChunkData]:
        """Разбивает текст на части с перекрытием."""
        if not text:
            return []
        
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        chunk_id = 0
        start_pos = 0
        
        for word in words:
            word_len = len(word) + 1  # +1 для пробела
            
            if current_length + word_len > chunk_size and current_chunk:
                # Создаем чанк
                chunk_text = " ".join(current_chunk)
                end_pos = start_pos + len(chunk_text)
                
                chunk = ChunkData(
                    content=chunk_text,
                    metadata=metadata,
                    chunk_id=f"{metadata.source_path}_chunk_{chunk_id}",
                    start_pos=start_pos,
                    end_pos=end_pos,
                    tokens_count=len(current_chunk)
                )
                chunks.append(chunk)
                
                # Подготавливаем следующий чанк с перекрытием
                chunk_id += 1
                overlap_words = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
                current_chunk = overlap_words
                current_length = sum(len(w) + 1 for w in current_chunk)
                start_pos = end_pos - len(chunk_text)
            else:
                current_chunk.append(word)
                current_length += word_len
        
        # Добавляем последний чанк
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            end_pos = start_pos + len(chunk_text)
            
            chunk = ChunkData(
                content=chunk_text,
                metadata=metadata,
                chunk_id=f"{metadata.source_path}_chunk_{chunk_id}",
                start_pos=start_pos,
                end_pos=end_pos,
                tokens_count=len(current_chunk)
            )
            chunks.append(chunk)
        
        return chunks


class IngestPipeline:
    """Полноценный IngestPipeline для обработки различных типов документов."""
    
    def __init__(
        self,
        memory: MemoryManager,
        dao: InMemoryDAO,
        bm25: InMemoryBM25Index,
        vec: InMemoryVectorIndex,
        graph_idx: InMemoryGraphIndex,
        graph_view: InMemoryGraphView,
        object_store: InMemoryObjectStore,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> None:
        self.memory = memory
        self.dao = dao
        self.bm25 = bm25
        self.vec = vec
        self.graph_idx = graph_idx
        self.graph_view = graph_view
        self.object_store = object_store
        
        # Настройки
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Инициализация компонентов
        self.text_extractor = TextExtractor()
        self.text_chunker = TextChunker()
        self.git_processor = GitRepositoryProcessor()
        
        # Статистика
        self.stats = defaultdict(int)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("IngestPipeline инициализирован")
    
    def validate_file(self, file_path: str) -> bool:
        """Валидирует файл для обработки."""
        try:
            path = Path(file_path)
            
            # Проверяем существование
            if not path.exists():
                self.logger.error(f"Файл не существует: {file_path}")
                return False
            
            # Проверяем размер (не более 100MB)
            if path.stat().st_size > 100 * 1024 * 1024:
                self.logger.error(f"Файл слишком большой: {file_path}")
                return False
            
            # Проверяем поддерживаемые расширения
            supported_extensions = {
                '.pdf', '.md', '.txt', '.docx', '.html', '.csv', '.json',
                '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.cs', '.go', '.rs', '.php', '.rb'
            }
            
            if path.suffix.lower() not in supported_extensions:
                self.logger.warning(f"Неподдерживаемое расширение: {path.suffix}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при валидации файла {file_path}: {e}")
            return False
    
    def ingest_document(self, file_path: str, chunk_override: Optional[int] = None) -> Event:
        """Универсальный метод обработки документа."""
        try:
            self.logger.info(f"Начинаем обработку документа: {file_path}")
            
            if not self.validate_file(file_path):
                raise ValueError(f"Файл не прошел валидацию: {file_path}")
            
            file_ext = Path(file_path).suffix.lower()
            chunk_size = chunk_override or self.chunk_size
            
            # Обработка в зависимости от типа файла
            if file_ext == '.pdf':
                text_content, metadata = self.text_extractor.extract_text_from_pdf(file_path)
            elif file_ext == '.md':
                text_content, metadata = self.text_extractor.extract_text_from_markdown(file_path)
            elif file_ext == '.docx':
                text_content, metadata = self.text_extractor.extract_text_from_docx(file_path)
            elif file_ext == '.html':
                text_content, metadata = self.text_extractor.extract_text_from_html(file_path)
            elif file_ext == '.csv':
                text_content, metadata = self.text_extractor.extract_text_from_csv(file_path)
            elif file_ext == '.json':
                text_content, metadata = self.text_extractor.extract_text_from_json(file_path)
            else:  # .txt и исходные файлы
                text_content, metadata = self.text_extractor.extract_text_from_txt(file_path)
            
            # Разбиение на чанки
            chunks = self.text_chunker.chunk_text(text_content, metadata, chunk_size, self.overlap)
            
            # Создание события
            event_id = f"doc::{metadata.source_type}::{Path(file_path).stem}"
            event = self._create_event(event_id, metadata, text_content[:500])  # Первые 500 символов для summary
            
            # Сохранение в индексы и память
            self._index_chunks(chunks, event.id)
            self._store_in_memory(event, metadata, chunks)
            
            # Сохранение метаданных в Vault
            self._store_metadata_in_vault(metadata, event.id)
            
            self.stats['documents_processed'] += 1
            self.stats['chunks_created'] += len(chunks)
            
            self.logger.info(f"Документ успешно обработан: {len(chunks)} чанков создано")
            return event
            
        except Exception as e:
            self.logger.error(f"Ошибка при обработке документа {file_path}: {e}")
            self.stats['errors'] += 1
            raise
    
    def process_git_repo(self, repo_url: str, branch: str = "main", process_readme: bool = True, 
                        process_source: bool = True) -> List[Event]:
        """Обрабатывает Git-репозиторий."""
        events = []
        temp_dir = None
        
        try:
            self.logger.info(f"Начинаем обработку репозитория: {repo_url}")
            
            # Клонирование репозитория
            temp_dir = self.git_processor.clone_repository(repo_url, branch)
            
            # Обработка README
            if process_readme:
                readme_result = self.git_processor.extract_readme_content(temp_dir)
                if readme_result:
                    text_content, metadata = readme_result
                    metadata.source_type = "git_readme"
                    metadata.title = f"README из {repo_url}"
                    
                    # Создание события для README
                    event_id = f"git::readme::{hashlib.md5(repo_url.encode()).hexdigest()[:8]}"
                    event = self._create_event(event_id, metadata, text_content[:500])
                    
                    # Обработка и сохранение
                    chunks = self.text_chunker.chunk_text(text_content, metadata, self.chunk_size, self.overlap)
                    self._index_chunks(chunks, event.id)
                    self._store_in_memory(event, metadata, chunks)
                    self._store_metadata_in_vault(metadata, event.id)
                    
                    events.append(event)
            
            # Обработка исходных файлов
            if process_source:
                source_files = self.git_processor.extract_source_files(temp_dir)
                for i, (text_content, metadata) in enumerate(source_files[:20]):  # Ограничиваем количество
                    event_id = f"git::source::{hashlib.md5(f'{repo_url}_{i}'.encode()).hexdigest()[:8]}"
                    event = self._create_event(event_id, metadata, text_content[:500])
                    
                    chunks = self.text_chunker.chunk_text(text_content, metadata, self.chunk_size, self.overlap)
                    self._index_chunks(chunks, event.id)
                    self._store_in_memory(event, metadata, chunks)
                    self._store_metadata_in_vault(metadata, event.id)
                    
                    events.append(event)
            
            self.logger.info(f"Репозиторий успешно обработано: {len(events)} событий создано")
            
        except Exception as e:
            self.logger.error(f"Ошибка при обработке репозитория {repo_url}: {e}")
            self.stats['errors'] += 1
            raise
        finally:
            # Очистка временной директории
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                self.logger.info(f"Временная директория очищена: {temp_dir}")
        
        return events
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, DocumentMetadata]:
        """Извлекает текст из PDF (алиас для совместимости)."""
        return self.text_extractor.extract_text_from_pdf(pdf_path)
    
    def chunk_text(self, text: str, metadata: DocumentMetadata, 
                   chunk_size: Optional[int] = None, overlap: Optional[int] = None) -> List[ChunkData]:
        """Разбивает текст на части (алиас для совместимости)."""
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.overlap
        return self.text_chunker.chunk_text(text, metadata, chunk_size, overlap)
    
    def ingest_facts(self, facts: Iterable[Fact]) -> None:
        """Сохраняет факты в систему."""
        try:
            for fact in facts:
                self.dao.upsert_node(fact.model_dump())
                self.bm25.upsert(fact.id, fact.object)
                self.vec.upsert(fact.id, [0.5, 0.4, 0.3])
                self.stats['facts_processed'] += 1
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении фактов: {e}")
            self.stats['errors'] += 1
            raise
    
    def ingest_pdf(self, pdf_path: str) -> Event:
        """Обрабатывает PDF (совместимость с существующим API)."""
        return self.ingest_document(pdf_path)
    
    def batch_process(self, sources: List[str], chunk_override: Optional[int] = None) -> List[Event]:
        """Пакетная обработка нескольких источников."""
        events = []
        
        self.logger.info(f"Начинаем пакетную обработку {len(sources)} источников")
        
        for i, source in enumerate(sources):
            try:
                self.logger.info(f"Обрабатываем источник {i+1}/{len(sources)}: {source}")
                
                if self._is_git_url(source):
                    repo_events = self.process_git_repo(source)
                    events.extend(repo_events)
                else:
                    event = self.ingest_document(source, chunk_override)
                    events.append(event)
                    
            except Exception as e:
                self.logger.error(f"Ошибка при обработке источника {source}: {e}")
                self.stats['errors'] += 1
                continue
        
        self.logger.info(f"Пакетная обработка завершена: {len(events)} событий создано")
        return events
    
    def get_statistics(self) -> Dict[str, int]:
        """Возвращает статистику обработки."""
        return dict(self.stats)
    
    def reset_statistics(self) -> None:
        """Сбрасывает статистику."""
        self.stats.clear()
    
    def _is_git_url(self, source: str) -> bool:
        """Проверяет, является ли источник Git URL."""
        patterns = [
            'github.com', 'gitlab.com', 'bitbucket.org',
            '.git', 'git://', 'git@'
        ]
        return any(pattern in source.lower() for pattern in patterns)
    
    def _create_event(self, event_id: str, metadata: DocumentMetadata, summary: str) -> Event:
        """Создает событие для документа."""
        event = Event(
            id=event_id,
            ntype="Event",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            owner="system",
            privacy="team",
            confidence=0.9,
            attrs={
                "text": summary,
                "source_type": metadata.source_type,
                "source_path": metadata.source_path,
                "file_size": metadata.file_size,
                "file_hash": metadata.file_hash,
                "title": metadata.title or "",
                "author": metadata.author or "",
                "tags": list(metadata.tags),
                "language": metadata.language or "unknown",
                "encoding": metadata.encoding or "utf-8"
            },
            t_start=datetime.now(UTC),
            actors=["pipeline"],
            channel="document_ingest",
            raw_ref=metadata.source_path,
        )
        
        self.dao.upsert_node(event.model_dump())
        self.graph_view.upsert_event(event)
        return event
    
    def _index_chunks(self, chunks: List[ChunkData], event_id: str) -> None:
        """Индексирует чанки в поисковых системах."""
        try:
            for chunk in chunks:
                # BM25 индекс
                self.bm25.upsert(chunk.chunk_id, chunk.content)
                
                # Векторный индекс (заглушка - в реальности нужен embedding)
                self.vec.upsert(chunk.chunk_id, [0.1, 0.2, 0.3])
                
                # Графовый индекс
                neighbors = []
                self.graph_idx.set_neighbors(chunk.chunk_id, neighbors)
                
                self.stats['chunks_indexed'] += 1
                
        except Exception as e:
            self.logger.error(f"Ошибка при индексации чанков: {e}")
            raise
    
    def _store_in_memory(self, event: Event, metadata: DocumentMetadata, chunks: List[ChunkData]) -> None:
        """Сохраняет данные в слои памяти."""
        try:
            # Эпизодическая память - события
            self.memory.episodic.store_event({
                "id": event.id,
                "summary": event.attrs["text"],
                "timestamp": datetime.now(UTC).isoformat(),
                "source_type": metadata.source_type,
                "source_path": metadata.source_path
            })
            
            # Семантическая память - концепты и извлеченные знания
            if metadata.title:
                self.memory.semantic.store_concept(
                    metadata.title, 
                    f"Документ: {metadata.summary or 'Без описания'}"
                )
            
            # Процедурная память - шаги обработки
            processing_steps = [
                "Валидация файла",
                "Извлечение текста", 
                "Разбиение на чанки",
                "Индексация",
                "Сохранение в память"
            ]
            
            for step in processing_steps:
                self.memory.procedural.store_procedure(
                    f"Обработка {metadata.source_type}",
                    [step]
                )
            
            # Блюпринт трекер
            self.memory.blueprint_tracker.link_resource(event.id, {
                "type": metadata.source_type,
                "path": metadata.source_path,
                "file_size": metadata.file_size,
                "chunks_count": len(chunks),
                "tags": list(metadata.tags)
            })
            
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении в память: {e}")
            raise
    
    def _store_metadata_in_vault(self, metadata: DocumentMetadata, event_id: str) -> None:
        """Сохраняет метаданные в Vault слой."""
        try:
            vault_key = f"document_metadata::{event_id}"
            vault_data = {
                "event_id": event_id,
                "source_type": metadata.source_type,
                "source_path": metadata.source_path,
                "file_size": metadata.file_size,
                "file_hash": metadata.file_hash,
                "title": metadata.title,
                "author": metadata.author,
                "created_at": metadata.created_at.isoformat() if metadata.created_at else None,
                "modified_at": metadata.modified_at.isoformat() if metadata.modified_at else None,
                "tags": list(metadata.tags),
                "language": metadata.language,
                "summary": metadata.summary,
                "checksum": metadata.checksum,
                "encoding": metadata.encoding,
                "stored_at": datetime.now(UTC).isoformat()
            }
            
            self.memory.vault.store_secret(vault_key, vault_data)
            self.logger.debug(f"Метаданные сохранены в Vault: {vault_key}")
            
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении метаданных в Vault: {e}")
            raise


# Фабрика методы для разных типов источников
class IngestPipelineFactory:
    """Фабрика для создания IngestPipeline с различными конфигурациями."""
    
    @staticmethod
    def create_basic_pipeline(memory: MemoryManager, dao: InMemoryDAO,
                            bm25: InMemoryBM25Index, vec: InMemoryVectorIndex,
                            graph_idx: InMemoryGraphIndex, graph_view: InMemoryGraphView,
                            object_store: InMemoryObjectStore) -> IngestPipeline:
        """Создает базовый pipeline с настройками по умолчанию."""
        return IngestPipeline(
            memory=memory,
            dao=dao,
            bm25=bm25,
            vec=vec,
            graph_idx=graph_idx,
            graph_view=graph_view,
            object_store=object_store,
            chunk_size=1000,
            overlap=200
        )
    
    @staticmethod
    def create_detailed_pipeline(memory: MemoryManager, dao: InMemoryDAO,
                               bm25: InMemoryBM25Index, vec: InMemoryVectorIndex,
                               graph_idx: InMemoryGraphIndex, graph_view: InMemoryGraphView,
                               object_store: InMemoryObjectStore, chunk_size: int = 1500) -> IngestPipeline:
        """Создает детальный pipeline с увеличенными чанками."""
        return IngestPipeline(
            memory=memory,
            dao=dao,
            bm25=bm25,
            vec=vec,
            graph_idx=graph_idx,
            graph_view=graph_view,
            object_store=object_store,
            chunk_size=chunk_size,
            overlap=300
        )
    
    @staticmethod
    def create_fast_pipeline(memory: MemoryManager, dao: InMemoryDAO,
                           bm25: InMemoryBM25Index, vec: InMemoryVectorIndex,
                           graph_idx: InMemoryGraphIndex, graph_view: InMemoryGraphView,
                           object_store: InMemoryObjectStore) -> IngestPipeline:
        """Создает быстрый pipeline с маленькими чанками."""
        return IngestPipeline(
            memory=memory,
            dao=dao,
            bm25=bm25,
            vec=vec,
            graph_idx=graph_idx,
            graph_view=graph_view,
            object_store=object_store,
            chunk_size=500,
            overlap=100
        )


# Класс для совместимости
class IngestionRecord(IngestRecord):
    """Класс для совместимости с существующим API."""
    pass