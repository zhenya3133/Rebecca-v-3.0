"""Тесты для обновленного IngestPipeline."""

import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from ingest.loader import (
    IngestPipeline, 
    IngestPipelineFactory,
    TextExtractor,
    DocumentMetadata,
    TextChunker
)
from memory_manager.memory_manager import MemoryManager
from storage.pg_dao import InMemoryDAO
from retrieval.indexes import InMemoryBM25Index, InMemoryVectorIndex, InMemoryGraphIndex
from storage.graph_view import InMemoryGraphView
from storage.object_store import InMemoryObjectStore


class TestIngestPipeline:
    """Тесты для IngestPipeline."""
    
    @pytest.fixture
    def pipeline_components(self):
        """Создает компоненты для pipeline."""
        memory = MemoryManager()
        dao = InMemoryDAO()
        bm25 = InMemoryBM25Index()
        vec = InMemoryVectorIndex()
        graph_idx = InMemoryGraphIndex()
        graph_view = InMemoryGraphView()
        object_store = InMemoryObjectStore()
        
        return memory, dao, bm25, vec, graph_idx, graph_view, object_store
    
    @pytest.fixture
    def pipeline(self, pipeline_components):
        """Создает IngestPipeline."""
        memory, dao, bm25, vec, graph_idx, graph_view, object_store = pipeline_components
        
        return IngestPipeline(
            memory=memory,
            dao=dao,
            bm25=bm25,
            vec=vec,
            graph_idx=graph_idx,
            graph_view=graph_view,
            object_store=object_store,
            chunk_size=100,
            overlap=20
        )
    
    @pytest.fixture
    def test_files(self):
        """Создает тестовые файлы."""
        temp_dir = tempfile.mkdtemp()
        
        # Текстовый файл
        txt_path = Path(temp_dir) / "test.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("Тестовый текстовый файл для проверки обработки.")
        
        # Markdown файл
        md_path = Path(temp_dir) / "test.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("""---
title: Тестовый документ
author: Тест
tags: [test, example]
---

# Заголовок документа

Это тестовый документ для проверки IngestPipeline.

## Содержание

Содержит основную информацию для тестирования.
""")
        
        # JSON файл
        json_path = Path(temp_dir) / "test.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write('{"test": "data", "number": 42, "items": [1, 2, 3]}')
        
        return {
            'temp_dir': temp_dir,
            'txt_path': str(txt_path),
            'md_path': str(md_path), 
            'json_path': str(json_path)
        }
    
    def test_pipeline_creation(self, pipeline_components):
        """Тест создания pipeline через фабрику."""
        memory, dao, bm25, vec, graph_idx, graph_view, object_store = pipeline_components
        
        # Базовая фабрика
        pipeline = IngestPipelineFactory.create_basic_pipeline(
            memory, dao, bm25, vec, graph_idx, graph_view, object_store
        )
        assert isinstance(pipeline, IngestPipeline)
        assert pipeline.chunk_size == 1000
        assert pipeline.overlap == 200
        
        # Детальная фабрика
        pipeline = IngestPipelineFactory.create_detailed_pipeline(
            memory, dao, bm25, vec, graph_idx, graph_view, object_store, chunk_size=1500
        )
        assert pipeline.chunk_size == 1500
        
        # Быстрая фабрика
        pipeline = IngestPipelineFactory.create_fast_pipeline(
            memory, dao, bm25, vec, graph_idx, graph_view, object_store
        )
        assert pipeline.chunk_size == 500
        assert pipeline.overlap == 100
    
    def test_file_validation(self, pipeline, test_files):
        """Тест валидации файлов."""
        # Валидные файлы
        assert pipeline.validate_file(test_files['txt_path']) == True
        assert pipeline.validate_file(test_files['md_path']) == True
        assert pipeline.validate_file(test_files['json_path']) == True
        
        # Несуществующий файл
        assert pipeline.validate_file("/nonexistent/file.txt") == False
        
        # Файл с неподдерживаемым расширением
        unsupported_path = Path(test_files['temp_dir']) / "test.xyz"
        unsupported_path.touch()
        assert pipeline.validate_file(str(unsupported_path)) == False
    
    def test_text_extractor_txt(self, test_files):
        """Тест извлечения текста из TXT файла."""
        text, metadata = TextExtractor.extract_text_from_txt(test_files['txt_path'])
        
        assert "Тестовый текстовый файл" in text
        assert metadata.source_type == "text"
        assert metadata.source_path == test_files['txt_path']
        assert metadata.encoding == "utf-8"
        assert metadata.checksum is not None
    
    def test_text_extractor_markdown(self, test_files):
        """Тест извлечения текста из Markdown."""
        text, metadata = TextExtractor.extract_text_from_markdown(test_files['md_path'])
        
        assert "Заголовок документа" in text
        assert metadata.title == "Тестовый документ"
        assert metadata.author == "Тест"
        assert "test" in metadata.tags
        assert "example" in metadata.tags
    
    def test_text_extractor_json(self, test_files):
        """Тест извлечения текста из JSON."""
        text, metadata = TextExtractor.extract_text_from_json(test_files['json_path'])
        
        assert "test" in text
        assert "data" in text
        assert metadata.source_type == "json"
        assert metadata.checksum is not None
    
    def test_text_chunker(self, test_files):
        """Тест разбиения текста на чанки."""
        text = "Это тестовое предложение для проверки разбиения на чанки." * 10
        
        metadata = DocumentMetadata(
            source_type="test",
            source_path="/test/path",
            file_size=len(text),
            file_hash="test_hash"
        )
        
        chunks = TextChunker.chunk_text(text, metadata, chunk_size=50, overlap=10)
        
        assert len(chunks) > 1
        assert all(isinstance(chunk.content, str) for chunk in chunks)
        assert all(chunk.metadata == metadata for chunk in chunks)
        assert all(chunk.chunk_id.startswith("/test/path_chunk_") for chunk in chunks)
    
    def test_ingest_document_txt(self, pipeline, test_files):
        """Тест обработки текстового документа."""
        event = pipeline.ingest_document(test_files['txt_path'])
        
        assert event is not None
        assert event.id.startswith("doc::text::")
        assert "Тестовый текстовый файл" in event.attrs["text"]
        assert event.attrs["source_type"] == "text"
        assert event.attrs["source_path"] == test_files['txt_path']
        
        # Проверка сохранения в память
        events = pipeline.memory.episodic.get_events()
        assert len(events) > 0
    
    def test_ingest_document_markdown(self, pipeline, test_files):
        """Тест обработки Markdown документа."""
        event = pipeline.ingest_document(test_files['md_path'])
        
        assert event is not None
        assert event.attrs["source_type"] == "markdown"
        assert event.attrs["title"] == "Тестовый документ"
        assert event.attrs["author"] == "Тест"
        assert "test" in event.attrs["tags"]
        
        # Проверка семантической памяти
        concepts = pipeline.memory.semantic.concepts
        assert len(concepts) > 0
    
    def test_ingest_document_custom_chunk_size(self, pipeline, test_files):
        """Тест обработки с кастомным размером чанков."""
        event = pipeline.ingest_document(test_files['txt_path'], chunk_override=20)
        
        assert event is not None
        # Проверка статистики
        stats = pipeline.get_statistics()
        assert stats['documents_processed'] >= 1
    
    def test_batch_processing(self, pipeline, test_files):
        """Тест пакетной обработки."""
        sources = [test_files['txt_path'], test_files['md_path']]
        
        events = pipeline.batch_process(sources)
        
        assert len(events) >= 2
        assert all(event.id.startswith("doc::") for event in events)
        
        # Проверка статистики
        stats = pipeline.get_statistics()
        assert stats['documents_processed'] >= 2
    
    def test_statistics(self, pipeline, test_files):
        """Тест статистики обработки."""
        # Проверка начальной статистики
        stats = pipeline.get_statistics()
        assert stats == {}
        
        # Обработка документа
        pipeline.ingest_document(test_files['txt_path'])
        
        stats = pipeline.get_statistics()
        assert stats['documents_processed'] == 1
        
        # Сброс статистики
        pipeline.reset_statistics()
        
        stats = pipeline.get_statistics()
        assert stats == {}
    
    def test_is_git_url(self, pipeline):
        """Тест определения Git URL."""
        # Git URL
        assert pipeline._is_git_url("https://github.com/user/repo.git") == True
        assert pipeline._is_git_url("git@github.com:user/repo.git") == True
        assert pipeline._is_git_url("git://github.com/user/repo.git") == True
        
        # Не Git URL
        assert pipeline._is_git_url("https://example.com/document.pdf") == False
        assert pipeline._is_git_url("/local/path/file.txt") == False
        assert pipeline._is_git_url("document.pdf") == False
    
    def test_error_handling(self, pipeline):
        """Тест обработки ошибок."""
        # Несуществующий файл
        with pytest.raises(ValueError):
            pipeline.ingest_document("/nonexistent/file.txt")
        
        # Неподдерживаемый файл
        unsupported_path = Path(pipeline.memory.core.base_path) / "test.xyz"
        unsupported_path.touch()
        
        with pytest.raises(ValueError):
            pipeline.ingest_document(str(unsupported_path))
    
    def test_metadata_storage_in_vault(self, pipeline, test_files):
        """Тест сохранения метаданных в Vault."""
        event = pipeline.ingest_document(test_files['md_path'])
        
        # Проверка сохранения в Vault
        vault_key = f"document_metadata::{event.id}"
        assert vault_key in pipeline.memory.vault.secrets
        
        vault_data = pipeline.memory.vault.secrets[vault_key]
        assert vault_data['event_id'] == event.id
        assert vault_data['source_type'] == 'markdown'
        assert vault_data['source_path'] == test_files['md_path']
        assert vault_data['title'] == 'Тестовый документ'
    
    def test_cleanup_test_files(self, test_files):
        """Тест очистки тестовых файлов."""
        temp_dir = Path(test_files['temp_dir'])
        assert temp_dir.exists()
        
        # Очистка
        import shutil
        shutil.rmtree(temp_dir)
        assert not temp_dir.exists()


class TestTextExtractor:
    """Тесты для TextExtractor."""
    
    def test_calculate_file_hash(self):
        """Тест вычисления хеша файла."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_path = f.name
        
        try:
            hash1 = TextExtractor._calculate_file_hash(temp_path)
            hash2 = TextExtractor._calculate_file_hash(temp_path)
            
            assert hash1 == hash2
            assert len(hash1) == 32  # MD5 хеш
        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])