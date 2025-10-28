"""Пример использования обновленного IngestPipeline.

Демонстрирует:
- Обработку PDF документов
- Обработку Markdown файлов  
- Обработку Git репозиториев
- Пакетную обработку
- Интеграцию с MemoryManager
"""

import asyncio
import tempfile
from pathlib import Path

from ingest.loader import IngestPipelineFactory
from memory_manager.memory_manager import MemoryManager
from storage.pg_dao import InMemoryDAO
from retrieval.indexes import InMemoryBM25Index, InMemoryVectorIndex, InMemoryGraphIndex
from storage.graph_view import InMemoryGraphView
from storage.object_store import InMemoryObjectStore


async def main():
    """Основная функция примера."""
    
    # Инициализация компонентов
    memory = MemoryManager()
    dao = InMemoryDAO()
    bm25 = InMemoryBM25Index()
    vec = InMemoryVectorIndex()
    graph_idx = InMemoryGraphIndex()
    graph_view = InMemoryGraphView()
    object_store = InMemoryObjectStore()
    
    # Создание pipeline с помощью фабрики
    pipeline = IngestPipelineFactory.create_basic_pipeline(
        memory=memory,
        dao=dao,
        bm25=bm25,
        vec=vec,
        graph_idx=graph_idx,
        graph_view=graph_view,
        object_store=object_store
    )
    
    print("🚀 Пример использования IngestPipeline")
    print("=" * 50)
    
    # 1. Создание тестового PDF документа
    print("\n1. Создание тестового PDF документа...")
    pdf_path = create_test_pdf()
    print(f"   Создан: {pdf_path}")
    
    # 2. Обработка PDF документа
    print("\n2. Обработка PDF документа...")
    try:
        event = pipeline.ingest_document(pdf_path)
        print(f"   ✅ PDF обработан: {event.id}")
        print(f"   📊 Создано чанков: {len(pipeline.get_statistics())}")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
    
    # 3. Создание тестового Markdown файла
    print("\n3. Создание тестового Markdown файла...")
    md_path = create_test_markdown()
    print(f"   Создан: {md_path}")
    
    # 4. Обработка Markdown файла
    print("\n4. Обработка Markdown файла...")
    try:
        event = pipeline.ingest_document(md_path)
        print(f"   ✅ Markdown обработан: {event.id}")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
    
    # 5. Обработка Git репозитория (симуляция)
    print("\n5. Обработка Git репозитория...")
    try:
        # В реальности здесь был бы настоящий Git URL
        # events = pipeline.process_git_repo("https://github.com/example/repo.git")
        print("   ⚠️  Git обработка пропущена в примере")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
    
    # 6. Пакетная обработка
    print("\n6. Пакетная обработка...")
    try:
        sources = [pdf_path, md_path]
        events = pipeline.batch_process(sources)
        print(f"   ✅ Пакетная обработка завершена: {len(events)} событий")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
    
    # 7. Статистика
    print("\n7. Статистика обработки:")
    stats = pipeline.get_statistics()
    for key, value in stats.items():
        print(f"   📈 {key}: {value}")
    
    # 8. Проверка сохранения в MemoryManager
    print("\n8. Проверка сохранения в MemoryManager:")
    print(f"   🧠 Эпизодическая память: {len(memory.episodic.get_events())} событий")
    print(f"   💎 Vault память: {len(memory.vault.secrets)} секретов")
    print(f"   🔧 Семантическая память: {len(memory.semantic.concepts)} концептов")
    
    # Очистка временных файлов
    cleanup_test_files([pdf_path, md_path])
    
    print("\n🎉 Пример завершен успешно!")


def create_test_pdf() -> str:
    """Создает тестовый PDF файл."""
    # В реальном примере здесь был бы PDF файл
    # Для демонстрации создаем текстовый файл
    temp_dir = tempfile.mkdtemp()
    pdf_path = Path(temp_dir) / "test_document.txt"
    
    with open(pdf_path, 'w', encoding='utf-8') as f:
        f.write("""# Тестовый документ

Это тестовый документ для демонстрации IngestPipeline.

## Содержание

Документ содержит:
- Текст на русском языке
- Markdown форматирование
- Различные секции

## Выводы

IngestPipeline поддерживает обработку различных типов документов:
- PDF файлы
- Markdown документы
- Git репозитории
- Исходный код

Система автоматически:
1. Валидирует файлы
2. Извлекает текст
3. Разбивает на чанки
4. Индексирует содержимое
5. Сохраняет в слои памяти
""")
    
    return str(pdf_path)


def create_test_markdown() -> str:
    """Создает тестовый Markdown файл."""
    temp_dir = tempfile.mkdtemp()
    md_path = Path(temp_dir) / "test_guide.md"
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("""---
title: Руководство по IngestPipeline
author: System
date: 2025-10-28
tags: [ingest, pipeline, documents]
description: Полное руководство по использованию IngestPipeline
---

# Руководство по IngestPipeline

## Введение

IngestPipeline - это универсальная система для обработки документов различных типов.

## Поддерживаемые форматы

### Документы
- PDF (.pdf)
- Microsoft Word (.docx)
- HTML (.html)
- Markdown (.md)
- Обычный текст (.txt)
- CSV (.csv)
- JSON (.json)

### Исходный код
- Python (.py)
- JavaScript (.js, .ts)
- Java (.java)
- C/C++ (.c, .cpp, .h)
- И другие

## Основные возможности

1. **Валидация файлов**
2. **Извлечение текста**
3. **Разбиение на чанки**
4. **Индексация**
5. **Сохранение в память**

## Примеры использования

### Обработка отдельного файла

```python
event = pipeline.ingest_document("document.pdf")
```

### Обработка Git репозитория

```python
events = pipeline.process_git_repo("https://github.com/user/repo.git")
```

### Пакетная обработка

```python
events = pipeline.batch_process(["doc1.pdf", "doc2.md", "doc3.txt"])
```

## Заключение

IngestPipeline обеспечивает полную обработку документов с интеграцией в систему памяти Rebecca Platform.
""")
    
    return str(md_path)


def cleanup_test_files(file_paths: list):
    """Удаляет тестовые файлы."""
    for file_path in file_paths:
        try:
            path = Path(file_path)
            if path.exists():
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    import shutil
                    shutil.rmtree(path)
        except Exception as e:
            print(f"Не удалось удалить {file_path}: {e}")


if __name__ == "__main__":
    asyncio.run(main())