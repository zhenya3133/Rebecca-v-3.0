#!/usr/bin/env python3
"""Простая демонстрация использования PDF процессора без импорта ingest."""

import sys
from pathlib import Path
import importlib.util

def demo_pdf_processor():
    """Демонстрация основных возможностей PDF процессора."""
    print("🎯 ДЕМОНСТРАЦИЯ PDF PROCESSOR")
    print("=" * 50)
    
    try:
        # Загружаем модуль напрямую
        pdf_module_path = Path(__file__).parent.parent / "src" / "ingest" / "pdf_processor.py"
        spec = importlib.util.spec_from_file_location("pdf_processor", pdf_module_path)
        pdf_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pdf_module)
        
        print("✓ Модуль pdf_processor загружен успешно!")
        
        # Демонстрируем основные классы и функции
        PDFProcessor = pdf_module.PDFProcessor
        PDFType = pdf_module.PDFType
        PDFMetadata = pdf_module.PDFMetadata
        ExtractionResult = pdf_module.ExtractionResult
        ProgressCallback = pdf_module.ProgressCallback
        
        print("✓ Все классы доступны:")
        
        # Создаем экземпляр процессора
        processor = PDFProcessor(ocr_enabled=False)
        print(f"  - PDFProcessor создан (OCR: {processor.ocr_enabled})")
        
        # Демонстрируем типы PDF
        print(f"\n📄 Типы PDF:")
        for pdf_type in PDFType:
            print(f"  - {pdf_type.value}: {pdf_type.name}")
        
        # Создаем метаданные
        metadata = PDFMetadata(
            title="Демо документ",
            author="PDF Processor Demo",
            subject="Тестовая документация",
            page_count=42,
            file_size=1024000
        )
        print(f"\n📊 Метаданные созданы:")
        print(f"  - Заголовок: {metadata.title}")
        print(f"  - Автор: {metadata.author}")
        print(f"  - Страниц: {metadata.page_count}")
        print(f"  - Размер: {metadata.file_size} байт")
        
        # Создаем результат извлечения
        result = ExtractionResult(
            success=True,
            data="Пример извлеченного текста",
            pdf_type=PDFType.TEXT_BASED
        )
        print(f"\n✅ Результат извлечения:")
        print(f"  - Успех: {result.success}")
        print(f"  - Тип PDF: {result.pdf_type.value}")
        print(f"  - Данные: {len(result.data)} символов")
        
        # Демонстрируем progress tracking
        progress = ProgressCallback()
        progress.update(50, 100, "Демонстрация прогресса")
        print(f"\n📈 Отслеживание прогресса:")
        print(f"  - Текущий: {progress.current_page}")
        print(f"  - Всего: {progress.total_pages}")
        
        # Показываем доступные методы
        print(f"\n🔧 Доступные методы PDFProcessor:")
        methods = [
            'extract_text', 'extract_images', 'extract_tables',
            'extract_metadata', 'extract_page_count', 'detect_language',
            'detect_pdf_type', 'process_full_pdf'
        ]
        for method in methods:
            print(f"  - {method}()")
        
        print(f"\n🎉 Демонстрация завершена успешно!")
        print(f"\n📚 Для полной документации см:")
        print(f"  - docs/pdf_processor_README.md")
        print(f"  - docs/pdf_processor_examples.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = demo_pdf_processor()
    sys.exit(0 if success else 1)