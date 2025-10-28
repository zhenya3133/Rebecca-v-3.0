#!/usr/bin/env python3
"""Простая демонстрация использования PDF процессора."""

import sys
from pathlib import Path

# Добавляем путь к src
sys.path.append(str(Path(__file__).parent.parent / "src"))

def demo_basic_usage():
    """Демонстрация базового использования PDF процессора."""
    print("🎯 ДЕМОНСТРАЦИЯ PDF PROCESSOR")
    print("=" * 50)
    
    try:
        from ingest.pdf_processor import PDFProcessor, PDFType, PDFMetadata, ExtractionResult, ProgressCallback
        
        print("✓ Импорт успешен!")
        
        # Создаем процессор
        processor = PDFProcessor(ocr_enabled=False)  # OCR отключен для демо
        print("✓ PDF Processor создан")
        
        # Демонстрируем структуры данных
        metadata = PDFMetadata(
            title="Демо документ",
            author="PDF Processor",
            page_count=10
        )
        print(f"✓ PDFMetadata создан: {metadata.title}")
        
        # Создаем результат извлечения
        result = ExtractionResult(
            success=True,
            data="Демо текст",
            pdf_type=PDFType.TEXT_BASED
        )
        print(f"✓ ExtractionResult создан: {result.success}")
        
        # Демонстрируем progress callback
        progress = ProgressCallback()
        progress.update(5, 10, "Демо прогресс")
        print(f"✓ Progress callback: {progress.current_page}/{progress.total_pages}")
        
        print("\n🎉 Демонстрация завершена успешно!")
        print("\n📋 Доступные методы PDFProcessor:")
        print("  - extract_text(pdf_path)")
        print("  - extract_images(pdf_path)")
        print("  - extract_tables(pdf_path)")
        print("  - extract_metadata(pdf_path)")
        print("  - extract_page_count(pdf_path)")
        print("  - detect_language(text)")
        print("  - detect_pdf_type(pdf_path)")
        print("  - process_full_pdf(pdf_path)")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

if __name__ == "__main__":
    success = demo_basic_usage()
    sys.exit(0 if success else 1)