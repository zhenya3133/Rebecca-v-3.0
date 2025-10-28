"""Примеры использования PDFProcessor с различными типами PDF файлов."""

import os
import logging
from pathlib import Path

# Импортируем наш процессор
import sys
sys.path.append(str(Path(__file__).parent.parent))

from pdf_processor import PDFProcessor, ProgressCallback


class PDFProcessorDemo:
    """Демонстрация возможностей PDF процессора."""
    
    def __init__(self):
        # Настройка логирования
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Создаем процессор
        self.processor = PDFProcessor(ocr_enabled=True, lang="rus+eng")
    
    def demo_basic_usage(self, pdf_path: str):
        """Основное использование - извлечение текста."""
        print(f"=== Демо: Основное использование ===")
        print(f"Обработка файла: {pdf_path}")
        
        # Извлекаем текст
        result = self.processor.extract_text(pdf_path)
        
        if result.success:
            print(f"✓ Текст извлечен успешно")
            print(f"Тип PDF: {result.pdf_type.value}")
            print(f"Первые 200 символов текста:")
            print(result.data[:200] + "..." if len(result.data) > 200 else result.data)
        else:
            print(f"✗ Ошибка: {result.error}")
        
        return result
    
    def demo_with_progress(self, pdf_path: str):
        """Использование с отслеживанием прогресса."""
        print(f"\\n=== Демо: Отслеживание прогресса ===")
        
        class CustomProgressCallback(ProgressCallback):
            def update(self, current: int, total: int, message: str = ""):
                """Кастомное обновление прогресса с панелью прогресса."""
                if total > 0:
                    percent = (current / total) * 100
                    bar_length = 30
                    filled_length = int(bar_length * current // total)
                    bar = '█' * filled_length + '░' * (bar_length - filled_length)
                    print(f'\\r|{bar}| {percent:6.1f}% - {message}', end='', flush=True)
                    if current == total:
                        print()  # Новая строка в конце
        
        progress_callback = CustomProgressCallback()
        
        result = self.processor.extract_text(pdf_path, progress_callback)
        
        if result.success:
            print(f"✓ Обработка завершена!")
            print(f"Извлечено символов: {len(result.data)}")
        else:
            print(f"\\n✗ Ошибка: {result.error}")
        
        return result
    
    def demo_metadata_extraction(self, pdf_path: str):
        """Извлечение метаданных PDF."""
        print(f"\\n=== Демо: Извлечение метаданных ===")
        
        result = self.processor.extract_metadata(pdf_path)
        
        if result.success:
            metadata = result.data
            print(f"✓ Метаданные извлечены:")
            print(f"  Название: {metadata.title or 'Не указано'}")
            print(f"  Автор: {metadata.author or 'Не указан'}")
            print(f"  Тема: {metadata.subject or 'Не указана'}")
            print(f"  Создатель: {metadata.creator or 'Не указан'}")
            print(f"  Количество страниц: {metadata.page_count}")
            print(f"  Размер файла: {metadata.file_size or 'Неизвестен'} байт")
            print(f"  Контрольная сумма: {metadata.checksum[:16]}...")
        else:
            print(f"✗ Ошибка: {result.error}")
        
        return result
    
    def demo_table_extraction(self, pdf_path: str):
        """Извлечение таблиц из PDF."""
        print(f"\\n=== Демо: Извлечение таблиц ===")
        
        result = self.processor.extract_tables(pdf_path)
        
        if result.success:
            tables = result.data
            print(f"✓ Найдено таблиц: {len(tables)}")
            
            for i, table in enumerate(tables[:3]):  # Показываем первые 3 таблицы
                print(f"\\nТаблица {i+1}:")
                print(f"  Страница: {table['page']}")
                print(f"  Метод: {table['method']}")
                print(f"  Размер данных: {len(table['data'])} строк")
                
                # Показываем первые несколько строк
                print(f"  Первые строки:")
                for idx, row in table['data'].head(3).iterrows():
                    print(f"    {row.to_dict()}")
                    
        else:
            print(f"✗ Ошибка: {result.error}")
        
        return result
    
    def demo_image_extraction(self, pdf_path: str, output_dir: str = None):
        """Извлечение изображений из PDF."""
        print(f"\\n=== Демо: Извлечение изображений ===")
        
        if output_dir is None:
            output_dir = f"{os.path.splitext(pdf_path)[0]}_images"
        
        result = self.processor.extract_images(pdf_path, output_dir)
        
        if result.success:
            images = result.data
            print(f"✓ Извлечено изображений: {len(images)}")
            print(f"Папка с изображениями: {output_dir}")
            
            for img_path in images[:5]:  # Показываем первые 5
                print(f"  {img_path}")
        else:
            print(f"✗ Ошибка: {result.error}")
        
        return result
    
    def demo_language_detection(self, pdf_path: str):
        """Определение языка текста в PDF."""
        print(f"\\n=== Демо: Определение языка ===")
        
        # Сначала извлекаем текст
        text_result = self.processor.extract_text(pdf_path)
        
        if text_result.success:
            # Определяем язык
            lang_result = self.processor.detect_language(text_result.data)
            
            if lang_result.success:
                lang_data = lang_result.data
                print(f"✓ Язык определен: {lang_data['language']}")
                print(f"Уровень уверенности: {lang_data['confidence_scores'][0]}")
            else:
                print(f"✗ Ошибка определения языка: {lang_result.error}")
        else:
            print(f"✗ Не удалось извлечь текст для определения языка: {text_result.error}")
        
        return text_result
    
    def demo_full_processing(self, pdf_path: str, output_dir: str = None):
        """Полная обработка PDF со всеми возможностями."""
        print(f"\\n=== Демо: Полная обработка PDF ===")
        
        if output_dir is None:
            output_dir = f"{os.path.splitext(pdf_path)[0]}_extracted"
        
        class DemoProgressCallback(ProgressCallback):
            def update(self, current: int, total: int, message: str = ""):
                """Прогресс для демо."""
                if total > 0:
                    percent = (current / total) * 100
                    print(f"\\rПрогресс: {percent:6.1f}% - {message}", end='', flush=True)
                    if current == total:
                        print()  # Новая строка в конце
        
        progress_callback = DemoProgressCallback()
        
        # Запускаем полную обработку
        results = self.processor.process_full_pdf(
            pdf_path,
            output_dir=output_dir,
            extract_images=True,
            extract_tables=True,
            progress_callback=progress_callback
        )
        
        # Анализируем результаты
        print(f"\\n✓ Полная обработка завершена!")
        print(f"\\nРезультаты:")
        
        for key, result in results.items():
            if hasattr(result, 'success'):
                if result.success:
                    print(f"  ✓ {key}: Успешно")
                    if key == 'metadata' and result.data:
                        print(f"    Страниц: {result.data.page_count}")
                    elif key == 'text' and result.data:
                        print(f"    Символов текста: {len(result.data)}")
                    elif key == 'language' and result.data:
                        print(f"    Язык: {result.data.get('language', 'неизвестен')}")
                    elif key == 'tables' and result.data:
                        print(f"    Таблиц: {len(result.data)}")
                    elif key == 'images' and result.data:
                        print(f"    Изображений: {len(result.data)}")
                else:
                    print(f"  ✗ {key}: {result.error}")
        
        return results
    
    def demo_pdf_type_detection(self, pdf_path: str):
        """Демонстрация определения типа PDF."""
        print(f"\\n=== Демо: Определение типа PDF ===")
        
        # Определяем тип
        pdf_type = self.processor.detect_pdf_type(pdf_path)
        
        print(f"Тип PDF: {pdf_type.value}")
        
        type_descriptions = {
            PDFType.TEXT_BASED: "Текстовый PDF - содержит извлекаемый текст",
            PDFType.SCANNED: "Сканированный PDF - только изображения, требует OCR",
            PDFType.MIXED: "Смешанный PDF - содержит и текст, и изображения",
            PDFType.UNKNOWN: "Неопределенный тип"
        }
        
        print(f"Описание: {type_descriptions.get(pdf_type, 'Неизвестно')}")
        
        # В зависимости от типа предлагаем разные подходы
        if pdf_type == PDFType.TEXT_BASED:
            print("Рекомендация: Используйте прямое извлечение текста")
        elif pdf_type == PDFType.SCANNED:
            print("Рекомендация: Используйте OCR для извлечения текста")
        elif pdf_type == PDFType.MIXED:
            print("Рекомендация: Попробуйте оба метода извлечения")
        
        return pdf_type
    
    def run_comprehensive_demo(self, pdf_path: str):
        """Запуск полной демонстрации всех возможностей."""
        print("🎯 ДЕМОНСТРАЦИЯ ВОЗМОЖНОСТЕЙ PDF PROCESSOR")
        print("=" * 60)
        
        if not os.path.exists(pdf_path):
            print(f"❌ Файл не найден: {pdf_path}")
            return
        
        try:
            # 1. Определение типа PDF
            self.demo_pdf_type_detection(pdf_path)
            
            # 2. Извлечение метаданных
            self.demo_metadata_extraction(pdf_path)
            
            # 3. Извлечение текста с прогрессом
            self.demo_with_progress(pdf_path)
            
            # 4. Определение языка
            self.demo_language_detection(pdf_path)
            
            # 5. Извлечение таблиц
            self.demo_table_extraction(pdf_path)
            
            # 6. Извлечение изображений
            images_dir = f"{os.path.splitext(pdf_path)[0]}_demo_images"
            self.demo_image_extraction(pdf_path, images_dir)
            
            # 7. Полная обработка
            full_dir = f"{os.path.splitext(pdf_path)[0]}_full_demo"
            self.demo_full_processing(pdf_path, full_dir)
            
            print(f"\\n🎉 Демонстрация завершена!")
            print(f"📁 Проверьте созданные директории с результатами")
            
        except Exception as e:
            print(f"❌ Ошибка во время демонстрации: {e}")


def main():
    """Основная функция для запуска демонстрации."""
    demo = PDFProcessorDemo()
    
    # Пример использования
    pdf_path = "/path/to/your/document.pdf"  # Замените на путь к вашему PDF
    
    if os.path.exists(pdf_path):
        # Запускаем полную демонстрацию
        demo.run_comprehensive_demo(pdf_path)
    else:
        print("💡 Для запуска демонстрации:")
        print("1. Установите необходимые зависимости:")
        print("   pip install pdfplumber PyPDF2 pdf2image pytesseract camelot-py pandas pillow")
        print("2. Установите tesseract-ocr")
        print("3. Укажите путь к вашему PDF файлу в переменной pdf_path")
        print("\\n📋 Пример простого использования:")
        
        # Простой пример
        processor = PDFProcessor(ocr_enabled=True)
        print(f"\\nprocessor = PDFProcessor(ocr_enabled=True)")
        print(f"result = processor.extract_text('{pdf_path}')")
        print(f"if result.success:")
        print(f"    print('Текст извлечен:', len(result.data), 'символов')")


if __name__ == "__main__":
    main()


# Дополнительные примеры использования:

def example_batch_processing():
    """Пример пакетной обработки нескольких PDF файлов."""
    pdf_files = [
        "/path/to/doc1.pdf",
        "/path/to/doc2.pdf",
        "/path/to/doc3.pdf"
    ]
    
    processor = PDFProcessor(ocr_enabled=True)
    
    results = []
    for pdf_path in pdf_files:
        if os.path.exists(pdf_path):
            print(f"Обработка: {pdf_path}")
            
            # Быстрое извлечение основной информации
            metadata_result = processor.extract_metadata(pdf_path)
            text_result = processor.extract_text(pdf_path)
            
            results.append({
                'file': pdf_path,
                'metadata': metadata_result,
                'text': text_result,
                'word_count': len(text_result.data.split()) if text_result.success else 0
            })
    
    # Анализ результатов
    total_words = sum(r['word_count'] for r in results)
    print(f"Общее количество слов во всех документах: {total_words}")


def example_conditional_processing():
    """Пример условной обработки в зависимости от типа PDF."""
    pdf_path = "/path/to/document.pdf"
    
    processor = PDFProcessor(ocr_enabled=True)
    
    # Определяем тип PDF
    pdf_type = processor.detect_pdf_type(pdf_path)
    
    if pdf_type == PDFType.TEXT_BASED:
        # Для текстовых PDF - только прямое извлечение
        text_result = processor.extract_text(pdf_path)
        print(f"Извлечено символов: {len(text_result.data)}")
        
    elif pdf_type == PDFType.SCANNED:
        # Для сканированных PDF - OCR + извлечение изображений
        text_result = processor.extract_text(pdf_path)
        images_result = processor.extract_images(pdf_path)
        print(f"OCR текст: {len(text_result.data)} символов")
        print(f"Изображений: {len(images_result.data)}")
        
    elif pdf_type == PDFType.MIXED:
        # Для смешанных PDF - полная обработка
        results = processor.process_full_pdf(pdf_path)
        print(f"Обработка завершена, тип: {results['metadata'].data.page_count} страниц")


def example_custom_configuration():
    """Пример кастомной конфигурации процессора."""
    # Создаем процессор с настройками для конкретной задачи
    processor = PDFProcessor(
        ocr_enabled=True,
        lang="rus"  # Только русский язык
    )
    
    # Настройка для обработки с приоритетом качества
    def quality_focused_callback(ProgressCallback):
        def update(self, current, total, message=""):
            # Детальное логирование для отладки
            if total > 0:
                percent = (current / total) * 100
                logging.info(f"Качество: {percent:.1f}% - {message}")
    
    pdf_path = "/path/to/high_quality_document.pdf"
    
    # Обработка с акцентом на качество
    results = processor.process_full_pdf(
        pdf_path,
        output_dir="/path/to/output",
        extract_images=True,
        extract_tables=True,
        progress_callback=quality_focused_callback()
    )
    
    print("Обработка с фокусом на качество завершена")