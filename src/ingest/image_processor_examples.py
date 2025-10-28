"""
Примеры использования ImageProcessor для OCR и обработки изображений.

Этот файл содержит практические примеры того, как использовать
класс ImageProcessor для различных задач обработки изображений.
"""

import os
from pathlib import Path
from src.ingest.image_processor import ImageProcessor, create_image_processor, quick_ocr

# Создание экземпляра процессора
def example_basic_usage():
    """Базовое использование ImageProcessor."""
    print("=== Базовое использование ImageProcessor ===")
    
    # Создаем процессор
    processor = ImageProcessor()
    
    # Пример пути к изображению (замените на реальный)
    image_path = "sample_document.jpg"
    
    try:
        # Получаем информацию об изображении
        info = processor.get_image_info(image_path)
        print(f"Информация об изображении:")
        print(f"  Путь: {info.path}")
        print(f"  Формат: {info.format}")
        print(f"  Размер: {info.size}")
        print(f"  Размер файла: {info.file_size} байт")
        print(f"  Контрольная сумма: {info.checksum}")
        
        # Извлекаем текст
        ocr_result = processor.extract_text(image_path, language='rus+eng')
        print(f"\nИзвлеченный текст:")
        print(f"  Текст: {ocr_result.text}")
        print(f"  Уверенность: {ocr_result.confidence}")
        print(f"  Язык: {ocr_result.language}")
        print(f"  Время обработки: {ocr_result.processing_time:.2f} сек")
        
        # Определяем язык извлеченного текста
        detected_lang = processor.detect_language(ocr_result.text)
        print(f"  Определенный язык: {detected_lang}")
        
    except FileNotFoundError:
        print(f"Файл {image_path} не найден. Создайте тестовое изображение.")
    except Exception as e:
        print(f"Ошибка: {e}")


def example_preprocessing():
    """Пример предобработки изображений для улучшения OCR."""
    print("\n=== Предобработка изображений ===")
    
    processor = ImageProcessor()
    image_path = "low_quality_document.jpg"
    
    try:
        # Предобработка изображения
        processed_path = processor.preprocess_image(
            image_path=image_path,
            enhance_contrast=True,
            enhance_brightness=True,
            denoise=True,
            grayscale=True
        )
        
        print(f"Исходное изображение: {image_path}")
        print(f"Обработанное изображение: {processed_path}")
        
        # Сравниваем результаты OCR до и после предобработки
        original_result = processor.extract_text(image_path, preprocessing=False)
        processed_result = processor.extract_text(processed_path, preprocessing=False)
        
        print(f"\nРезультаты OCR:")
        print(f"  До предобработки: {len(original_result.text)} символов")
        print(f"  После предобработки: {len(processed_result.text)} символов")
        print(f"  Примененные методы: {processed_result.preprocessing_applied}")
        
    except FileNotFoundError:
        print(f"Файл {image_path} не найден.")
    except Exception as e:
        print(f"Ошибка: {e}")


def example_table_extraction():
    """Пример извлечения таблиц из изображений."""
    print("\n=== Извлечение таблиц ===")
    
    processor = ImageProcessor()
    image_path = "table_document.png"
    
    try:
        # Извлекаем таблицу
        table_result = processor.extract_tables_from_image(image_path)
        
        print(f"Извлеченная таблица:")
        print(f"  Количество строк: {len(table_result.rows)}")
        print(f"  Уверенность: {table_result.confidence}")
        print(f"  Метод: {table_result.method}")
        
        # Выводим содержимое таблицы
        for i, row in enumerate(table_result.rows):
            print(f"  Строка {i + 1}: {row}")
            
    except FileNotFoundError:
        print(f"Файл {image_path} не найден.")
    except Exception as e:
        print(f"Ошибка: {e}")


def example_face_detection():
    """Пример обнаружения лиц на изображении."""
    print("\n=== Обнаружение лиц ===")
    
    processor = ImageProcessor()
    image_path = "people_photo.jpg"
    
    try:
        # Обнаруживаем лица
        face_result = processor.extract_faces(image_path)
        
        print(f"Обнаруженные лица:")
        print(f"  Количество: {face_result.count}")
        print(f"  Метод: {face_result.method}")
        
        # Выводим координаты bounding boxes
        for i, bbox in enumerate(face_result.bounding_boxes):
            print(f"  Лицо {i + 1}: {bbox}")
            
        if face_result.encodings:
            print(f"  Количество кодировок: {len(face_result.encodings)}")
            
    except FileNotFoundError:
        print(f"Файл {image_path} не найден.")
    except Exception as e:
        print(f"Ошибка: {e}")


def example_batch_processing():
    """Пример batch обработки нескольких изображений."""
    print("\n=== Batch обработка изображений ===")
    
    processor = ImageProcessor()
    
    # Список изображений для обработки
    image_paths = [
        "document1.jpg",
        "document2.png", 
        "document3.png"
    ]
    
    # Проверяем существование файлов
    existing_paths = [path for path in image_paths if os.path.exists(path)]
    
    if not existing_paths:
        print("Ни одного изображения не найдено. Создайте тестовые файлы.")
        return
    
    # Batch обработка
    batch_result = processor.batch_process_images(
        image_paths=existing_paths,
        operations=['extract_text', 'extract_tables', 'extract_faces', 'get_info'],
        output_dir="processed_images"
    )
    
    print(f"Результаты batch обработки:")
    print(f"  Всего обработано: {batch_result.total_processed}")
    print(f"  Успешно: {batch_result.successful}")
    print(f"  С ошибками: {batch_result.failed}")
    
    # Выводим результаты для каждого изображения
    for image_path, results in batch_result.results.items():
        print(f"\n  Файл: {os.path.basename(image_path)}")
        
        if 'get_info' in results:
            info = results['get_info']
            print(f"    Информация: {info.get('format', 'N/A')} {info.get('size', 'N/A')}")
        
        if 'extract_text' in results:
            text_result = results['extract_text']
            print(f"    Текст: {len(text_result.get('text', ''))} символов")
        
        if 'extract_tables' in results:
            table_result = results['extract_tables']
            print(f"    Таблица: {len(table_result.get('rows', []))} строк")
        
        if 'extract_faces' in results:
            face_result = results['extract_faces']
            print(f"    Лица: {face_result.get('count', 0)} обнаружено")
    
    # Выводим ошибки, если есть
    if batch_result.errors:
        print(f"\n  Ошибки:")
        for image_path, error in batch_result.errors.items():
            print(f"    {os.path.basename(image_path)}: {error}")


def example_memory_integration():
    """Пример интеграции с MemoryManager."""
    print("\n=== Интеграция с MemoryManager ===")
    
    # Создаем mock MemoryManager для примера
    class MockMemoryManager:
        def __init__(self):
            self.storage = {}
        
        class MockVault:
            def store_secret(self, key, value):
                print(f"Сохранено в MemoryManager: {key}")
        
        def __init__(self):
            self.vault = self.MockVault()
    
    memory_manager = MockMemoryManager()
    processor = ImageProcessor(memory_manager=memory_manager)
    
    image_path = "test_document.jpg"
    
    try:
        # Обработка с сохранением в MemoryManager
        batch_result = processor.batch_process_images(
            [image_path], 
            operations=['extract_text', 'get_info']
        )
        
        print(f"Результаты обработки сохранены в MemoryManager")
        
    except FileNotFoundError:
        print(f"Файл {image_path} не найден.")
    except Exception as e:
        print(f"Ошибка: {e}")


def example_quick_functions():
    """Пример использования быстрых функций."""
    print("\n=== Быстрые функции ===")
    
    # Быстрое извлечение текста из одного изображения
    image_path = "quick_test.jpg"
    
    try:
        text = quick_ocr(image_path, language='rus+eng')
        print(f"Быстрое OCR: {text[:100]}..." if len(text) > 100 else f"Быстрое OCR: {text}")
        
    except FileNotFoundError:
        print(f"Файл {image_path} не найден.")
    except Exception as e:
        print(f"Ошибка: {e}")
    
    # Быстрое batch OCR
    image_paths = ["doc1.jpg", "doc2.png"]
    existing_paths = [path for path in image_paths if os.path.exists(path)]
    
    if existing_paths:
        try:
            results = quick_batch_ocr(existing_paths, language='eng')
            print(f"\nБыстрый batch OCR:")
            for path, text in results.items():
                print(f"  {os.path.basename(path)}: {len(text)} символов")
        except Exception as e:
            print(f"Ошибка batch OCR: {e}")
    else:
        print("Нет изображений для batch OCR")


def example_supported_formats():
    """Пример работы с поддерживаемыми форматами."""
    print("\n=== Поддерживаемые форматы и языки ===")
    
    processor = ImageProcessor()
    
    # Поддерживаемые форматы
    formats = processor.get_supported_formats()
    print(f"Поддерживаемые форматы: {', '.join(formats)}")
    
    # Поддерживаемые языки
    languages = processor.get_supported_languages()
    print(f"\nПоддерживаемые языки OCR:")
    for code, name in languages.items():
        print(f"  {code}: {name}")


def example_error_handling():
    """Пример обработки ошибок и fallback режимов."""
    print("\n=== Обработка ошибок и fallback ===")
    
    processor = ImageProcessor()
    
    # Тест с несуществующим файлом
    non_existent_file = "does_not_exist.jpg"
    
    try:
        result = processor.extract_text(non_existent_file)
        print(f"Результат: {result.text}")
    except FileNotFoundError as e:
        print(f"Файл не найден: {e}")
    except Exception as e:
        print(f"Общая ошибка: {e}")
    
    # Тест с некорректным форматом
    invalid_format_file = "test.xyz"
    with open(invalid_format_file, 'w') as f:
        f.write("test")
    
    try:
        info = processor.get_image_info(invalid_format_file)
        print(f"Информация: {info}")
    except ValueError as e:
        print(f"Неподдерживаемый формат: {e}")
    except Exception as e:
        print(f"Общая ошибка: {e}")
    finally:
        # Очистка тестового файла
        if os.path.exists(invalid_format_file):
            os.remove(invalid_format_file)
    
    # Mock режим при отсутствии зависимостей
    print(f"\nТекущий режим работы: {'Mock' if processor.use_mock else 'Реальный OCR'}")
    print(f"Доступные зависимости:")
    print(f"  PIL: {'Да' if processor._check_dependencies() else 'Нет'}")


def create_test_images():
    """Создание простых тестовых изображений для демонстрации."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Создаем директорию для тестовых изображений
        os.makedirs("test_images", exist_ok=True)
        
        # Простое изображение с текстом
        img = Image.new('RGB', (400, 200), color='white')
        draw = ImageDraw.Draw(img)
        
        # Добавляем текст (используем стандартный шрифт)
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        draw.text((10, 50), "Тестовый документ", fill='black', font=font)
        draw.text((10, 80), "OCR Test Document", fill='black', font=font)
        draw.text((10, 110), "Дата: 2023-01-01", fill='black', font=font)
        
        img.save("test_images/document.jpg", "JPEG")
        
        # Простое изображение с таблицей
        table_img = Image.new('RGB', (300, 150), color='white')
        draw = ImageDraw.Draw(table_img)
        
        # Рисуем простую таблицу
        for row in range(4):
            y = row * 30
            for col in range(3):
                x = col * 100
                draw.rectangle([x, y, x+100, y+30], outline='black')
        
        table_img.save("test_images/table.png", "PNG")
        
        # Изображение с лицом (простой круг)
        face_img = Image.new('RGB', (200, 200), color='lightblue')
        draw = ImageDraw.Draw(face_img)
        draw.ellipse([50, 50, 150, 150], fill='peachpuff', outline='black', width=2)
        
        face_img.save("test_images/face.png", "PNG")
        
        print("Тестовые изображения созданы в директории 'test_images'")
        
    except ImportError:
        print("PIL не установлен. Создание тестовых изображений пропущено.")
    except Exception as e:
        print(f"Ошибка создания тестовых изображений: {e}")


def main():
    """Главная функция с примерами использования."""
    print("=== Демонстрация возможностей ImageProcessor ===\n")
    
    # Создание тестовых изображений
    create_test_images()
    
    # Запуск примеров
    example_supported_formats()
    example_basic_usage()
    example_preprocessing()
    example_table_extraction()
    example_face_detection()
    example_batch_processing()
    example_memory_integration()
    example_quick_functions()
    example_error_handling()
    
    print("\n=== Демонстрация завершена ===")


if __name__ == "__main__":
    main()