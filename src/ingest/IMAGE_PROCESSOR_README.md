# ImageProcessor - OCR и обработка изображений

Модуль `image_processor.py` предоставляет полнофункциональные возможности для обработки изображений с поддержкой оптического распознавания символов (OCR) на основе pytesseract.

## Возможности

### Основные функции
- **Извлечение текста** из изображений различных форматов (JPG, PNG, GIF, BMP, TIFF, WebP)
- **Многоязычный OCR** с поддержкой русского, английского и других языков
- **Предобработка изображений** для улучшения качества распознавания
- **Определение языка** извлеченного текста
- **Извлечение таблиц** из изображений документов
- **Обнаружение лиц** на фотографиях
- **Batch обработка** множественных изображений
- **Интеграция с MemoryManager** для сохранения результатов

### Поддерживаемые форматы
- JPG/JPEG
- PNG
- GIF
- BMP
- TIFF
- WebP

### Поддерживаемые языки OCR
- Русский (rus)
- Английский (eng)
- Немецкий (de)
- Французский (fr)
- Испанский (es)
- Итальянский (it)
- Португальский (pt)
- Голландский (nl)
- Польский (pl)
- Украинский (uk)
- Белорусский (be)
- Китайский (zh)
- Японский (ja)
- Корейский (ko)

## Установка зависимостей

### Системные зависимости

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng
sudo apt-get install libgl1-mesa-glx  # для OpenCV
```

**macOS:**
```bash
brew install tesseract
brew install tesseract-lang  # для дополнительных языков
```

**Windows:**
1. Скачайте и установите Tesseract с [официального сайта](https://github.com/UB-Mannheim/tesseract/wiki)
2. Добавьте путь к tesseract.exe в переменную окружения PATH
3. Скачайте языковые данные и поместите их в папку tessdata

### Python зависимости

```bash
pip install -r src/ingest/requirements_ocr.txt
```

Или установите вручную:
```bash
pip install Pillow opencv-python pytesseract face-recognition langdetect tabula-py scikit-image scipy pdf2image tqdm exifread
```

## Использование

### Базовое использование

```python
from src.ingest.image_processor import ImageProcessor

# Создание экземпляра процессора
processor = ImageProcessor()

# Извлечение текста из изображения
result = processor.extract_text("document.jpg", language='rus+eng')
print(f"Извлеченный текст: {result.text}")
print(f"Уверенность: {result.confidence}")
print(f"Язык: {result.language}")
```

### Получение информации об изображении

```python
# Получение метаданных изображения
info = processor.get_image_info("photo.png")
print(f"Формат: {info.format}")
print(f"Размер: {info.size}")
print(f"Размер файла: {info.file_size}")
print(f"Контрольная сумма: {info.checksum}")
```

### Предобработка изображений

```python
# Улучшение качества изображения для лучшего OCR
processed_path = processor.preprocess_image(
    "low_quality.jpg",
    enhance_contrast=True,
    enhance_brightness=True,
    denoise=True,
    grayscale=True,
    output_path="improved.jpg"
)
```

### Извлечение таблиц

```python
# Извлечение таблиц из документа
table_result = processor.extract_tables_from_image("table_document.png")
print(f"Найдено строк: {len(table_result.rows)}")
for row in table_result.rows:
    print(row)
```

### Обнаружение лиц

```python
# Поиск лиц на фотографии
face_result = processor.extract_faces("people_photo.jpg")
print(f"Обнаружено лиц: {face_result.count}")
for bbox in face_result.bounding_boxes:
    print(f"Координаты: {bbox}")
```

### Batch обработка

```python
# Обработка нескольких изображений одновременно
image_paths = ["doc1.jpg", "doc2.png", "doc3.jpg"]
batch_result = processor.batch_process_images(
    image_paths,
    operations=['extract_text', 'extract_tables', 'extract_faces'],
    output_dir="processed"
)

print(f"Обработано: {batch_result.successful}/{batch_result.total_processed}")
```

### Интеграция с MemoryManager

```python
from src.memory_manager.memory_manager import MemoryManager

# Создание с интеграцией MemoryManager
memory_manager = MemoryManager()
processor = ImageProcessor(memory_manager=memory_manager)

# Результаты автоматически сохраняются в MemoryManager
result = processor.extract_text("document.jpg")
```

### Быстрые функции

```python
from src.ingest.image_processor import quick_ocr, quick_batch_ocr

# Быстрое извлечение текста
text = quick_ocr("document.jpg", language='rus+eng')

# Быстрая batch обработка
results = quick_batch_ocr(["doc1.jpg", "doc2.png"], language='eng')
```

## API Reference

### Классы

#### ImageProcessor

Основной класс для обработки изображений.

**Конструктор:**
```python
ImageProcessor(memory_manager=None, tesseract_config=None)
```

**Основные методы:**

- `get_image_info(image_path) -> ImageInfo`
- `extract_text(image_path, language='rus+eng', preprocessing=True, handwriting=False) -> OCRResult`
- `preprocess_image(image_path, **kwargs) -> str`
- `detect_language(text) -> str`
- `extract_tables_from_image(image_path) -> TableDetection`
- `extract_faces(image_path) -> FaceDetection`
- `batch_process_images(image_paths, operations, output_dir=None) -> BatchResult`

#### Модели данных

**ImageInfo**: Информация об изображении
**OCRResult**: Результат OCR операции
**TableDetection**: Результат извлечения таблицы
**FaceDetection**: Результат обнаружения лиц
**BatchResult**: Результат batch обработки

### Функции-помощники

- `create_image_processor(memory_manager=None, tesseract_config=None) -> ImageProcessor`
- `quick_ocr(image_path, language='rus+eng') -> str`
- `quick_batch_ocr(image_paths, language='rus+eng') -> Dict[str, str]`

## Конфигурация

### Настройка Tesseract

Можно передать кастомную конфигурацию Tesseract:

```python
processor = ImageProcessor(tesseract_config='--oem 3 --psm 6 --user-patterns custom_patterns.txt')
```

**Полезные параметры конфигурации:**
- `--oem 0`: Legacy engine only
- `--oem 1`: Neural nets LSTM engine only
- `--oem 2`: Legacy + LSTM engines
- `--oem 3`: Default, based on what is available
- `--psm 0`: Orientation and script detection (OSD) only
- `--psm 1`: Automatic page segmentation with OSD
- `--psm 3`: Fully automatic page segmentation, but no OSD
- `--psm 6`: Assume a single uniform block of text
- `--psm 8`: Treat the image as a single text line

### Настройка предобработки

```python
# Извлечение текста с улучшенным предобработкой
result = processor.extract_text(
    "image.jpg", 
    preprocessing=True
)

# Ручная предобработка с кастомными параметрами
processed_path = processor.preprocess_image(
    "image.jpg",
    enhance_contrast=True,
    enhance_brightness=True,
    denoise=True,
    grayscale=True
)
```

## Обработка ошибок

### Fallback режимы

Если необходимые зависимости не установлены, ImageProcessor автоматически переключается в mock режим:

```python
processor = ImageProcessor()
print(f"Режим работы: {'Mock' if processor.use_mock else 'Реальный OCR'}")
```

### Обработка ошибок

```python
try:
    result = processor.extract_text("document.jpg")
except FileNotFoundError:
    print("Файл не найден")
except ValueError as e:
    print(f"Недопустимое значение: {e}")
except Exception as e:
    print(f"Общая ошибка: {e}")
```

## Тестирование

Запуск unit тестов:

```bash
# Из корневой директории проекта
pytest src/ingest/test_image_processor.py -v

# С покрытием кода
pytest src/ingest/test_image_processor.py --cov=src.ingest.image_processor --cov-report=html

# Только определенный тест
pytest src/ingest/test_image_processor.py::TestImageProcessor::test_extract_text -v
```

## Примеры

Запуск примеров использования:

```bash
python src/ingest/image_processor_examples.py
```

Это создаст тестовые изображения и продемонстрирует все возможности ImageProcessor.

## Производительность

### Оптимизация скорости

1. **Batch обработка**: Используйте `batch_process_images()` для обработки множественных изображений
2. **Предобработка**: Включайте предобработку только при необходимости
3. **Языки**: Указывайте конкретные языки вместо 'rus+eng' для ускорения
4. **Кэширование**: Результаты сохраняются в MemoryManager для повторного использования

### Мониторинг производительности

```python
result = processor.extract_text("document.jpg")
print(f"Время обработки: {result.processing_time:.2f} сек")
print(f"Примененные методы: {result.preprocessing_applied}")
```

## Ограничения

- Качество распознавания зависит от качества исходного изображения
- Для сложных документов может потребоваться ручная предобработка
- Rешение капчи и защищенного текста не поддерживается
- OCR не заменяет человеческое чтение для критически важных задач

## Расширение функциональности

### Добавление новых языков

1. Установите языковые данные Tesseract
2. Добавьте язык в `SUPPORTED_LANGUAGES`
3. Обновите документацию

### Интеграция с другими OCR движками

```python
# Пример интеграции с EasyOCR
def extract_text_easyocr(self, image_path, language='ru'):
    try:
        import easyocr
        reader = easyocr.Reader([language])
        result = reader.readtext(image_path)
        text = ' '.join([detection[1] for detection in result])
        return OCRResult(text=text, confidence=0.9, language=language, method='easyocr')
    except ImportError:
        # Fallback к pytesseract
        return self.extract_text(image_path, language)
```

## Лицензия

Модуль является частью проекта Rebecca Platform и распространяется под той же лицензией.

## Поддержка

Для получения поддержки и сообщения об ошибках:

1. Проверьте документацию и примеры
2. Убедитесь, что все зависимости установлены корректно
3. Проверьте логи на наличие ошибок
4. Создайте issue с подробным описанием проблемы

## Changelog

### Версия 1.0.0
- Базовая функциональность OCR
- Поддержка множественных форматов изображений
- Batch обработка
- Интеграция с MemoryManager
- Mock режим для тестирования
- Unit тесты и документация