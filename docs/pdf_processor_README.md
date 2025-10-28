# PDF Processor - Полноценная обработка PDF файлов

PDF Processor - это мощный инструмент для извлечения данных из PDF файлов различных типов с поддержкой OCR, автоматического определения типа документа и обработкой таблиц.

## 🌟 Основные возможности

- **Извлечение текста** из текстовых и сканированных PDF
- **Автоматическое определение типа PDF** (текстовый/сканированный/смешанный)
- **OCR поддержка** для обработки сканированных документов
- **Извлечение изображений** из PDF файлов
- **Извлечение таблиц** с помощью Camelot и OCR
- **Метаданные PDF** - информация о документе
- **Определение языка** текста
- **Progress tracking** для больших файлов
- **Обработка ошибок** с fallback механизмами

## 📦 Установка

### 1. Установка Python зависимостей

```bash
pip install pdfplumber PyPDF2 pdf2image pytesseract PyMuPDF camelot-py pandas pillow langdetect opencv-python
```

### 2. Установка Tesseract OCR

#### Windows
```bash
# Скачайте tesseract-ocr-w64-setup-v5.3.3.20231005.exe с GitHub
# Установите и добавьте в PATH
```

#### macOS
```bash
brew install tesseract
```

#### Ubuntu/Debian
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-rus
```

### 3. Установка Poppler (для pdf2image)

#### Windows
- Скачайте Poppler для Windows с http://blog.alivate.com.au/poppler-windows/
- Добавьте путь к bin в переменную окружения PATH

#### macOS
```bash
brew install poppler
```

#### Ubuntu/Debian
```bash
sudo apt-get install poppler-utils
```

## 🚀 Быстрый старт

### Простое извлечение текста

```python
from pdf_processor import PDFProcessor

# Создаем процессор
processor = PDFProcessor(ocr_enabled=True)

# Извлекаем текст
result = processor.extract_text("/path/to/document.pdf")

if result.success:
    print(f"Извлечено {len(result.data)} символов текста")
    print(f"Тип PDF: {result.pdf_type.value}")
else:
    print(f"Ошибка: {result.error}")
```

### Извлечение метаданных

```python
metadata_result = processor.extract_metadata("/path/to/document.pdf")

if metadata_result.success:
    metadata = metadata_result.data
    print(f"Название: {metadata.title}")
    print(f"Автор: {metadata.author}")
    print(f"Страниц: {metadata.page_count}")
```

### Полная обработка PDF

```python
from pdf_processor import ProgressCallback

class CustomProgress(ProgressCallback):
    def update(self, current, total, message=""):
        percent = (current / total) * 100 if total > 0 else 0
        print(f"\\r[{percent:6.1f}%] {message}", end="", flush=True)

results = processor.process_full_pdf(
    "/path/to/document.pdf",
    output_dir="/path/to/output",
    extract_images=True,
    extract_tables=True,
    progress_callback=CustomProgress()
)

print("\\nОбработка завершена!")
```

## 📖 Подробное использование

### Классы и структуры данных

#### PDFProcessor
Основной класс для обработки PDF файлов.

```python
processor = PDFProcessor(
    ocr_enabled=True,    # Включить OCR
    lang="rus+eng"      # Языки для OCR
)
```

#### PDFType
Перечисление типов PDF:
- `TEXT_BASED` - текстовый PDF с извлекаемым текстом
- `SCANNED` - сканированный PDF (только изображения)
- `MIXED` - смешанный PDF (текст + изображения)
- `UNKNOWN` - неопределенный тип

#### PDFMetadata
Структура метаданных:
```python
@dataclass
class PDFMetadata:
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    keywords: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    page_count: int = 0
    file_size: Optional[int] = None
    checksum: Optional[str] = None
```

#### ExtractionResult
Результат извлечения данных:
```python
@dataclass
class ExtractionResult:
    success: bool
    data: Any = None
    error: Optional[str] = None
    pdf_type: Optional[PDFType] = None
    metadata: Optional[PDFMetadata] = None
```

### Методы PDFProcessor

#### extract_text(pdf_path, progress_callback=None)
Извлекает текст из PDF с автоматическим определением метода:
- Для текстовых PDF - прямое извлечение
- Для сканированных PDF - OCR обработка
- Для смешанных PDF - комбинированный подход

```python
result = processor.extract_text("/path/to/document.pdf")
print(result.data)  # Извлеченный текст
print(result.pdf_type)  # Тип PDF
```

#### extract_metadata(pdf_path)
Извлекает метаданные PDF документа.

```python
metadata_result = processor.extract_metadata("/path/to/document.pdf")
if metadata_result.success:
    print(f"Страниц: {metadata_result.data.page_count}")
    print(f"Автор: {metadata_result.data.author}")
```

#### detect_pdf_type(pdf_path, sample_pages=3)
Автоматически определяет тип PDF файла.

```python
pdf_type = processor.detect_pdf_type("/path/to/document.pdf")
print(f"Тип: {pdf_type.value}")
```

#### extract_images(pdf_path, output_dir=None, progress_callback=None)
Извлекает изображения из PDF:
- Для сканированных PDF - конвертирует страницы в изображения
- Для текстовых PDF - извлекает встроенные изображения

```python
images_result = processor.extract_images(
    "/path/to/document.pdf", 
    output_dir="/path/to/images"
)

if images_result.success:
    print(f"Извлечено изображений: {len(images_result.data)}")
    for img_path in images_result.data:
        print(img_path)
```

#### extract_tables(pdf_path, progress_callback=None)
Извлекает таблицы из PDF:
- Для текстовых PDF - использует Camelot
- Для сканированных PDF - использует OCR

```python
tables_result = processor.extract_tables("/path/to/document.pdf")

if tables_result.success:
    for i, table in enumerate(tables_result.data):
        print(f"Таблица {i+1}:")
        print(f"  Страница: {table['page']}")
        print(f"  Метод: {table['method']}")
        print(f"  Размер: {len(table['data'])} строк")
        print(table['data'].head())  # Показываем первые строки
```

#### detect_language(text)
Определяет язык текста.

```python
# Сначала извлекаем текст
text_result = processor.extract_text("/path/to/document.pdf")

if text_result.success:
    # Определяем язык
    lang_result = processor.detect_language(text_result.data)
    
    if lang_result.success:
        print(f"Язык: {lang_result.data['language']}")
        print(f"Уверенность: {lang_result.data['confidence_scores']}")
```

#### process_full_pdf()
Полная обработка PDF со всеми возможностями.

```python
results = processor.process_full_pdf(
    pdf_path="/path/to/document.pdf",
    output_dir="/path/to/output",
    extract_images=True,
    extract_tables=True
)

# Результат содержит:
# results['metadata'] - метаданные
# results['text'] - извлеченный текст
# results['language'] - определенный язык
# results['tables'] - извлеченные таблицы
# results['images'] - извлеченные изображения
```

### Progress Tracking

Для отслеживания прогресса используйте класс ProgressCallback:

```python
class MyProgressCallback(ProgressCallback):
    def update(self, current, total, message=""):
        if total > 0:
            percent = (current / total) * 100
            bar_length = 30
            filled_length = int(bar_length * current // total)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f'\\r|{bar}| {percent:6.1f}% - {message}', end='', flush=True)
            if current == total:
                print()

progress_callback = MyProgressCallback()
result = processor.extract_text("/path/to/document.pdf", progress_callback)
```

## 🔧 Примеры использования

### Пакетная обработка PDF файлов

```python
import os
from pdf_processor import PDFProcessor

def process_pdf_batch(pdf_directory):
    processor = PDFProcessor(ocr_enabled=True)
    results = []
    
    for filename in os.listdir(pdf_directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_directory, filename)
            
            print(f"Обработка: {filename}")
            
            # Извлекаем основную информацию
            metadata = processor.extract_metadata(pdf_path)
            text = processor.extract_text(pdf_path)
            
            results.append({
                'filename': filename,
                'pages': metadata.data.page_count if metadata.success else 0,
                'word_count': len(text.data.split()) if text.success else 0,
                'pdf_type': text.pdf_type.value if text.success else 'unknown'
            })
    
    return results

# Использование
results = process_pdf_batch("/path/to/pdfs")
for result in results:
    print(f"{result['filename']}: {result['pages']} стр., {result['word_count']} слов")
```

### Условная обработка по типу PDF

```python
def smart_process_pdf(pdf_path):
    processor = PDFProcessor(ocr_enabled=True)
    
    # Определяем тип PDF
    pdf_type = processor.detect_pdf_type(pdf_path)
    
    if pdf_type == PDFType.TEXT_BASED:
        # Для текстовых PDF - быстрая обработка
        result = processor.extract_text(pdf_path)
        print(f"Текстовый PDF, извлечено {len(result.data)} символов")
        
    elif pdf_type == PDFType.SCANNED:
        # Для сканированных PDF - полная OCR обработка
        result = processor.process_full_pdf(
            pdf_path, 
            extract_images=True,
            extract_tables=True
        )
        print(f"Сканированный PDF, OCR обработка завершена")
        
    elif pdf_type == PDFType.MIXED:
        # Для смешанных PDF - комбинированная обработка
        result = processor.extract_text(pdf_path)  # Сначала текст
        if not result.success:
            # Если текст не найден, используем OCR
            result = processor._extract_text_ocr(pdf_path)
        print(f"Смешанный PDF, метод: {result.pdf_type.value}")
    
    return result
```

### Кастомная конфигурация

```python
# Процессор только для русского языка
processor_ru = PDFProcessor(ocr_enabled=True, lang="rus")

# Процессор без OCR (только текстовые PDF)
processor_text = PDFProcessor(ocr_enabled=False)

# Процессор для документов с таблицами
processor_tables = PDFProcessor(ocr_enabled=True)

def process_business_documents(pdf_path):
    processor = PDFProcessor(ocr_enabled=True, lang="rus+eng")
    
    # Всегда извлекаем таблицы для бизнес-документов
    results = processor.process_full_pdf(
        pdf_path,
        extract_tables=True,
        extract_images=False  # Изображения не нужны
    )
    
    # Анализируем таблицы
    if results['tables'].success:
        tables = results['tables'].data
        for table in tables:
            print(f"Таблица с {len(table['data'])} строками")
            # Дополнительный анализ таблиц...
    
    return results
```

## 🧪 Тестирование

Для запуска тестов:

```bash
cd /path/to/Rebecca-Platform
python -m pytest tests/test_pdf_processor.py -v
```

### Покрытие тестами

Тесты покрывают:
- Инициализацию процессора
- Извлечение метаданных
- Определение типа PDF
- Извлечение текста (прямое и OCR)
- Извлечение таблиц
- Извлечение изображений
- Определение языка
- Progress tracking
- Обработку ошибок

## ⚠️ Обработка ошибок

PDF Processor включает надежную обработку ошибок:

```python
try:
    result = processor.extract_text("/path/to/document.pdf")
    if not result.success:
        print(f"Ошибка извлечения: {result.error}")
        
        # Возможные причины:
        # - Файл поврежден
        # - PDF защищен паролем
        # - Недостаточно памяти
        # - Проблемы с OCR
        
except Exception as e:
    print(f"Критическая ошибка: {e}")
```

### Fallback механизмы

1. **Множественные методы извлечения текста:**
   - pdfplumber → PyPDF2 → PyMuPDF

2. **Извлечение таблиц:**
   - Camelot lattice → Camelot stream → OCR

3. **Определение количества страниц:**
   - pdfplumber → PyPDF2 → PyMuPDF

## 🔍 Ограничения и требования

### Системные требования
- Python 3.8+
- Tesseract OCR установлен и доступен в PATH
- Poppler для конвертации PDF в изображения

### Ограничения
- OCR работает только с установленным tesseract
- Некоторые сложные таблицы могут не извлекаться корректно
- OCR качество зависит от качества сканирования

### Производительность
- Обработка больших PDF может занимать значительное время
- OCR значительно медленнее прямого извлечения текста
- Рекомендуется использовать progress tracking для больших файлов

## 🤝 Вклад в проект

Для добавления новых возможностей:

1. Создайте ветку от main
2. Реализуйте функциональность
3. Добавьте тесты
4. Создайте pull request

## 📝 Лицензия

Этот проект распространяется под той же лицензией, что и основной проект Rebecca Platform.

## 🆘 Поддержка

При возникновении проблем:

1. Проверьте установку всех зависимостей
2. Убедитесь, что tesseract доступен в PATH
3. Проверьте права доступа к файлам PDF
4. Используйте логирование для отладки

```python
import logging
logging.basicConfig(level=logging.DEBUG)

processor = PDFProcessor(ocr_enabled=True)
# Теперь все операции будут логироваться
```