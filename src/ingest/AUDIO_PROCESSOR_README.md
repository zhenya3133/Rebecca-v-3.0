# AudioProcessor с Whisper

AudioProcessor - это полнофункциональный модуль для обработки аудио и видео файлов с поддержкой транскрипции через OpenAI Whisper API, локальные модели Whisper и mock режим для разработки.

## Особенности

- **Поддержка форматов**: mp3, wav, m4a, flac, mp4, avi, mov
- **Транскрипция**: OpenAI Whisper API, локальная модель, mock режим
- **Batch обработка**: Обработка множества файлов параллельно
- **Извлечение аудио**: Автоматическое извлечение аудио из видео
- **Определение языка**: Автоматическое определение языка аудио
- **Сегментация**: Разбивка длинных транскриптов на части
- **MemoryManager интеграция**: Сохранение результатов в память
- **Метаданные**: Получение информации о файлах
- **Обработка ошибок**: Надежная обработка различных сценариев

## Установка зависимостей

### Основные зависимости
```bash
pip install librosa soundfile pydub ffmpeg-python
```

### Whisper (опционально)
```bash
pip install openai-whisper
```

### OpenAI API (опционально)
```bash
pip install openai
```

## Быстрый старт

### 1. Mock режим (без API ключей)
```python
from audio_processor import create_audio_processor

# Создание процессора в mock режиме
processor = create_audio_processor(mock_mode=True)

# Транскрипция файла
result = processor.transcribe_audio("sample_audio.mp3")
print(result.text)
```

### 2. OpenAI API режим
```python
import os
from audio_processor import create_audio_processor

# Создание процессора с API ключом
api_key = os.getenv("OPENAI_API_KEY")
processor = create_audio_processor(openai_api_key=api_key)

result = processor.transcribe_audio("sample_audio.mp3")
print(f"Язык: {result.language}, Уверенность: {result.confidence}")
```

### 3. Локальная модель Whisper
```python
from audio_processor import create_audio_processor

processor = create_audio_processor(whisper_model="base")
result = processor.transcribe_audio("sample_audio.mp3")
print(f"Результат: {result.text}")
```

## Основные методы

### Транскрипция аудио/видео

```python
# Транскрипция с автоматическим определением языка
result = processor.transcribe_audio("audio.mp3")

# Транскрипция с указанием языка
result = processor.transcribe_audio("audio.mp3", language="ru")

# Транскрипция с сохранением в MemoryManager
result = processor.transcribe_audio("audio.mp3", save_to_memory=True)
```

### Batch обработка

```python
# Синхронная обработка
file_paths = ["audio1.mp3", "audio2.wav", "video.mp4"]
results = processor.batch_transcribe(file_paths)

# Асинхронная обработка
results = await processor.batch_transcribe_async(file_paths)
```

### Работа с видео

```python
# Автоматическое извлечение аудио из видео
result = processor.transcribe_audio("video.mp4")

# Извлечение аудио вручную
audio_path = processor.extract_audio_from_video("video.mp4")
result = processor.transcribe_audio(audio_path)
```

### Метаданные файлов

```python
# Получение информации о файле
info = processor.get_audio_info("audio.mp3")
print(f"Длительность: {info.duration}с")
print(f"Частота дискретизации: {info.sample_rate}Гц")
print(f"Каналы: {info.channels}")
```

### Определение языка

```python
# Определение языка аудио
language = processor.detect_language("audio.mp3")
print(f"Язык: {language}")

# Использование определенного языка
result = processor.transcribe_audio("audio.mp3", language=language)
```

### Сегментация транскрипта

```python
# Разбивка длинного текста на части
segments = processor.segment_transcript(long_text, max_length=500)
for segment in segments:
    print(f"Сегмент: {segment}")
```

## Структуры данных

### AudioInfo
```python
@dataclass
class AudioInfo:
    duration: float                    # Длительность в секундах
    sample_rate: int                   # Частота дискретизации
    channels: int                      # Количество каналов
    format: str                        # Формат файла
    bit_rate: Optional[int]            # Битрейт (если доступен)
    file_size: int                     # Размер файла в байтах
    path: str                          # Путь к файлу
```

### TranscriptionResult
```python
@dataclass
class TranscriptionResult:
    text: str                          # Текст транскрипции
    language: str                      # Определенный язык
    confidence: float                  # Уровень уверенности
    segments: List[Dict[str, Any]]     # Сегменты с временными метками
    processing_time: float             # Время обработки в секундах
    method: str                        # Метод: 'openai_api', 'local_model', 'mock'
    metadata: Dict[str, Any]           # Дополнительные метаданные
```

## Интеграция с MemoryManager

```python
from memory_manager.memory_manager_interface import MemoryManager
from audio_processor import create_audio_processor

# Создание MemoryManager
memory_manager = MemoryManager()

# Создание AudioProcessor с интеграцией памяти
processor = create_audio_processor(memory_manager=memory_manager)

# Транскрипция с автоматическим сохранением
result = processor.transcribe_audio("audio.mp3", save_to_memory=True)

# Поиск сохраненных транскриптов
from memory_manager.memory_manager_interface import MemoryLayer, MemoryFilter

saved_items = await memory_manager.retrieve(
    layer=MemoryLayer.EPISODIC,
    query="audio_transcription",
    filters=MemoryFilter(metadata={"type": "audio_transcription"})
)
```

## Конфигурация

### Параметры AudioProcessor

```python
processor = AudioProcessor(
    memory_manager=None,           # MemoryManager для сохранения результатов
    openai_api_key=None,           # API ключ OpenAI
    whisper_model="base",          # Размер модели: tiny, base, small, medium, large
    mock_mode=False,               # Принудительный mock режим
    max_workers=4                  # Количество потоков для batch обработки
)
```

### Поддерживаемые форматы

- **Аудио**: .mp3, .wav, .m4a, .flac, .aac, .ogg, .wma
- **Видео**: .mp4, .avi, .mov, .mkv, .wmv, .flv

## Статистика и мониторинг

```python
# Получение статистики по результатам
results = processor.batch_transcribe(file_paths)
stats = processor.get_transcription_stats(results)

print(f"Общее количество файлов: {stats['total_files']}")
print(f"Общее время обработки: {stats['total_processing_time_seconds']:.2f}с")
print(f"Успешность: {stats['success_rate']:.1f}%")
print(f"Распределение методов: {stats['method_distribution']}")
print(f"Распределение языков: {stats['language_distribution']}")
```

## Сохранение результатов

```python
# Сохранение результата в JSON файл
output_path = "transcript_result.json"
processor.save_transcript(result, output_path)

# Результат содержит всю информацию:
# - Исходный текст
# - Язык и уверенность
# - Временные сегменты
# - Метаданные обработки
```

## Обработка ошибок

Модуль обрабатывает различные типы ошибок:

- **FileNotFoundError**: Файл не найден
- **ValueError**: Неподдерживаемый формат файла
- **RuntimeError**: Отсутствуют необходимые зависимости
- **Exception**: Ошибки API или модели

```python
try:
    result = processor.transcribe_audio("audio.mp3")
except FileNotFoundError:
    print("Файл не найден")
except ValueError as e:
    print(f"Ошибка формата: {e}")
except Exception as e:
    print(f"Общая ошибка: {e}")
```

## Удобные функции

### Транскрипция одного файла
```python
from audio_processor import transcribe_single_file

result = transcribe_single_file("audio.mp3", mock_mode=True)
```

### Batch транскрипция
```python
from audio_processor import batch_transcribe_files

results = batch_transcribe_files(
    file_paths=["audio1.mp3", "audio2.wav"],
    mock_mode=True
)
```

## Производительность

### Рекомендации по оптимизации

1. **Выбор модели**: 
   - `tiny`: Быстрая, менее точная
   - `base`: Баланс скорости и качества
   - `small`, `medium`, `large`: Более точные, медленнее

2. **Batch обработка**: Используйте `max_workers` для параллельной обработки

3. **Формат файлов**: WAV с частотой 16кГц обеспечивает лучшую совместимость

4. **Предварительная обработка**: Обрезка тишины улучшает качество транскрипции

### Ограничения

- OpenAI API имеет лимиты на размер файлов и скорость запросов
- Локальные модели требуют значительных ресурсов памяти
- FFmpeg необходим для обработки видео файлов

## Тестирование

Запуск тестов:
```bash
cd /path/to/Rebecca-Platform/src/ingest
python -m pytest test_audio_processor.py -v
```

Запуск примеров:
```bash
python audio_processor_examples.py
```

## Логирование

```python
import logging

# Настройка уровня логирования
logging.basicConfig(level=logging.INFO)

# Включение debug логов
logging.getLogger('audio_processor').setLevel(logging.DEBUG)
```

## Примеры интеграции

### FastAPI endpoint
```python
from fastapi import FastAPI, UploadFile
from audio_processor import create_audio_processor

app = FastAPI()
processor = create_audio_processor(mock_mode=True)

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile):
    # Сохранение загруженного файла
    with open(f"temp_{file.filename}", "wb") as buffer:
        buffer.write(await file.read())
    
    # Транскрипция
    result = processor.transcribe_audio(f"temp_{file.filename}")
    
    return {
        "text": result.text,
        "language": result.language,
        "confidence": result.confidence
    }
```

### CLI утилита
```bash
python -m audio_processor sample_audio.mp3 --api-key sk-xxx --model base
```

## Лицензия

Модуль является частью Rebecca Platform и распространяется под соответствующей лицензией.

## Поддержка

Для вопросов и предложений создавайте Issues в репозитории проекта.
