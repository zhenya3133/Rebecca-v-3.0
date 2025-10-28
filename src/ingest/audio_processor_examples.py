"""
Примеры использования AudioProcessor для обработки аудио и видео файлов.
Демонстрирует различные сценарии использования и интеграцию с MemoryManager.
"""

import asyncio
import os
from pathlib import Path

# Импорт AudioProcessor
from audio_processor import (
    AudioProcessor,
    create_audio_processor,
    transcribe_single_file,
    batch_transcribe_files
)

# Импорт MemoryManager (если доступен)
try:
    from memory_manager.memory_manager_interface import MemoryManager
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    print("MemoryManager недоступен")


def example_1_basic_transcription():
    """Пример 1: Базовая транскрипция одного файла."""
    print("=== Пример 1: Базовая транскрипция ===")
    
    # Создание процессора в mock режиме
    processor = create_audio_processor(mock_mode=True)
    
    # Транскрипция файла
    file_path = "sample_audio.mp3"
    
    try:
        result = processor.transcribe_audio(file_path)
        
        print(f"Текст: {result.text}")
        print(f"Язык: {result.language}")
        print(f"Уверенность: {result.confidence}")
        print(f"Время обработки: {result.processing_time:.2f}с")
        print(f"Метод: {result.method}")
        
    except FileNotFoundError:
        print(f"Файл {file_path} не найден")
    except Exception as e:
        print(f"Ошибка: {e}")


def example_2_with_api_key():
    """Пример 2: Транскрипция с OpenAI API."""
    print("\n=== Пример 2: Транскрипция с OpenAI API ===")
    
    # Получение API ключа из переменной окружения
    api_key = os.getenv("OPENAI_API_KEY")
    
    if api_key:
        processor = create_audio_processor(
            openai_api_key=api_key,
            whisper_model="base"  # Можно выбрать: tiny, base, small, medium, large
        )
        
        result = processor.transcribe_audio("sample_audio.mp3")
        print(f"Результат: {result.text[:100]}...")
        print(f"Метод: {result.method}")
    else:
        print("API ключ OpenAI не найден в переменных окружения")


def example_3_with_local_model():
    """Пример 3: Использование локальной модели Whisper."""
    print("\n=== Пример 3: Локальная модель Whisper ===")
    
    processor = create_audio_processor(
        whisper_model="base",  # tiny, base, small, medium, large
        mock_mode=False
    )
    
    try:
        result = processor.transcribe_audio("sample_audio.mp3")
        print(f"Результат: {result.text[:100]}...")
        print(f"Метод: {result.method}")
    except Exception as e:
        print(f"Ошибка: {e}")
        print("Убедитесь, что модель Whisper установлена")


def example_4_with_memory_manager():
    """Пример 4: Интеграция с MemoryManager."""
    print("\n=== Пример 4: Интеграция с MemoryManager ===")
    
    if not MEMORY_AVAILABLE:
        print("MemoryManager недоступен")
        return
    
    # Создание MemoryManager
    memory_manager = MemoryManager()
    
    # Создание AudioProcessor с интеграцией памяти
    processor = create_audio_processor(
        memory_manager=memory_manager,
        mock_mode=True
    )
    
    try:
        result = processor.transcribe_audio(
            "sample_audio.mp3",
            save_to_memory=True  # Сохранение в MemoryManager
        )
        
        print(f"Транскрипция сохранена в памяти")
        print(f"Результат: {result.text[:100]}...")
        
        # Поиск сохраненных данных в памяти
        from memory_manager.memory_manager_interface import MemoryLayer, MemoryFilter
        
        saved_items = asyncio.run(
            memory_manager.retrieve(
                layer=MemoryLayer.EPISODIC,
                query="audio_transcription",
                filters=MemoryFilter(metadata={"type": "audio_transcription"})
            )
        )
        
        print(f"Найдено сохраненных элементов: {len(saved_items)}")
        
    except Exception as e:
        print(f"Ошибка: {e}")


def example_5_batch_processing():
    """Пример 5: Batch обработка множества файлов."""
    print("\n=== Пример 5: Batch обработка ===")
    
    processor = create_audio_processor(mock_mode=True)
    
    # Список файлов для обработки
    file_paths = [
        "audio1.mp3",
        "audio2.wav", 
        "audio3.m4a",
        "video1.mp4",
        "video2.avi"
    ]
    
    # Проверка существования файлов
    existing_files = [f for f in file_paths if os.path.exists(f)]
    
    if existing_files:
        print(f"Обработка {len(existing_files)} файлов...")
        
        results = processor.batch_transcribe(existing_files)
        
        print(f"Обработано файлов: {len(results)}")
        
        # Статистика
        stats = processor.get_transcription_stats(results)
        print(f"Статистика:")
        print(f"  - Общее время: {stats['total_processing_time_seconds']:.2f}с")
        print(f"  - Среднее время: {stats['average_processing_time_seconds']:.2f}с")
        print(f"  - Распределение методов: {stats['method_distribution']}")
        print(f"  - Успешность: {stats['success_rate']:.1f}%")
        
        # Детали по каждому файлу
        for i, result in enumerate(results):
            print(f"  Файл {i+1}: {result.language}, {result.processing_time:.2f}с")
    else:
        print("Файлы для обработки не найдены")


def example_6_video_processing():
    """Пример 6: Обработка видео файлов."""
    print("\n=== Пример 6: Обработка видео ===")
    
    processor = create_audio_processor(mock_mode=True)
    
    video_file = "sample_video.mp4"
    
    try:
        # Получение информации о видео
        audio_info = processor.get_audio_info(video_file)
        print(f"Информация о видео:")
        print(f"  - Длительность: {audio_info.duration:.2f}с")
        print(f"  - Формат: {audio_info.format}")
        print(f"  - Размер: {audio_info.file_size} байт")
        
        # Транскрипция (автоматически извлекает аудио)
        result = processor.transcribe_audio(video_file)
        print(f"Транскрипция: {result.text[:100]}...")
        
    except FileNotFoundError:
        print(f"Видео файл {video_file} не найден")
    except Exception as e:
        print(f"Ошибка: {e}")


def example_7_language_detection():
    """Пример 7: Определение языка."""
    print("\n=== Пример 7: Определение языка ===")
    
    processor = create_audio_processor(mock_mode=True)
    
    audio_file = "sample_audio.mp3"
    
    try:
        language = processor.detect_language(audio_file)
        print(f"Определенный язык: {language}")
        
        # Транскрипция с указанным языком
        result = processor.transcribe_audio(audio_file, language=language)
        print(f"Результат: {result.text[:100]}...")
        
    except FileNotFoundError:
        print(f"Аудио файл {audio_file} не найден")
    except Exception as e:
        print(f"Ошибка: {e}")


def example_8_transcript_segmentation():
    """Пример 8: Сегментация транскрипта."""
    print("\n=== Пример 8: Сегментация транскрипта ===")
    
    processor = create_audio_processor(mock_mode=True)
    
    # Длинный текст для демонстрации
    long_text = """
    Это очень длинный текст транскрипта. Он содержит много предложений для демонстрации 
    возможностей сегментации. Первое предложение заканчивается здесь. Второе предложение 
    продолжает повествование. Третье предложение добавляет новую информацию. 
    Четвертое предложение содержит важные детали. Пятое предложение завершает мысль.
    """.strip()
    
    # Сегментация на части по 100 символов
    segments = processor.segment_transcript(long_text, max_length=100)
    
    print(f"Исходный текст ({len(long_text)} символов):")
    print(long_text[:100] + "...")
    
    print(f"\nСегменты ({len(segments)} частей):")
    for i, segment in enumerate(segments, 1):
        print(f"  {i}. {segment} ({len(segment)} символов)")


def example_9_save_and_load_results():
    """Пример 9: Сохранение и загрузка результатов."""
    print("\n=== Пример 9: Сохранение результатов ===")
    
    processor = create_audio_processor(mock_mode=True)
    
    try:
        # Транскрипция
        result = processor.transcribe_audio("sample_audio.mp3")
        
        # Сохранение в файл
        output_path = "transcript_result.json"
        processor.save_transcript(result, output_path)
        print(f"Результат сохранен: {output_path}")
        
        # Проверка содержимого файла
        if os.path.exists(output_path):
            import json
            with open(output_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            
            print(f"Сохраненные данные:")
            print(f"  - Текст: {saved_data['text'][:50]}...")
            print(f"  - Язык: {saved_data['language']}")
            print(f"  - Метод: {saved_data['method']}")
        
    except FileNotFoundError:
        print("Файл для транскрипции не найден")
    except Exception as e:
        print(f"Ошибка: {e}")


async def example_10_async_processing():
    """Пример 10: Асинхронная обработка."""
    print("\n=== Пример 10: Асинхронная обработка ===")
    
    processor = create_audio_processor(mock_mode=True)
    
    file_paths = ["audio1.mp3", "audio2.wav", "audio3.m4a"]
    existing_files = [f for f in file_paths if os.path.exists(f)]
    
    if existing_files:
        print(f"Асинхронная обработка {len(existing_files)} файлов...")
        
        # Асинхронная batch обработка
        results = await processor.batch_transcribe_async(existing_files)
        
        print(f"Обработано файлов: {len(results)}")
        
        for i, result in enumerate(results):
            print(f"  Файл {i+1}: {result.language}, {result.processing_time:.2f}с")
    else:
        print("Файлы для обработки не найдены")


def example_11_error_handling():
    """Пример 11: Обработка ошибок."""
    print("\n=== Пример 11: Обработка ошибок ===")
    
    processor = create_audio_processor(mock_mode=True)
    
    # Тест различных ошибок
    test_cases = [
        ("несуществующий_файл.mp3", "Файл не найден"),
        ("test.xyz", "Неподдерживаемый формат"),
        ("", "Пустой путь")
    ]
    
    for file_path, description in test_cases:
        try:
            if file_path:
                # Создаем фиктивный файл для теста ошибки формата
                if file_path == "test.xyz":
                    with open(file_path, 'w') as f:
                        f.write("test")
                    
                    result = processor.transcribe_audio(file_path)
                    os.remove(file_path)  # Очистка
                else:
                    result = processor.transcribe_audio(file_path)
            else:
                result = processor.transcribe_audio(file_path)
                
        except FileNotFoundError:
            print(f"✓ {description}: Файл не найден (ожидаемая ошибка)")
        except ValueError as e:
            print(f"✓ {description}: Ошибка формата (ожидаемая ошибка)")
        except Exception as e:
            print(f"✗ {description}: Неожиданная ошибка: {e}")


def main():
    """Запуск всех примеров."""
    print("AudioProcessor - Примеры использования")
    print("=" * 50)
    
    # Запуск примеров
    example_1_basic_transcription()
    example_2_with_api_key()
    example_3_with_local_model()
    example_4_with_memory_manager()
    example_5_batch_processing()
    example_6_video_processing()
    example_7_language_detection()
    example_8_transcript_segmentation()
    example_9_save_and_load_results()
    
    # Асинхронный пример
    asyncio.run(example_10_async_processing())
    
    example_11_error_handling()
    
    print("\n" + "=" * 50)
    print("Все примеры завершены")


if __name__ == "__main__":
    main()
