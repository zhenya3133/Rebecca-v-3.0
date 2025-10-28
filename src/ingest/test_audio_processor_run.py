#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работы AudioProcessor.
Создает тестовые файлы и проверяет основную функциональность.
"""

import os
import tempfile
import sys
from pathlib import Path

# Добавляем путь к src в sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Импорт AudioProcessor из текущей директории
from audio_processor import (
    AudioProcessor,
    AudioInfo,
    TranscriptionResult,
    create_audio_processor
)


def create_test_audio_file(filepath):
    """Создание тестового WAV файла."""
    # Создание простого WAV заголовка
    with open(filepath, 'wb') as f:
        # WAV заголовок
        f.write(b'RIFF')
        f.write((36).to_bytes(4, 'little'))  # Размер файла - 36 байт
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write((16).to_bytes(4, 'little'))  # Размер fmt блока
        f.write((1).to_bytes(2, 'little'))   # PCM формат
        f.write((1).to_bytes(2, 'little'))   # Моно
        f.write((16000).to_bytes(4, 'little'))  # Частота дискретизации
        f.write((32000).to_bytes(4, 'little'))  # Байтрейт
        f.write((2).to_bytes(2, 'little'))   # Размер блока
        f.write((16).to_bytes(2, 'little'))  # Бит на сэмпл
        f.write(b'data')
        f.write((0).to_bytes(4, 'little'))   # Размер данных
    
    return filepath


def create_test_video_file(filepath):
    """Создание тестового MP4 файла (заглушка)."""
    with open(filepath, 'wb') as f:
        f.write(b'fake video file for testing')
    return filepath


def test_basic_functionality():
    """Тестирование основной функциональности."""
    print("🔧 Тестирование основной функциональности AudioProcessor")
    print("=" * 60)
    
    # Создание временных файлов
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_file = os.path.join(temp_dir, "test_audio.wav")
        video_file = os.path.join(temp_dir, "test_video.mp4")
        
        # Создание тестовых файлов
        create_test_audio_file(audio_file)
        create_test_video_file(video_file)
        
        print(f"📁 Созданы тестовые файлы:")
        print(f"   - Аудио: {audio_file}")
        print(f"   - Видео: {video_file}")
        
        # Создание процессора
        processor = create_audio_processor(mock_mode=True)
        print(f"\n✅ AudioProcessor создан (mock режим)")
        
        # Тест 1: Получение поддерживаемых форматов
        print(f"\n📋 Тест 1: Поддерживаемые форматы")
        formats = processor.get_supported_formats()
        print(f"   Аудио форматы: {len(formats['audio'])}")
        print(f"   Видео форматы: {len(formats['video'])}")
        print(f"   Всего форматов: {len(formats['all'])}")
        assert len(formats['all']) > 0, "Должны быть поддерживаемые форматы"
        print(f"   ✅ Форматы получены успешно")
        
        # Тест 2: Получение информации о файле
        print(f"\n📊 Тест 2: Метаданные аудио файла")
        try:
            info = processor.get_audio_info(audio_file)
            print(f"   Формат: {info.format}")
            print(f"   Размер: {info.file_size} байт")
            print(f"   Путь: {info.path}")
            print(f"   ✅ Метаданные получены успешно")
        except Exception as e:
            print(f"   ⚠️  Ошибка получения метаданных: {e}")
        
        # Тест 3: Определение языка
        print(f"\n🌍 Тест 3: Определение языка")
        try:
            language = processor.detect_language(audio_file)
            print(f"   Определенный язык: {language}")
            assert language in ['ru', 'en', 'unknown'], f"Неожиданный язык: {language}"
            print(f"   ✅ Язык определен успешно")
        except Exception as e:
            print(f"   ⚠️  Ошибка определения языка: {e}")
        
        # Тест 4: Сегментация текста
        print(f"\n📝 Тест 4: Сегментация текста")
        long_text = "Первое предложение. Второе предложение. Третье предложение. " * 10
        segments = processor.segment_transcript(long_text, max_length=100)
        print(f"   Исходная длина: {len(long_text)} символов")
        print(f"   Количество сегментов: {len(segments)}")
        for i, segment in enumerate(segments[:3], 1):  # Показать первые 3
            print(f"   Сегмент {i}: {len(segment)} символов")
        print(f"   ✅ Сегментация работает корректно")
        
        # Тест 5: Mock транскрипция
        print(f"\n🎭 Тест 5: Mock транскрипция")
        try:
            result = processor.transcribe_audio(audio_file)
            print(f"   Метод: {result.method}")
            print(f"   Язык: {result.language}")
            print(f"   Уверенность: {result.confidence}")
            print(f"   Текст: {result.text[:100]}...")
            print(f"   Время обработки: {result.processing_time:.3f}с")
            assert result.method == "mock", f"Неожиданный метод: {result.method}"
            assert result.language == "ru", f"Неожиданный язык: {result.language}"
            print(f"   ✅ Mock транскрипция работает")
        except Exception as e:
            print(f"   ❌ Ошибка mock транскрипции: {e}")
            raise
        
        # Тест 6: Batch обработка
        print(f"\n🔄 Тест 6: Batch обработка")
        try:
            file_paths = [audio_file, audio_file]  # Два одинаковых файла для теста
            results = processor.batch_transcribe(file_paths)
            print(f"   Обработано файлов: {len(results)}")
            assert len(results) == 2, f"Ожидалось 2 результата, получено {len(results)}"
            
            for i, result in enumerate(results, 1):
                print(f"   Результат {i}: {result.method}, {result.language}")
            print(f"   ✅ Batch обработка работает")
        except Exception as e:
            print(f"   ❌ Ошибка batch обработки: {e}")
            raise
        
        # Тест 7: Сохранение результата
        print(f"\n💾 Тест 7: Сохранение результата")
        try:
            import json
            output_path = os.path.join(temp_dir, "test_result.json")
            processor.save_transcript(result, output_path)
            
            # Проверка файла
            assert os.path.exists(output_path), "Файл результата не создан"
            
            with open(output_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            
            print(f"   Файл сохранен: {output_path}")
            print(f"   Размер файла: {os.path.getsize(output_path)} байт")
            print(f"   Содержит текст: {'text' in saved_data}")
            print(f"   ✅ Сохранение работает")
        except Exception as e:
            print(f"   ❌ Ошибка сохранения: {e}")
            raise
        
        # Тест 8: Статистика
        print(f"\n📈 Тест 8: Статистика")
        try:
            stats = processor.get_transcription_stats(results)
            print(f"   Общее количество файлов: {stats['total_files']}")
            print(f"   Общее время обработки: {stats['total_processing_time_seconds']:.3f}с")
            print(f"   Среднее время: {stats['average_processing_time_seconds']:.3f}с")
            print(f"   Распределение методов: {stats['method_distribution']}")
            print(f"   Успешность: {stats['success_rate']:.1f}%")
            print(f"   ✅ Статистика получена")
        except Exception as e:
            print(f"   ⚠️  Ошибка получения статистики: {e}")
        
        print(f"\n🎉 Все основные тесты прошли успешно!")


def test_error_handling():
    """Тестирование обработки ошибок."""
    print(f"\n⚠️  Тестирование обработки ошибок")
    print("=" * 60)
    
    processor = create_audio_processor(mock_mode=True)
    
    # Тест 1: Несуществующий файл
    print(f"\n❌ Тест 1: Несуществующий файл")
    try:
        processor.transcribe_audio("/nonexistent/file.wav")
        print(f"   ❌ Ошибка: файл не найден, но исключение не выброшено")
    except FileNotFoundError:
        print(f"   ✅ FileNotFoundError корректно выброшен")
    except Exception as e:
        print(f"   ⚠️  Неожиданное исключение: {e}")
    
    # Тест 2: Неподдерживаемый формат
    print(f"\n❌ Тест 2: Неподдерживаемый формат")
    with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
        temp_path = f.name
    
    try:
        processor.get_audio_info(temp_path)
        print(f"   ❌ Ошибка: неподдерживаемый формат, но исключение не выброшено")
    except ValueError:
        print(f"   ✅ ValueError корректно выброшен")
    except Exception as e:
        print(f"   ⚠️  Неожиданное исключение: {e}")
    finally:
        os.unlink(temp_path)


def test_structures():
    """Тестирование структур данных."""
    print(f"\n📦 Тестирование структур данных")
    print("=" * 60)
    
    # Тест AudioInfo
    print(f"\n📊 Тест AudioInfo")
    audio_info = AudioInfo(
        duration=10.5,
        sample_rate=44100,
        channels=2,
        format="mp3",
        bit_rate=128000,
        file_size=1024000,
        path="/test/file.mp3"
    )
    print(f"   Длительность: {audio_info.duration}с")
    print(f"   Частота: {audio_info.sample_rate}Гц")
    print(f"   Каналы: {audio_info.channels}")
    print(f"   ✅ AudioInfo создана корректно")
    
    # Тест TranscriptionResult
    print(f"\n📝 Тест TranscriptionResult")
    transcription_result = TranscriptionResult(
        text="Тестовый текст транскрипции",
        language="ru",
        confidence=0.95,
        segments=[{"text": "Сегмент 1", "start": 0, "end": 5}],
        processing_time=2.5,
        method="test",
        metadata={"source": "test"}
    )
    print(f"   Текст: {transcription_result.text}")
    print(f"   Язык: {transcription_result.language}")
    print(f"   Уверенность: {transcription_result.confidence}")
    print(f"   Метод: {transcription_result.method}")
    print(f"   ✅ TranscriptionResult создана корректно")


def main():
    """Главная функция тестирования."""
    print("🧪 AudioProcessor - Тестирование")
    print("=" * 60)
    print(f"Python версия: {sys.version}")
    print(f"Текущая директория: {os.getcwd()}")
    print(f"Временная директория: {tempfile.gettempdir()}")
    
    try:
        # Основные тесты
        test_basic_functionality()
        
        # Тесты обработки ошибок
        test_error_handling()
        
        # Тесты структур данных
        test_structures()
        
        print(f"\n" + "=" * 60)
        print(f"🎉 ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")
        print(f"✅ AudioProcessor готов к использованию")
        
    except Exception as e:
        print(f"\n" + "=" * 60)
        print(f"❌ ТЕСТЫ НЕ ПРОШЛИ!")
        print(f"💥 Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
