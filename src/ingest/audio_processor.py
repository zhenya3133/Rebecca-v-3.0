"""
Audio Processor с интеграцией Whisper для транскрипции аудио и видео файлов.
Поддерживает OpenAI Whisper API, локальную модель и mock режим для работы без ключей.
"""

import os
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Импорты для работы с аудио
try:
    import librosa
    import soundfile as sf
    from pydub import AudioSegment
    from pydub.utils import mediainfo
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False
    logging.warning("Библиотеки для работы с аудио не установлены. Установите: pip install librosa soundfile pydub")

# Импорты для работы с видео
try:
    import ffmpeg
    VIDEO_LIBS_AVAILABLE = True
except ImportError:
    VIDEO_LIBS_AVAILABLE = False
    logging.warning("FFmpeg не установлен. Установите: pip install ffmpeg-python")

# Импорты для Whisper
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logging.warning("Whisper не установлен. Установите: pip install openai-whisper")

# Импорты для работы с памятью
try:
    from ..memory_manager.memory_manager_interface import MemoryManager, MemoryLayer, MemoryItem, MemoryFilter
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    logging.warning("MemoryManager недоступен. Будет использоваться локальное хранение.")

# Импорт asyncio для асинхронных операций
import asyncio


@dataclass
class AudioInfo:
    """Метаданные аудио файла."""
    duration: float
    sample_rate: int
    channels: int
    format: str
    bit_rate: Optional[int] = None
    file_size: int = 0
    path: str = ""


@dataclass
class TranscriptionResult:
    """Результат транскрипции."""
    text: str
    language: str
    confidence: float
    segments: List[Dict[str, Any]]
    processing_time: float
    method: str  # 'openai_api', 'local_model', 'mock'
    metadata: Dict[str, Any] = None


class AudioProcessor:
    """
    Процессор аудио с поддержкой транскрипции через Whisper API и локальную модель.
    
    Поддерживаемые форматы:
    - Аудио: mp3, wav, m4a, flac
    - Видео: mp4, avi, mov
    
    Методы работы:
    - OpenAI Whisper API (требует API ключ)
    - Локальная модель Whisper (требует установки модели)
    - Mock режим (без ключей, для разработки)
    """
    
    SUPPORTED_AUDIO_FORMATS = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma'}
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
    SUPPORTED_FORMATS = SUPPORTED_AUDIO_FORMATS | SUPPORTED_VIDEO_FORMATS
    
    def __init__(self, 
                 memory_manager: Optional[Any] = None,
                 openai_api_key: Optional[str] = None,
                 whisper_model: str = "base",
                 mock_mode: bool = False,
                 max_workers: int = 4):
        """
        Инициализация AudioProcessor.
        
        Args:
            memory_manager: Экземпляр MemoryManager для сохранения транскриптов
            openai_api_key: API ключ OpenAI (опционально)
            whisper_model: Размер модели для локального Whisper (tiny, base, small, medium, large)
            mock_mode: Режим mock для работы без API ключей
            max_workers: Количество потоков для batch обработки
        """
        self.memory_manager = memory_manager
        self.openai_api_key = openai_api_key
        self.whisper_model = whisper_model
        self.mock_mode = mock_mode or not openai_api_key
        self.max_workers = max_workers
        
        # Настройка логгера
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Инициализация локальной модели Whisper
        self.local_model = None
        if WHISPER_AVAILABLE and not self.mock_mode:
            try:
                self.local_model = whisper.load_model(whisper_model)
                self.logger.info(f"Загружена локальная модель Whisper: {whisper_model}")
            except Exception as e:
                self.logger.warning(f"Не удалось загрузить модель Whisper: {e}")
                self.mock_mode = True
        
        # Проверка доступности зависимостей
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Проверка доступности необходимых библиотек."""
        missing_deps = []
        
        if not AUDIO_LIBS_AVAILABLE:
            missing_deps.append("audio libraries (librosa, soundfile, pydub)")
        
        if not VIDEO_LIBS_AVAILABLE and self.mock_mode:
            missing_deps.append("video libraries (ffmpeg-python)")
        
        if not WHISPER_AVAILABLE:
            missing_deps.append("whisper")
        
        if missing_deps:
            self.logger.warning(f"Отсутствующие зависимости: {', '.join(missing_deps)}")
            
            if self.mock_mode:
                self.logger.info("Работа в mock режиме без реальной транскрипции")
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Получение списка поддерживаемых форматов."""
        return {
            "audio": list(self.SUPPORTED_AUDIO_FORMATS),
            "video": list(self.SUPPORTED_VIDEO_FORMATS),
            "all": list(self.SUPPORTED_FORMATS)
        }
    
    def get_audio_info(self, audio_path: str) -> AudioInfo:
        """
        Получение метаданных аудио файла.
        
        Args:
            audio_path: Путь к аудио файлу
            
        Returns:
            AudioInfo с метаданными файла
        """
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Файл не найден: {audio_path}")
            
            # Проверка формата
            file_ext = Path(audio_path).suffix.lower()
            if file_ext not in self.SUPPORTED_FORMATS:
                raise ValueError(f"Неподдерживаемый формат файла: {file_ext}")
            
            # Извлечение метаданных через pydub
            if AUDIO_LIBS_AVAILABLE:
                info = mediainfo(audio_path)
                
                # Для аудио файлов
                if file_ext in self.SUPPORTED_AUDIO_FORMATS:
                    try:
                        # Загрузка с librosa для точных метрик
                        y, sr = librosa.load(audio_path, sr=None)
                        duration = librosa.get_duration(y=y, sr=sr)
                        
                        return AudioInfo(
                            duration=duration,
                            sample_rate=sr,
                            channels=1 if len(y.shape) == 1 else y.shape[0],
                            format=file_ext[1:],
                            bit_rate=int(info.get('bit_rate', 0)) if info.get('bit_rate') else None,
                            file_size=os.path.getsize(audio_path),
                            path=audio_path
                        )
                    except Exception as e:
                        self.logger.warning(f"Не удалось получить точные метаданные: {e}")
                
                # Общие метаданные через pydub
                audio = AudioSegment.from_file(audio_path)
                return AudioInfo(
                    duration=len(audio) / 1000.0,  # pydub возвращает миллисекунды
                    sample_rate=audio.frame_rate,
                    channels=audio.channels,
                    format=file_ext[1:],
                    bit_rate=audio.frame_width * audio.frame_rate * 8 if audio.frame_width else None,
                    file_size=os.path.getsize(audio_path),
                    path=audio_path
                )
            else:
                # Fallback без библиотек
                return AudioInfo(
                    duration=0.0,
                    sample_rate=0,
                    channels=0,
                    format=file_ext[1:],
                    file_size=os.path.getsize(audio_path),
                    path=audio_path
                )
                
        except Exception as e:
            self.logger.error(f"Ошибка при получении метаданных файла {audio_path}: {e}")
            raise
    
    def extract_audio_from_video(self, video_path: str, output_path: Optional[str] = None) -> str:
        """
        Извлечение аудио из видео файла.
        
        Args:
            video_path: Путь к видео файлу
            output_path: Путь для сохранения аудио (опционально)
            
        Returns:
            Путь к извлеченному аудио файлу
        """
        try:
            if not VIDEO_LIBS_AVAILABLE:
                raise RuntimeError("FFmpeg не установлен для обработки видео")
            
            file_ext = Path(video_path).suffix.lower()
            if file_ext not in self.SUPPORTED_VIDEO_FORMATS:
                raise ValueError(f"Неподдерживаемый формат видео: {file_ext}")
            
            if output_path is None:
                # Создание временного файла
                temp_dir = tempfile.gettempdir()
                output_path = os.path.join(temp_dir, f"extracted_audio_{int(time.time())}.wav")
            
            # Извлечение аудио через ffmpeg
            (
                ffmpeg
                .input(video_path)
                .output(output_path, acodec='pcm_s16le', ac=1, ar=16000)
                .overwrite_output()
                .run(quiet=True)
            )
            
            self.logger.info(f"Аудио извлечено из {video_path} в {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Ошибка при извлечении аудио из {video_path}: {e}")
            raise
    
    def detect_language(self, audio_path: str) -> str:
        """
        Определение языка аудио.
        
        Args:
            audio_path: Путь к аудио файлу
            
        Returns:
            Код языка (например, 'ru', 'en')
        """
        try:
            if self.mock_mode or not WHISPER_AVAILABLE:
                # Mock определение языка
                return "ru"  # По умолчанию русский
            
            # Использование Whisper для определения языка
            if self.local_model:
                result = self.local_model.detect_language(audio_path)
                return result["language"]
            else:
                # Загрузка временной модели для определения языка
                model = whisper.load_model("base")
                audio = whisper.load_audio(audio_path)
                mel = whisper.log_mel_spectrogram(audio).to(model.device)
                
                _, probs = model.detect_language(mel)
                detected_lang = max(probs, key=probs.get)
                return detected_lang
                
        except Exception as e:
            self.logger.warning(f"Не удалось определить язык для {audio_path}: {e}")
            return "unknown"
    
    def segment_transcript(self, transcript: str, max_length: int = 500) -> List[str]:
        """
        Сегментация транскрипта на части заданной длины.
        
        Args:
            transcript: Полный текст транскрипта
            max_length: Максимальная длина сегмента в символах
            
        Returns:
            Список сегментов текста
        """
        if not transcript:
            return []
        
        if len(transcript) <= max_length:
            return [transcript]
        
        segments = []
        sentences = transcript.replace('!', '.').replace('?', '.').split('.')
        current_segment = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Проверка, поместится ли предложение в текущий сегмент
            test_segment = current_segment + ". " + sentence if current_segment else sentence
            
            if len(test_segment) <= max_length:
                current_segment = test_segment
            else:
                # Сохранение текущего сегмента
                if current_segment:
                    segments.append(current_segment.strip())
                
                # Проверка длины самого предложения
                if len(sentence) > max_length:
                    # Деление длинного предложения по словам
                    words = sentence.split()
                    temp_segment = ""
                    
                    for word in words:
                        if len(temp_segment + " " + word) <= max_length:
                            temp_segment += " " + word if temp_segment else word
                        else:
                            if temp_segment:
                                segments.append(temp_segment)
                            temp_segment = word
                    
                    if temp_segment:
                        current_segment = temp_segment
                    else:
                        current_segment = ""
                else:
                    current_segment = sentence
        
        # Добавление последнего сегмента
        if current_segment:
            segments.append(current_segment.strip())
        
        return segments
    
    async def _save_to_memory(self, 
                             file_path: str, 
                             result: TranscriptionResult,
                             metadata: Dict[str, Any]) -> Optional[str]:
        """
        Сохранение результата транскрипции в MemoryManager.
        
        Args:
            file_path: Путь к исходному файлу
            result: Результат транскрипции
            metadata: Дополнительные метаданные
            
        Returns:
            ID сохраненного элемента или None
        """
        if not self.memory_manager or not MEMORY_AVAILABLE:
            return None
        
        try:
            memory_data = {
                "transcript": result.text,
                "language": result.language,
                "confidence": result.confidence,
                "segments": result.segments,
                "processing_time": result.processing_time,
                "method": result.method,
                "file_path": file_path,
                "file_size": os.path.getsize(file_path),
                "timestamp": time.time()
            }
            
            memory_data.update(metadata)
            
            # Сохранение в эпизодическую память
            item_id = await self.memory_manager.store(
                layer=MemoryLayer.EPISODIC,
                data=memory_data,
                metadata={
                    "type": "audio_transcription",
                    "format": Path(file_path).suffix,
                    "language": result.language
                }
            )
            
            self.logger.info(f"Транскрипция сохранена в памяти: {item_id}")
            return item_id
            
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении в память: {e}")
            return None
    
    def _mock_transcription(self, file_path: str) -> TranscriptionResult:
        """
        Mock транскрипция для разработки без API ключей.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Мок результат транскрипции
        """
        file_name = Path(file_path).stem
        mock_text = f"Это mock транскрипция для файла: {file_name}. " \
                   f"Для получения реальной транскрипции настройте API ключ OpenAI " \
                   f"или установите локальную модель Whisper."
        
        return TranscriptionResult(
            text=mock_text,
            language="ru",
            confidence=0.5,
            segments=[{"text": mock_text, "start": 0, "end": 5}],
            processing_time=0.1,
            method="mock",
            metadata={"file_path": file_path}
        )
    
    def transcribe_audio(self, file_path: str, 
                        language: Optional[str] = None,
                        save_to_memory: bool = True) -> TranscriptionResult:
        """
        Транскрипция аудио файла.
        
        Args:
            file_path: Путь к аудио/видео файлу
            language: Язык транскрипции (опционально)
            save_to_memory: Сохранить ли результат в MemoryManager
            
        Returns:
            TranscriptionResult с результатом транскрипции
        """
        start_time = time.time()
        
        try:
            # Проверка существования файла
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Файл не найден: {file_path}")
            
            # Проверка формата
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.SUPPORTED_FORMATS:
                raise ValueError(f"Неподдерживаемый формат файла: {file_ext}")
            
            # Извлечение аудио из видео при необходимости
            audio_path = file_path
            if file_ext in self.SUPPORTED_VIDEO_FORMATS:
                if VIDEO_LIBS_AVAILABLE:
                    audio_path = self.extract_audio_from_video(file_path)
                else:
                    raise RuntimeError("Обработка видео требует установки FFmpeg")
            
            # Определение языка если не указан
            if language is None:
                language = self.detect_language(audio_path)
            
            transcript_text = ""
            segments = []
            confidence = 0.0
            
            # Выбор метода транскрипции
            if self.mock_mode:
                result = self._mock_transcription(audio_path)
                method = "mock"
            elif self.openai_api_key and not self.mock_mode:
                # OpenAI Whisper API (требует API ключ)
                result = self._transcribe_with_openai_api(audio_path, language)
                method = "openai_api"
            elif self.local_model:
                # Локальная модель Whisper
                result = self._transcribe_with_local_model(audio_path, language)
                method = "local_model"
            else:
                # Fallback на mock
                result = self._mock_transcription(audio_path)
                method = "mock"
            
            processing_time = time.time() - start_time
            
            # Обновление времени обработки
            result.processing_time = processing_time
            result.method = method
            
            # Сохранение в память
            if save_to_memory:
                try:
                    # Попытка запуска в асинхронном режиме
                    import asyncio as async_lib
                    loop = async_lib.get_running_loop()
                    task = loop.create_task(self._save_to_memory(
                        file_path=file_path,
                        result=result,
                        metadata={"audio_info": asdict(self.get_audio_info(file_path))}
                    ))
                    # Не ждем завершения задачи, чтобы не блокировать основной поток
                except RuntimeError:
                    # Если нет запущенного цикла событий, запускаем задачу в фоновом режиме
                    try:
                        import threading
                        import asyncio as async_lib
                        
                        def run_save_task():
                            new_loop = async_lib.new_event_loop()
                            async_lib.set_event_loop(new_loop)
                            try:
                                new_loop.run_until_complete(self._save_to_memory(
                                    file_path=file_path,
                                    result=result,
                                    metadata={"audio_info": asdict(self.get_audio_info(file_path))}
                                ))
                            finally:
                                new_loop.close()
                        
                        thread = threading.Thread(target=run_save_task, daemon=True)
                        thread.start()
                    except Exception as e:
                        self.logger.warning(f"Не удалось сохранить в память: {e}")
            
            self.logger.info(f"Транскрипция завершена для {file_path} за {processing_time:.2f}с")
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка при транскрипции {file_path}: {e}")
            raise
        finally:
            # Очистка временных файлов
            if audio_path != file_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except:
                    pass
    
    def _transcribe_with_openai_api(self, audio_path: str, language: str) -> TranscriptionResult:
        """
        Транскрипция через OpenAI Whisper API.
        
        Args:
            audio_path: Путь к аудио файлу
            language: Язык транскрипции
            
        Returns:
            TranscriptionResult
        """
        try:
            import openai
            
            client = openai.OpenAI(api_key=self.openai_api_key)
            
            with open(audio_path, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language if language != "unknown" else None,
                    response_format="verbose_json"
                )
            
            transcript_text = response.text
            segments = []
            confidence = 0.9  # OpenAI API не возвращает confidence score
            
            # Извлечение сегментов если доступны
            if hasattr(response, 'segments') and response.segments:
                for segment in response.segments:
                    segments.append({
                        "text": segment.text,
                        "start": segment.start,
                        "end": segment.end
                    })
            
            return TranscriptionResult(
                text=transcript_text,
                language=language,
                confidence=confidence,
                segments=segments,
                processing_time=0.0,  # Будет обновлено
                method="openai_api",
                metadata={"openai_response": asdict(response) if hasattr(response, '__dict__') else {}}
            )
            
        except Exception as e:
            self.logger.error(f"Ошибка OpenAI API: {e}")
            # Fallback на локальную модель или mock
            if self.local_model:
                return self._transcribe_with_local_model(audio_path, language)
            else:
                return self._mock_transcription(audio_path)
    
    def _transcribe_with_local_model(self, audio_path: str, language: str) -> TranscriptionResult:
        """
        Транскрипция через локальную модель Whisper.
        
        Args:
            audio_path: Путь к аудио файлу
            language: Язык транскрипции
            
        Returns:
            TranscriptionResult
        """
        try:
            if not self.local_model:
                raise RuntimeError("Локальная модель не загружена")
            
            # Подготовка параметров
            options = {}
            if language and language != "unknown":
                options["language"] = language
            
            # Транскрипция
            result = self.local_model.transcribe(audio_path, **options)
            
            transcript_text = result["text"].strip()
            detected_language = result.get("language", language)
            confidence = 0.8  # Whisper не возвращает confidence score
            
            # Извлечение сегментов
            segments = []
            if "segments" in result:
                for segment in result["segments"]:
                    segments.append({
                        "text": segment["text"].strip(),
                        "start": segment["start"],
                        "end": segment["end"]
                    })
            
            return TranscriptionResult(
                text=transcript_text,
                language=detected_language,
                confidence=confidence,
                segments=segments,
                processing_time=0.0,  # Будет обновлено
                method="local_model",
                metadata={"whisper_result": result}
            )
            
        except Exception as e:
            self.logger.error(f"Ошибка локальной модели: {e}")
            return self._mock_transcription(audio_path)
    
    def batch_transcribe(self, 
                        file_paths: List[str],
                        language: Optional[str] = None,
                        save_to_memory: bool = True) -> List[TranscriptionResult]:
        """
        Batch обработка множества аудио/видео файлов.
        
        Args:
            file_paths: Список путей к файлам
            language: Язык транскрипции (опционально)
            save_to_memory: Сохранить ли результаты в MemoryManager
            
        Returns:
            Список результатов транскрипции
        """
        results = []
        
        if not file_paths:
            return results
        
        self.logger.info(f"Начало batch обработки {len(file_paths)} файлов")
        
        # Ограничение количества потоков
        max_workers = min(self.max_workers, len(file_paths))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Запуск задач
            future_to_path = {
                executor.submit(
                    self.transcribe_audio, 
                    file_path, 
                    language, 
                    save_to_memory
                ): file_path 
                for file_path in file_paths
            }
            
            # Сбор результатов
            for future in as_completed(future_to_path):
                file_path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                    self.logger.debug(f"Обработан файл: {file_path}")
                except Exception as e:
                    self.logger.error(f"Ошибка при обработке {file_path}: {e}")
                    
                    # Добавление ошибки в результат
                    error_result = TranscriptionResult(
                        text=f"Ошибка транскрипции: {str(e)}",
                        language="error",
                        confidence=0.0,
                        segments=[],
                        processing_time=0.0,
                        method="error",
                        metadata={"file_path": file_path, "error": str(e)}
                    )
                    results.append(error_result)
        
        self.logger.info(f"Batch обработка завершена: {len(results)} результатов")
        return results
    
    async def batch_transcribe_async(self, 
                                   file_paths: List[str],
                                   language: Optional[str] = None,
                                   save_to_memory: bool = True) -> List[TranscriptionResult]:
        """
        Асинхронная batch обработка файлов.
        
        Args:
            file_paths: Список путей к файлам
            language: Язык транскрипции (опционально)
            save_to_memory: Сохранить ли результаты в MemoryManager
            
        Returns:
            Список результатов транскрипции
        """
        # Для простоты используем синхронную версию в отдельном потоке
        import asyncio
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            results = await loop.run_in_executor(
                executor, 
                self.batch_transcribe, 
                file_paths, 
                language, 
                save_to_memory
            )
        
        return results
    
    def save_transcript(self, result: TranscriptionResult, output_path: str) -> str:
        """
        Сохранение результата транскрипции в файл.
        
        Args:
            result: Результат транскрипции
            output_path: Путь для сохранения
            
        Returns:
            Путь к сохраненному файлу
        """
        try:
            # Создание директории если не существует
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Сохранение в JSON формате
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(result), f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Результат транскрипции сохранен: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении транскрипции: {e}")
            raise
    
    def get_transcription_stats(self, results: List[TranscriptionResult]) -> Dict[str, Any]:
        """
        Получение статистики по результатам транскрипции.
        
        Args:
            results: Список результатов транскрипции
            
        Returns:
            Словарь со статистикой
        """
        if not results:
            return {"total_files": 0}
        
        total_files = len(results)
        total_duration = sum(r.metadata.get("duration", 0) for r in results if r.metadata)
        total_processing_time = sum(r.processing_time for r in results)
        total_text_length = sum(len(r.text) for r in results)
        
        # Статистика по методам
        method_counts = {}
        language_counts = {}
        
        for result in results:
            # Подсчет методов
            method_counts[result.method] = method_counts.get(result.method, 0) + 1
            
            # Подсчет языков
            lang = result.language
            language_counts[lang] = language_counts.get(lang, 0) + 1
        
        return {
            "total_files": total_files,
            "total_duration_seconds": total_duration,
            "total_processing_time_seconds": total_processing_time,
            "average_processing_time_seconds": total_processing_time / total_files,
            "total_text_characters": total_text_length,
            "average_text_length": total_text_length / total_files,
            "method_distribution": method_counts,
            "language_distribution": language_counts,
            "success_rate": len([r for r in results if r.language != "error"]) / total_files * 100
        }
    
    def cleanup_temp_files(self):
        """Очистка временных файлов."""
        # В текущей реализации временные файлы удаляются сразу
        # Этот метод оставлен для будущих расширений
        pass


# Дополнительные функции для удобства использования

def create_audio_processor(memory_manager=None, 
                          openai_api_key=None, 
                          whisper_model="base", 
                          mock_mode=False) -> AudioProcessor:
    """
    Фабричная функция для создания AudioProcessor.
    
    Args:
        memory_manager: Экземпляр MemoryManager
        openai_api_key: API ключ OpenAI
        whisper_model: Размер модели Whisper
        mock_mode: Режим mock
        
    Returns:
        Экземпляр AudioProcessor
    """
    return AudioProcessor(
        memory_manager=memory_manager,
        openai_api_key=openai_api_key,
        whisper_model=whisper_model,
        mock_mode=mock_mode
    )


def transcribe_single_file(file_path: str, 
                         openai_api_key=None, 
                         whisper_model="base",
                         mock_mode=False) -> TranscriptionResult:
    """
    Удобная функция для транскрипции одного файла.
    
    Args:
        file_path: Путь к файлу
        openai_api_key: API ключ OpenAI
        whisper_model: Размер модели Whisper
        mock_mode: Режим mock
        
    Returns:
        Результат транскрипции
    """
    processor = create_audio_processor(
        openai_api_key=openai_api_key,
        whisper_model=whisper_model,
        mock_mode=mock_mode
    )
    
    return processor.transcribe_audio(file_path)


def batch_transcribe_files(file_paths: List[str], 
                         openai_api_key=None,
                         whisper_model="base", 
                         mock_mode=False) -> List[TranscriptionResult]:
    """
    Удобная функция для batch транскрипции файлов.
    
    Args:
        file_paths: Список путей к файлам
        openai_api_key: API ключ OpenAI
        whisper_model: Размер модели Whisper
        mock_mode: Режим mock
        
    Returns:
        Список результатов транскрипции
    """
    processor = create_audio_processor(
        openai_api_key=openai_api_key,
        whisper_model=whisper_model,
        mock_mode=mock_mode
    )
    
    return processor.batch_transcribe(file_paths)


if __name__ == "__main__":
    # Пример использования
    import argparse
    
    parser = argparse.ArgumentParser(description="Audio Processor с Whisper")
    parser.add_argument("file_path", help="Путь к аудио/видео файлу")
    parser.add_argument("--api-key", help="OpenAI API ключ")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--mock", action="store_true", help="Режим mock")
    parser.add_argument("--language", help="Язык транскрипции")
    parser.add_argument("--output", help="Путь для сохранения результата")
    
    args = parser.parse_args()
    
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    # Создание процессора
    processor = create_audio_processor(
        openai_api_key=args.api_key,
        whisper_model=args.model,
        mock_mode=args.mock
    )
    
    try:
        # Транскрипция
        result = processor.transcribe_audio(args.file_path, args.language)
        
        print(f"Результат транскрипции:")
        print(f"Текст: {result.text}")
        print(f"Язык: {result.language}")
        print(f"Уверенность: {result.confidence}")
        print(f"Метод: {result.method}")
        print(f"Время обработки: {result.processing_time:.2f}с")
        
        # Сохранение если указан output
        if args.output:
            processor.save_transcript(result, args.output)
            print(f"Результат сохранен: {args.output}")
        
    except Exception as e:
        print(f"Ошибка: {e}")
