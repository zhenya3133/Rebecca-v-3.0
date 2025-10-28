"""
Unit тесты для AudioProcessor.
Тестирование всех основных функций и возможностей.
"""

import pytest
import os
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import time

# Импорты тестируемого модуля
from audio_processor import (
    AudioProcessor,
    AudioInfo,
    TranscriptionResult,
    create_audio_processor,
    transcribe_single_file,
    batch_transcribe_files
)


class TestAudioProcessor:
    """Набор тестов для AudioProcessor."""
    
    @pytest.fixture
    def temp_audio_file(self):
        """Создание временного аудио файла для тестирования."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            # Создание минимального WAV файла (заглушка)
            f.write(b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00')
            temp_path = f.name
        
        yield temp_path
        
        # Очистка
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def temp_video_file(self):
        """Создание временного видео файла для тестирования."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            temp_path = f.name
        
        yield temp_path
        
        # Очистка
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def mock_memory_manager(self):
        """Mock MemoryManager для тестирования."""
        return Mock()
    
    @pytest.fixture
    def audio_processor(self, mock_memory_manager):
        """Создание экземпляра AudioProcessor для тестирования."""
        return AudioProcessor(
            memory_manager=mock_memory_manager,
            mock_mode=True  # Используем mock режим для тестов
        )
    
    def test_initialization(self, mock_memory_manager):
        """Тест инициализации AudioProcessor."""
        processor = AudioProcessor(
            memory_manager=mock_memory_manager,
            mock_mode=True
        )
        
        assert processor.memory_manager == mock_memory_manager
        assert processor.mock_mode is True
        assert processor.openai_api_key is None
        assert processor.whisper_model == "base"
    
    def test_supported_formats(self, audio_processor):
        """Тест получения поддерживаемых форматов."""
        formats = audio_processor.get_supported_formats()
        
        assert "audio" in formats
        assert "video" in formats
        assert "all" in formats
        
        assert ".mp3" in formats["audio"]
        assert ".wav" in formats["audio"]
        assert ".mp4" in formats["video"]
        assert ".avi" in formats["video"]
    
    def test_get_audio_info_mock(self, audio_processor, temp_audio_file):
        """Тест получения метаданных аудио файла (mock режим)."""
        # Создание фиктивного файла для теста
        audio_processor._check_dependencies = Mock()  # Мокаем проверку зависимостей
        
        # Мокаем mediainfo
        with patch('ingest.audio_processor.mediainfo') as mock_mediainfo:
            mock_mediainfo.return_value = {
                'bit_rate': '128000',
                'duration': '10.5'
            }
            
            info = audio_processor.get_audio_info(temp_audio_file)
            
            assert isinstance(info, AudioInfo)
            assert info.path == temp_audio_file
            assert info.format == 'wav'
            assert info.file_size > 0
    
    def test_get_audio_info_file_not_found(self, audio_processor):
        """Тест обработки ошибки отсутствующего файла."""
        with pytest.raises(FileNotFoundError):
            audio_processor.get_audio_info("/nonexistent/file.wav")
    
    def test_get_audio_info_unsupported_format(self, audio_processor):
        """Тест обработки неподдерживаемого формата."""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError):
                audio_processor.get_audio_info(temp_path)
        finally:
            os.unlink(temp_path)
    
    @patch('ingest.audio_processor.ffmpeg')
    def test_extract_audio_from_video(self, mock_ffmpeg, audio_processor, temp_video_file):
        """Тест извлечения аудио из видео."""
        # Мокаем ffmpeg
        mock_ffmpeg.input.return_value.output.return_value.overwrite_output.return_value.run.return_value = None
        
        output_path = audio_processor.extract_audio_from_video(temp_video_file)
        
        assert output_path.endswith('.wav')
        mock_ffmpeg.input.assert_called_once_with(temp_video_file)
    
    def test_extract_audio_without_ffmpeg(self, audio_processor, temp_video_file):
        """Тест обработки отсутствия FFmpeg."""
        with patch('ingest.audio_processor.VIDEO_LIBS_AVAILABLE', False):
            with pytest.raises(RuntimeError):
                audio_processor.extract_audio_from_video(temp_video_file)
    
    def test_detect_language_mock(self, audio_processor, temp_audio_file):
        """Тест определения языка (mock режим)."""
        language = audio_processor.detect_language(temp_audio_file)
        assert language == "ru"  # Mock возвращает русский
    
    @patch('ingest.audio_processor.whisper')
    def test_detect_language_with_whisper(self, mock_whisper, temp_audio_file):
        """Тест определения языка с реальным Whisper."""
        audio_processor.mock_mode = False
        audio_processor.local_model = Mock()
        
        # Мокаем результат определения языка
        mock_result = {"language": "en"}
        audio_processor.local_model.detect_language.return_value = mock_result
        
        language = audio_processor.detect_language(temp_audio_file)
        assert language == "en"
    
    def test_segment_transcript(self, audio_processor):
        """Тест сегментации транскрипта."""
        long_text = "Это первое предложение. Это второе предложение. " * 50
        segments = audio_processor.segment_transcript(long_text, max_length=100)
        
        assert len(segments) > 1
        for segment in segments:
            assert len(segment) <= 100
    
    def test_segment_transcript_short_text(self, audio_processor):
        """Тест сегментации короткого текста."""
        short_text = "Короткий текст"
        segments = audio_processor.segment_transcript(short_text, max_length=100)
        
        assert len(segments) == 1
        assert segments[0] == short_text
    
    def test_segment_transcript_empty(self, audio_processor):
        """Тест сегментации пустого текста."""
        segments = audio_processor.segment_transcript("", max_length=100)
        assert segments == []
    
    def test_mock_transcription(self, audio_processor, temp_audio_file):
        """Тест mock транскрипции."""
        result = audio_processor._mock_transcription(temp_audio_file)
        
        assert isinstance(result, TranscriptionResult)
        assert result.method == "mock"
        assert result.language == "ru"
        assert "mock транскрипция" in result.text
        assert result.confidence == 0.5
    
    @pytest.mark.asyncio
    async def test_save_to_memory(self, audio_processor, mock_memory_manager, temp_audio_file):
        """Тест сохранения в MemoryManager."""
        mock_memory_manager.store = Mock(return_value="test_id")
        
        result = TranscriptionResult(
            text="Тестовый текст",
            language="ru",
            confidence=0.9,
            segments=[],
            processing_time=1.0,
            method="test",
            metadata={}
        )
        
        item_id = await audio_processor._save_to_memory(
            temp_audio_file, result, {"test": "data"}
        )
        
        assert item_id == "test_id"
        mock_memory_manager.store.assert_called_once()
    
    def test_transcribe_audio_mock_mode(self, audio_processor, temp_audio_file):
        """Тест транскрипции в mock режиме."""
        result = audio_processor.transcribe_audio(temp_audio_file)
        
        assert isinstance(result, TranscriptionResult)
        assert result.method == "mock"
        assert result.processing_time > 0
        assert result.language == "ru"
    
    def test_transcribe_audio_file_not_found(self, audio_processor):
        """Тест обработки отсутствующего файла при транскрипции."""
        with pytest.raises(FileNotFoundError):
            audio_processor.transcribe_audio("/nonexistent/file.wav")
    
    def test_transcribe_audio_unsupported_format(self, audio_processor):
        """Тест обработки неподдерживаемого формата при транскрипции."""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError):
                audio_processor.transcribe_audio(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_batch_transcribe(self, audio_processor, temp_audio_file):
        """Тест batch транскрипции."""
        file_paths = [temp_audio_file, temp_audio_file]
        results = audio_processor.batch_transcribe(file_paths)
        
        assert len(results) == 2
        for result in results:
            assert isinstance(result, TranscriptionResult)
            assert result.method == "mock"
    
    def test_batch_transcribe_empty_list(self, audio_processor):
        """Тест batch транскрипции пустого списка."""
        results = audio_processor.batch_transcribe([])
        assert results == []
    
    def test_save_transcript(self, audio_processor, temp_dir="test_dir"):
        """Тест сохранения транскрипции в файл."""
        result = TranscriptionResult(
            text="Тестовый текст",
            language="ru",
            confidence=0.9,
            segments=[],
            processing_time=1.0,
            method="test",
            metadata={}
        )
        
        # Создание временной директории
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "transcript.json")
            
            saved_path = audio_processor.save_transcript(result, output_path)
            
            assert saved_path == output_path
            assert os.path.exists(output_path)
            
            # Проверка содержимого файла
            with open(output_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            
            assert saved_data["text"] == "Тестовый текст"
            assert saved_data["language"] == "ru"
    
    def test_get_transcription_stats(self, audio_processor):
        """Тест получения статистики транскрипции."""
        results = [
            TranscriptionResult(
                text="Первый текст",
                language="ru",
                confidence=0.9,
                segments=[],
                processing_time=1.0,
                method="mock",
                metadata={"duration": 10.0}
            ),
            TranscriptionResult(
                text="Second text",
                language="en",
                confidence=0.8,
                segments=[],
                processing_time=2.0,
                method="mock",
                metadata={"duration": 15.0}
            )
        ]
        
        stats = audio_processor.get_transcription_stats(results)
        
        assert stats["total_files"] == 2
        assert stats["total_duration_seconds"] == 25.0
        assert stats["total_processing_time_seconds"] == 3.0
        assert stats["method_distribution"]["mock"] == 2
        assert stats["language_distribution"]["ru"] == 1
        assert stats["language_distribution"]["en"] == 1
    
    def test_get_transcription_stats_empty(self, audio_processor):
        """Тест получения статистики для пустого списка."""
        stats = audio_processor.get_transcription_stats([])
        
        assert stats["total_files"] == 0
    
    def test_factory_function(self, mock_memory_manager):
        """Тест фабричной функции create_audio_processor."""
        processor = create_audio_processor(
            memory_manager=mock_memory_manager,
            mock_mode=True
        )
        
        assert isinstance(processor, AudioProcessor)
        assert processor.memory_manager == mock_memory_manager
        assert processor.mock_mode is True
    
    def test_convenience_functions(self, temp_audio_file):
        """Тест удобных функций transcribe_single_file и batch_transcribe_files."""
        # Тест transcribe_single_file
        result = transcribe_single_file(temp_audio_file, mock_mode=True)
        assert isinstance(result, TranscriptionResult)
        assert result.method == "mock"
        
        # Тест batch_transcribe_files
        results = batch_transcribe_files([temp_audio_file], mock_mode=True)
        assert len(results) == 1
        assert isinstance(results[0], TranscriptionResult)


class TestAudioInfo:
    """Тесты для структуры AudioInfo."""
    
    def test_audio_info_creation(self):
        """Тест создания AudioInfo."""
        info = AudioInfo(
            duration=10.5,
            sample_rate=44100,
            channels=2,
            format="mp3",
            bit_rate=128000,
            file_size=1024000,
            path="/test/file.mp3"
        )
        
        assert info.duration == 10.5
        assert info.sample_rate == 44100
        assert info.channels == 2
        assert info.format == "mp3"
        assert info.bit_rate == 128000
        assert info.file_size == 1024000
        assert info.path == "/test/file.mp3"


class TestTranscriptionResult:
    """Тесты для структуры TranscriptionResult."""
    
    def test_transcription_result_creation(self):
        """Тест создания TranscriptionResult."""
        result = TranscriptionResult(
            text="Тестовый текст",
            language="ru",
            confidence=0.9,
            segments=[{"text": "Сегмент 1", "start": 0, "end": 5}],
            processing_time=2.5,
            method="test_method",
            metadata={"key": "value"}
        )
        
        assert result.text == "Тестовый текст"
        assert result.language == "ru"
        assert result.confidence == 0.9
        assert len(result.segments) == 1
        assert result.processing_time == 2.5
        assert result.method == "test_method"
        assert result.metadata["key"] == "value"


class TestIntegrationWithMemoryManager:
    """Интеграционные тесты с MemoryManager."""
    
    @pytest.fixture
    def mock_memory_manager_integration(self):
        """Mock MemoryManager для интеграционных тестов."""
        manager = Mock()
        manager.store = Mock(return_value="integration_test_id")
        return manager
    
    @pytest.mark.asyncio
    async def test_full_integration(self, mock_memory_manager_integration, temp_audio_file):
        """Тест полной интеграции с MemoryManager."""
        processor = AudioProcessor(
            memory_manager=mock_memory_manager_integration,
            mock_mode=True
        )
        
        # Транскрипция с сохранением в память
        result = processor.transcribe_audio(temp_audio_file, save_to_memory=True)
        
        assert isinstance(result, TranscriptionResult)
        assert result.method == "mock"
        
        # Проверка что данные были сохранены в память
        mock_memory_manager_integration.store.assert_called_once()
        
        # Проверка аргументов сохранения
        call_args = mock_memory_manager_integration.store.call_args
        assert call_args[1]["layer"].value == "episodic"
        assert call_args[1]["data"]["transcript"] == result.text
        assert call_args[1]["metadata"]["type"] == "audio_transcription"


class TestErrorHandling:
    """Тесты обработки ошибок."""
    
    def test_api_key_fallback(self, temp_audio_file):
        """Тест fallback при отсутствии API ключа."""
        processor = AudioProcessor(
            openai_api_key=None,
            mock_mode=False
        )
        
        # Должен переключиться в mock режим
        assert processor.mock_mode is True
    
    def test_whisper_model_loading_error(self, mock_memory_manager):
        """Тест обработки ошибки загрузки модели Whisper."""
        with patch('ingest.audio_processor.WHISPER_AVAILABLE', True):
            with patch('ingest.audio_processor.whisper.load_model') as mock_load:
                mock_load.side_effect = Exception("Model loading failed")
                
                processor = AudioProcessor(
                    memory_manager=mock_memory_manager,
                    mock_mode=False
                )
                
                # Должен переключиться в mock режим при ошибке загрузки модели
                assert processor.mock_mode is True


if __name__ == "__main__":
    # Запуск тестов
    pytest.main([__file__, "-v"])
