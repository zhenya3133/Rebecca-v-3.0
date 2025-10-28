"""
Unit тесты для ImageProcessor.

Запуск тестов:
    pytest src/ingest/test_image_processor.py -v
"""

import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

# Тестируемый модуль
from src.ingest.image_processor import (
    ImageProcessor, ImageInfo, OCRResult, TableDetection, 
    FaceDetection, BatchResult
)


class TestImageProcessor:
    """Набор тестов для ImageProcessor."""
    
    @pytest.fixture
    def temp_dir(self):
        """Создание временной директории для тестов."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def sample_image_path(self, temp_dir):
        """Создание образца изображения для тестов."""
        try:
            from PIL import Image
            # Создаем простое тестовое изображение
            img = Image.new('RGB', (100, 100), color='white')
            # Добавляем немного текста (просто для визуализации)
            path = os.path.join(temp_dir, 'test_image.png')
            img.save(path)
            return path
        except ImportError:
            # Если PIL не установлен, создаем пустой файл
            path = os.path.join(temp_dir, 'test_image.png')
            with open(path, 'w') as f:
                f.write('')
            return path
    
    @pytest.fixture
    def mock_memory_manager(self):
        """Мок для MemoryManager."""
        mock_memory = Mock()
        mock_memory.vault = Mock()
        return Mock(vault=mock_memory)
    
    def test_initialization(self, mock_memory_manager):
        """Тест инициализации ImageProcessor."""
        processor = ImageProcessor(memory_manager=mock_memory_manager)
        assert processor.memory_manager == mock_memory_manager
        assert processor.tesseract_config == '--oem 3 --psm 6'
    
    def test_initialization_with_config(self):
        """Тест инициализации с кастомной конфигурацией."""
        config = '--oem 1 --psm 8'
        processor = ImageProcessor(tesseract_config=config)
        assert processor.tesseract_config == config
    
    def test_calculate_checksum(self, sample_image_path):
        """Тест вычисления контрольной суммы."""
        processor = ImageProcessor()
        checksum = processor._calculate_checksum(sample_image_path)
        assert isinstance(checksum, str)
        assert len(checksum) == 32  # MD5 хеш
    
    def test_calculate_checksum_different_files(self, temp_dir):
        """Тест различия контрольных сумм для разных файлов."""
        processor = ImageProcessor()
        
        # Создаем два разных файла
        file1 = os.path.join(temp_dir, 'file1.txt')
        file2 = os.path.join(temp_dir, 'file2.txt')
        
        with open(file1, 'w') as f:
            f.write('content1')
        with open(file2, 'w') as f:
            f.write('content2')
        
        checksum1 = processor._calculate_checksum(file1)
        checksum2 = processor._calculate_checksum(file2)
        
        assert checksum1 != checksum2
    
    @patch('src.ingest.image_processor.PILImage')
    def test_get_image_info(self, mock_pil, sample_image_path):
        """Тест получения информации об изображении."""
        # Мок PIL Image
        mock_img = Mock()
        mock_img.format = 'PNG'
        mock_img.size = (100, 100)
        mock_img.mode = 'RGB'
        mock_img.info = {'dpi': (300, 300)}
        
        mock_context = MagicMock()
        mock_context.__enter__ = Mock(return_value=mock_img)
        mock_context.__exit__ = Mock(return_value=None)
        
        with patch('builtins.open', create=True):
            mock_pil.Image.open.return_value = mock_context
        
        processor = ImageProcessor()
        info = processor.get_image_info(sample_image_path)
        
        assert isinstance(info, ImageInfo)
        assert info.path == sample_image_path
        assert info.format == 'PNG'
        assert info.size == (100, 100)
        assert info.mode == 'RGB'
        assert info.file_size > 0
        assert info.checksum is not None
    
    def test_get_image_info_file_not_found(self):
        """Тест ошибки при отсутствии файла."""
        processor = ImageProcessor()
        with pytest.raises(FileNotFoundError):
            processor.get_image_info('nonexistent_file.jpg')
    
    def test_get_image_info_unsupported_format(self, temp_dir):
        """Тест ошибки при неподдерживаемом формате."""
        # Создаем файл с неподдерживаемым расширением
        file_path = os.path.join(temp_dir, 'test.xyz')
        with open(file_path, 'w') as f:
            f.write('test')
        
        processor = ImageProcessor()
        with pytest.raises(ValueError):
            processor.get_image_info(file_path)
    
    @patch('src.ingest.image_processor.PILImage')
    def test_preprocess_image(self, mock_pil, sample_image_path):
        """Тест предобработки изображения."""
        # Мок PIL Image
        mock_img = Mock()
        mock_context = MagicMock()
        mock_context.__enter__ = Mock(return_value=mock_img)
        mock_context.__exit__ = Mock(return_value=None)
        
        with patch('builtins.open', create=True):
            mock_pil.Image.open.return_value = mock_context
        
        processor = ImageProcessor()
        output_path = processor.preprocess_image(sample_image_path, temp_dir)
        
        assert output_path.endswith('_processed.png')
        assert 'test_image_processed.png' in output_path
    
    def test_preprocess_image_without_pil(self):
        """Тест ошибки при отсутствии PIL."""
        processor = ImageProcessor()
        # Создаем пустой файл для теста
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b'test')
            temp_path = tmp.name
        
        try:
            with patch('src.ingest.image_processor.PILImage', None):
                with pytest.raises(ImportError):
                    processor.preprocess_image(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_detect_language_cyrillic(self):
        """Тест определения русского языка."""
        processor = ImageProcessor()
        text = "Привет мир! Это тест на русском языке."
        lang = processor.detect_language(text)
        assert lang == 'ru'
    
    def test_detect_language_latin(self):
        """Тест определения английского языка."""
        processor = ImageProcessor()
        text = "Hello world! This is a test in English."
        lang = processor.detect_language(text)
        assert lang == 'en'
    
    def test_detect_language_empty(self):
        """Тест определения языка для пустого текста."""
        processor = ImageProcessor()
        lang = processor.detect_language("")
        assert lang == 'unknown'
    
    def test_detect_language_whitespace(self):
        """Тест определения языка для пробельного текста."""
        processor = ImageProcessor()
        lang = processor.detect_language("   \n\t   ")
        assert lang == 'unknown'
    
    @patch('src.ingest.image_processor.cv2')
    @patch('src.ingest.image_processor.pytesseract')
    def test_extract_text_mock_mode(self, mock_tesseract, mock_cv2, sample_image_path):
        """Тест извлечения текста в mock режиме."""
        processor = ImageProcessor()
        processor.use_mock = True
        
        result = processor.extract_text(sample_image_path)
        
        assert isinstance(result, OCRResult)
        assert result.text == f"[MOCK OCR] Извлеченный текст из {os.path.basename(sample_image_path)}"
        assert result.confidence == 0.85
        assert result.language == 'ru'
        assert result.method == 'mock'
    
    @patch('src.ingest.image_processor.cv2')
    @patch('src.ingest.image_processor.pytesseract')
    def test_extract_text_with_preprocessing(self, mock_tesseract, mock_cv2, sample_image_path):
        """Тест извлечения текста с предобработкой."""
        # Настройка моков
        mock_cv2.imread.return_value = Mock()
        mock_cv2.cvtColor.return_value = Mock()
        mock_cv2.createCLAHE.return_value.apply.return_value = Mock()
        mock_cv2.medianBlur.return_value = Mock()
        mock_cv2.threshold.return_value = (None, Mock())
        mock_tesseract.image_to_string.return_value = "Extracted text"
        
        processor = ImageProcessor()
        result = processor.extract_text(sample_image_path, preprocessing=True)
        
        assert isinstance(result, OCRResult)
        assert result.text == "Extracted text"
        assert result.method == 'pytesseract'
        assert 'grayscale' in result.preprocessing_applied
        assert 'contrast' in result.preprocessing_applied
    
    @patch('src.ingest.image_processor.cv2')
    def test_extract_text_cv2_not_available(self, sample_image_path):
        """Тест ошибки при отсутствии OpenCV."""
        with patch('src.ingest.image_processor.cv2', None):
            processor = ImageProcessor()
            with pytest.raises(ImportError):
                processor.extract_text(sample_image_path)
    
    @patch('src.ingest.image_processor.cv2')
    def test_extract_tables_mock_mode(self, mock_cv2, sample_image_path):
        """Тест извлечения таблиц в mock режиме."""
        processor = ImageProcessor()
        processor.use_mock = True
        
        result = processor.extract_tables_from_image(sample_image_path)
        
        assert isinstance(result, TableDetection)
        assert len(result.rows) > 0
        assert result.confidence == 0.8
        assert result.method == "mock"
    
    @patch('src.ingest.image_processor.cv2')
    def test_extract_tables_with_real_cv2(self, mock_cv2, sample_image_path):
        """Тест извлечения таблиц с реальным OpenCV."""
        # Настройка моков для OpenCV
        mock_image = Mock()
        mock_gray = Mock()
        
        mock_cv2.imread.return_value = mock_image
        mock_cv2.cvtColor.return_value = mock_gray
        mock_cv2.threshold.return_value = (None, mock_image)
        mock_cv2.getStructuringElement.return_value = Mock()
        mock_cv2.morphologyEx.return_value = mock_image
        mock_cv2.addWeighted.return_value = mock_image
        mock_cv2.findContours.return_value = ([Mock()], None)
        
        # Мок pytesseract
        with patch('src.ingest.image_processor.pytesseract') as mock_tesseract:
            mock_tesseract.image_to_string.return_value = "Column1 Column2\nData1 Data2"
            
            processor = ImageProcessor()
            result = processor.extract_tables_from_image(sample_image_path)
            
            assert isinstance(result, TableDetection)
            assert len(result.rows) > 0
            assert result.method == "opencv+pytesseract"
    
    @patch('src.ingest.image_processor.cv2')
    def test_extract_faces_mock_mode(self, mock_cv2, sample_image_path):
        """Тест обнаружения лиц в mock режиме."""
        processor = ImageProcessor()
        processor.use_mock = True
        
        result = processor.extract_faces(sample_image_path)
        
        assert isinstance(result, FaceDetection)
        assert result.count == 1
        assert len(result.bounding_boxes) == 1
        assert result.method == "mock"
    
    @patch('src.ingest.image_processor.cv2')
    def test_extract_faces_with_real_cv2(self, mock_cv2, sample_image_path):
        """Тест обнаружения лиц с реальным OpenCV."""
        # Настройка моков
        mock_image = Mock()
        mock_gray = Mock()
        mock_cascade = Mock()
        mock_face = [50, 50, 100, 100]  # [x, y, w, h]
        
        mock_cv2.imread.return_value = mock_image
        mock_cv2.cvtColor.return_value = mock_gray
        mock_cv2.CascadeClassifier.return_value = mock_cascade
        mock_cascade.detectMultiScale.return_value = [mock_face]
        
        processor = ImageProcessor()
        result = processor.extract_faces(sample_image_path)
        
        assert isinstance(result, FaceDetection)
        assert result.count == 1
        assert result.method == "haarcascade"
    
    def test_batch_process_images(self, sample_image_path, mock_memory_manager):
        """Тест batch обработки изображений."""
        processor = ImageProcessor(memory_manager=mock_memory_manager)
        processor.use_mock = True  # Используем mock для тестирования
        
        result = processor.batch_process_images(
            [sample_image_path], 
            operations=['extract_text', 'get_info']
        )
        
        assert isinstance(result, BatchResult)
        assert result.total_processed == 1
        assert result.successful == 1
        assert result.failed == 0
        assert sample_image_path in result.results
        
        # Проверка интеграции с MemoryManager
        mock_memory_manager.vault.store_secret.assert_called()
    
    def test_batch_process_images_with_errors(self, temp_dir, mock_memory_manager):
        """Тест batch обработки с ошибками."""
        processor = ImageProcessor(memory_manager=mock_memory_manager)
        processor.use_mock = True
        
        # Список с несуществующими файлами
        non_existent_files = ['nonexistent1.jpg', 'nonexistent2.png']
        
        result = processor.batch_process_images(non_existent_files)
        
        assert result.total_processed == 2
        assert result.successful == 0
        assert result.failed == 2
        assert len(result.errors) == 2
    
    @patch('src.ingest.image_processor.json')
    def test_save_results_to_file(self, mock_json, temp_dir):
        """Тест сохранения результатов в файл."""
        processor = ImageProcessor()
        
        # Создаем mock результат
        result = BatchResult(
            total_processed=1,
            successful=1,
            failed=0,
            results={'test': {'data': 'value'}},
            errors={}
        )
        
        output_path = os.path.join(temp_dir, 'results.json')
        processor.save_results_to_file(result, output_path)
        
        mock_json.dump.assert_called_once()
    
    def test_save_results_to_file_error(self, temp_dir):
        """Тест ошибки при сохранении результатов."""
        processor = ImageProcessor()
        
        result = BatchResult(
            total_processed=0,
            successful=0,
            failed=0,
            results={},
            errors={}
        )
        
        output_path = os.path.join(temp_dir, 'results.json')
        
        # Имитируем ошибку записи
        with patch('builtins.open', side_effect=IOError("Permission denied")):
            with pytest.raises(Exception):
                processor.save_results_to_file(result, output_path)
    
    def test_get_supported_languages(self):
        """Тест получения списка поддерживаемых языков."""
        processor = ImageProcessor()
        languages = processor.get_supported_languages()
        
        assert isinstance(languages, dict)
        assert 'ru' in languages
        assert 'en' in languages
        assert languages['ru'] == 'Русский'
        assert languages['en'] == 'English'
    
    def test_get_supported_formats(self):
        """Тест получения списка поддерживаемых форматов."""
        processor = ImageProcessor()
        formats = processor.get_supported_formats()
        
        assert isinstance(formats, list)
        assert '.png' in formats
        assert '.jpg' in formats
        assert '.jpeg' in formats
        assert '.bmp' in formats


class TestImageInfo:
    """Тесты для модели ImageInfo."""
    
    def test_image_info_creation(self):
        """Тест создания ImageInfo."""
        info = ImageInfo(
            path="/test/image.png",
            format="PNG",
            size=(1920, 1080),
            mode="RGB",
            file_size=1024000,
            checksum="abc123"
        )
        
        assert info.path == "/test/image.png"
        assert info.format == "PNG"
        assert info.size == (1920, 1080)
        assert info.mode == "RGB"
        assert info.file_size == 1024000
        assert info.checksum == "abc123"
    
    def test_image_info_with_optional_fields(self):
        """Тест создания ImageInfo с опциональными полями."""
        info = ImageInfo(
            path="/test/image.png",
            format="PNG",
            size=(1920, 1080),
            mode="RGBA",
            file_size=1024000,
            checksum="abc123",
            dpi=(300, 300),
            has_transparency=True,
            created_at="2023-01-01T00:00:00",
            modified_at="2023-01-01T12:00:00"
        )
        
        assert info.dpi == (300, 300)
        assert info.has_transparency == True
        assert info.created_at == "2023-01-01T00:00:00"
        assert info.modified_at == "2023-01-01T12:00:00"


class TestOCRResult:
    """Тесты для модели OCRResult."""
    
    def test_ocr_result_creation(self):
        """Тест создания OCRResult."""
        result = OCRResult(
            text="Test text",
            confidence=0.95,
            language="ru",
            method="pytesseract"
        )
        
        assert result.text == "Test text"
        assert result.confidence == 0.95
        assert result.language == "ru"
        assert result.method == "pytesseract"
        assert result.words == []
        assert result.preprocessing_applied == []
    
    def test_ocr_result_with_words(self):
        """Тест создания OCRResult с данными о словах."""
        words = [
            {"text": "word1", "confidence": 95, "bbox": [10, 10, 50, 30]},
            {"text": "word2", "confidence": 90, "bbox": [60, 10, 100, 30]}
        ]
        
        result = OCRResult(
            text="word1 word2",
            confidence=0.92,
            language="en",
            words=words,
            preprocessing_applied=["grayscale", "contrast"]
        )
        
        assert len(result.words) == 2
        assert result.words[0]["text"] == "word1"
        assert result.preprocessing_applied == ["grayscale", "contrast"]


class TestBatchResult:
    """Тесты для модели BatchResult."""
    
    def test_batch_result_creation(self):
        """Тест создания BatchResult."""
        result = BatchResult(
            total_processed=5,
            successful=4,
            failed=1,
            results={"image1.jpg": {"ocr": "text1"}},
            errors={"image2.jpg": "error message"}
        )
        
        assert result.total_processed == 5
        assert result.successful == 4
        assert result.failed == 1
        assert "image1.jpg" in result.results
        assert "image2.jpg" in result.errors


class TestUtilityFunctions:
    """Тесты для вспомогательных функций."""
    
    def test_create_image_processor(self):
        """Тест создания ImageProcessor через функцию-помощник."""
        from src.ingest.image_processor import create_image_processor
        
        processor = create_image_processor()
        assert isinstance(processor, ImageProcessor)
    
    def test_create_image_processor_with_memory_manager(self, mock_memory_manager):
        """Тест создания ImageProcessor с MemoryManager."""
        from src.ingest.image_processor import create_image_processor
        
        processor = create_image_processor(memory_manager=mock_memory_manager)
        assert processor.memory_manager == mock_memory_manager
    
    @patch('src.ingest.image_processor.ImageProcessor')
    def test_quick_ocr(self, mock_processor_class):
        """Тест функции быстрого OCR."""
        from src.ingest.image_processor import quick_ocr
        
        mock_result = Mock()
        mock_result.text = "Extracted text"
        
        mock_processor = Mock()
        mock_processor.extract_text.return_value = mock_result
        mock_processor_class.return_value = mock_processor
        
        text = quick_ocr("test.jpg")
        assert text == "Extracted text"
    
    @patch('src.ingest.image_processor.ImageProcessor')
    def test_quick_batch_ocr(self, mock_processor_class):
        """Тест функции быстрого batch OCR."""
        from src.ingest.image_processor import quick_batch_ocr
        
        mock_batch_result = Mock()
        mock_batch_result.results = {
            "img1.jpg": {"extract_text": {"text": "text1"}},
            "img2.jpg": {"extract_text": {"text": "text2"}}
        }
        
        mock_processor = Mock()
        mock_processor.batch_process_images.return_value = mock_batch_result
        mock_processor_class.return_value = mock_processor
        
        results = quick_batch_ocr(["img1.jpg", "img2.jpg"])
        assert results == {"img1.jpg": "text1", "img2.jpg": "text2"}


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Настройка тестового окружения."""
    # Установка переменных окружения для тестирования
    os.environ["TESTING"] = "1"
    
    yield
    
    # Очистка после тестов
    if "TESTING" in os.environ:
        del os.environ["TESTING"]


if __name__ == "__main__":
    # Запуск тестов напрямую
    pytest.main([__file__, "-v"])