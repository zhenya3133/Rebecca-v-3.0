"""Unit тесты для PDFProcessor."""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import io
import sys

# Добавляем путь к src для импортов
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Импортируем только нужные модули напрямую, минуя ingest.__init__
import importlib.util
spec = importlib.util.spec_from_file_location("pdf_processor", Path(__file__).parent.parent / "src" / "ingest" / "pdf_processor.py")
pdf_processor_module = importlib.util.module_from_spec(spec)
sys.modules["pdf_processor"] = pdf_processor_module
spec.loader.exec_module(pdf_processor_module)

PDFProcessor = pdf_processor_module.PDFProcessor
PDFType = pdf_processor_module.PDFType
PDFMetadata = pdf_processor_module.PDFMetadata
ExtractionResult = pdf_processor_module.ExtractionResult
ProgressCallback = pdf_processor_module.ProgressCallback


class TestPDFProcessor:
    """Тесты для класса PDFProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Создает экземпляр процессора для тестов."""
        return PDFProcessor(ocr_enabled=True, lang="rus+eng")
    
    @pytest.fixture
    def sample_pdf_path(self):
        """Создает путь к тестовому PDF файлу."""
        # В реальных тестах здесь был бы путь к настоящему PDF файлу
        return "/path/to/sample.pdf"
    
    def test_init(self, processor):
        """Тест инициализации процессора."""
        assert processor.ocr_enabled is True
        assert processor.ocr_lang == "rus+eng"
        assert processor.logger is not None
    
    def test_calculate_checksum(self, processor):
        """Тест вычисления контрольной суммы."""
        # Создаем временный файл для тестирования
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
            tmp_file.write("test content")
            tmp_path = tmp_file.name
        
        try:
            checksum = processor._calculate_checksum(tmp_path)
            assert isinstance(checksum, str)
            assert len(checksum) == 64  # SHA256 hash length
            assert all(c in '0123456789abcdef' for c in checksum)
        finally:
            os.unlink(tmp_path)
    
    @patch('pdf_processor.pdfplumber')
    def test_extract_page_count_text_based(self, mock_pdfplumber, processor, sample_pdf_path):
        """Тест извлечения количества страниц для текстового PDF."""
        # Мокируем pdfplumber
        mock_pdf = Mock()
        mock_pdf.pages = [Mock(), Mock(), Mock()]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        result = processor.extract_page_count(sample_pdf_path)
        
        assert result.success is True
        assert result.data == 3
        assert result.error is None
    
    @patch('pdf_processor.PyPDF2')
    @patch('pdf_processor.pdfplumber')
    def test_extract_page_count_fallback(self, mock_pyPDF2, mock_pdfplumber, processor, sample_pdf_path):
        """Тест fallback при извлечении количества страниц."""
        # pdfplumber падает
        mock_pdfplumber.open.side_effect = Exception("PDFPlumber error")
        
        # PyPDF2 работает
        mock_reader = Mock()
        mock_reader.pages = [Mock(), Mock()]
        mock_pyPDF2.PdfReader.return_value.__enter__.return_value = mock_reader
        
        result = processor.extract_page_count(sample_pdf_path)
        
        assert result.success is True
        assert result.data == 2
    
    @patch('pdf_processor.pdfplumber')
    def test_extract_metadata(self, mock_pdfplumber, processor, sample_pdf_path):
        """Тест извлечения метаданных."""
        # Мокируем метаданные
        mock_pdf = Mock()
        mock_pdf.metadata = {
            '/Title': 'Test Document',
            '/Author': 'Test Author',
            '/Subject': 'Test Subject',
            '/Keywords': 'test, keywords',
            '/Creator': 'Test Creator',
            '/Producer': 'Test Producer',
            '/CreationDate': 'D:20230101000000Z',
            '/ModDate': 'D:20230101000000Z'
        }
        mock_pdf.pages = [Mock(), Mock()]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        with patch('pdf_processor.os.stat') as mock_stat:
            mock_stat.return_value.st_size = 1024
            
            result = processor.extract_metadata(sample_pdf_path)
        
        assert result.success is True
        assert isinstance(result.data, PDFMetadata)
        assert result.data.title == 'Test Document'
        assert result.data.author == 'Test Author'
        assert result.data.page_count == 2
        assert result.data.file_size == 1024
        assert result.data.checksum is not None
    
    def test_detect_pdf_type_text_based(self, processor, sample_pdf_path):
        """Тест определения типа PDF - текстовый."""
        with patch('pdf_processor.pdfplumber') as mock_pdfplumber:
            mock_page1 = Mock()
            mock_page1.extract_text.return_value = "This is text content"
            mock_page1.images = []
            
            mock_page2 = Mock()
            mock_page2.extract_text.return_value = "More text content"
            mock_page2.images = []
            
            mock_pdf = Mock()
            mock_pdf.pages = [mock_page1, mock_page2]
            mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
            
            pdf_type = processor.detect_pdf_type(sample_pdf_path, sample_pages=2)
            
            assert pdf_type == PDFType.TEXT_BASED
    
    def test_detect_pdf_type_scanned(self, processor, sample_pdf_path):
        """Тест определения типа PDF - сканированный."""
        with patch('pdf_processor.pdfplumber') as mock_pdfplumber:
            mock_page1 = Mock()
            mock_page1.extract_text.return_value = ""
            mock_page1.images = [{'image': 'data'}]
            
            mock_pdf = Mock()
            mock_pdf.pages = [mock_page1]
            mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
            
            pdf_type = processor.detect_pdf_type(sample_pdf_path, sample_pages=1)
            
            assert pdf_type == PDFType.SCANNED
    
    def test_detect_pdf_type_mixed(self, processor, sample_pdf_path):
        """Тест определения типа PDF - смешанный."""
        with patch('pdf_processor.pdfplumber') as mock_pdfplumber:
            mock_page1 = Mock()
            mock_page1.extract_text.return_value = "Some text"
            mock_page1.images = [{'image': 'data'}]
            
            mock_pdf = Mock()
            mock_pdf.pages = [mock_page1]
            mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
            
            pdf_type = processor.detect_pdf_type(sample_pdf_path, sample_pages=1)
            
            assert pdf_type == PDFType.MIXED
    
    @patch('pdf_processor.pdfplumber')
    def test_extract_text_direct(self, mock_pdfplumber, processor, sample_pdf_path):
        """Тест прямого извлечения текста."""
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page 1 text"
        
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Page 2 text"
        
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        result = processor._extract_text_direct(sample_pdf_path)
        
        assert result.success is True
        assert isinstance(result.data, str)
        assert "Page 1 text" in result.data
        assert "Page 2 text" in result.data
        assert result.pdf_type == PDFType.TEXT_BASED
    
    @patch('pdf_processor.convert_from_path')
    @patch('pdf_processor.pytesseract')
    def test_extract_text_ocr(self, mock_tesseract, mock_convert, processor, sample_pdf_path):
        """Тест извлечения текста через OCR."""
        # Мокируем конвертацию PDF в изображения
        mock_image1 = Mock()
        mock_image1.close = Mock()
        mock_image2 = Mock()
        mock_image2.close = Mock()
        mock_convert.return_value = [mock_image1, mock_image2]
        
        # Мокируем OCR
        mock_tesseract.image_to_string.side_effect = ["OCR text 1", "OCR text 2"]
        
        result = processor._extract_text_ocr(sample_pdf_path)
        
        assert result.success is True
        assert isinstance(result.data, str)
        assert "OCR text 1" in result.data
        assert "OCR text 2" in result.data
        assert result.pdf_type == PDFType.SCANNED
    
    @patch('pdf_processor.pdfplumber')
    def test_extract_text_with_progress(self, mock_pdfplumber, processor, sample_pdf_path):
        """Тест извлечения текста с отслеживанием прогресса."""
        progress_callback = Mock()
        
        mock_page = Mock()
        mock_page.extract_text.return_value = "Test text"
        
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        result = processor._extract_text_direct(sample_pdf_path, progress_callback)
        
        assert result.success is True
        # Проверяем что progress callback был вызван
        progress_callback.update.assert_called()
    
    @patch('pdf_processor.convert_from_path')
    def test_extract_images_scanned(self, mock_convert, processor, sample_pdf_path):
        """Тест извлечения изображений из сканированного PDF."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Мокируем конвертацию
            mock_image = Mock()
            mock_image.save = Mock()
            mock_image.close = Mock()
            mock_convert.return_value = [mock_image]
            
            result = processor._extract_scanned_images(sample_pdf_path, tmp_dir)
            
            assert result.success is True
            assert isinstance(result.data, list)
            assert len(result.data) == 1
            assert result.pdf_type == PDFType.SCANNED
    
    @patch('pdf_processor.camelot')
    def test_extract_tables_camelot(self, mock_camelot, processor, sample_pdf_path):
        """Тест извлечения таблиц с camelot."""
        # Мокируем camelot
        mock_table = Mock()
        mock_table.df = Mock()
        mock_table.page = 1
        
        mock_tables = Mock()
        mock_tables.__len__ = Mock(return_value=1)
        mock_tables.__getitem__ = Mock(return_value=mock_table)
        mock_camelot.read_pdf.return_value = mock_tables
        
        result = processor._extract_tables_camelot(sample_pdf_path)
        
        assert result.success is True
        assert isinstance(result.data, list)
        assert len(result.data) == 1
        assert result.data[0]['method'] == 'lattice'
    
    def test_detect_language_success(self, processor):
        """Тест успешного определения языка."""
        test_text = "This is a sample text in English for language detection testing."
        
        with patch('pdf_processor.LANGDETECT_AVAILABLE', True):
            with patch('pdf_processor.detect') as mock_detect:
                mock_detect.return_value = 'en'
                
                with patch('pdf_processor.detect_langs') as mock_detect_langs:
                    mock_lang = Mock()
                    mock_lang.lang = 'en'
                    mock_lang.prob = 0.95
                    mock_detect_langs.return_value = [mock_lang]
                    
                    result = processor.detect_language(test_text)
        
        assert result.success is True
        assert result.data['language'] == 'en'
        assert 'confidence_scores' in result.data
    
    def test_detect_language_no_langdetect(self, processor):
        """Тест определения языка без langdetect."""
        test_text = "Sample text"
        
        with patch('pdf_processor.LANGDETECT_AVAILABLE', False):
            result = processor.detect_language(test_text)
        
        assert result.success is False
        assert "langdetect не установлена" in result.error
    
    def test_detect_language_short_text(self, processor):
        """Тест определения языка для короткого текста."""
        short_text = "Hi"
        
        with patch('pdf_processor.LANGDETECT_AVAILABLE', True):
            result = processor.detect_language(short_text)
        
        assert result.success is False
        assert "слишком короткий" in result.error
    
    def test_progress_callback(self):
        """Тест callback для отслеживания прогресса."""
        callback = ProgressCallback()
        
        assert callback.current_page == 0
        assert callback.total_pages == 0
        
        callback.update(5, 20, "Тестовое обновление")
        
        assert callback.current_page == 5
        assert callback.total_pages == 20
    
    @patch('pdf_processor.PDFProcessor.extract_metadata')
    @patch('pdf_processor.PDFProcessor.extract_text')
    @patch('pdf_processor.PDFProcessor.detect_language')
    @patch('pdf_processor.PDFProcessor.extract_tables')
    @patch('pdf_processor.PDFProcessor.extract_images')
    def test_process_full_pdf(self, mock_extract_images, mock_extract_tables, 
                            mock_detect_lang, mock_extract_text, mock_extract_metadata,
                            processor, sample_pdf_path):
        """Тест полной обработки PDF."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Мокируем все методы извлечения
            mock_extract_metadata.return_value = ExtractionResult(
                success=True, data=PDFMetadata(), pdf_type=PDFType.TEXT_BASED
            )
            
            mock_extract_text.return_value = ExtractionResult(
                success=True, data="Sample text", pdf_type=PDFType.TEXT_BASED
            )
            
            mock_detect_lang.return_value = ExtractionResult(
                success=True, data={'language': 'en'}, pdf_type=PDFType.UNKNOWN
            )
            
            mock_extract_tables.return_value = ExtractionResult(
                success=True, data=[], pdf_type=PDFType.TEXT_BASED
            )
            
            mock_extract_images.return_value = ExtractionResult(
                success=True, data=[], pdf_type=PDFType.TEXT_BASED
            )
            
            results = processor.process_full_pdf(
                sample_pdf_path, 
                output_dir=tmp_dir,
                extract_images=True,
                extract_tables=True
            )
        
        # Проверяем что все методы были вызваны
        mock_extract_metadata.assert_called_once()
        mock_extract_text.assert_called_once()
        mock_detect_lang.assert_called_once()
        mock_extract_tables.assert_called_once()
        mock_extract_images.assert_called_once()
        
        # Проверяем результаты
        assert 'metadata' in results
        assert 'text' in results
        assert 'language' in results
        assert 'tables' in results
        assert 'images' in results
        
        assert all(results[key].success for key in results if key != 'error')
    
    def test_error_handling(self, processor, sample_pdf_path):
        """Тест обработки ошибок."""
        # Тест с несуществующим файлом
        with patch('pdf_processor.pdfplumber') as mock_pdfplumber:
            mock_pdfplumber.open.side_effect = Exception("File not found")
            
            with patch('pdf_processor.PyPDF2') as mock_pyPDF2:
                with patch('pdf_processor.fitz') as mock_fitz:
                    # Все методы падают
                    mock_pyPDF2.PdfReader.side_effect = Exception("PyPDF2 error")
                    mock_fitz.open.side_effect = Exception("PyMuPDF error")
                    
                    result = processor.extract_page_count(sample_pdf_path)
        
        assert result.success is False
        assert result.error is not None
    
    @patch('pdf_processor.os.makedirs')
    def test_extract_images_directory_creation(self, mock_makedirs, processor, sample_pdf_path):
        """Тест создания директории для изображений."""
        with patch('pdf_processor.PDFProcessor.detect_pdf_type') as mock_detect_type:
            with patch('pdf_processor.PDFProcessor._extract_scanned_images') as mock_extract:
                mock_detect_type.return_value = PDFType.SCANNED
                mock_extract.return_value = ExtractionResult(success=True, data=[])
                
                result = processor.extract_images(sample_pdf_path)
        
        # Проверяем что директория создается
        mock_makedirs.assert_called()


class TestExtractionResult:
    """Тесты для класса ExtractionResult."""
    
    def test_successful_result(self):
        """Тест успешного результата."""
        result = ExtractionResult(
            success=True,
            data="test data",
            pdf_type=PDFType.TEXT_BASED
        )
        
        assert result.success is True
        assert result.data == "test data"
        assert result.pdf_type == PDFType.TEXT_BASED
        assert result.error is None
    
    def test_error_result(self):
        """Тест результата с ошибкой."""
        result = ExtractionResult(
            success=False,
            error="Test error"
        )
        
        assert result.success is False
        assert result.error == "Test error"
        assert result.data is None


class TestPDFMetadata:
    """Тесты для класса PDFMetadata."""
    
    def test_metadata_creation(self):
        """Тест создания метаданных."""
        metadata = PDFMetadata(
            title="Test Title",
            author="Test Author",
            page_count=5
        )
        
        assert metadata.title == "Test Title"
        assert metadata.author == "Test Author"
        assert metadata.page_count == 5
        assert metadata.file_size is None
        assert metadata.checksum is None
    
    def test_metadata_defaults(self):
        """Тест значений по умолчанию для метаданных."""
        metadata = PDFMetadata()
        
        assert metadata.title is None
        assert metadata.author is None
        assert metadata.page_count == 0
        assert metadata.file_size is None
        assert metadata.checksum is None


if __name__ == "__main__":
    pytest.main([__file__])