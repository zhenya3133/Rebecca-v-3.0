"""Полноценный PDF процессор с извлечением текста, изображений, таблиц и метаданных."""

import hashlib
import io
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum

# Импорты для работы с PDF
try:
    import PyPDF2
    import pdfplumber
    from pdf2image import convert_from_path
    import pytesseract
    from PIL import Image
    import camelot
    import pandas as pd
    import fitz  # PyMuPDF
except ImportError as e:
    logging.warning(f"Не удалось импортировать некоторые библиотеки: {e}")
    # Будем использовать базовые функции где возможно

# Для определения языка
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0  # Для воспроизводимых результатов
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


class PDFType(Enum):
    """Типы PDF файлов."""
    TEXT_BASED = "text_based"  # Содержит извлекаемый текст
    SCANNED = "scanned"  # Сканированные изображения
    MIXED = "mixed"  # Смешанный тип (текст + изображения)
    UNKNOWN = "unknown"  # Неопределенный тип


@dataclass
class PDFMetadata:
    """Метаданные PDF документа."""
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


@dataclass
class ExtractionResult:
    """Результат извлечения данных из PDF."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    pdf_type: Optional[PDFType] = None
    metadata: Optional[PDFMetadata] = None


class ProgressCallback:
    """Callback для отслеживания прогресса."""
    
    def __init__(self):
        self.current_page = 0
        self.total_pages = 0
    
    def update(self, current: int, total: int, message: str = ""):
        """Обновление прогресса."""
        self.current_page = current
        self.total_pages = total
        progress_percent = (current / total) * 100 if total > 0 else 0
        logging.info(f"Прогресс: {progress_percent:.1f}% - {message}")


class PDFProcessor:
    """Полноценный процессор PDF файлов с поддержкой различных типов."""
    
    def __init__(self, ocr_enabled: bool = True, lang: str = "rus+eng"):
        """
        Инициализация процессора PDF.
        
        Args:
            ocr_enabled: Включить OCR для сканированных PDF
            lang: Языки для OCR (по умолчанию русский + английский)
        """
        self.ocr_enabled = ocr_enabled
        self.ocr_lang = lang
        self.logger = logging.getLogger(__name__)
        
        # Настройка OCR если доступен
        if self.ocr_enabled and self._check_tesseract():
            self.logger.info("OCR поддержка включена")
        else:
            self.ocr_enabled = False
            self.logger.warning("OCR поддержка недоступна")
    
    def _check_tesseract(self) -> bool:
        """Проверка доступности tesseract."""
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Вычисление контрольной суммы файла."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def extract_page_count(self, pdf_path: str) -> ExtractionResult:
        """Извлечение количества страниц из PDF."""
        try:
            page_count = 0
            
            # Попытка через pdfplumber
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    page_count = len(pdf.pages)
            except Exception:
                # Fallback через PyPDF2
                try:
                    with open(pdf_path, 'rb') as file:
                        reader = PyPDF2.PdfReader(file)
                        page_count = len(reader.pages)
                except Exception:
                    # Fallback через PyMuPDF
                    try:
                        doc = fitz.open(pdf_path)
                        page_count = len(doc)
                        doc.close()
                    except Exception as e:
                        return ExtractionResult(
                            success=False,
                            error=f"Не удалось определить количество страниц: {str(e)}"
                        )
            
            return ExtractionResult(
                success=True,
                data=page_count,
                pdf_type=PDFType.UNKNOWN
            )
            
        except Exception as e:
            return ExtractionResult(
                success=False,
                error=f"Ошибка при определении количества страниц: {str(e)}"
            )
    
    def extract_metadata(self, pdf_path: str) -> ExtractionResult:
        """Извлечение метаданных PDF."""
        try:
            metadata = PDFMetadata()
            file_stats = os.stat(pdf_path)
            metadata.file_size = file_stats.st_size
            metadata.checksum = self._calculate_checksum(pdf_path)
            
            # Извлечение метаданных через pdfplumber
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    pdf_meta = pdf.metadata or {}
                    metadata.title = pdf_meta.get('/Title')
                    metadata.author = pdf_meta.get('/Author')
                    metadata.subject = pdf_meta.get('/Subject')
                    metadata.keywords = pdf_meta.get('/Keywords')
                    metadata.creator = pdf_meta.get('/Creator')
                    metadata.producer = pdf_meta.get('/Producer')
                    metadata.creation_date = pdf_meta.get('/CreationDate')
                    metadata.modification_date = pdf_meta.get('/ModDate')
                    metadata.page_count = len(pdf.pages)
            except Exception as e:
                self.logger.warning(f"Не удалось извлечь метаданные через pdfplumber: {e}")
                
                # Fallback через PyPDF2
                try:
                    with open(pdf_path, 'rb') as file:
                        reader = PyPDF2.PdfReader(file)
                        if reader.metadata:
                            metadata_dict = reader.metadata
                            metadata.title = metadata_dict.get('/Title')
                            metadata.author = metadata_dict.get('/Author')
                            metadata.subject = metadata_dict.get('/Subject')
                            metadata.keywords = metadata_dict.get('/Keywords')
                            metadata.creator = metadata_dict.get('/Creator')
                            metadata.producer = metadata_dict.get('/Producer')
                            metadata.creation_date = str(metadata_dict.get('/CreationDate'))
                            metadata.modification_date = str(metadata_dict.get('/ModDate'))
                        metadata.page_count = len(reader.pages)
                except Exception as e2:
                    self.logger.warning(f"Не удалось извлечь метаданные через PyPDF2: {e2}")
                    
                    # Fallback через PyMuPDF
                    try:
                        doc = fitz.open(pdf_path)
                        metadata_dict = doc.metadata
                        metadata.title = metadata_dict.get('title')
                        metadata.author = metadata_dict.get('author')
                        metadata.subject = metadata_dict.get('subject')
                        metadata.keywords = metadata_dict.get('keywords')
                        metadata.creator = metadata_dict.get('creator')
                        metadata.producer = metadata_dict.get('producer')
                        metadata.creation_date = metadata_dict.get('creationDate')
                        metadata.modification_date = metadata_dict.get('modDate')
                        metadata.page_count = len(doc)
                        doc.close()
                    except Exception as e3:
                        self.logger.warning(f"Не удалось извлечь метаданные через PyMuPDF: {e3}")
            
            return ExtractionResult(
                success=True,
                data=metadata,
                pdf_type=PDFType.UNKNOWN
            )
            
        except Exception as e:
            return ExtractionResult(
                success=False,
                error=f"Ошибка при извлечении метаданных: {str(e)}"
            )
    
    def detect_pdf_type(self, pdf_path: str, sample_pages: int = 3) -> PDFType:
        """Автоматическое определение типа PDF."""
        try:
            text_pages = 0
            image_pages = 0
            
            with pdfplumber.open(pdf_path) as pdf:
                pages_to_check = min(sample_pages, len(pdf.pages))
                
                for i in range(pages_to_check):
                    page = pdf.pages[i]
                    
                    # Проверяем наличие текста
                    text = page.extract_text()
                    if text and text.strip():
                        text_pages += 1
                    
                    # Проверяем наличие изображений
                    images = page.images
                    if images:
                        image_pages += 1
                
                # Определяем тип
                if text_pages == pages_to_check and image_pages == 0:
                    return PDFType.TEXT_BASED
                elif text_pages == 0 and image_pages > 0:
                    return PDFType.SCANNED
                elif text_pages > 0 and image_pages > 0:
                    return PDFType.MIXED
                else:
                    return PDFType.UNKNOWN
                    
        except Exception as e:
            self.logger.error(f"Ошибка при определении типа PDF: {e}")
            return PDFType.UNKNOWN
    
    def extract_text(self, pdf_path: str, 
                    progress_callback: Optional[ProgressCallback] = None) -> ExtractionResult:
        """Извлечение текста из PDF с поддержкой OCR."""
        try:
            # Определяем тип PDF
            pdf_type = self.detect_pdf_type(pdf_path)
            
            if progress_callback:
                page_count_result = self.extract_page_count(pdf_path)
                if page_count_result.success:
                    progress_callback.total_pages = page_count_result.data
            
            # Для текстовых PDF используем прямое извлечение
            if pdf_type == PDFType.TEXT_BASED:
                return self._extract_text_direct(pdf_path, progress_callback)
            
            # Для сканированных PDF используем OCR
            elif pdf_type == PDFType.SCANNED:
                if self.ocr_enabled:
                    return self._extract_text_ocr(pdf_path, progress_callback)
                else:
                    return ExtractionResult(
                        success=False,
                        error="OCR отключен, невозможно извлечь текст из сканированного PDF"
                    )
            
            # Для смешанных PDF пробуем оба метода
            elif pdf_type == PDFType.MIXED:
                # Сначала пробуем прямое извлечение
                direct_result = self._extract_text_direct(pdf_path, progress_callback)
                if direct_result.success and direct_result.data:
                    return direct_result
                
                # Если текст не найден, используем OCR
                if self.ocr_enabled:
                    return self._extract_text_ocr(pdf_path, progress_callback)
                else:
                    return direct_result
            
            else:
                return ExtractionResult(
                    success=False,
                    error="Не удалось определить тип PDF или извлечь текст"
                )
                
        except Exception as e:
            return ExtractionResult(
                success=False,
                error=f"Ошибка при извлечении текста: {str(e)}"
            )
    
    def _extract_text_direct(self, pdf_path: str, 
                           progress_callback: Optional[ProgressCallback] = None) -> ExtractionResult:
        """Прямое извлечение текста из PDF."""
        try:
            full_text = []
            page_count = 0
            
            with pdfplumber.open(pdf_path) as pdf:
                page_count = len(pdf.pages)
                
                for i, page in enumerate(pdf.pages):
                    if progress_callback:
                        progress_callback.update(i + 1, page_count, f"Извлечение страницы {i + 1}")
                    
                    text = page.extract_text()
                    if text:
                        full_text.append(f"--- Страница {i + 1} ---\n{text}")
            
            extracted_text = "\n\n".join(full_text)
            
            return ExtractionResult(
                success=True,
                data=extracted_text,
                pdf_type=PDFType.TEXT_BASED
            )
            
        except Exception as e:
            return ExtractionResult(
                success=False,
                error=f"Ошибка при прямом извлечении текста: {str(e)}"
            )
    
    def _extract_text_ocr(self, pdf_path: str,
                         progress_callback: Optional[ProgressCallback] = None) -> ExtractionResult:
        """Извлечение текста через OCR."""
        try:
            full_text = []
            
            # Конвертируем PDF в изображения
            if progress_callback:
                progress_callback.update(0, 100, "Конвертация PDF в изображения")
            
            images = convert_from_path(pdf_path)
            total_pages = len(images)
            
            for i, image in enumerate(images):
                if progress_callback:
                    progress = 10 + (i / total_pages) * 80  # 10% на конвертацию, 80% на OCR
                    progress_callback.update(int(progress), 100, f"OCR обработка страницы {i + 1}")
                
                # Применяем OCR
                text = pytesseract.image_to_string(image, lang=self.ocr_lang)
                if text.strip():
                    full_text.append(f"--- Страница {i + 1} ---\n{text}")
                
                # Очищаем память
                image.close()
            
            extracted_text = "\n\n".join(full_text)
            
            return ExtractionResult(
                success=True,
                data=extracted_text,
                pdf_type=PDFType.SCANNED
            )
            
        except Exception as e:
            return ExtractionResult(
                success=False,
                error=f"Ошибка при OCR извлечении: {str(e)}"
            )
    
    def extract_images(self, pdf_path: str, 
                      output_dir: Optional[str] = None,
                      progress_callback: Optional[ProgressCallback] = None) -> ExtractionResult:
        """Извлечение изображений из PDF."""
        try:
            if output_dir is None:
                output_dir = os.path.splitext(pdf_path)[0] + "_images"
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Определяем тип PDF
            pdf_type = self.detect_pdf_type(pdf_path)
            
            if pdf_type == PDFType.SCANNED:
                # Для сканированных PDF - конвертируем страницы в изображения
                return self._extract_scanned_images(pdf_path, output_dir, progress_callback)
            else:
                # Для текстовых PDF - извлекаем встроенные изображения
                return self._extract_embedded_images(pdf_path, output_dir, progress_callback)
                
        except Exception as e:
            return ExtractionResult(
                success=False,
                error=f"Ошибка при извлечении изображений: {str(e)}"
            )
    
    def _extract_scanned_images(self, pdf_path: str, output_dir: str,
                              progress_callback: Optional[ProgressCallback] = None) -> ExtractionResult:
        """Извлечение изображений из сканированного PDF."""
        try:
            if progress_callback:
                progress_callback.update(0, 100, "Конвертация страниц в изображения")
            
            images = convert_from_path(pdf_path)
            extracted_images = []
            total_pages = len(images)
            
            for i, image in enumerate(images):
                if progress_callback:
                    progress = (i / total_pages) * 100
                    progress_callback.update(int(progress), 100, f"Сохранение изображения {i + 1}")
                
                image_path = os.path.join(output_dir, f"page_{i + 1:03d}.png")
                image.save(image_path, "PNG")
                extracted_images.append(image_path)
                
                image.close()
            
            return ExtractionResult(
                success=True,
                data=extracted_images,
                pdf_type=PDFType.SCANNED
            )
            
        except Exception as e:
            return ExtractionResult(
                success=False,
                error=f"Ошибка при извлечении изображений из сканированного PDF: {str(e)}"
            )
    
    def _extract_embedded_images(self, pdf_path: str, output_dir: str,
                               progress_callback: Optional[ProgressCallback] = None) -> ExtractionResult:
        """Извлечение встроенных изображений из PDF."""
        try:
            extracted_images = []
            image_count = 0
            
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                
                for i, page in enumerate(pdf.pages):
                    if progress_callback:
                        progress = (i / total_pages) * 100
                        progress_callback.update(int(progress), 100, f"Обработка страницы {i + 1}")
                    
                    for image in page.images:
                        try:
                            # Извлекаем изображение
                            image_obj = page.within(image['x0'], image['top'], 
                                                  image['x1'], image['bottom'])
                            
                            if image_obj.images:
                                for img in image_obj.images:
                                    image_count += 1
                                    image_path = os.path.join(output_dir, 
                                                            f"image_{image_count:03d}.png")
                                    
                                    # Сохраняем изображение
                                    with open(image_path, 'wb') as f:
                                        f.write(img['stream'].get_rawdata())
                                    
                                    extracted_images.append(image_path)
                        except Exception as e:
                            self.logger.warning(f"Не удалось извлечь изображение: {e}")
            
            return ExtractionResult(
                success=True,
                data=extracted_images,
                pdf_type=PDFType.TEXT_BASED
            )
            
        except Exception as e:
            return ExtractionResult(
                success=False,
                error=f"Ошибка при извлечении встроенных изображений: {str(e)}"
            )
    
    def extract_tables(self, pdf_path: str, 
                      progress_callback: Optional[ProgressCallback] = None) -> ExtractionResult:
        """Извлечение таблиц из PDF с поддержкой OCR."""
        try:
            # Определяем тип PDF
            pdf_type = self.detect_pdf_type(pdf_path)
            
            if pdf_type == PDFType.SCANNED:
                # Для сканированных PDF используем OCR
                return self._extract_tables_ocr(pdf_path, progress_callback)
            else:
                # Для текстовых PDF используем camelot
                return self._extract_tables_camelot(pdf_path, progress_callback)
                
        except Exception as e:
            return ExtractionResult(
                success=False,
                error=f"Ошибка при извлечении таблиц: {str(e)}"
            )
    
    def _extract_tables_camelot(self, pdf_path: str,
                               progress_callback: Optional[ProgressCallback] = None) -> ExtractionResult:
        """Извлечение таблиц с помощью camelot."""
        try:
            if progress_callback:
                progress_callback.update(0, 100, "Анализ PDF для поиска таблиц")
            
            # Пробуем разные методы извлечения
            tables_data = []
            
            # Метод lattice (для таблиц с четкими границами)
            try:
                if progress_callback:
                    progress_callback.update(25, 100, "Извлечение таблиц с границами")
                
                lattice_tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
                
                for i, table in enumerate(lattice_tables):
                    tables_data.append({
                        'table_id': len(tables_data) + 1,
                        'method': 'lattice',
                        'data': table.df,
                        'page': table.page
                    })
            except Exception as e:
                self.logger.warning(f"Не удалось извлечь таблицы методом lattice: {e}")
            
            # Метод stream (для таблиц без четких границ)
            try:
                if progress_callback:
                    progress_callback.update(50, 100, "Извлечение таблиц без границ")
                
                stream_tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
                
                for table in stream_tables:
                    # Избегаем дублирования
                    is_duplicate = any(
                        t['page'] == table.page and t['method'] == 'stream' 
                        for t in tables_data
                    )
                    
                    if not is_duplicate:
                        tables_data.append({
                            'table_id': len(tables_data) + 1,
                            'method': 'stream',
                            'data': table.df,
                            'page': table.page
                        })
            except Exception as e:
                self.logger.warning(f"Не удалось извлечь таблицы методом stream: {e}")
            
            if progress_callback:
                progress_callback.update(100, 100, f"Найдено таблиц: {len(tables_data)}")
            
            return ExtractionResult(
                success=True,
                data=tables_data,
                pdf_type=PDFType.TEXT_BASED
            )
            
        except Exception as e:
            return ExtractionResult(
                success=False,
                error=f"Ошибка при извлечении таблиц с camelot: {str(e)}"
            )
    
    def _extract_tables_ocr(self, pdf_path: str,
                           progress_callback: Optional[ProgressCallback] = None) -> ExtractionResult:
        """Извлечение таблиц через OCR (упрощенная версия)."""
        try:
            if not self.ocr_enabled:
                return ExtractionResult(
                    success=False,
                    error="OCR отключен, невозможно извлечь таблицы из сканированного PDF"
                )
            
            # Конвертируем PDF в изображения
            images = convert_from_path(pdf_path)
            tables_data = []
            total_pages = len(images)
            
            for i, image in enumerate(images):
                if progress_callback:
                    progress = (i / total_pages) * 100
                    progress_callback.update(int(progress), 100, f"OCR анализ страницы {i + 1}")
                
                # Получаем данные таблицы через OCR
                try:
                    table_data = pytesseract.image_to_string(image, lang=self.ocr_lang, 
                                                            config='--psm 6')
                    
                    if table_data.strip():
                        # Простое разбиение на строки
                        lines = table_data.strip().split('\n')
                        
                        # Создаем DataFrame из распознанного текста
                        table_df = pd.DataFrame(lines, columns=['Table Data'])
                        
                        tables_data.append({
                            'table_id': len(tables_data) + 1,
                            'method': 'ocr',
                            'data': table_df,
                            'page': i + 1
                        })
                except Exception as e:
                    self.logger.warning(f"Не удалось извлечь таблицу со страницы {i + 1}: {e}")
                
                image.close()
            
            if progress_callback:
                progress_callback.update(100, 100, f"Найдено таблиц через OCR: {len(tables_data)}")
            
            return ExtractionResult(
                success=True,
                data=tables_data,
                pdf_type=PDFType.SCANNED
            )
            
        except Exception as e:
            return ExtractionResult(
                success=False,
                error=f"Ошибка при OCR извлечении таблиц: {str(e)}"
            )
    
    def detect_language(self, text: str) -> ExtractionResult:
        """Определение языка текста."""
        try:
            if not LANGDETECT_AVAILABLE:
                return ExtractionResult(
                    success=False,
                    error="Библиотека langdetect не установлена"
                )
            
            if not text or len(text.strip()) < 10:
                return ExtractionResult(
                    success=False,
                    error="Текст слишком короткий для определения языка"
                )
            
            # Определяем язык
            detected_lang = detect(text)
            
            # Получаем дополнительную информацию
            confidence_scores = []
            for _ in range(3):  # Проверяем несколько раз для стабильности
                try:
                    from langdetect import detect_langs
                    langs = detect_langs(text)
                    confidence_scores.append({lang.lang: lang.prob for lang in langs[:3]})
                except:
                    continue
            
            return ExtractionResult(
                success=True,
                data={
                    'language': detected_lang,
                    'confidence_scores': confidence_scores
                },
                pdf_type=PDFType.UNKNOWN
            )
            
        except Exception as e:
            return ExtractionResult(
                success=False,
                error=f"Ошибка при определении языка: {str(e)}"
            )
    
    def process_full_pdf(self, pdf_path: str, 
                        output_dir: Optional[str] = None,
                        extract_images: bool = True,
                        extract_tables: bool = True,
                        progress_callback: Optional[ProgressCallback] = None) -> Dict[str, ExtractionResult]:
        """Полная обработка PDF с извлечением всех доступных данных."""
        
        if output_dir is None:
            output_dir = os.path.splitext(pdf_path)[0] + "_extracted"
        
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        try:
            # 1. Извлекаем метаданные
            if progress_callback:
                progress_callback.update(0, 100, "Извлечение метаданных")
            
            results['metadata'] = self.extract_metadata(pdf_path)
            
            # 2. Извлекаем текст
            if progress_callback:
                progress_callback.update(20, 100, "Извлечение текста")
            
            results['text'] = self.extract_text(pdf_path, progress_callback)
            
            # 3. Определяем язык текста если текст извлечен
            if results['text'].success and results['text'].data:
                if progress_callback:
                    progress_callback.update(40, 100, "Определение языка")
                
                results['language'] = self.detect_language(results['text'].data)
            
            # 4. Извлекаем таблицы
            if extract_tables:
                if progress_callback:
                    progress_callback.update(60, 100, "Извлечение таблиц")
                
                results['tables'] = self.extract_tables(pdf_path, progress_callback)
            
            # 5. Извлекаем изображения
            if extract_images:
                if progress_callback:
                    progress_callback.update(80, 100, "Извлечение изображений")
                
                images_output_dir = os.path.join(output_dir, "images")
                results['images'] = self.extract_images(pdf_path, images_output_dir, progress_callback)
            
            if progress_callback:
                progress_callback.update(100, 100, "Обработка завершена")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Ошибка при полной обработке PDF: {e}")
            results['error'] = ExtractionResult(
                success=False,
                error=f"Критическая ошибка при обработке: {str(e)}"
            )
            return results