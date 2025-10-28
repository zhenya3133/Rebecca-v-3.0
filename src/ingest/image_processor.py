"""
Модуль для обработки изображений с поддержкой OCR.

Включает:
- Извлечение текста с помощью pytesseract
- Предобработку изображений
- Определение языка
- Извлечение таблиц
- Обнаружение лиц
- Многоязычную поддержку
- Batch обработку
"""

import os
import io
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import base64

# Обработка изображений
try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    from PIL.Image import Image as PILImage
except ImportError:
    PILImage = None
    logging.warning("PIL не установлен. Установите: pip install Pillow")

# OpenCV для обработки изображений
try:
    import cv2
    import numpy as np
    import pytesseract
except ImportError:
    cv2 = None
    np = None
    logging.warning("OpenCV или pytesseract не установлены. Установите: pip install opencv-python pytesseract")

# Детекция лиц
try:
    from face_recognition import face_locations
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    logging.warning("face_recognition не установлен. Установите: pip install face_recognition")

# pydantic для моделей данных
from pydantic import BaseModel, Field

# Локальные импорты
from .ingestion_models import IngestRecord

# Настройка логгера
logger = logging.getLogger(__name__)


class ImageInfo(BaseModel):
    """Модель информации об изображении."""
    path: str
    format: str
    size: Tuple[int, int]  # (width, height)
    mode: str
    file_size: int
    dpi: Optional[Tuple[int, int]] = None
    has_transparency: bool = False
    created_at: Optional[str] = None
    modified_at: Optional[str] = None
    checksum: str = Field(description="MD5 хеш файла")


class OCRResult(BaseModel):
    """Результат OCR операции."""
    text: str
    confidence: float
    language: str
    method: str = "pytesseract"  # или "mock"
    words: List[Dict[str, Any]] = []
    preprocessing_applied: List[str] = []
    processing_time: float = 0.0


class TableDetection(BaseModel):
    """Результат извлечения таблицы."""
    rows: List[List[str]]
    confidence: float
    bounding_boxes: List[List[int]] = []  # [[x1, y1, x2, y2], ...]
    method: str


class FaceDetection(BaseModel):
    """Результат обнаружения лиц."""
    count: int
    bounding_boxes: List[List[int]]
    encodings: List[List[float]] = []
    method: str


class BatchResult(BaseModel):
    """Результат batch обработки."""
    total_processed: int
    successful: int
    failed: int
    results: Dict[str, Any]
    errors: Dict[str, str]


class ImageProcessor:
    """
    Класс для обработки изображений с поддержкой OCR.
    
    Поддерживает:
    - Извлечение текста с различных форматов изображений
    - Предобработку для улучшения качества
    - Многоязычный OCR (русский, английский, и др.)
    - Определение языка
    - Извлечение таблиц
    - Обнаружение лиц
    - Batch обработку
    """
    
    # Поддерживаемые форматы
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    
    # Поддерживаемые языки для OCR
    SUPPORTED_LANGUAGES = {
        'ru': 'Русский',
        'en': 'English',
        'de': 'Deutsch',
        'fr': 'Français',
        'es': 'Español',
        'it': 'Italiano',
        'pt': 'Português',
        'nl': 'Nederlands',
        'pl': 'Polski',
        'uk': 'Українська',
        'be': 'Беларуская',
        'zh': '中文',
        'ja': '日本語',
        'ko': '한국어'
    }
    
    def __init__(self, memory_manager=None, tesseract_config: Optional[str] = None):
        """
        Инициализация ImageProcessor.
        
        Args:
            memory_manager: Экземпляр MemoryManager для интеграции
            tesseract_config: Конфигурация для tesseract (например, '--oem 3 --psm 6')
        """
        self.memory_manager = memory_manager
        self.tesseract_config = tesseract_config or '--oem 3 --psm 6'
        self.use_mock = not self._check_dependencies()
        
        if self.use_mock:
            logger.warning("Используется mock режим из-за отсутствия зависимостей")
    
    def _check_dependencies(self) -> bool:
        """Проверка наличия всех необходимых зависимостей."""
        try:
            import pytesseract
            from PIL import Image
            import cv2
            import numpy as np
            return True
        except ImportError:
            return False
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Вычисление MD5 хеша файла."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def get_image_info(self, image_path: str) -> ImageInfo:
        """
        Получение метаданных изображения.
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            ImageInfo: Информация об изображении
        """
        try:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Файл не найден: {image_path}")
            
            # Проверка формата
            if path.suffix.lower() not in self.SUPPORTED_FORMATS:
                raise ValueError(f"Неподдерживаемый формат: {path.suffix}")
            
            # Получение базовой информации
            stat = path.stat()
            
            if PILImage:
                with Image.open(path) as img:
                    info = ImageInfo(
                        path=image_path,
                        format=img.format or path.suffix.upper(),
                        size=img.size,
                        mode=img.mode,
                        file_size=stat.st_size,
                        dpi=img.info.get('dpi'),
                        has_transparency=img.mode in ('RGBA', 'LA') or 'transparency' in img.info,
                        created_at=datetime.fromtimestamp(stat.st_ctime).isoformat() if stat.st_ctime > 0 else None,
                        modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat() if stat.st_mtime > 0 else None,
                        checksum=self._calculate_checksum(image_path)
                    )
            else:
                # Fallback без PIL
                info = ImageInfo(
                    path=image_path,
                    format=path.suffix.upper()[1:] or 'UNKNOWN',
                    size=(0, 0),  # Не можем определить без PIL
                    mode='UNKNOWN',
                    file_size=stat.st_size,
                    checksum=self._calculate_checksum(image_path)
                )
            
            return info
            
        except Exception as e:
            logger.error(f"Ошибка при получении информации об изображении {image_path}: {e}")
            raise
    
    def _preprocess_image_internal(self, image: PILImage, 
                                   enhance_contrast: bool = True,
                                   enhance_brightness: bool = True,
                                   denoise: bool = True,
                                   grayscale: bool = False) -> PILImage:
        """
        Внутренний метод предобработки изображения.
        
        Args:
            image: PIL изображение
            enhance_contrast: Увеличить контраст
            enhance_brightness: Увеличить яркость
            denoise: Удаление шума
            grayscale: Конвертация в оттенки серого
            
        Returns:
            Обработанное изображение
        """
        processed_steps = []
        
        # Конвертация в оттенки серого для лучшего OCR
        if grayscale:
            image = image.convert('L')
            processed_steps.append("grayscale")
        
        # Увеличение контраста
        if enhance_contrast:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            processed_steps.append("contrast")
        
        # Увеличение яркости
        if enhance_brightness:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.2)
            processed_steps.append("brightness")
        
        # Удаление шума
        if denoise:
            image = image.filter(ImageFilter.MedianFilter())
            processed_steps.append("denoise")
        
        # Дополнительное повышение резкости
        image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
        processed_steps.append("sharpening")
        
        return image, processed_steps
    
    def preprocess_image(self, image_path: str, 
                        output_path: Optional[str] = None,
                        enhance_contrast: bool = True,
                        enhance_brightness: bool = True,
                        denoise: bool = True,
                        grayscale: bool = False) -> str:
        """
        Предобработка изображения для улучшения качества OCR.
        
        Args:
            image_path: Путь к исходному изображению
            output_path: Путь для сохранения обработанного изображения
            enhance_contrast: Увеличить контраст
            enhance_brightness: Увеличить яркость
            denoise: Удаление шума
            grayscale: Конвертация в оттенки серого
            
        Returns:
            Путь к обработанному изображению
        """
        try:
            if not PILImage:
                raise ImportError("PIL не установлен")
            
            with Image.open(image_path) as img:
                processed_img, steps = self._preprocess_image_internal(
                    img, enhance_contrast, enhance_brightness, denoise, grayscale
                )
                
                # Сохранение результата
                if output_path is None:
                    path = Path(image_path)
                    output_path = str(path.parent / f"{path.stem}_processed{path.suffix}")
                
                processed_img.save(output_path)
                logger.info(f"Изображение обработано: {image_path} -> {output_path}, шаги: {steps}")
                
                return output_path
                
        except Exception as e:
            logger.error(f"Ошибка при предобработке изображения {image_path}: {e}")
            raise
    
    def detect_language(self, text: str) -> str:
        """
        Определение языка текста.
        
        Args:
            text: Текст для анализа
            
        Returns:
            Код языка (ru, en, и т.д.)
        """
        try:
            if not text.strip():
                return 'unknown'
            
            # Простое определение на основе символов
            cyrillic_chars = sum(1 for c in text if ord(c) >= 1040 and ord(c) <= 1103)
            latin_chars = sum(1 for c in text if (ord(c) >= 65 and ord(c) <= 90) or (ord(c) >= 97 and ord(c) <= 122))
            
            # Подсчет символов
            total_chars = len([c for c in text if c.isalpha()])
            
            if total_chars == 0:
                return 'unknown'
            
            cyrillic_ratio = cyrillic_chars / total_chars
            latin_ratio = latin_chars / total_chars
            
            if cyrillic_ratio > 0.5:
                return 'ru'
            elif latin_ratio > 0.5:
                return 'en'
            else:
                # Возвращаем первый попавшийся язык из поддерживаемых
                return 'en'
                
        except Exception as e:
            logger.error(f"Ошибка при определении языка: {e}")
            return 'unknown'
    
    def extract_text(self, image_path: str, 
                    language: str = 'rus+eng',
                    preprocessing: bool = True,
                    handwriting: bool = False) -> OCRResult:
        """
        Извлечение текста из изображения с помощью OCR.
        
        Args:
            image_path: Путь к изображению
            language: Языки для OCR (например, 'rus+eng')
            preprocessing: Применить предобработку
            handwriting: Поддержка рукописного текста (если доступно)
            
        Returns:
            OCRResult: Результат извлечения текста
        """
        start_time = datetime.now()
        
        try:
            if self.use_mock:
                # Mock режим для тестирования
                mock_text = f"[MOCK OCR] Извлеченный текст из {Path(image_path).name}"
                return OCRResult(
                    text=mock_text,
                    confidence=0.85,
                    language='ru',
                    method='mock',
                    words=[],
                    preprocessing_applied=[],
                    processing_time=(datetime.now() - start_time).total_seconds()
                )
            
            if not PILImage or not cv2:
                raise ImportError("Необходимые библиотеки не установлены")
            
            # Загрузка изображения
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Не удалось загрузить изображение: {image_path}")
            
            preprocessing_steps = []
            
            # Предобработка изображения
            if preprocessing:
                # Конвертация в оттенки серого
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                preprocessing_steps.append("grayscale")
                
                # Увеличение контраста
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                gray = clahe.apply(gray)
                preprocessing_steps.append("contrast")
                
                # Удаление шума
                gray = cv2.medianBlur(gray, 3)
                preprocessing_steps.append("denoise")
                
                # Бинаризация
                _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                preprocessing_steps.append("threshold")
                
                image = processed
            
            # Извлечение текста
            custom_config = self.tesseract_config
            if handwriting:
                custom_config += ' --user-words custom_words.txt'
            
            text = pytesseract.image_to_string(image, lang=language, config=custom_config)
            
            # Получение данных о словах и их координатах
            words_data = []
            try:
                data = pytesseract.image_to_data(image, lang=language, config=custom_config, output_type=pytesseract.Output.DICT)
                n_boxes = len(data['level'])
                for i in range(n_boxes):
                    confidence = int(data['conf'][i])
                    if confidence > 0:  # Игнорируем неопределенные слова
                        word = data['text'][i].strip()
                        if word:
                            words_data.append({
                                'text': word,
                                'confidence': confidence,
                                'bbox': [data['left'][i], data['top'][i], 
                                        data['left'][i] + data['width'][i], 
                                        data['top'][i] + data['height'][i]]
                            })
            except Exception as e:
                logger.warning(f"Не удалось извлечь данные о словах: {e}")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return OCRResult(
                text=text,
                confidence=0.9 if preprocessing else 0.8,  # Примерная оценка
                language=language,
                method='pytesseract',
                words=words_data,
                preprocessing_applied=preprocessing_steps,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Ошибка при извлечении текста из {image_path}: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                language='unknown',
                method='error',
                words=[],
                preprocessing_applied=[],
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    def extract_tables_from_image(self, image_path: str) -> TableDetection:
        """
        Извлечение таблиц из изображения.
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            TableDetection: Результат извлечения таблицы
        """
        try:
            if self.use_mock:
                return TableDetection(
                    rows=[["Колонка 1", "Колонка 2"], ["Данные 1", "Данные 2"]],
                    confidence=0.8,
                    method="mock"
                )
            
            if not cv2:
                raise ImportError("OpenCV не установлен")
            
            # Загрузка изображения
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Не удалось загрузить изображение: {image_path}")
            
            # Применение адаптивной бинаризации
            _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Поиск горизонтальных линий
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
            
            # Поиск вертикальных линий
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
            
            # Объединение линий
            table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
            
            # Нахождение контуров
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            rows = []
            bounding_boxes = []
            confidence = 0.0
            
            # Простая обработка для демонстрации
            if contours:
                # Извлечение текста из областей таблицы
                try:
                    data = pytesseract.image_to_string(image, config='--psm 6').split('\n')
                    # Простое разделение на строки (упрощенная реализация)
                    for line in data:
                        if line.strip():
                            cells = [cell.strip() for cell in line.split() if cell.strip()]
                            if cells:
                                rows.append(cells)
                    
                    confidence = min(0.9, len(rows) * 0.1)
                except:
                    rows = [["Не удалось извлечь таблицу"]]
                    confidence = 0.1
            
            return TableDetection(
                rows=rows,
                confidence=confidence,
                bounding_boxes=bounding_boxes,
                method="opencv+pytesseract"
            )
            
        except Exception as e:
            logger.error(f"Ошибка при извлечении таблицы из {image_path}: {e}")
            return TableDetection(
                rows=[["Ошибка извлечения"]],
                confidence=0.0,
                method="error"
            )
    
    def extract_faces(self, image_path: str) -> FaceDetection:
        """
        Обнаружение лиц на изображении.
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            FaceDetection: Результат обнаружения лиц
        """
        try:
            if self.use_mock:
                return FaceDetection(
                    count=1,
                    bounding_boxes=[[50, 50, 150, 150]],
                    method="mock"
                )
            
            if not cv2:
                raise ImportError("OpenCV не установлен")
            
            # Загрузка изображения
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Не удалось загрузить изображение: {image_path}")
            
            # Преобразование в оттенки серого
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Загрузка каскада для обнаружения лиц
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Обнаружение лиц
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            bounding_boxes = []
            encodings = []
            
            for (x, y, w, h) in faces:
                bounding_boxes.append([x, y, x + w, y + h])
                
                # Если доступен face_recognition, добавляем кодировки
                if FACE_RECOGNITION_AVAILABLE:
                    try:
                        face_image = image[y:y+h, x:x+w]
                        # Простая кодировка (упрощенная версия)
                        encoding = face_image.flatten().tolist()[:128]  # Ограничиваем размер
                        encodings.append(encoding)
                    except:
                        pass
            
            return FaceDetection(
                count=len(faces),
                bounding_boxes=bounding_boxes,
                encodings=encodings,
                method="haarcascade"
            )
            
        except Exception as e:
            logger.error(f"Ошибка при обнаружении лиц в {image_path}: {e}")
            return FaceDetection(
                count=0,
                bounding_boxes=[],
                method="error"
            )
    
    def batch_process_images(self, image_paths: List[str], 
                           operations: List[str] = ['extract_text'],
                           output_dir: Optional[str] = None) -> BatchResult:
        """
        Batch обработка нескольких изображений.
        
        Args:
            image_paths: Список путей к изображениям
            operations: Список операций (extract_text, extract_tables, extract_faces, preprocess)
            output_dir: Директория для сохранения результатов
            
        Returns:
            BatchResult: Результат batch обработки
        """
        results = {}
        errors = {}
        successful = 0
        failed = 0
        
        for image_path in image_paths:
            try:
                image_results = {}
                
                for operation in operations:
                    try:
                        if operation == 'extract_text':
                            result = self.extract_text(image_path)
                            image_results[operation] = result.dict()
                        
                        elif operation == 'extract_tables':
                            result = self.extract_tables_from_image(image_path)
                            image_results[operation] = result.dict()
                        
                        elif operation == 'extract_faces':
                            result = self.extract_faces(image_path)
                            image_results[operation] = result.dict()
                        
                        elif operation == 'preprocess':
                            processed_path = self.preprocess_image(
                                image_path, 
                                output_path=os.path.join(output_dir, f"processed_{os.path.basename(image_path)}") 
                                if output_dir else None
                            )
                            image_results[operation] = {'processed_path': processed_path}
                        
                        elif operation == 'get_info':
                            info = self.get_image_info(image_path)
                            image_results[operation] = info.dict()
                        
                    except Exception as e:
                        image_results[operation] = {'error': str(e)}
                
                results[image_path] = image_results
                successful += 1
                
                # Интеграция с MemoryManager
                if self.memory_manager:
                    self.memory_manager.vault.store_secret(
                        f"ocr_result::{image_path}",
                        image_results
                    )
                
            except Exception as e:
                errors[image_path] = str(e)
                failed += 1
        
        return BatchResult(
            total_processed=len(image_paths),
            successful=successful,
            failed=failed,
            results=results,
            errors=errors
        )
    
    def save_results_to_file(self, result: BatchResult, output_path: str) -> None:
        """
        Сохранение результатов batch обработки в файл.
        
        Args:
            result: Результат batch обработки
            output_path: Путь к файлу для сохранения
        """
        try:
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result.dict(), f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"Результаты сохранены в {output_path}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении результатов: {e}")
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Получение списка поддерживаемых языков."""
        return self.SUPPORTED_LANGUAGES.copy()
    
    def get_supported_formats(self) -> List[str]:
        """Получение списка поддерживаемых форматов."""
        return list(self.SUPPORTED_FORMATS)


# Функции-помощники для удобного использования

def create_image_processor(memory_manager=None, tesseract_config=None) -> ImageProcessor:
    """
    Создание экземпляра ImageProcessor.
    
    Args:
        memory_manager: Экземпляр MemoryManager
        tesseract_config: Конфигурация для tesseract
        
    Returns:
        ImageProcessor: Экземпляр процессора изображений
    """
    return ImageProcessor(memory_manager, tesseract_config)


def quick_ocr(image_path: str, language: str = 'rus+eng') -> str:
    """
    Быстрое извлечение текста из изображения.
    
    Args:
        image_path: Путь к изображению
        language: Языки для OCR
        
    Returns:
        str: Извлеченный текст
    """
    processor = ImageProcessor()
    result = processor.extract_text(image_path, language)
    return result.text


def quick_batch_ocr(image_paths: List[str], language: str = 'rus+eng') -> Dict[str, str]:
    """
    Быстрое batch извлечение текста из нескольких изображений.
    
    Args:
        image_paths: Список путей к изображениям
        language: Языки для OCR
        
    Returns:
        Dict[str, str]: Словарь {путь: извлеченный_текст}
    """
    processor = ImageProcessor()
    batch_result = processor.batch_process_images(image_paths, ['extract_text'])
    
    results = {}
    for image_path, image_results in batch_result.results.items():
        if 'extract_text' in image_results:
            results[image_path] = image_results['extract_text'].get('text', '')
    
    return results


if __name__ == "__main__":
    # Демонстрация использования
    processor = ImageProcessor()
    
    # Пример использования
    example_image = "example.jpg"  # Замените на реальный путь
    
    try:
        # Получение информации об изображении
        info = processor.get_image_info(example_image)
        print(f"Информация об изображении: {info}")
        
        # Извлечение текста
        ocr_result = processor.extract_text(example_image)
        print(f"Извлеченный текст: {ocr_result.text}")
        
        # Предобработка изображения
        processed_path = processor.preprocess_image(example_image)
        print(f"Обработанное изображение сохранено: {processed_path}")
        
    except Exception as e:
        print(f"Ошибка: {e}")