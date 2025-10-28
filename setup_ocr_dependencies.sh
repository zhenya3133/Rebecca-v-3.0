#!/bin/bash

# Скрипт установки зависимостей для ImageProcessor OCR
# Автор: Rebecca Platform
# Дата: 2025-10-28

set -e  # Остановка при ошибке

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функция вывода сообщений
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Определение операционной системы
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        DISTRO=$(lsb_release -si 2>/dev/null || echo "Unknown")
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
    else
        OS="unknown"
    fi
}

# Проверка наличия команды
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Установка системных зависимостей для Linux
install_linux_deps() {
    print_info "Установка системных зависимостей для Linux..."
    
    # Обновление пакетного менеджера
    if command_exists apt-get; then
        sudo apt-get update
        
        # Основные зависимости
        sudo apt-get install -y \
            tesseract-ocr \
            tesseract-ocr-rus \
            tesseract-ocr-eng \
            tesseract-ocr-deu \
            tesseract-ocr-fra \
            tesseract-ocr-spa \
            tesseract-ocr-ita \
            tesseract-ocr-por \
            tesseract-ocr-nld \
            tesseract-ocr-pol \
            tesseract-ocr-ukr \
            tesseract-ocr-bel \
            tesseract-ocr-chi_sim \
            tesseract-ocr-chi_tra \
            tesseract-ocr-jpn \
            tesseract-ocr-kor
            
        # OpenCV и другие зависимости
        sudo apt-get install -y \
            python3-opencv \
            libopencv-dev \
            python3-dev \
            python3-pip \
            python3-venv \
            libgl1-mesa-glx \
            libglib2.0-0 \
            libsm6 \
            libxext6 \
            libxrender-dev \
            libgomp1 \
            libgtk-3-dev \
            pkg-config
            
    elif command_exists yum; then
        sudo yum install -y \
            tesseract \
            tesseract-langpack-rus \
            tesseract-langpack-eng \
            opencv-python3 \
            python3-pip \
            python3-devel \
            gcc \
            gcc-c++ \
            make
            
    elif command_exists pacman; then
        sudo pacman -S --noconfirm \
            tesseract \
            tesseract-data-rus \
            tesseract-data-eng \
            opencv \
            python-pip \
            python-opencv \
            python-numpy
            
    else
        print_warning "Неизвестный пакетный менеджер. Установите зависимости вручную."
        print_info "Необходимо установить: tesseract-ocr, python3-opencv, python3-pip"
    fi
}

# Установка системных зависимостей для macOS
install_macos_deps() {
    print_info "Установка системных зависимостей для macOS..."
    
    # Проверка наличия Homebrew
    if ! command_exists brew; then
        print_info "Устанавливаю Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    # Установка зависимостей через Homebrew
    brew install tesseract
    brew install tesseract-lang  # Все доступные языки
    brew install opencv
    brew install pkg-config
    
    # Проверка доступности Python
    if ! command_exists python3; then
        print_warning "Python 3 не найден. Рекомендуется установить Python 3."
        print_info "Можно установить через: brew install python"
    fi
}

# Установка системных зависимостей для Windows
install_windows_deps() {
    print_info "Установка системных зависимостей для Windows..."
    
    print_warning "Для Windows необходимо установить зависимости вручную:"
    print_info "1. Скачайте и установите Tesseract с: https://github.com/UB-Mannheim/tesseract/wiki"
    print_info "2. Добавьте путь к tesseract.exe в переменную PATH"
    print_info "3. Установите Python зависимости через pip"
    
    # Проверка наличия tesseract в PATH
    if command_exists tesseract; then
        print_success "Tesseract найден в PATH"
    else
        print_error "Tesseract не найден в PATH. Добавьте его вручную."
    fi
}

# Создание виртуального окружения Python
setup_python_env() {
    print_info "Настройка Python окружения..."
    
    # Создание виртуального окружения
    if [ ! -d "venv_ocr" ]; then
        python3 -m venv venv_ocr
        print_success "Создано виртуальное окружение: venv_ocr"
    fi
    
    # Активация виртуального окружения
    source venv_ocr/bin/activate
    
    # Обновление pip
    pip install --upgrade pip
    
    print_success "Python окружение готово"
}

# Установка Python зависимостей
install_python_deps() {
    print_info "Установка Python зависимостей..."
    
    # Активация виртуального окружения
    if [ -f "venv_ocr/bin/activate" ]; then
        source venv_ocr/bin/activate
    fi
    
    # Установка основных зависимостей
    pip install Pillow>=10.0.0
    pip install opencv-python>=4.8.0
    pip install pytesseract>=0.3.10
    
    # Опциональные зависимости
    pip install face-recognition>=1.3.0 || print_warning "Не удалось установить face-recognition"
    pip install langdetect>=1.0.9 || print_warning "Не удалось установить langdetect"
    pip install tabula-py>=2.7.0 || print_warning "Не удалось установить tabula-py"
    pip install scikit-image>=0.21.0 || print_warning "Не удалось установить scikit-image"
    pip install scipy>=1.11.0 || print_warning "Не удалось установить scipy"
    pip install pdf2image>=1.16.0 || print_warning "Не удалось установить pdf2image"
    pip install tqdm>=4.65.0 || print_warning "Не удалось установить tqdm"
    pip install exifread>=3.0.0 || print_warning "Не удалось установить exifread"
    
    print_success "Python зависимости установлены"
}

# Проверка установки
verify_installation() {
    print_info "Проверка установки..."
    
    # Активация виртуального окружения
    if [ -f "venv_ocr/bin/activate" ]; then
        source venv_ocr/bin/activate
    fi
    
    # Проверка Python пакетов
    python3 -c "
import sys
print('Python version:', sys.version)
try:
    import PIL
    print('✓ PIL/Pillow установлен')
except ImportError:
    print('✗ PIL/Pillow НЕ установлен')

try:
    import cv2
    print('✓ OpenCV установлен, версия:', cv2.__version__)
except ImportError:
    print('✗ OpenCV НЕ установлен')

try:
    import pytesseract
    print('✓ pytesseract установлен')
    try:
        version = pytesseract.get_tesseract_version()
        print('✓ Tesseract найден, версия:', version)
    except:
        print('! Tesseract не найден в PATH')
except ImportError:
    print('✗ pytesseract НЕ установлен')

try:
    import face_recognition
    print('✓ face_recognition установлен')
except ImportError:
    print('! face_recognition НЕ установлен (опционально)')

print()
print('Проверка доступных языков Tesseract:')
import pytesseract
langs = pytesseract.get_languages(config='')
print('Доступные языки:', langs)
"
}

# Создание тестового скрипта
create_test_script() {
    print_info "Создание тестового скрипта..."
    
    cat > test_ocr_installation.py << 'EOF'
#!/usr/bin/env python3
"""
Тестовый скрипт для проверки установки OCR зависимостей.
"""

import sys
import traceback

def test_imports():
    """Тест импортов библиотек."""
    tests = []
    
    try:
        from PIL import Image
        tests.append(("PIL/Pillow", True, ""))
    except Exception as e:
        tests.append(("PIL/Pillow", False, str(e)))
    
    try:
        import cv2
        tests.append(("OpenCV", True, f"версия {cv2.__version__}"))
    except Exception as e:
        tests.append(("OpenCV", False, str(e)))
    
    try:
        import pytesseract
        tests.append(("pytesseract", True, ""))
    except Exception as e:
        tests.append(("pytesseract", False, str(e)))
    
    try:
        import face_recognition
        tests.append(("face_recognition", True, ""))
    except Exception as e:
        tests.append(("face_recognition", False, str(e)))
    
    try:
        import numpy as np
        tests.append(("NumPy", True, f"версия {np.__version__}"))
    except Exception as e:
        tests.append(("NumPy", False, str(e)))
    
    return tests

def test_tesseract():
    """Тест доступности Tesseract."""
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        
        # Тест OCR на простом изображении
        from PIL import Image
        import io
        
        # Создаем простое тестовое изображение
        img = Image.new('RGB', (200, 50), color='white')
        
        try:
            text = pytesseract.image_to_string(img, lang='eng')
            return True, f"Tesseract {version}, OCR тест прошел"
        except Exception as e:
            return False, f"Tesseract {version} найден, но OCR не работает: {e}"
            
    except Exception as e:
        return False, f"Tesseract недоступен: {e}"

def test_image_processing():
    """Тест обработки изображений."""
    try:
        from PIL import Image, ImageEnhance, ImageFilter
        import cv2
        import numpy as np
        
        # Создание тестового изображения
        img = Image.new('RGB', (100, 100), color='red')
        
        # Тест предобработки
        gray = img.convert('L')
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(2.0)
        
        # Тест OpenCV
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        return True, "Обработка изображений работает"
        
    except Exception as e:
        return False, f"Ошибка обработки изображений: {e}"

def main():
    """Главная функция тестирования."""
    print("=== Тестирование установки OCR зависимостей ===\n")
    
    # Тест импортов
    print("1. Проверка Python библиотек:")
    import_tests = test_imports()
    
    for name, success, info in import_tests:
        status = "✓" if success else "✗"
        print(f"   {status} {name}: {info}")
    
    # Тест Tesseract
    print("\n2. Проверка Tesseract:")
    tesseract_success, tesseract_info = test_tesseract()
    status = "✓" if tesseract_success else "✗"
    print(f"   {status} {tesseract_info}")
    
    # Тест обработки изображений
    print("\n3. Проверка обработки изображений:")
    img_success, img_info = test_image_processing()
    status = "✓" if img_success else "✗"
    print(f"   {status} {img_info}")
    
    # Итоговый результат
    print("\n=== Результат ===")
    
    all_good = all(test[1] for test in import_tests) and tesseract_success and img_success
    
    if all_good:
        print("🎉 Все компоненты успешно установлены!")
        print("   Можно использовать ImageProcessor для OCR")
        
        # Тест быстрого OCR
        try:
            from src.ingest.image_processor import quick_ocr
            print("\nТест быстрого OCR:")
            print("   ℹ️  Создайте тестовое изображение и запустите:")
            print("      python3 -c \"from src.ingest.image_processor import quick_ocr; print(quick_ocr('test.jpg'))\"")
            
        except Exception as e:
            print(f"   ⚠️  Не удалось импортировать ImageProcessor: {e}")
            
    else:
        print("❌ Обнаружены проблемы с установкой:")
        print("   Проверьте логи выше и установите недостающие компоненты")
        print("\nДля устранения проблем:")
        print("   1. Запустите скрипт установки заново")
        print("   2. Проверьте права доступа")
        print("   3. Установите системные зависимости вручную")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n💥 Критическая ошибка тестирования:")
        print(f"   {e}")
        traceback.print_exc()
        sys.exit(1)
EOF

    chmod +x test_ocr_installation.py
    print_success "Создан тестовый скрипт: test_ocr_installation.py"
}

# Создание документации по настройке
create_setup_guide() {
    print_info "Создание руководства по настройке..."
    
    cat > OCR_SETUP_GUIDE.md << 'EOF'
# Руководство по настройке ImageProcessor OCR

## Быстрая установка

```bash
# Скачайте и запустите скрипт установки
chmod +x setup_ocr_dependencies.sh
./setup_ocr_dependencies.sh

# Протестируйте установку
python3 test_ocr_installation.py
```

## Ручная установка

### Linux (Ubuntu/Debian)

```bash
# Системные зависимости
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng python3-opencv python3-pip

# Python зависимости
pip3 install Pillow opencv-python pytesseract
```

### macOS

```bash
# Через Homebrew
brew install tesseract opencv
pip3 install Pillow opencv-python pytesseract
```

### Windows

1. Скачайте Tesseract с [официального сайта](https://github.com/UB-Mannheim/tesseract/wiki)
2. Установите и добавьте в PATH
3. Установите Python зависимости: `pip install Pillow opencv-python pytesseract`

## Проверка установки

```bash
python3 test_ocr_installation.py
```

## Тестирование ImageProcessor

```bash
# Запуск примеров
python3 src/ingest/image_processor_examples.py

# Запуск unit тестов
pytest src/ingest/test_image_processor.py -v
```

## Решение проблем

### Tesseract не найден
- Убедитесь, что tesseract установлен и добавлен в PATH
- Проверьте: `tesseract --version`

### Ошибки импорта Python библиотек
- Установите зависимости: `pip install -r src/ingest/requirements_ocr.txt`
- Проверьте версию Python (требуется 3.7+)

### Проблемы с OpenCV
- Linux: `sudo apt-get install python3-opencv libopencv-dev`
- macOS: `brew install opencv`
- Windows: `pip install opencv-python`

### Низкое качество OCR
- Используйте предобработку: `preprocess_image()`
- Проверьте качество исходного изображения
- Укажите правильные языки: `language='rus+eng'`

## Производительность

### Рекомендации для больших объемов:
- Используйте batch обработку: `batch_process_images()`
- Отключайте ненужную предобработку
- Кэшируйте результаты через MemoryManager

### Мониторинг производительности:
```python
result = processor.extract_text("image.jpg")
print(f"Время: {result.processing_time:.2f}с")
print(f"Применено: {result.preprocessing_applied}")
```
EOF

    print_success "Создано руководство: OCR_SETUP_GUIDE.md"
}

# Основная функция установки
main() {
    print_info "🚀 Начало установки OCR зависимостей для ImageProcessor"
    print_info "Определение операционной системы..."
    
    detect_os
    print_info "Обнаружена ОС: $OS"
    
    case $OS in
        "linux")
            print_info "Установка для Linux..."
            install_linux_deps
            ;;
        "macos")
            print_info "Установка для macOS..."
            install_macos_deps
            ;;
        "windows")
            print_info "Установка для Windows..."
            install_windows_deps
            ;;
        *)
            print_error "Неизвестная операционная система. Пропуск установки системных зависимостей."
            ;;
    esac
    
    # Настройка Python окружения
    setup_python_env
    
    # Установка Python зависимостей
    install_python_deps
    
    # Проверка установки
    verify_installation
    
    # Создание вспомогательных файлов
    create_test_script
    create_setup_guide
    
    print_success "🎉 Установка завершена!"
    print_info "Следующие шаги:"
    echo "   1. Запустите тест: python3 test_ocr_installation.py"
    echo "   2. Изучите примеры: python3 src/ingest/image_processor_examples.py"
    echo "   3. Запустите unit тесты: pytest src/ingest/test_image_processor.py -v"
    echo "   4. Прочитайте документацию: src/ingest/IMAGE_PROCESSOR_README.md"
    echo ""
    print_info "Для активации Python окружения:"
    echo "   source venv_ocr/bin/activate"
}

# Проверка аргументов командной строки
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Скрипт установки OCR зависимостей для ImageProcessor"
    echo ""
    echo "Использование: $0 [опции]"
    echo ""
    echo "Опции:"
    echo "  --help, -h     Показать эту справку"
    echo "  --verify-only  Только проверить установку"
    echo ""
    echo "Поддерживаемые ОС: Linux, macOS, Windows (частично)"
    exit 0
elif [ "$1" == "--verify-only" ]; then
    verify_installation
    exit 0
fi

# Запуск основной функции
main "$@"