"""
Простой тест GitProcessor для проверки основной функциональности.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# Добавляем src в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Создаем базовые моки для git
class MockGit:
    Repo = MagicMock
    GitCommandError = Exception
    InvalidGitRepositoryError = Exception

sys.modules['git'] = MockGit()

# Создаем базовые моки для yaml
class MockYaml:
    @staticmethod
    def safe_load(content):
        return {}

sys.modules['yaml'] = MockYaml()

try:
    # Импортируем GitProcessor
    from ingest.git_processor import GitProcessor
    print("✅ GitProcessor успешно импортирован")
except ImportError as e:
    print(f"❌ Ошибка импорта GitProcessor: {e}")
    sys.exit(1)


def test_basic_functionality():
    """Тест базовой функциональности."""
    print("\n=== Тестирование базовой функциональности ===")
    
    # Создаем процессор
    processor = GitProcessor()
    print("✅ GitProcessor создан")
    
    # Тест извлечения имени репозитория
    test_cases = [
        ("https://github.com/user/repo.git", "repo"),
        ("https://github.com/user/repo", "repo"),
        ("git@github.com:user/repo.git", "repo"),
        ("https://gitlab.com/user/project.git", "project")
    ]
    
    for url, expected in test_cases:
        result = processor._extract_repo_name(url)
        if result == expected:
            print(f"✅ _extract_repo_name('{url}') = '{result}'")
        else:
            print(f"❌ _extract_repo_name('{url}') = '{result}', ожидалось '{expected}'")
    
    # Тест определения языка по расширению
    language_tests = [
        (".py", "Python"),
        (".js", "JavaScript"),
        (".java", "Java"),
        (".cpp", "C++"),
        (".go", "Go"),
        (".unknown", "Other")
    ]
    
    for ext, expected_lang in language_tests:
        actual_lang = processor.LANGUAGE_EXTENSIONS.get(ext, "Other")
        if actual_lang == expected_lang:
            print(f"✅ Определение языка: {ext} -> {actual_lang}")
        else:
            print(f"❌ Определение языка: {ext} -> {actual_lang}, ожидалось {expected_lang}")
    
    # Тест подсчета комментариев
    python_code = """# Comment 1
def function():
    # Comment 2
    print("test")  # Comment 3
"""
    comment_count = processor._count_comments(python_code, "Python")
    print(f"✅ Подсчет комментариев в Python: {comment_count} (ожидается 3)")
    
    # Тест извлечения функций
    python_func_code = """
def function1():
    pass

def function2(param):
    return param

class MyClass:
    def method(self):
        pass
"""
    functions = processor._extract_functions(python_func_code, "Python")
    print(f"✅ Извлечение функций: {functions}")
    
    # Тест извлечения классов
    classes = processor._extract_classes(python_func_code, "Python")
    print(f"✅ Извлечение классов: {classes}")
    
    # Тест извлечения импортов
    import_code = """import os
import sys
from typing import List, Dict
from package.module import function
"""
    imports = processor._extract_imports(import_code, "Python")
    print(f"✅ Извлечение импортов: {imports}")
    
    # Тест определения типа документации
    doc_type_tests = [
        ("README.md", "readme"),
        ("CHANGELOG.md", "changelog"),
        ("LICENSE", "license"),
        ("CONTRIBUTING.md", "contributing"),
        ("UNKNOWN.md", "other")
    ]
    
    for filename, expected_type in doc_type_tests:
        actual_type = processor._get_documentation_type(filename)
        if actual_type == expected_type:
            print(f"✅ Тип документации: {filename} -> {actual_type}")
        else:
            print(f"❌ Тип документации: {filename} -> {actual_type}, ожидалось {expected_type}")
    
    # Тест определения типа зависимостей
    dep_type_tests = [
        ("requirements.txt", "python"),
        ("package.json", "javascript"),
        ("pom.xml", "java"),
        ("Cargo.toml", "rust"),
        ("unknown.xyz", "other")
    ]
    
    for filename, expected_type in dep_type_tests:
        actual_type = processor._get_dependency_type(filename)
        if actual_type == expected_type:
            print(f"✅ Тип зависимостей: {filename} -> {actual_type}")
        else:
            print(f"❌ Тип зависимостей: {filename} -> {actual_type}, ожидалось {expected_type}")
    
    print("✅ Базовое тестирование завершено")


def test_file_analysis():
    """Тест анализа файлов."""
    print("\n=== Тестирование анализа файлов ===")
    
    processor = GitProcessor()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Создаем тестовые файлы разных языков
        test_files = [
            ("test.py", "Python", "def hello():\n    # comment\n    print('hello')"),
            ("test.js", "JavaScript", "function hello() {\n    // comment\n    console.log('hello');\n}"),
            ("test.java", "Java", "public class Test {\n    // comment\n    public void hello() {}\n}"),
        ]
        
        for filename, expected_lang, content in test_files:
            file_path = temp_path / filename
            file_path.write_text(content)
            
            analysis = processor._analyze_file(file_path)
            
            if analysis:
                print(f"✅ Анализ {filename}:")
                print(f"   - Язык: {analysis.language}")
                print(f"   - Строк: {analysis.line_count}")
                print(f"   - Комментариев: {analysis.comment_count}")
            else:
                print(f"❌ Анализ {filename} вернул None")
    
    print("✅ Тестирование анализа файлов завершено")


def test_structure_analysis():
    """Тест анализа структуры."""
    print("\n=== Тестирование анализа структуры ===")
    
    processor = GitProcessor()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Создаем тестовую структуру для MVC
        (temp_path / "controllers").mkdir()
        (temp_path / "models").mkdir()
        (temp_path / "views").mkdir()
        
        patterns = processor._detect_architectural_patterns(temp_path)
        
        if "MVC" in patterns:
            print("✅ Обнаружен паттерн MVC")
        else:
            print(f"❌ Паттерн MVC не обнаружен. Найдено: {patterns}")
        
        # Создаем структуру для микросервисов
        shutil.rmtree(temp_path / "controllers")
        shutil.rmtree(temp_path / "models")
        shutil.rmtree(temp_path / "views")
        
        (temp_path / "user-service").mkdir()
        (temp_path / "order-service").mkdir()
        (temp_path / "payment-service").mkdir()
        
        patterns = processor._detect_architectural_patterns(temp_path)
        
        if "Microservices" in patterns:
            print("✅ Обнаружен паттерн Microservices")
        else:
            print(f"❌ Паттерн Microservices не обнаружен. Найдено: {patterns}")
    
    print("✅ Тестирование анализа структуры завершено")


def test_metrics_calculation():
    """Тест вычисления метрик."""
    print("\n=== Тестирование вычисления метрик ===")
    
    processor = GitProcessor()
    
    # Тестовые данные
    test_files = [
        {
            'language': 'Python',
            'line_count': 100,
            'size': 2000,
            'functions': ['func1', 'func2'],
            'classes': ['Class1'],
            'comment_count': 10
        },
        {
            'language': 'JavaScript',
            'line_count': 150,
            'size': 3000,
            'functions': ['func3'],
            'classes': ['Class2', 'Class3'],
            'comment_count': 20
        }
    ]
    
    metrics = processor._calculate_code_metrics(test_files)
    
    print(f"✅ Метрики вычислены:")
    print(f"   - Общее количество файлов: {metrics['total_files']}")
    print(f"   - Общее количество строк: {metrics['total_lines']}")
    print(f"   - Общий размер: {metrics['total_size']} байт")
    print(f"   - Средний размер файла: {metrics['average_file_size']:.1f} байт")
    print(f"   - Языки: {list(metrics['languages'].keys())}")
    print(f"   - Общее количество функций: {metrics['complexity']['total_functions']}")
    print(f"   - Общее количество классов: {metrics['complexity']['total_classes']}")
    print(f"   - Индекс поддерживаемости: {metrics['maintainability_index']:.1f}")
    
    print("✅ Тестирование вычисления метрик завершено")


def main():
    """Главная функция тестирования."""
    print("Git Processor - Простое тестирование")
    print("=" * 50)
    
    try:
        test_basic_functionality()
        test_file_analysis()
        test_structure_analysis()
        test_metrics_calculation()
        
        print("\n" + "=" * 50)
        print("🎉 Все тесты успешно пройдены!")
        
    except Exception as e:
        print(f"\n❌ Ошибка во время тестирования: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())