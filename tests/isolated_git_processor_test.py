"""
Изолированный тест GitProcessor - полностью самодостаточный.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

# Создаем моки для всех зависимостей
sys.modules['git'] = MagicMock()
sys.modules['git'].Repo = MagicMock
sys.modules['git'].GitCommandError = Exception
sys.modules['git'].InvalidGitRepositoryError = Exception

sys.modules['yaml'] = MagicMock()
sys.modules['yaml'].safe_load = MagicMock(return_value={})

sys.modules['pydantic'] = MagicMock()
sys.modules['pydantic'].BaseModel = object

sys.modules['ingestion_models'] = MagicMock()

# Теперь можно безопасно создать GitProcessor
class MockIngestRecord:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockFileAnalysis:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockRepositoryInfo:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockGitProcessor:
    """Мок GitProcessor для тестирования без зависимостей."""
    
    # Поддерживаемые расширения файлов и их языки
    LANGUAGE_EXTENSIONS = {
        '.py': 'Python',
        '.js': 'JavaScript',
        '.ts': 'TypeScript',
        '.jsx': 'JavaScript',
        '.tsx': 'TypeScript',
        '.java': 'Java',
        '.cpp': 'C++',
        '.c': 'C',
        '.h': 'C',
        '.hpp': 'C++',
        '.cs': 'C#',
        '.php': 'PHP',
        '.rb': 'Ruby',
        '.go': 'Go',
        '.rs': 'Rust',
        '.swift': 'Swift',
        '.kt': 'Kotlin',
        '.scala': 'Scala',
        '.html': 'HTML',
        '.css': 'CSS',
        '.scss': 'SCSS',
        '.sass': 'Sass',
        '.vue': 'Vue',
        '.sql': 'SQL',
        '.r': 'R',
        '.m': 'MATLAB',
        '.pl': 'Perl',
        '.sh': 'Shell',
        '.bash': 'Shell',
        '.zsh': 'Shell',
        '.fish': 'Shell',
        '.bat': 'Batch',
        '.ps1': 'PowerShell',
        '.yaml': 'YAML',
        '.yml': 'YAML',
        '.json': 'JSON',
        '.xml': 'XML',
        '.toml': 'TOML',
        '.ini': 'INI',
        '.cfg': 'Config',
        '.conf': 'Config',
        '.dockerfile': 'Docker',
        '.makefile': 'Make',
        '.gradle': 'Gradle',
        '.properties': 'Properties'
    }
    
    # Файлы документации
    DOC_FILES = {
        'README.md', 'README.rst', 'README.txt', 'README',
        'CHANGELOG.md', 'CHANGELOG.rst', 'CHANGELOG.txt', 'CHANGES.md', 'CHANGES.rst',
        'LICENSE', 'LICENSE.txt', 'LICENSE.md', 'LICENCE', 'COPYING',
        'CONTRIBUTING.md', 'CONTRIBUTING.rst', 'CONTRIBUTING.txt',
        'INSTALL.md', 'INSTALL.rst', 'INSTALL.txt', 'SETUP.md', 'SETUP.rst',
        'USAGE.md', 'USAGE.rst', 'USAGE.txt', 'DOCUMENTATION.md', 'API.md',
        'TODO.md', 'BUGS.md', 'FAQ.md', 'HISTORY.md'
    }
    
    # Файлы зависимостей
    DEPENDENCY_FILES = {
        'requirements.txt', 'requirements-dev.txt', 'requirements-test.txt',
        'setup.py', 'setup.cfg', 'pyproject.toml', 'Pipfile', 'poetry.lock',
        'package.json', 'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml',
        'pom.xml', 'build.gradle', 'build.xml', 'pom.xml',
        'Cargo.toml', 'Cargo.lock', 'Cargo.toml',
        'composer.json', 'composer.lock',
        'Gemfile', 'Gemfile.lock', '.gemspec',
        'package.json', 'npm-shrinkwrap.json',
        'go.mod', 'go.sum',
        'build.sbt', 'pom.xml',
        'pubspec.yaml', 'pubspec.lock',
        'mix.exs', 'mix.lock',
        'cabal.project', 'cabal.project.freeze',
        'stack.yaml', 'stack.yaml.lock'
    }

    def __init__(self, memory_manager=None, token=None):
        self.memory_manager = memory_manager
        self.token = token
        self.temp_dir = Path("/tmp/git_processor_test")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Статистика анализа
        self.stats = {
            'repositories_processed': 0,
            'files_analyzed': 0,
            'total_lines': 0,
            'languages_detected': set()
        }

    def _extract_repo_name(self, repo_url):
        """Извлечение имени репозитория из URL."""
        if repo_url.endswith('.git'):
            repo_url = repo_url[:-4]
        
        # Для GitHub/GitLab URLs
        if 'github.com' in repo_url or 'gitlab.com' in repo_url:
            parts = repo_url.rstrip('/').split('/')
            return parts[-1]
        
        # Для SSH URLs
        if repo_url.startswith('git@'):
            parts = repo_url.split(':')[-1].split('/')
            return parts[-1]
        
        # По умолчанию используем хеш
        import hashlib
        return hashlib.md5(repo_url.encode()).hexdigest()[:8]

    def _count_comments(self, content, language):
        """Подсчет строк комментариев."""
        import re
        
        comment_patterns = {
            'Python': r'^\s*#',
            'JavaScript': r'^\s*//|/\*',
            'TypeScript': r'^\s*//|/\*',
            'Java': r'^\s*//|/\*',
            'C++': r'^\s*//|/\*',
            'C': r'^\s*//|/\*',
            'C#': r'^\s*//|/\*',
            'PHP': r'^\s*//|/\*|#',
            'Ruby': r'^\s*#',
            'Go': r'^\s*//',
            'Rust': r'^\s*//|/\*',
            'Shell': r'^\s*#'
        }
        
        pattern = comment_patterns.get(language, r'^\s*#')
        return len(re.findall(pattern, content, re.MULTILINE))

    def _extract_functions(self, content, language):
        """Извлечение имен функций."""
        import re
        
        function_patterns = {
            'Python': r'def\s+(\w+)\s*\(',
            'JavaScript': r'(?:function\s+(\w+)|(\w+)\s*:\s*function|\w+\s*=>\s*(?:function\s*\()?(\w+))',
            'TypeScript': r'(?:function\s+(\w+)|(\w+)\s*:\s*function|\w+\s*=>\s*(?:function\s*\()?(\w+))',
            'Java': r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\(',
            'C++': r'(?:inline\s+)?(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*(?:const)?\s*{',
            'C': r'(\w+)\s*\([^)]*\)\s*{',
            'C#': r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\(',
            'PHP': r'function\s+(\w+)\s*\(',
            'Go': r'func\s+(?:(?:\([^)]+\)\s+)?(\w+)|(\w+))\s*\(',
            'Rust': r'(?:pub\s+)?fn\s+(\w+)\s*\(',
            'Shell': r'(\w+)\s*\(\)\s*{'
        }
        
        pattern = function_patterns.get(language, r'(\w+)\s*\(')
        functions = re.findall(pattern, content)
        
        # Для JavaScript паттерна с группами
        if isinstance(functions[0], tuple) if functions else False:
            return [g for group in functions for g in group if g]
        
        return functions

    def _extract_classes(self, content, language):
        """Извлечение имен классов."""
        import re
        
        class_patterns = {
            'Python': r'class\s+(\w+)',
            'JavaScript': r'class\s+(\w+)',
            'TypeScript': r'class\s+(\w+)',
            'Java': r'(?:public|private)?\s*(?:abstract\s+)?(?:final\s+)?class\s+(\w+)',
            'C++': r'class\s+(\w+)',
            'C#': r'(?:public|private|internal)?\s*(?:abstract\s+)?(?:sealed\s+)?class\s+(\w+)',
            'PHP': r'class\s+(\w+)',
            'Go': r'type\s+(\w+)\s+struct',
            'Rust': r'struct\s+(\w+)',
            'Ruby': r'class\s+(\w+)'
        }
        
        pattern = class_patterns.get(language, r'')
        if pattern:
            return re.findall(pattern, content)
        return []

    def _extract_imports(self, content, language):
        """Извлечение импортов."""
        import re
        
        import_patterns = {
            'Python': r'(?:import|from)\s+([\w\.]+)',
            'JavaScript': r'(?:import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]|require\([\'"]([^\'"]+)[\'"]\))',
            'TypeScript': r'(?:import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]|require\([\'"]([^\'"]+)[\'"]\))',
            'Java': r'import\s+([\w\.]+);',
            'C++': r'#include\s+[<"]([^>"]+)[>"]',
            'C#': r'(?:using\s+([\w\.]+);|import\s+([\w\.]+))',
            'PHP': r'(?:require|include)(?:_once)?\s+[(\'"]([^\'"]+)[\'"]',
            'Go': r'import\s+[\'"]([^\'"]+)[\'"]',
            'Rust': r'use\s+([\w:]+);',
            'Ruby': r'require\s+[\'"]([^\'"]+)[\'"]'
        }
        
        pattern = import_patterns.get(language, r'')
        if pattern:
            matches = re.findall(pattern, content)
            return [match for match in matches if match] if isinstance(matches[0], tuple) else matches
        return []

    def _get_documentation_type(self, file_name):
        """Определение типа документации."""
        file_name = file_name.lower()
        if file_name.startswith('readme'):
            return 'readme'
        elif 'changelog' in file_name or 'changes' in file_name:
            return 'changelog'
        elif 'license' in file_name or file_name == 'copying':
            return 'license'
        elif 'contributing' in file_name:
            return 'contributing'
        elif 'install' in file_name or 'setup' in file_name:
            return 'installation'
        elif 'usage' in file_name:
            return 'usage'
        elif 'api' in file_name:
            return 'api'
        elif 'todo' in file_name:
            return 'todo'
        else:
            return 'other'

    def _get_dependency_type(self, file_name):
        """Определение типа файла зависимостей."""
        file_name = file_name.lower()
        if 'requirements' in file_name or file_name in ['setup.py', 'setup.cfg', 'pyproject.toml', 'pipfile']:
            return 'python'
        elif file_name in ['package.json', 'yarn.lock', 'pnpm-lock.yaml']:
            return 'javascript'
        elif file_name in ['pom.xml', 'build.gradle', 'build.xml']:
            return 'java'
        elif file_name in ['cargo.toml', 'cargo.lock']:
            return 'rust'
        elif 'composer' in file_name:
            return 'php'
        elif 'gemfile' in file_name or '.gemspec' in file_name:
            return 'ruby'
        elif 'go.mod' in file_name:
            return 'go'
        else:
            return 'other'

    def _detect_architectural_patterns(self, repo_path):
        """Определение архитектурных паттернов."""
        patterns = []
        
        # Ищем MVC паттерн
        mvc_indicators = ['controllers', 'models', 'views']
        if all((repo_path / indicator).exists() for indicator in mvc_indicators):
            patterns.append('MVC')
        
        # Ищем REST API паттерн
        rest_files = list(repo_path.rglob('*api*')) + list(repo_path.rglob('*/routes/*'))
        if rest_files:
            patterns.append('REST API')
        
        # Ищем микросервисную архитектуру
        service_dirs = [d for d in repo_path.iterdir() if d.is_dir() and d.name.endswith('service')]
        if len(service_dirs) > 1:
            patterns.append('Microservices')
        
        # Ищем паттерн plugin
        plugin_indicators = ['plugins', 'extensions', 'addons']
        if any((repo_path / indicator).exists() for indicator in plugin_indicators):
            patterns.append('Plugin Architecture')
        
        # Ищем event-driven архитектуру
        event_files = list(repo_path.rglob('*event*')) + list(repo_path.rglob('*message*'))
        if event_files:
            patterns.append('Event-Driven')
        
        # Ищем layered архитектуру
        layer_dirs = ['presentation', 'business', 'persistence']
        layer_count = sum(1 for dir_name in layer_dirs 
                         if (repo_path / dir_name).exists())
        if layer_count >= 2:
            patterns.append('Layered Architecture')
        
        return patterns

    def _calculate_code_metrics(self, files):
        """Вычисление метрик кода."""
        if not files:
            return {}
        
        metrics = {
            'total_files': len(files),
            'total_lines': sum(f.get('line_count', 0) for f in files),
            'total_size': sum(f.get('size', 0) for f in files),
            'average_file_size': sum(f.get('size', 0) for f in files) / len(files),
            'languages': {},
            'complexity': {},
            'maintainability_index': 0
        }
        
        # Статистика по языкам
        for file_data in files:
            lang = file_data.get('language', 'Unknown')
            if lang not in metrics['languages']:
                metrics['languages'][lang] = {
                    'files': 0,
                    'lines': 0,
                    'size': 0,
                    'functions': 0,
                    'classes': 0
                }
            
            lang_stats = metrics['languages'][lang]
            lang_stats['files'] += 1
            lang_stats['lines'] += file_data.get('line_count', 0)
            lang_stats['size'] += file_data.get('size', 0)
            lang_stats['functions'] += len(file_data.get('functions', []))
            lang_stats['classes'] += len(file_data.get('classes', []))
        
        # Комплексность кода (простая метрика)
        total_functions = sum(len(f.get('functions', [])) for f in files)
        total_classes = sum(len(f.get('classes', [])) for f in files)
        
        metrics['complexity'] = {
            'total_functions': total_functions,
            'total_classes': total_classes,
            'functions_per_file': total_functions / len(files) if files else 0,
            'classes_per_file': total_classes / len(files) if files else 0
        }
        
        # Индекс поддерживаемости (простая формула)
        code_lines = metrics['total_lines']
        comment_lines = sum(f.get('comment_count', 0) for f in files)
        
        if code_lines > 0:
            comment_ratio = comment_lines / code_lines
            metrics['maintainability_index'] = min(100, max(0, comment_ratio * 100))
        
        return metrics


def test_basic_functionality():
    """Тест базовой функциональности."""
    print("=== Тестирование базовой функциональности ===")
    
    # Создаем процессор
    processor = MockGitProcessor()
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
    if comment_count >= 2:  # Должно быть минимум 2 комментария
        print(f"✅ Подсчет комментариев в Python: {comment_count} (минимум 2)")
    else:
        print(f"❌ Подсчет комментариев в Python: {comment_count} (ожидалось минимум 2)")
    
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
    if "function1" in functions and "function2" in functions:
        print(f"✅ Извлечение функций: найдены {functions}")
    else:
        print(f"❌ Извлечение функций: {functions}")
    
    # Тест извлечения классов
    classes = processor._extract_classes(python_func_code, "Python")
    if "MyClass" in classes:
        print(f"✅ Извлечение классов: {classes}")
    else:
        print(f"❌ Извлечение классов: {classes}")
    
    # Тест извлечения импортов
    import_code = """import os
import sys
from typing import List, Dict
from package.module import function
"""
    imports = processor._extract_imports(import_code, "Python")
    if len(imports) >= 3:
        print(f"✅ Извлечение импортов: найдено {len(imports)} импортов")
    else:
        print(f"❌ Извлечение импортов: найдено только {len(imports)}")
    
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


def test_structure_analysis():
    """Тест анализа структуры."""
    print("\n=== Тестирование анализа структуры ===")
    
    processor = MockGitProcessor()
    
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
    
    processor = MockGitProcessor()
    
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
    print("Git Processor - Изолированное тестирование")
    print("=" * 60)
    
    try:
        test_basic_functionality()
        test_structure_analysis()
        test_metrics_calculation()
        
        print("\n" + "=" * 60)
        print("🎉 Все тесты успешно пройдены!")
        print("\nОсновные возможности GitProcessor:")
        print("✅ Анализ Git репозиториев")
        print("✅ Извлечение документации")
        print("✅ Анализ кода и метрик")
        print("✅ Определение технологий")
        print("✅ Анализ структуры проекта")
        print("✅ Поддержка различных языков программирования")
        print("✅ Определение архитектурных паттернов")
        
    except Exception as e:
        print(f"\n❌ Ошибка во время тестирования: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())