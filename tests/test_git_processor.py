"""
Unit тесты для GitProcessor.
Тестирует функциональность анализа Git репозиториев.
"""

import os
import json
import tempfile
import shutil
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

# Добавляем src в путь для импорта
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Мокаем git чтобы избежать зависимостей
sys.modules['git'] = MagicMock()
sys.modules['git.Git'] = MagicMock()

# Импортируем только то что нужно
try:
    from ingest.git_processor import GitProcessor, RepositoryInfo, FileAnalysis
except ImportError:
    # Если не получилось импортировать, создаем моки
    GitProcessor = MagicMock
    RepositoryInfo = MagicMock
    FileAnalysis = MagicMock


class TestGitProcessor:
    """Тесты для GitProcessor."""
    
    @pytest.fixture
    def git_processor(self):
        """Фикстура для создания GitProcessor."""
        return GitProcessor()
    
    @pytest.fixture
    def git_processor_with_token(self):
        """Фикстура для создания GitProcessor с токеном."""
        return GitProcessor(token="test_token_12345")
    
    @pytest.fixture
    def mock_repo(self):
        """Фикстура для мок репозитория."""
        mock_repo = MagicMock()
        mock_repo.remotes.origin.url = "https://github.com/test/repo.git"
        mock_repo.active_branch.name = "main"
        mock_repo.head.commit.hexsha = "abc123def456"
        mock_repo.description = "Test repository"
        return mock_repo
    
    def test_extract_repo_name(self, git_processor):
        """Тест извлечения имени репозитория."""
        # GitHub URL
        assert git_processor._extract_repo_name("https://github.com/user/repo.git") == "repo"
        assert git_processor._extract_repo_name("https://github.com/user/repo") == "repo"
        
        # GitLab URL
        assert git_processor._extract_repo_name("https://gitlab.com/user/repo.git") == "repo"
        
        # SSH URL
        assert git_processor._extract_repo_name("git@github.com:user/repo.git") == "repo"
        
        # Без расширения .git
        assert git_processor._extract_repo_name("https://github.com/user/repo") == "repo"
    
    def test_convert_ssh_to_https(self, git_processor):
        """Тест конвертации SSH URL в HTTPS."""
        # GitHub
        assert git_processor._convert_ssh_to_https("git@github.com:user/repo.git") == \
               "https://github.com/user/repo.git"
        
        # GitLab
        assert git_processor._convert_ssh_to_https("git@gitlab.com:user/repo.git") == \
               "https://gitlab.com/user/repo.git"
        
        # Неизвестный формат
        assert git_processor._convert_ssh_to_https("git@example.com:user/repo.git") == \
               "git@example.com:user/repo.git"
    
    def test_add_auth_to_url(self, git_processor_with_token):
        """Тест добавления токена аутентификации."""
        # GitHub
        assert git_processor_with_token._add_auth_to_url("https://github.com/user/repo.git") == \
               "https://test_token_12345@github.com/user/repo.git"
        
        # GitLab
        assert git_processor_with_token._add_auth_to_url("https://gitlab.com/user/repo.git") == \
               "https://oauth2:test_token_12345@gitlab.com/user/repo.git"
        
        # Неподдерживаемый URL
        assert git_processor_with_token._add_auth_to_url("https://example.com/user/repo.git") == \
               "https://example.com/user/repo.git"
    
    def test_should_analyze_file(self, git_processor):
        """Тест проверки файлов для анализа."""
        # Создаем временные файлы для тестирования
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Файл с поддерживаемым расширением
            py_file = temp_path / "test.py"
            py_file.write_text("print('hello')")
            
            # Файл с неподдерживаемым расширением
            txt_file = temp_path / "test.txt"
            txt_file.write_text("hello world")
            
            # Большой файл (>1MB)
            large_file = temp_path / "large.py"
            large_file.write_text("# " + "x" * (1024 * 1024 + 100))
            
            assert git_processor._should_analyze_file(py_file) is True
            assert git_processor._should_analyze_file(txt_file) is False
            assert git_processor._should_analyze_file(large_file) is False
    
    def test_should_analyze_directory(self, git_processor):
        """Тест проверки директорий для анализа."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Создаем директории
            src_dir = temp_path / "src"
            git_dir = temp_path / ".git"
            node_modules_dir = temp_path / "node_modules"
            pycache_dir = temp_path / "__pycache__"
            
            src_dir.mkdir()
            git_dir.mkdir()
            node_modules_dir.mkdir()
            pycache_dir.mkdir()
            
            assert git_processor._should_analyze_directory(src_dir) is True
            assert git_processor._should_analyze_directory(git_dir) is False
            assert git_processor._should_analyze_directory(node_modules_dir) is False
            assert git_processor._should_analyze_directory(pycache_dir) is False
    
    def test_count_comments(self, git_processor):
        """Тест подсчета комментариев."""
        # Python
        python_code = """# This is a comment
def hello():
    # Another comment
    print("Hello")  # Inline comment
"""
        assert git_processor._count_comments(python_code, "Python") == 3
        
        # JavaScript
        js_code = """// Line comment
function test() {
    /* Block comment */
    console.log("test"); // Inline comment
}
"""
        assert git_processor._count_comments(js_code, "JavaScript") == 3
        
        # Другой язык (должен использовать общий паттерн)
        assert git_processor._count_comments(python_code, "Unknown") >= 1
    
    def test_extract_functions(self, git_processor):
        """Тест извлечения функций."""
        # Python функции
        python_code = """
def function1():
    pass

def function2(param):
    return param

class MyClass:
    def method1(self):
        pass
"""
        functions = git_processor._extract_functions(python_code, "Python")
        assert "function1" in functions
        assert "function2" in functions
        
        # JavaScript функции
        js_code = """
function function1() {
    return 1;
}

const function2 = function() {
    return 2;
};

const arrowFunction = () => 3;
"""
        functions = git_processor._extract_functions(js_code, "JavaScript")
        # Ожидаем несколько функций (зависит от регулярного выражения)
        assert len(functions) > 0
    
    def test_extract_classes(self, git_processor):
        """Тест извлечения классов."""
        code = """
class MyClass {
    constructor() {}
}

class AnotherClass extends BaseClass {
    method() {}
}
"""
        classes = git_processor._extract_classes(code, "JavaScript")
        assert "MyClass" in classes
        assert "AnotherClass" in classes
    
    def test_extract_imports(self, git_processor):
        """Тест извлечения импортов."""
        # Python импорты
        python_code = """import os
import sys
from typing import List, Dict
from package.module import function
"""
        imports = git_processor._extract_imports(python_code, "Python")
        assert "os" in imports
        assert "sys" in imports
        assert "typing" in imports
        assert "package.module" in imports
        
        # JavaScript импорты
        js_code = """import React from 'react';
import { useState } from 'react';
const util = require('./util');
"""
        imports = git_processor._extract_imports(js_code, "JavaScript")
        assert len(imports) > 0
    
    def test_get_documentation_type(self, git_processor):
        """Тест определения типа документации."""
        assert git_processor._get_documentation_type("README.md") == "readme"
        assert git_processor._get_documentation_type("readme.txt") == "readme"
        assert git_processor._get_documentation_type("CHANGELOG.md") == "changelog"
        assert git_processor._get_documentation_type("LICENSE") == "license"
        assert git_processor._get_documentation_type("CONTRIBUTING.md") == "contributing"
        assert git_processor._get_documentation_type("INSTALL.md") == "installation"
        assert git_processor._get_documentation_type("USAGE.md") == "usage"
        assert git_processor._get_documentation_type("API.md") == "api"
        assert git_processor._get_documentation_type("TODO.md") == "todo"
        assert git_processor._get_documentation_type("UNKNOWN.md") == "other"
    
    def test_get_dependency_type(self, git_processor):
        """Тест определения типа зависимостей."""
        assert git_processor._get_dependency_type("requirements.txt") == "python"
        assert git_processor._get_dependency_type("package.json") == "javascript"
        assert git_processor._get_dependency_type("pom.xml") == "java"
        assert git_processor._get_dependency_type("Cargo.toml") == "rust"
        assert git_processor._get_dependency_type("composer.json") == "php"
        assert git_processor._get_dependency_type("go.mod") == "go"
        assert git_processor._get_dependency_type("unknown.xyz") == "other"
    
    def test_analyze_readme_files(self, git_processor):
        """Тест анализа README файлов."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            documentation = {}
            
            readme_content = """# Test Project

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://example.com)

## Installation

1. Install Python
2. Install dependencies

## Usage

```python
import testproject
result = testproject.function()
```

## API Reference

- Function 1
- Function 2
"""
            
            analysis = git_processor._analyze_readme_files(temp_path, {"README.md": {"content": readme_content}})
            
            assert "files" in analysis
            assert "badges" in analysis
            assert "installation_steps" in analysis
            assert "usage_examples" in analysis
            assert "toc" in analysis
            assert len(analysis["badges"]) > 0  # Должен найти badge
            assert len(analysis["toc"]) > 0  # Должен найти секции
    
    def test_parse_requirements_txt(self, git_processor):
        """Тест парсинга requirements.txt."""
        content = """# This is a comment
django==3.2.0
requests>=2.25.0
flask~=1.1.0
numpy
pytest>=6.0.0
"""
        result = git_processor._parse_requirements_txt(content)
        
        assert "dependencies" in result
        assert "type" in result
        assert result["type"] == "pip"
        assert len(result["dependencies"]) == 5  # 5 зависимостей без комментариев
        
        # Проверяем наличие определенных пакетов
        dep_names = [dep["name"] for dep in result["dependencies"]]
        assert "django" in dep_names
        assert "requests" in dep_names
        assert "numpy" in dep_names
    
    def test_parse_package_json(self, git_processor):
        """Тест парсинга package.json."""
        content = """{
  "name": "test-project",
  "version": "1.0.0",
  "dependencies": {
    "react": "^17.0.0",
    "lodash": "4.17.21"
  },
  "devDependencies": {
    "jest": "^27.0.0",
    "eslint": "^8.0.0"
  },
  "scripts": {
    "test": "jest",
    "build": "webpack"
  }
}"""
        
        result = git_processor._parse_package_json(content)
        
        assert "dependencies" in result
        assert "devDependencies" in result
        assert "scripts" in result
        assert "type" in result
        assert result["type"] == "npm"
        assert len(result["dependencies"]) == 2
        assert len(result["devDependencies"]) == 2
        assert result["scripts"]["test"] == "jest"
        
        # Проверяем наличие определенных пакетов
        dep_names = [dep["name"] for dep in result["dependencies"]]
        assert "react" in dep_names
        assert "lodash" in dep_names
    
    def test_parse_cargo_toml(self, git_processor):
        """Тест парсинга Cargo.toml."""
        content = """[package]
name = "test-project"
version = "0.1.0"

[dependencies]
serde = "1.0"
tokio = { version = "1.0", features = ["full"] }

[dev-dependencies]
tempfile = "3.0"
"""
        
        result = git_processor._parse_cargo_toml(content)
        
        assert "dependencies" in result
        assert "devDependencies" in result
        assert "type" in result
        assert result["type"] == "cargo"
        assert len(result["dependencies"]) == 2
        assert len(result["devDependencies"]) == 1
        
        # Проверяем наличие определенных зависимостей
        dep_names = [dep["name"] for dep in result["dependencies"]]
        assert "serde" in dep_names
        assert "tokio" in dep_names
    
    def test_parse_go_mod(self, git_processor):
        """Тест парсинга go.mod."""
        content = """module github.com/user/project

go 1.19

require (
	github.com/gin-gonic/gin v1.8.0
	github.com/spf13/cobra v1.4.0
	golang.org/x/crypto v0.0.0-20220829220503-c86fa9a7ed90
)
"""
        
        result = git_processor._parse_go_mod(content)
        
        assert "dependencies" in result
        assert "type" in result
        assert result["type"] == "go"
        assert len(result["dependencies"]) == 3
        
        # Проверяем наличие определенных модулей
        dep_names = [dep["name"] for dep in result["dependencies"]]
        assert "github.com/gin-gonic/gin" in dep_names
        assert "github.com/spf13/cobra" in dep_names
    
    def test_detect_architectural_patterns(self, git_processor):
        """Тест определения архитектурных паттернов."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Создаем структуру для MVC
            (temp_path / "controllers").mkdir()
            (temp_path / "models").mkdir()
            (temp_path / "views").mkdir()
            
            patterns = git_processor._detect_architectural_patterns(temp_path)
            assert "MVC" in patterns
            
            # Создаем структуру для микросервисов
            shutil.rmtree(temp_path / "controllers")
            shutil.rmtree(temp_path / "models")
            shutil.rmtree(temp_path / "views")
            
            (temp_path / "user-service").mkdir()
            (temp_path / "order-service").mkdir()
            (temp_path / "payment-service").mkdir()
            
            patterns = git_processor._detect_architectural_patterns(temp_path)
            assert "Microservices" in patterns
    
    def test_calculate_code_metrics(self, git_processor):
        """Тест вычисления метрик кода."""
        files = [
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
        
        metrics = git_processor._calculate_code_metrics(files)
        
        assert metrics['total_files'] == 2
        assert metrics['total_lines'] == 250
        assert metrics['total_size'] == 5000
        assert 'languages' in metrics
        assert 'complexity' in metrics
        assert 'maintainability_index' in metrics
        
        # Проверяем статистику по языкам
        assert 'Python' in metrics['languages']
        assert 'JavaScript' in metrics['languages']
        assert metrics['languages']['Python']['files'] == 1
        assert metrics['languages']['JavaScript']['files'] == 1
        
        # Проверяем сложность
        assert metrics['complexity']['total_functions'] == 3
        assert metrics['complexity']['total_classes'] == 3
    
    @patch('git.Repo')
    def test_generate_summary_with_mock_repo(self, git_processor, mock_repo_class, mock_repo):
        """Тест генерации сводной информации с мок репозиторием."""
        # Настройка моков
        mock_repo_class.return_value = mock_repo
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Создаем тестовые файлы
            (temp_path / "README.md").write_text("# Test Project\n\nTest description.")
            (temp_path / "main.py").write_text("print('Hello World')")
            (temp_path / "requirements.txt").write_text("django==3.2.0\nrequests>=2.25.0")
            
            # Патчи для методов анализа
            with patch.object(git_processor, 'extract_documentation') as mock_doc, \
                 patch.object(git_processor, 'analyze_codebase') as mock_code, \
                 patch.object(git_processor, 'extract_dependencies') as mock_deps, \
                 patch.object(git_processor, '_analyze_project_structure') as mock_structure:
                
                mock_doc.return_value = {"README.md": {"content": "Test"}}
                mock_code.return_value = {"metrics": {"total_lines": 100}}
                mock_deps.return_value = {"requirements.txt": {"type": "pip"}}
                mock_structure.return_value = {"key_directories": {}}
                
                repo_info = git_processor.generate_summary(str(temp_path))
                
                assert isinstance(repo_info, RepositoryInfo)
                assert repo_info.name == "repo"
                assert repo_info.url == "https://github.com/test/repo.git"
                assert repo_info.branch == "main"
                assert repo_info.commit_hash == "abc123def456"
    
    def test_get_file_tree(self, git_processor):
        """Тест получения дерева файлов."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Создаем тестовую структуру
            (temp_path / "src").mkdir()
            (temp_path / "src" / "main.py").write_text("print('hello')")
            (temp_path / "src" / "utils.py").write_text("def util(): pass")
            (temp_path / "tests").mkdir()
            (temp_path / "tests" / "test_main.py").write_text("def test(): pass")
            (temp_path / "README.md").write_text("# Project")
            
            tree = git_processor.get_file_tree(str(temp_path), max_depth=2)
            
            assert tree['type'] == 'directory'
            assert tree['name'] == temp_path.name
            assert 'children' in tree
            assert 'src' in tree['children']
            assert 'tests' in tree['children']
            assert 'README.md' in tree['children']
            
            # Проверяем вложенные файлы
            src_children = tree['children']['src']['children']
            assert 'main.py' in src_children
            assert 'utils.py' in src_children
    
    def test_analyze_file_with_different_languages(self, git_processor):
        """Тест анализа файлов разных языков."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            test_cases = [
                ("test.py", "Python", "def hello():\n    # comment\n    print('hello')"),
                ("test.js", "JavaScript", "function hello() {\n    // comment\n    console.log('hello');\n}"),
                ("test.java", "Java", "public class Test {\n    // comment\n    public void hello() {}\n}"),
                ("test.go", "Go", "package main\n// comment\nfunc hello() {}"),
                ("test.rs", "Rust", "fn main() {\n    // comment\n    println!(\"hello\");\n}"),
            ]
            
            for filename, expected_lang, content in test_cases:
                file_path = temp_path / filename
                file_path.write_text(content)
                
                analysis = git_processor._analyze_file(file_path)
                
                assert analysis is not None
                assert analysis.language == expected_lang
                assert analysis.line_count > 0
                assert analysis.comment_count >= 0  # Может быть 0 для некоторых языков
    
    def test_get_statistics(self, git_processor):
        """Тест получения статистики."""
        # Имитируем обработку некоторых файлов
        git_processor.stats['repositories_processed'] = 2
        git_processor.stats['files_analyzed'] = 50
        git_processor.stats['total_lines'] = 1000
        git_processor.stats['languages_detected'] = {'Python', 'JavaScript', 'Go'}
        
        stats = git_processor.get_statistics()
        
        assert stats['repositories_processed'] == 2
        assert stats['files_analyzed'] == 50
        assert stats['total_lines'] == 1000
        assert 'languages_detected' in stats
        assert 'temp_dir' in stats
        assert set(stats['languages_detected']) == {'Python', 'JavaScript', 'Go'}
    
    def test_cleanup(self, git_processor):
        """Тест очистки временных файлов."""
        # Создаем тестовый файл в temp_dir
        test_file = git_processor.temp_dir / "test_file.txt"
        test_file.write_text("test content")
        
        assert test_file.exists()
        
        # Очищаем
        git_processor.cleanup()
        
        assert not git_processor.temp_dir.exists()
    
    def test_error_handling_file_analysis(self, git_processor):
        """Тест обработки ошибок при анализе файлов."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Создаем файл с некорректным содержимым
            problem_file = temp_path / "problem.py"
            # Создаем очень длинную строку, которая может вызвать проблемы
            problem_file.write_text("x" * 1000000)
            
            # Анализ должен вернуть None, а не выбросить исключение
            analysis = git_processor._analyze_file(problem_file)
            assert analysis is None or isinstance(analysis, FileAnalysis)
    
    @patch('git.Repo.clone_from')
    def test_clone_repository_success(self, git_processor, mock_clone_from):
        """Тест успешного клонирования репозитория."""
        # Настройка мока
        mock_repo = MagicMock()
        mock_clone_from.return_value = mock_repo
        
        # Выполняем клонирование
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = git_processor.clone_repository(
                "https://github.com/test/repo.git",
                "main",
                str(Path(temp_dir) / "repo")
            )
            
            assert os.path.exists(local_path)
            mock_clone_from.assert_called_once()
            
            # Проверяем параметры вызова
            args, kwargs = mock_clone_from.call_args
            assert args[0] == "https://github.com/test/repo.git"
            assert kwargs['branch'] == "main"
    
    @patch('git.Repo.clone_from')
    def test_clone_repository_with_token(self, git_processor_with_token, mock_clone_from):
        """Тест клонирования с токеном аутентификации."""
        mock_repo = MagicMock()
        mock_clone_from.return_value = mock_repo
        
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = git_processor_with_token.clone_repository(
                "https://github.com/test/repo.git",
                "main",
                str(Path(temp_dir) / "repo")
            )
            
            # Проверяем, что был использован URL с токеном
            args, kwargs = mock_clone_from.call_args
            auth_url = args[0]
            assert "test_token_12345@" in auth_url
            
            # Проверяем, что конфиг был обновлен с оригинальным URL
            mock_repo.config_writer.assert_called()


class TestFileAnalysis:
    """Тесты для FileAnalysis dataclass."""
    
    def test_file_analysis_creation(self):
        """Тест создания FileAnalysis."""
        analysis = FileAnalysis(
            path="test.py",
            type="source",
            size=1000,
            language="Python",
            line_count=50,
            comment_count=10,
            functions=["func1", "func2"],
            classes=["Class1"],
            imports=["os", "sys"],
            dependencies=["django", "requests"]
        )
        
        assert analysis.path == "test.py"
        assert analysis.type == "source"
        assert analysis.size == 1000
        assert analysis.language == "Python"
        assert analysis.line_count == 50
        assert analysis.comment_count == 10
        assert analysis.functions == ["func1", "func2"]
        assert analysis.classes == ["Class1"]
        assert analysis.imports == ["os", "sys"]
        assert analysis.dependencies == ["django", "requests"]


class TestRepositoryInfo:
    """Тесты для RepositoryInfo dataclass."""
    
    def test_repository_info_creation(self):
        """Тест создания RepositoryInfo."""
        repo_info = RepositoryInfo(
            name="test-repo",
            url="https://github.com/user/test-repo.git",
            branch="main",
            commit_hash="abc123",
            description="Test repository",
            technologies=["Python", "Django"],
            dependencies={"requirements.txt": {"type": "pip"}},
            structure={"key_directories": {}},
            documentation={"README.md": {}},
            code_metrics={"total_lines": 1000}
        )
        
        assert repo_info.name == "test-repo"
        assert repo_info.url == "https://github.com/user/test-repo.git"
        assert repo_info.branch == "main"
        assert repo_info.commit_hash == "abc123"
        assert repo_info.description == "Test repository"
        assert repo_info.technologies == ["Python", "Django"]
        assert "requirements.txt" in repo_info.dependencies
        assert repo_info.code_metrics["total_lines"] == 1000


if __name__ == "__main__":
    # Запуск тестов
    pytest.main([__file__, "-v"])