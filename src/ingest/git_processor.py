"""
Git Repository Processor для анализа кодовых баз.
Обеспечивает клонирование, анализ и извлечение информации из Git репозиториев.
"""

import os
import re
import json
import shutil
import hashlib
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

import git
from git import Repo, GitCommandError, InvalidGitRepositoryError

try:
    import yaml
except ImportError:
    yaml = None

from .ingestion_models import IngestRecord

# Настройка логирования
logger = logging.getLogger(__name__)


@dataclass
class RepositoryInfo:
    """Информация о репозитории."""
    name: str
    url: str
    branch: str
    commit_hash: str
    description: str
    technologies: List[str] = field(default_factory=list)
    dependencies: Dict[str, Any] = field(default_factory=dict)
    structure: Dict[str, Any] = field(default_factory=dict)
    documentation: Dict[str, Any] = field(default_factory=dict)
    code_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FileAnalysis:
    """Анализ файла."""
    path: str
    type: str
    size: int
    language: str
    line_count: int
    comment_count: int
    functions: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)


class GitProcessor:
    """
    Процессор для анализа Git репозиториев.
    
    Поддерживает:
    - Клонирование репозиториев
    - Анализ структуры проекта
    - Извлечение документации
    - Анализ зависимостей
    - Определение технологий
    - Обработку больших репозиториев
    """
    
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
    
    # Конфигурационные файлы
    CONFIG_FILES = {
        '.gitignore', '.gitattributes', '.gitmodules',
        '.dockerignore', 'Dockerfile', 'docker-compose.yml', 'docker-compose.yaml',
        'docker-compose.yml', 'docker-compose.yaml',
        '.env', '.env.example', '.env.local', '.env.development', '.env.production',
        'config.json', 'config.yaml', 'config.yml', 'config.ini',
        '.eslintrc', '.eslintrc.js', '.eslintrc.json', '.eslintrc.yaml',
        '.prettierrc', '.prettierrc.js', '.prettierrc.json', '.prettierrc.yaml',
        '.babelrc', '.babelrc.js', '.babelrc.json',
        'webpack.config.js', 'webpack.config.ts',
        'vite.config.js', 'vite.config.ts',
        'tsconfig.json', 'tsconfig.js', 'tsconfig.yaml',
        '.vscode/settings.json', '.idea/',
        '.travis.yml', '.github/workflows/', 'appveyor.yml', 'circle.yml'
    }

    def __init__(self, memory_manager=None, token: Optional[str] = None):
        """
        Инициализация Git процессора.
        
        Args:
            memory_manager: Менеджер памяти для сохранения анализа
            token: GitHub/GitLab API токен для private репозиториев
        """
        self.memory_manager = memory_manager
        self.token = token
        self.temp_dir = Path("/tmp/git_processor")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Статистика анализа
        self.stats = {
            'repositories_processed': 0,
            'files_analyzed': 0,
            'total_lines': 0,
            'languages_detected': set()
        }

    def clone_repository(
        self, 
        repo_url: str, 
        branch: str = "main", 
        local_path: Optional[str] = None,
        depth: int = 1
    ) -> str:
        """
        Клонирование репозитория.
        
        Args:
            repo_url: URL репозитория
            branch: Ветка для клонирования
            local_path: Локальный путь (генерируется автоматически если не указан)
            depth: Глубина клонирования (1 для shallow clone)
            
        Returns:
            str: Локальный путь к клонированному репозиторию
            
        Raises:
            GitCommandError: При ошибке клонирования
        """
        if local_path is None:
            # Генерируем уникальный путь на основе URL и хеша
            repo_name = self._extract_repo_name(repo_url)
            repo_hash = hashlib.md5(repo_url.encode()).hexdigest()[:8]
            local_path = str(self.temp_dir / f"{repo_name}_{repo_hash}")
        
        # Удаляем существующую директорию если она есть
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
        
        # Добавляем токен для private репозиториев
        if self.token and not repo_url.startswith('https://'):
            # Конвертируем SSH URL в HTTPS если нужен токен
            if repo_url.startswith('git@'):
                repo_url = self._convert_ssh_to_https(repo_url)
        
        auth_url = self._add_auth_to_url(repo_url) if self.token else repo_url
        
        try:
            logger.info(f"Клонирование репозитория {repo_url} (ветка: {branch})")
            
            # Клонируем репозиторий
            repo = Repo.clone_from(
                auth_url, 
                local_path, 
                branch=branch,
                depth=depth
            )
            
            # Если нужен токен, восстанавливаем оригинальный URL
            if self.token:
                with repo.config_writer() as config:
                    config.set_value('remote', 'origin', 'url', repo_url)
            
            logger.info(f"Репозиторий успешно клонирован в {local_path}")
            return local_path
            
        except GitCommandError as e:
            logger.error(f"Ошибка клонирования репозитория {repo_url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Неожиданная ошибка при клонировании: {e}")
            raise

    def extract_documentation(self, repo_path: str) -> Dict[str, Any]:
        """
        Извлечение документации из репозитория.
        
        Args:
            repo_path: Путь к репозиторию
            
        Returns:
            Dict[str, Any]: Словарь с документацией
        """
        logger.info("Извлечение документации...")
        documentation = {}
        repo_path = Path(repo_path)
        
        # Ищем файлы документации
        doc_files = self._find_documentation_files(repo_path)
        
        for file_path in doc_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                file_key = file_path.name
                documentation[file_key] = {
                    'path': str(file_path.relative_to(repo_path)),
                    'content': content,
                    'size': len(content),
                    'lines': len(content.splitlines()),
                    'type': self._get_documentation_type(file_path.name)
                }
                
            except Exception as e:
                logger.warning(f"Ошибка чтения файла {file_path}: {e}")
                continue
        
        # Анализируем README файлы более детально
        readme_analysis = self._analyze_readme_files(repo_path, documentation)
        documentation['readme_analysis'] = readme_analysis
        
        logger.info(f"Найдено {len(documentation)} файлов документации")
        return documentation

    def analyze_codebase(self, repo_path: str) -> Dict[str, Any]:
        """
        Глубокий анализ кодовой базы.
        
        Args:
            repo_path: Путь к репозиторию
            
        Returns:
            Dict[str, Any]: Результаты анализа кода
        """
        logger.info("Анализ кодовой базы...")
        repo_path = Path(repo_path)
        
        analysis = {
            'files': [],
            'structure': {},
            'languages': {},
            'metrics': {},
            'complexity': {},
            'architectural_patterns': []
        }
        
        # Обходим файлы рекурсивно
        file_analyses = self._analyze_files_recursive(repo_path)
        
        # Группируем по типам
        for file_analysis in file_analyses:
            analysis['files'].append(file_analysis.__dict__)
            
            # Языки программирования
            lang = file_analysis.language
            if lang not in analysis['languages']:
                analysis['languages'][lang] = {
                    'files': 0,
                    'lines': 0,
                    'size': 0
                }
            analysis['languages'][lang]['files'] += 1
            analysis['languages'][lang]['lines'] += file_analysis.line_count
            analysis['languages'][lang]['size'] += file_analysis.size
        
        # Структура проекта
        analysis['structure'] = self._analyze_project_structure(repo_path)
        
        # Метрики кода
        analysis['metrics'] = self._calculate_code_metrics(analysis['files'])
        
        # Обновляем статистику
        self.stats['files_analyzed'] += len(analysis['files'])
        self.stats['total_lines'] += analysis['metrics'].get('total_lines', 0)
        self.stats['languages_detected'].update(analysis['languages'].keys())
        
        logger.info(f"Анализ завершен: {len(analysis['files'])} файлов, {len(analysis['languages'])} языков")
        return analysis

    def extract_dependencies(self, repo_path: str) -> Dict[str, Any]:
        """
        Извлечение информации о зависимостях.
        
        Args:
            repo_path: Путь к репозиторию
            
        Returns:
            Dict[str, Any]: Информация о зависимостях
        """
        logger.info("Извлечение зависимостей...")
        dependencies = {}
        repo_path = Path(repo_path)
        
        # Ищем файлы зависимостей
        dep_files = self._find_dependency_files(repo_path)
        
        for file_path in dep_files:
            try:
                file_deps = self._parse_dependency_file(file_path)
                if file_deps:
                    dependencies[file_path.name] = {
                        'path': str(file_path.relative_to(repo_path)),
                        'dependencies': file_deps,
                        'type': self._get_dependency_type(file_path.name)
                    }
            except Exception as e:
                logger.warning(f"Ошибка парсинга файла зависимостей {file_path}: {e}")
                continue
        
        # Анализируем связки между зависимостями
        dependency_graph = self._analyze_dependency_graph(dependencies)
        dependencies['dependency_graph'] = dependency_graph
        
        logger.info(f"Найдено зависимостей в {len(dependencies)} файлах")
        return dependencies

    def generate_summary(self, repo_path: str) -> RepositoryInfo:
        """
        Генерация суммарной информации о репозитории.
        
        Args:
            repo_path: Путь к репозиторию
            
        Returns:
            RepositoryInfo: Сводная информация о репозитории
        """
        logger.info("Генерация сводной информации...")
        
        try:
            repo = Repo(repo_path)
            
            # Базовая информация о репозитории
            repo_info = RepositoryInfo(
                name=self._extract_repo_name(repo.remotes.origin.url if repo.remotes else ""),
                url=str(repo.remotes.origin.url) if repo.remotes else "",
                branch=repo.active_branch.name,
                commit_hash=repo.head.commit.hexsha,
                description=repo.description or "",
                technologies=[],
                dependencies={},
                structure={},
                documentation={},
                code_metrics={}
            )
            
            # Анализ компонентов
            try:
                repo_info.documentation = self.extract_documentation(repo_path)
            except Exception as e:
                logger.warning(f"Ошибка извлечения документации: {e}")
            
            try:
                code_analysis = self.analyze_codebase(repo_path)
                repo_info.code_metrics = code_analysis.get('metrics', {})
                repo_info.technologies = list(code_analysis.get('languages', {}).keys())
            except Exception as e:
                logger.warning(f"Ошибка анализа кода: {e}")
            
            try:
                repo_info.dependencies = self.extract_dependencies(repo_path)
            except Exception as e:
                logger.warning(f"Ошибка извлечения зависимостей: {e}")
            
            try:
                repo_info.structure = self._analyze_project_structure(Path(repo_path))
            except Exception as e:
                logger.warning(f"Ошибка анализа структуры: {e}")
            
            # Анализ архитектурных паттернов
            repo_info.structure['architectural_patterns'] = self._detect_architectural_patterns(repo_path)
            
            logger.info("Сводная информация сгенерирована")
            return repo_info
            
        except Exception as e:
            logger.error(f"Ошибка генерации сводной информации: {e}")
            raise

    def get_file_tree(self, repo_path: str, max_depth: int = 3) -> Dict[str, Any]:
        """
        Получение структуры файлов репозитория.
        
        Args:
            repo_path: Путь к репозиторию
            max_depth: Максимальная глубина вложенности
            
        Returns:
            Dict[str, Any]: Структура файлов
        """
        repo_path = Path(repo_path)
        
        def build_tree(directory: Path, current_depth: int = 0) -> Dict[str, Any]:
            if current_depth > max_depth:
                return {'type': 'directory', 'name': directory.name, 'truncated': True}
            
            tree = {
                'type': 'directory',
                'name': directory.name,
                'path': str(directory.relative_to(repo_path)),
                'children': {}
            }
            
            try:
                # Исключаем файлы .git и другие системные файлы
                items = [item for item in directory.iterdir() 
                        if not item.name.startswith('.') or item.name in ['.github', '.gitignore']]
                
                for item in sorted(items):
                    if item.is_file():
                        tree['children'][item.name] = {
                            'type': 'file',
                            'name': item.name,
                            'path': str(item.relative_to(repo_path)),
                            'size': item.stat().st_size,
                            'extension': item.suffix.lower(),
                            'language': self.LANGUAGE_EXTENSIONS.get(item.suffix.lower(), 'Other')
                        }
                    elif item.is_dir():
                        tree['children'][item.name] = build_tree(item, current_depth + 1)
                        
            except PermissionError:
                logger.warning(f"Нет доступа к директории {directory}")
            
            return tree
        
        return build_tree(repo_path)

    def _extract_repo_name(self, repo_url: str) -> str:
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
        return hashlib.md5(repo_url.encode()).hexdigest()[:8]

    def _convert_ssh_to_https(self, ssh_url: str) -> str:
        """Конвертация SSH URL в HTTPS."""
        if ssh_url.startswith('git@github.com:'):
            return ssh_url.replace('git@github.com:', 'https://github.com/')
        elif ssh_url.startswith('git@gitlab.com:'):
            return ssh_url.replace('git@gitlab.com:', 'https://gitlab.com/')
        return ssh_url

    def _add_auth_to_url(self, url: str) -> str:
        """Добавление токена аутентификации к URL."""
        if 'github.com' in url:
            return url.replace('https://', f'https://{self.token}@')
        elif 'gitlab.com' in url:
            return url.replace('https://', f'https://oauth2:{self.token}@')
        return url

    def _find_documentation_files(self, repo_path: Path) -> List[Path]:
        """Поиск файлов документации."""
        doc_files = []
        
        for file_name in self.DOC_FILES:
            file_path = repo_path / file_name
            if file_path.exists() and file_path.is_file():
                doc_files.append(file_path)
        
        # Ищем также в поддиректориях docs/
        docs_dir = repo_path / 'docs'
        if docs_dir.exists():
            for doc_file in docs_dir.glob('*.md'):
                if doc_file.is_file():
                    doc_files.append(doc_file)
        
        return doc_files

    def _get_documentation_type(self, file_name: str) -> str:
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

    def _analyze_readme_files(self, repo_path: Path, documentation: Dict[str, Any]) -> Dict[str, Any]:
        """Детальный анализ README файлов."""
        readme_files = {k: v for k, v in documentation.items() if 'readme' in k.lower()}
        
        analysis = {
            'files': list(readme_files.keys()),
            'main_readme': None,
            'badges': [],
            'features': [],
            'installation_steps': [],
            'usage_examples': [],
            'toc': []
        }
        
        if readme_files:
            # Используем README.md как основной, если он есть
            main_readme = readme_files.get('README.md') or list(readme_files.values())[0]
            analysis['main_readme'] = main_readme['path']
            
            content = main_readme['content']
            
            # Извлекаем badge-ы
            analysis['badges'] = re.findall(r'!\[.*?\]\(.*?\)', content)
            
            # Ищем секции
            sections = re.findall(r'^##?\s+(.+)$', content, re.MULTILINE)
            analysis['toc'] = sections
            
            # Ищем установку
            install_section = re.search(r'##?\s+Installation.*?(?=##?|\Z)', content, re.DOTALL | re.IGNORECASE)
            if install_section:
                analysis['installation_steps'] = install_section.group(0).split('\n')
            
            # Ищем примеры использования
            usage_section = re.search(r'##?\s+(?:Usage|Example).*?(?=##?|\Z)', content, re.DOTALL | re.IGNORECASE)
            if usage_section:
                analysis['usage_examples'] = usage_section.group(0).split('\n')
        
        return analysis

    def _find_dependency_files(self, repo_path: Path) -> List[Path]:
        """Поиск файлов зависимостей."""
        dep_files = []
        
        # Прямые файлы в корне
        for dep_file in self.DEPENDENCY_FILES:
            file_path = repo_path / dep_file
            if file_path.exists() and file_path.is_file():
                dep_files.append(file_path)
        
        # Файлы в поддиректориях
        for pattern in ['**/package.json', '**/Cargo.toml', '**/pom.xml', '**/build.gradle']:
            dep_files.extend(repo_path.glob(pattern))
        
        return dep_files

    def _get_dependency_type(self, file_name: str) -> str:
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

    def _parse_dependency_file(self, file_path: Path) -> Dict[str, Any]:
        """Парсинг файла зависимостей."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            file_name = file_path.name.lower()
            
            if file_name == 'requirements.txt':
                return self._parse_requirements_txt(content)
            elif file_name == 'package.json':
                return self._parse_package_json(content)
            elif file_name == 'pyproject.toml':
                return self._parse_pyproject_toml(content)
            elif file_name == 'pom.xml':
                return self._parse_pom_xml(content)
            elif file_name == 'cargo.toml':
                return self._parse_cargo_toml(content)
            elif 'go.mod' in file_name:
                return self._parse_go_mod(content)
            elif 'composer.json' in file_name:
                return self._parse_composer_json(content)
            else:
                return {'raw_content': content}
                
        except Exception as e:
            logger.warning(f"Ошибка парсинга файла {file_path}: {e}")
            return {}

    def _parse_requirements_txt(self, content: str) -> Dict[str, Any]:
        """Парсинг requirements.txt."""
        dependencies = []
        dev_dependencies = []
        
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                # Простая обработка спецификаторов версий
                dep = re.sub(r'[<>=!~].*', '', line)
                dependencies.append({
                    'name': dep.strip(),
                    'version_spec': line
                })
        
        return {
            'dependencies': dependencies,
            'type': 'pip',
            'count': len(dependencies)
        }

    def _parse_package_json(self, content: str) -> Dict[str, Any]:
        """Парсинг package.json."""
        try:
            data = json.loads(content)
            
            deps = data.get('dependencies', {})
            dev_deps = data.get('devDependencies', {})
            
            return {
                'dependencies': [{'name': k, 'version': v} for k, v in deps.items()],
                'devDependencies': [{'name': k, 'version': v} for k, v in dev_deps.items()],
                'scripts': data.get('scripts', {}),
                'type': 'npm',
                'count': len(deps) + len(dev_deps)
            }
        except json.JSONDecodeError:
            return {'error': 'Invalid JSON', 'raw_content': content}

    def _parse_pyproject_toml(self, content: str) -> Dict[str, Any]:
        """Парсинг pyproject.toml."""
        if yaml is None:
            return {'error': 'YAML parser not available', 'raw_content': content}
        
        try:
            data = yaml.safe_load(content)
            
            # Poetry формат
            if 'tool' in data and 'poetry' in data['tool']:
                poetry = data['tool']['poetry']
                deps = poetry.get('dependencies', {})
                dev_deps = poetry.get('dev-dependencies', {})
                
                return {
                    'dependencies': [{'name': k, 'version': v} for k, v in deps.items() if k != 'python'],
                    'devDependencies': [{'name': k, 'version': v} for k, v in dev_deps.items()],
                    'type': 'poetry',
                    'count': len(deps) + len(dev_deps)
                }
            
            #/setuptools формат
            if 'project' in data:
                project = data['project']
                deps = project.get('dependencies', [])
                
                return {
                    'dependencies': [{'name': dep} for dep in deps],
                    'type': 'setuptools',
                    'count': len(deps)
                }
            
            return {'raw_content': content}
            
        except Exception as e:
            return {'error': str(e), 'raw_content': content}

    def _parse_pom_xml(self, content: str) -> Dict[str, Any]:
        """Парсинг pom.xml."""
        # Простой парсинг через регулярные выражения
        dependencies = re.findall(r'<dependency>(.*?)</dependency>', content, re.DOTALL)
        deps = []
        
        for dep in dependencies:
            group_id = re.search(r'<groupId>(.*?)</groupId>', dep)
            artifact_id = re.search(r'<artifactId>(.*?)</artifactId>', dep)
            version = re.search(r'<version>(.*?)</version>', dep)
            
            dep_info = {}
            if group_id:
                dep_info['groupId'] = group_id.group(1)
            if artifact_id:
                dep_info['artifactId'] = artifact_id.group(1)
            if version:
                dep_info['version'] = version.group(1)
            
            if dep_info:
                deps.append(dep_info)
        
        return {
            'dependencies': deps,
            'type': 'maven',
            'count': len(deps)
        }

    def _parse_cargo_toml(self, content: str) -> Dict[str, Any]:
        """Парсинг Cargo.toml."""
        dependencies = []
        dev_dependencies = []
        
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('[') and line.endswith(']'):
                current_section = line
            elif current_section == '[dependencies]' and '=' in line:
                name, version = line.split('=', 1)
                dependencies.append({
                    'name': name.strip(),
                    'version': version.strip().strip('"')
                })
            elif current_section == '[dev-dependencies]' and '=' in line:
                name, version = line.split('=', 1)
                dev_dependencies.append({
                    'name': name.strip(),
                    'version': version.strip().strip('"')
                })
        
        return {
            'dependencies': dependencies,
            'devDependencies': dev_dependencies,
            'type': 'cargo',
            'count': len(dependencies) + len(dev_dependencies)
        }

    def _parse_go_mod(self, content: str) -> Dict[str, Any]:
        """Парсинг go.mod."""
        dependencies = []
        
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('require ') and not line.startswith('require //'):
                # Формат: require module version
                parts = line.split()
                if len(parts) >= 3:
                    module = parts[1]
                    version = parts[2]
                    dependencies.append({
                        'name': module,
                        'version': version
                    })
        
        return {
            'dependencies': dependencies,
            'type': 'go',
            'count': len(dependencies)
        }

    def _parse_composer_json(self, content: str) -> Dict[str, Any]:
        """Парсинг composer.json."""
        try:
            data = json.loads(content)
            
            deps = data.get('require', {})
            dev_deps = data.get('require-dev', {})
            
            # Убираем php из зависимостей
            deps = {k: v for k, v in deps.items() if k != 'php'}
            dev_deps = {k: v for k, v in dev_deps.items() if k != 'php'}
            
            return {
                'dependencies': [{'name': k, 'version': v} for k, v in deps.items()],
                'devDependencies': [{'name': k, 'version': v} for k, v in dev_deps.items()],
                'type': 'composer',
                'count': len(deps) + len(dev_deps)
            }
        except json.JSONDecodeError:
            return {'error': 'Invalid JSON', 'raw_content': content}

    def _analyze_dependency_graph(self, dependencies: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ графа зависимостей."""
        graph = {
            'nodes': [],
            'edges': [],
            'clusters': [],
            'conflicts': []
        }
        
        # Собираем все зависимости
        all_deps = {}
        for file_name, file_info in dependencies.items():
            if isinstance(file_info, dict) and 'dependencies' in file_info:
                for dep in file_info.get('dependencies', []):
                    dep_name = dep.get('name', '')
                    if dep_name and dep_name not in all_deps:
                        all_deps[dep_name] = []
                    all_deps[dep_name].append({
                        'file': file_name,
                        'version': dep.get('version', ''),
                        'type': file_info.get('type', '')
                    })
        
        # Создаем узлы и связи
        for dep_name, sources in all_deps.items():
            graph['nodes'].append({
                'id': dep_name,
                'label': dep_name,
                'sources': sources
            })
        
        return graph

    def _analyze_files_recursive(self, repo_path: Path, max_files: int = 1000) -> List[FileAnalysis]:
        """Рекурсивный анализ файлов с ограничением."""
        analyses = []
        files_processed = 0
        
        def analyze_directory(directory: Path):
            nonlocal files_processed
            
            try:
                for item in directory.iterdir():
                    if files_processed >= max_files:
                        break
                    
                    if item.is_file():
                        if self._should_analyze_file(item):
                            analysis = self._analyze_file(item)
                            if analysis:
                                analyses.append(analysis)
                                files_processed += 1
                    elif item.is_dir() and self._should_analyze_directory(item):
                        analyze_directory(item)
                        
            except PermissionError:
                logger.warning(f"Нет доступа к директории {directory}")
        
        analyze_directory(repo_path)
        return analyses

    def _should_analyze_file(self, file_path: Path) -> bool:
        """Проверка, нужно ли анализировать файл."""
        # Проверяем расширение
        if file_path.suffix.lower() not in self.LANGUAGE_EXTENSIONS:
            return False
        
        # Проверяем размер (не более 1MB)
        try:
            if file_path.stat().st_size > 1024 * 1024:
                return False
        except OSError:
            return False
        
        return True

    def _should_analyze_directory(self, directory: Path) -> bool:
        """Проверка, нужно ли анализировать директорию."""
        skip_dirs = {
            '.git', '.svn', '.hg', 'node_modules', '__pycache__',
            '.pytest_cache', '.mypy_cache', 'target', 'build',
            'dist', 'vendor', '.venv', 'venv', 'env',
            '.idea', '.vscode', 'coverage', '.tox'
        }
        
        return directory.name not in skip_dirs

    def _analyze_file(self, file_path: Path) -> Optional[FileAnalysis]:
        """Анализ отдельного файла."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            line_count = len(content.splitlines())
            
            # Определяем язык
            extension = file_path.suffix.lower()
            language = self.LANGUAGE_EXTENSIONS.get(extension, 'Other')
            
            # Подсчет комментариев (простая эвристика)
            comment_count = self._count_comments(content, language)
            
            # Извлечение функций и классов
            functions = self._extract_functions(content, language)
            classes = self._extract_classes(content, language)
            
            # Извлечение импортов
            imports = self._extract_imports(content, language)
            
            # Зависимости файла
            file_deps = self._extract_file_dependencies(content, language)
            
            return FileAnalysis(
                path=str(file_path.relative_to(file_path.parent.parent)),
                type='source' if language != 'Other' else 'other',
                size=len(content),
                language=language,
                line_count=line_count,
                comment_count=comment_count,
                functions=functions,
                classes=classes,
                imports=imports,
                dependencies=file_deps
            )
            
        except Exception as e:
            logger.warning(f"Ошибка анализа файла {file_path}: {e}")
            return None

    def _count_comments(self, content: str, language: str) -> int:
        """Подсчет строк комментариев."""
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

    def _extract_functions(self, content: str, language: str) -> List[str]:
        """Извлечение имен функций."""
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

    def _extract_classes(self, content: str, language: str) -> List[str]:
        """Извлечение имен классов."""
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

    def _extract_imports(self, content: str, language: str) -> List[str]:
        """Извлечение импортов."""
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

    def _extract_file_dependencies(self, content: str, language: str) -> List[str]:
        """Извлечение зависимостей файла."""
        # Используем ту же логику что и для импортов
        return self._extract_imports(content, language)

    def _analyze_project_structure(self, repo_path: Path) -> Dict[str, Any]:
        """Анализ структуры проекта."""
        structure = {
            'root_structure': {},
            'key_directories': {},
            'architecture_patterns': [],
            'complexity_score': 0
        }
        
        # Анализ ключевых директорий
        key_dirs = {
            'src': 'Исходный код',
            'lib': 'Библиотеки',
            'app': 'Приложение',
            'main': 'Главный модуль',
            'core': 'Ядро',
            'api': 'API',
            'server': 'Сервер',
            'client': 'Клиент',
            'frontend': 'Frontend',
            'backend': 'Backend',
            'docs': 'Документация',
            'test': 'Тесты',
            'tests': 'Тесты',
            'spec': 'Спецификации',
            'examples': 'Примеры',
            'scripts': 'Скрипты',
            'tools': 'Инструменты',
            'config': 'Конфигурация',
            'assets': 'Ресурсы',
            'public': 'Публичные файлы',
            'build': 'Сборка',
            'dist': 'Дистрибутив',
            'target': 'Цель сборки',
            'vendor': 'Сторонние библиотеки',
            'node_modules': 'Node модули'
        }
        
        for dir_name, description in key_dirs.items():
            dir_path = repo_path / dir_name
            if dir_path.exists() and dir_path.is_dir():
                structure['key_directories'][dir_name] = {
                    'exists': True,
                    'description': description,
                    'file_count': len(list(dir_path.rglob('*'))) if dir_path.exists() else 0
                }
        
        # Определяем архитектурные паттерны
        structure['architecture_patterns'] = self._detect_architectural_patterns(repo_path)
        
        return structure

    def _detect_architectural_patterns(self, repo_path: Path) -> List[str]:
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

    def _calculate_code_metrics(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
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

    async def process_repository_async(
        self, 
        repo_url: str, 
        branch: str = "main",
        save_to_memory: bool = True
    ) -> RepositoryInfo:
        """
        Асинхронная обработка репозитория.
        
        Args:
            repo_url: URL репозитория
            branch: Ветка
            save_to_memory: Сохранить в MemoryManager
            
        Returns:
            RepositoryInfo: Результаты анализа
        """
        logger.info(f"Асинхронная обработка репозитория {repo_url}")
        
        # Клонирование в отдельном потоке
        loop = asyncio.get_event_loop()
        repo_path = await loop.run_in_executor(
            None, self.clone_repository, repo_url, branch
        )
        
        try:
            # Полный анализ
            repo_info = await loop.run_in_executor(
                None, self.generate_summary, repo_path
            )
            
            # Сохранение в память
            if save_to_memory and self.memory_manager:
                await loop.run_in_executor(
                    None, self._save_to_memory, repo_info
                )
            
            # Обновляем статистику
            self.stats['repositories_processed'] += 1
            
            return repo_info
            
        finally:
            # Очистка временных файлов
            if os.path.exists(repo_path):
                shutil.rmtree(repo_path)
    
    def _save_to_memory(self, repo_info: RepositoryInfo):
        """Сохранение результатов анализа в MemoryManager."""
        if not self.memory_manager:
            return
        
        try:
            from ..memory_manager.memory_manager_interface import MemoryLayer
            
            # Сохраняем основную информацию в Semantic Memory
            memory_item = {
                'id': f"repo::{repo_info.name}",
                'layer': MemoryLayer.SEMANTIC,
                'data': {
                    'type': 'repository_analysis',
                    'repository_info': repo_info.__dict__,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'processor_version': '1.0.0'
                },
                'metadata': {
                    'repository_url': repo_info.url,
                    'repository_name': repo_info.name,
                    'branch': repo_info.branch,
                    'languages': repo_info.technologies,
                    'file_count': len(repo_info.code_metrics.get('languages', {}))
                }
            }
            
            self.memory_manager.store(memory_item)
            
            logger.info(f"Результаты анализа сохранены в память для {repo_info.name}")
            
        except Exception as e:
            logger.warning(f"Ошибка сохранения в память: {e}")

    def cleanup(self):
        """Очистка временных файлов."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info("Временные файлы очищены")

    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики обработки."""
        return {
            **self.stats,
            'languages_detected': list(self.stats['languages_detected']),
            'temp_dir': str(self.temp_dir)
        }