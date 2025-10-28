# Git Repository Processor

Мощный инструмент для анализа Git репозиториев, извлечения документации, анализа кода и определения технологий. Интегрирован с MemoryManager для сохранения результатов анализа.

## Возможности

### 🔄 Клонирование репозиториев
- Поддержка public и private репозиториев
- Автоматическая аутентификация через API токены
- Shallow cloning для больших репозиториев
- Поддержка различных веток

### 📋 Извлечение документации
- README файлы (всех форматов: .md, .rst, .txt)
- CHANGELOG и HISTORY файлы
- LICENSE файлы
- CONTRIBUTING и INSTALL файлы
- API документация

### 🔍 Анализ кодовой базы
- Определение языков программирования
- Подсчет строк кода и комментариев
- Извлечение функций и классов
- Анализ импортов и зависимостей
- Вычисление метрик кода

### 📦 Анализ зависимостей
- Python: requirements.txt, pyproject.toml, Pipfile
- JavaScript: package.json, yarn.lock, pnpm-lock.yaml
- Java: pom.xml, build.gradle
- Rust: Cargo.toml, Cargo.lock
- Go: go.mod, go.sum
- PHP: composer.json
- Ruby: Gemfile

### 🏗️ Анализ структуры проекта
- Определение архитектурных паттернов (MVC, microservices, etc.)
- Анализ ключевых директорий (src/, docs/, tests/, etc.)
- Обнаружение конфигурационных файлов
- Глубокая навигация по структуре

### 🧠 Интеграция с MemoryManager
- Сохранение анализа в слои памяти
- Semantic Memory для сводной информации
- Automatic metadata extraction
- Vector embeddings support

## Установка

### Зависимости
```bash
pip install GitPython pydantic PyYAML
```

### Инициализация
```python
from src.ingest.git_processor import GitProcessor

# Базовое использование
processor = GitProcessor()

# С токеном для private репозиториев
processor = GitProcessor(token="your_github_token")

# С интеграцией MemoryManager
processor = GitProcessor(memory_manager=memory_manager, token="token")
```

## Базовое использование

### 1. Клонирование репозитория
```python
# Клонирование public репозитория
repo_path = processor.clone_repository(
    "https://github.com/user/repo.git",
    branch="main"
)

# Клонирование с shallow clone для больших репозиториев
repo_path = processor.clone_repository(
    "https://github.com/torvalds/linux.git",
    branch="master",
    depth=1
)
```

### 2. Извлечение документации
```python
documentation = processor.extract_documentation(repo_path)

for doc_name, doc_info in documentation.items():
    if doc_name != 'readme_analysis':
        print(f"{doc_name}: {doc_info['lines']} строк")
```

### 3. Анализ кодовой базы
```python
codebase_analysis = processor.analyze_codebase(repo_path)

print(f"Проанализировано файлов: {codebase_analysis['total_files']}")
print(f"Языков: {len(codebase_analysis['languages'])}")
print(f"Общих строк: {codebase_analysis['metrics']['total_lines']}")

# Статистика по языкам
for lang, stats in codebase_analysis['languages'].items():
    print(f"{lang}: {stats['files']} файлов, {stats['lines']} строк")
```

### 4. Извлечение зависимостей
```python
dependencies = processor.extract_dependencies(repo_path)

for dep_file, dep_info in dependencies.items():
    if isinstance(dep_info, dict) and 'type' in dep_info:
        count = dep_info.get('count', 0)
        dep_type = dep_info.get('type', 'unknown')
        print(f"{dep_file}: {count} зависимостей ({dep_type})")
```

### 5. Получение структуры файлов
```python
file_tree = processor.get_file_tree(repo_path, max_depth=3)

def print_tree(tree, indent=0):
    prefix = "  " * indent
    if tree['type'] == 'directory':
        print(f"{prefix}📁 {tree['name']}/")
        for child in tree.get('children', {}).values():
            print_tree(child, indent + 1)
    else:
        lang_emoji = {
            'Python': '🐍',
            'JavaScript': '🟨',
            'TypeScript': '🔷',
            'Java': '☕',
            'Go': '🐹',
            'Rust': '🦀'
        }
        emoji = lang_emoji.get(tree.get('language'), '📄')
        print(f"{prefix}{emoji} {tree['name']}")

print_tree(file_tree)
```

### 6. Генерация сводной информации
```python
summary = processor.generate_summary(repo_path)

print(f"Репозиторий: {summary.name}")
print(f"URL: {summary.url}")
print(f"Ветка: {summary.branch}")
print(f"Коммит: {summary.commit_hash[:8]}...")
print(f"Технологии: {', '.join(summary.technologies)}")
print(f"Архитектурные паттерны: {', '.join(summary.structure.get('architectural_patterns', []))}")
```

## Асинхронное использование

```python
import asyncio

async def process_repository():
    processor = GitProcessor()
    
    try:
        summary = await processor.process_repository_async(
            "https://github.com/microsoft/vscode.git",
            save_to_memory=True
        )
        
        print(f"Обработан: {summary.name}")
        print(f"Файлов: {summary.code_metrics['total_files']}")
        
    finally:
        processor.cleanup()

# Запуск
asyncio.run(process_repository())
```

## Интеграция с MemoryManager

```python
from src.memory_manager.memory_manager_interface import MemoryManager

# Создаем MemoryManager
memory_manager = MemoryManager()
processor = GitProcessor(memory_manager=memory_manager)

# Обработка автоматически сохранит результаты в память
summary = processor.generate_summary(repo_path)

# Результаты сохраняются в Semantic Memory с метаданными:
# - repository_url
# - repository_name  
# - branch
# - languages
# - file_count
```

## Работа с Private репозиториями

### Настройка токена
```python
# Вариант 1: Через конструктор
processor = GitProcessor(token="ghp_xxxxxxxxxxxxxxxxxxxx")

# Вариант 2: Через переменную окружения
import os
os.environ["GITHUB_TOKEN"] = "ghp_xxxxxxxxxxxxxxxxxxxx"
processor = GitProcessor(token=os.getenv("GITHUB_TOKEN"))
```

### Поддерживаемые платформы
- GitHub (personal access token)
- GitLab (oauth2 token)
- Private instances (поддержка настраивается)

## Обработка больших репозиториев

### Shallow Clone
```python
# Для очень больших репозиториев
repo_path = processor.clone_repository(
    "https://github.com/torvalds/linux.git",
    depth=1  # Только последний коммит
)
```

### Ограничение анализа
```python
# Анализ только первых 500 файлов
file_analyses = processor._analyze_files_recursive(Path(repo_path), max_files=500)
```

### Поэтапный анализ
```python
# Сначала структура
structure = processor.get_file_tree(repo_path, max_depth=2)

# Затем документация
docs = processor.extract_documentation(repo_path)

# И наконец код (при необходимости)
code = processor.analyze_codebase(repo_path)
```

## Поддерживаемые языки

| Язык | Расширения | Анализ функций | Анализ классов | Комментарии |
|------|------------|----------------|----------------|-------------|
| Python | .py | ✅ | ✅ | ✅ |
| JavaScript | .js, .jsx | ✅ | ✅ | ✅ |
| TypeScript | .ts, .tsx | ✅ | ✅ | ✅ |
| Java | .java | ✅ | ✅ | ✅ |
| C++ | .cpp, .hpp | ✅ | ✅ | ✅ |
| C | .c, .h | ✅ | ✅ | ✅ |
| C# | .cs | ✅ | ✅ | ✅ |
| PHP | .php | ✅ | ✅ | ✅ |
| Go | .go | ✅ | ✅ | ✅ |
| Rust | .rs | ✅ | ✅ | ✅ |
| Ruby | .rb | ✅ | ✅ | ✅ |
| Shell | .sh, .bash | ✅ | ❌ | ✅ |

## Архитектурные паттерны

Процессор автоматически определяет:

- **MVC** - наличие папок controllers/, models/, views/
- **REST API** - файлы api/, routes/
- **Microservices** - несколько папок *-service/
- **Plugin Architecture** - папки plugins/, extensions/, addons/
- **Event-Driven** - файлы *event*, *message*
- **Layered Architecture** - папки presentation/, business/, persistence/

## Метрики кода

```python
metrics = processor._calculate_code_metrics(files)

# Доступные метрики:
# - total_files: общее количество файлов
# - total_lines: общее количество строк
# - total_size: общий размер в байтах
# - average_file_size: средний размер файла
# - languages: статистика по языкам
# - complexity: метрики сложности
# - maintainability_index: индекс поддерживаемости (0-100)
```

## Обработка ошибок

```python
try:
    repo_path = processor.clone_repository(repo_url)
    # Обработка...
except GitCommandError as e:
    print(f"Ошибка Git: {e}")
except Exception as e:
    print(f"Общая ошибка: {e}")
finally:
    processor.cleanup()  # Всегда очищаем временные файлы
```

## Статистика

```python
stats = processor.get_statistics()

print(f"Обработано репозиториев: {stats['repositories_processed']}")
print(f"Проанализировано файлов: {stats['files_analyzed']}")
print(f"Всего строк кода: {stats['total_lines']}")
print(f"Обнаружено языков: {len(stats['languages_detected'])}")
```

## Файлы конфигурации

Поддерживаемые файлы конфигурации:
- `.gitignore`, `.gitattributes`, `.gitmodules`
- `Dockerfile`, `docker-compose.yml`
- `.eslintrc.*`, `.prettierrc.*`, `.babelrc.*`
- `webpack.config.*`, `vite.config.*`
- `tsconfig.json`
- `.vscode/`, `.idea/`
- `.github/workflows/`

## Тестирование

```bash
# Запуск unit тестов
python -m pytest tests/test_git_processor.py -v

# Запуск с покрытием
python -m pytest tests/test_git_processor.py --cov=src.ingest.git_processor
```

## Примеры использования

Полные примеры смотрите в `examples/git_processor_examples.py`:

- Базовое использование
- Работа с токенами
- Асинхронная обработка
- Интеграция с MemoryManager
- Обработка больших репозиториев
- Пользовательский анализ

## Ограничения

1. **Размер файлов**: файлы больше 1MB не анализируются
2. **Глубина анализа**: по умолчанию ограничена для производительности
3. **Токены**: требует настройки для private репозиториев
4. **Сетевые ограничения**: зависит от Git доступности

## Производительность

- **Shallow clone** снижает время клонирования в 10-100 раз
- **Ограничение файлов** предотвращает зависание на больших репозиториях
- **Асинхронная обработка** не блокирует основной поток
- **Кэширование** результатов в MemoryManager

## Логирование

```python
import logging

# Настройка уровня логирования
logging.basicConfig(level=logging.INFO)

# Или для конкретного модуля
logging.getLogger('src.ingest.git_processor').setLevel(logging.DEBUG)
```

## Вклад в разработку

1. Форкните репозиторий
2. Создайте feature branch
3. Добавьте тесты для новой функциональности
4. Убедитесь что все тесты проходят
5. Создайте Pull Request

## Лицензия

Модуль является частью Rebecca Platform и распространяется под той же лицензией.