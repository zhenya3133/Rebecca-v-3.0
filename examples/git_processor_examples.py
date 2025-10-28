"""
Примеры использования GitProcessor.
Демонстрирует различные сценарии анализа Git репозиториев.
"""

import asyncio
import os
from pathlib import Path

# Импортируем GitProcessor
from src.ingest.git_processor import GitProcessor


def example_basic_usage():
    """Базовое использование GitProcessor."""
    print("=== Базовое использование GitProcessor ===")
    
    # Создаем процессор
    processor = GitProcessor()
    
    try:
        # Клонируем репозиторий
        repo_url = "https://github.com/octocat/Hello-World.git"
        repo_path = processor.clone_repository(repo_url, branch="master")
        print(f"Репозиторий клонирован в: {repo_path}")
        
        # Извлекаем документацию
        documentation = processor.extract_documentation(repo_path)
        print(f"Найдено файлов документации: {len(documentation)}")
        
        # Анализируем кодовую базу
        codebase_analysis = processor.analyze_codebase(repo_path)
        print(f"Проанализировано файлов: {codebase_analysis['total_files']}")
        print(f"Обнаружено языков: {len(codebase_analysis['languages'])}")
        
        # Извлекаем зависимости
        dependencies = processor.extract_dependencies(repo_path)
        print(f"Найдено файлов зависимостей: {len(dependencies)}")
        
        # Получаем структуру файлов
        file_tree = processor.get_file_tree(repo_path)
        print(f"Структура файлов получена")
        
        # Генерируем сводную информацию
        summary = processor.generate_summary(repo_path)
        print(f"Сводная информация:")
        print(f"  - Название: {summary.name}")
        print(f"  - Описание: {summary.description}")
        print(f"  - Ветка: {summary.branch}")
        print(f"  - Коммит: {summary.commit_hash[:8]}...")
        print(f"  - Технологии: {', '.join(summary.technologies)}")
        
    except Exception as e:
        print(f"Ошибка: {e}")
    
    finally:
        # Очищаем временные файлы
        processor.cleanup()


def example_with_token():
    """Использование с токеном для private репозиториев."""
    print("\n=== Использование с токеном аутентификации ===")
    
    # Создаем процессор с токеном
    token = os.getenv("GITHUB_TOKEN")  # Берем токен из переменной окружения
    if token:
        processor = GitProcessor(token=token)
        
        try:
            # Клонируем private репозиторий
            private_repo_url = "https://github.com/private-user/private-repo.git"
            repo_path = processor.clone_repository(private_repo_url)
            print(f"Private репозиторий клонирован успешно")
            
            # Анализируем
            summary = processor.generate_summary(repo_path)
            print(f"Анализ завершен: {summary.name}")
            
        except Exception as e:
            print(f"Ошибка клонирования private репозитория: {e}")
        
        finally:
            processor.cleanup()
    else:
        print("GITHUB_TOKEN не установлен в переменных окружения")


async def example_async_processing():
    """Асинхронная обработка репозитория."""
    print("\n=== Асинхронная обработка ===")
    
    processor = GitProcessor()
    
    try:
        # Асинхронная обработка репозитория
        repo_url = "https://github.com/python/cpython.git"
        summary = await processor.process_repository_async(repo_url)
        
        print(f"Асинхронная обработка завершена:")
        print(f"  - Репозиторий: {summary.name}")
        print(f"  - Файлов: {summary.code_metrics.get('total_files', 0)}")
        print(f"  - Строк кода: {summary.code_metrics.get('total_lines', 0)}")
        print(f"  - Языков: {len(summary.technologies)}")
        
    except Exception as e:
        print(f"Ошибка: {e}")
    
    finally:
        processor.cleanup()


def example_memory_integration():
    """Интеграция с MemoryManager."""
    print("\n=== Интеграция с MemoryManager ===")
    
    # Создаем мок MemoryManager для демонстрации
    class MockMemoryManager:
        def store(self, item):
            print(f"Сохранено в память:")
            print(f"  - ID: {item['id']}")
            print(f"  - Слой: {item['layer']}")
            print(f"  - Метаданные: {item['metadata']}")
    
    memory_manager = MockMemoryManager()
    processor = GitProcessor(memory_manager=memory_manager)
    
    try:
        # Обрабатываем репозиторий с сохранением в память
        repo_url = "https://github.com/microsoft/vscode.git"
        repo_path = processor.clone_repository(repo_url, branch="main")
        
        # Генерируем сводку (автоматически сохранится в память)
        summary = processor.generate_summary(repo_path)
        print(f"Репозиторий {summary.name} проанализирован и сохранен в память")
        
    except Exception as e:
        print(f"Ошибка: {e}")
    
    finally:
        processor.cleanup()


def example_large_repository():
    """Обработка большого репозитория с ограничениями."""
    print("\n=== Обработка большого репозитория ===")
    
    processor = GitProcessor()
    
    try:
        # Клонируем большой репозиторий с shallow clone
        repo_url = "https://github.com/torvalds/linux.git"
        repo_path = processor.clone_repository(
            repo_url, 
            branch="master", 
            depth=1  # Shallow clone для экономии времени
        )
        
        # Ограничиваем анализ первыми 500 файлами
        codebase = processor.analyze_codebase(repo_path)
        print(f"Ограниченный анализ:")
        print(f"  - Проанализировано файлов: {codebase['total_files']}")
        print(f"  - Языков: {len(codebase['languages'])}")
        
        # Получаем только структуру без глубокого анализа файлов
        structure = processor.get_file_tree(repo_path, max_depth=2)
        print(f"Структура получена (глубина: 2)")
        
    except Exception as e:
        print(f"Ошибка: {e}")
    
    finally:
        processor.cleanup()


def example_analyze_specific_files():
    """Анализ конкретных типов файлов."""
    print("\n=== Анализ конкретных файлов ===")
    
    processor = GitProcessor()
    
    try:
        # Клонируем репозиторий с различными типами файлов
        repo_url = "https://github.com/facebook/react.git"
        repo_path = processor.clone_repository(repo_url)
        
        # Извлекаем только документацию
        docs = processor.extract_documentation(repo_path)
        print("Документация:")
        for doc_name, doc_info in docs.items():
            if doc_name != 'readme_analysis':
                print(f"  - {doc_name}: {doc_info.get('lines', 0)} строк")
        
        # Анализируем только зависимости
        deps = processor.extract_dependencies(repo_path)
        print("\nЗависимости:")
        for dep_file, dep_info in deps.items():
            if isinstance(dep_info, dict) and 'type' in dep_info:
                count = dep_info.get('count', 0)
                dep_type = dep_info.get('type', 'unknown')
                print(f"  - {dep_file}: {count} зависимостей ({dep_type})")
        
        # Получаем статистику
        stats = processor.get_statistics()
        print(f"\nСтатистика обработки:")
        print(f"  - Репозиториев обработано: {stats['repositories_processed']}")
        print(f"  - Файлов проанализировано: {stats['files_analyzed']}")
        print(f"  - Всего строк: {stats['total_lines']}")
        
    except Exception as e:
        print(f"Ошибка: {e}")
    
    finally:
        processor.cleanup()


def example_custom_analysis():
    """Пользовательский анализ структуры проекта."""
    print("\n=== Пользовательский анализ ===")
    
    processor = GitProcessor()
    
    try:
        repo_url = "https://github.com/django/django.git"
        repo_path = processor.clone_repository(repo_url)
        
        # Получаем детальную структуру
        file_tree = processor.get_file_tree(repo_path, max_depth=3)
        
        # Анализируем структуру вручную
        def analyze_structure(tree, level=0, max_level=2):
            indent = "  " * level
            result = {}
            
            if tree['type'] == 'directory' and level <= max_level:
                for name, item in tree.get('children', {}).items():
                    if item['type'] == 'directory':
                        result[name] = analyze_structure(item, level + 1, max_level)
                    else:
                        result[name] = {
                            'extension': item.get('extension'),
                            'language': item.get('language'),
                            'size': item.get('size')
                        }
            
            return result
        
        structure = analyze_structure(file_tree)
        print("Структура проекта (первые 3 уровня):")
        
        # Показываем основные директории
        for dir_name, dir_content in structure.items():
            if isinstance(dir_content, dict) and dir_name in ['src', 'django', 'tests']:
                print(f"\n{dir_name}/:")
                file_count = sum(1 for item in dir_content.values() 
                               if isinstance(item, dict) and item.get('type') != 'directory')
                print(f"  - Файлов: {file_count}")
                
                # Показываем типы файлов
                extensions = set()
                for item in dir_content.values():
                    if isinstance(item, dict) and 'extension' in item:
                        if item['extension']:
                            extensions.add(item['extension'])
                print(f"  - Расширения: {', '.join(sorted(extensions))}")
        
    except Exception as e:
        print(f"Ошибка: {e}")
    
    finally:
        processor.cleanup()


def main():
    """Главная функция с примерами."""
    print("Git Processor - Примеры использования\n")
    
    # Запускаем примеры
    try:
        example_basic_usage()
    except Exception as e:
        print(f"Ошибка в базовом примере: {e}")
    
    try:
        example_with_token()
    except Exception as e:
        print(f"Ошибка в примере с токеном: {e}")
    
    try:
        example_large_repository()
    except Exception as e:
        print(f"Ошибка в примере с большим репозиторием: {e}")
    
    try:
        example_analyze_specific_files()
    except Exception as e:
        print(f"Ошибка в примере с конкретными файлами: {e}")
    
    try:
        example_memory_integration()
    except Exception as e:
        print(f"Ошибка в примере с памятью: {e}")
    
    try:
        example_custom_analysis()
    except Exception as e:
        print(f"Ошибка в пользовательском примере: {e}")
    
    # Запускаем асинхронный пример
    print("\n=== Запуск асинхронного примера ===")
    try:
        asyncio.run(example_async_processing())
    except Exception as e:
        print(f"Ошибка в асинхронном примере: {e}")
    
    print("\nВсе примеры завершены!")


if __name__ == "__main__":
    main()