#!/usr/bin/env python3
"""
Простой пример использования мета-агента Rebecca.

Демонстрирует основные возможности:
- Инициализацию мета-агента
- Поглощение источников
- Планирование задач
- Генерацию плейбуков
- Мониторинг выполнения
"""

import asyncio
import logging
import sys
import tempfile
from pathlib import Path

# Добавляем src в путь для импорта
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rebecca import RebeccaMetaAgent, MetaAgentConfig, MetaAgentDemo, MetaAgentTestData
from rebecca.utils import run_example_workflow, validate_agent_setup
from memory_manager.memory_manager import MemoryManager
from memory_manager.adaptive_blueprint import AdaptiveBlueprintTracker

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockIngestPipeline:
    """Mock класс для демонстрации IngestPipeline."""
    
    def ingest_document(self, file_path: str, chunk_override=None):
        """Mock метод обработки документа."""
        class MockEvent:
            id = f"mock_event_{hash(file_path) % 10000}"
            attrs = {
                "text": f"Mock content from {file_path}",
                "source_path": file_path,
                "file_type": Path(file_path).suffix
            }
        
        logger.info(f"Mock: Обработка документа {file_path}")
        return MockEvent()
    
    def process_git_repo(self, repo_url: str, branch="main", process_readme=True, process_source=True):
        """Mock метод обработки Git репозитория."""
        class MockEvent:
            id = f"mock_git_event_{hash(repo_url) % 10000}"
            attrs = {
                "text": f"Mock Git content from {repo_url}",
                "source_path": repo_url,
                "file_type": "git"
            }
        
        logger.info(f"Mock: Обработка Git репозитория {repo_url}")
        return [MockEvent()]


class MockContextHandler:
    """Mock класс для ContextHandler."""
    pass


async def create_mock_components():
    """Создает mock компоненты для демонстрации."""
    logger.info("Создание mock компонентов...")
    
    # Создаем MemoryManager
    memory_manager = MemoryManager()
    
    # Создаем BlueprintTracker
    blueprint_tracker = AdaptiveBlueprintTracker(memory_manager.semantic)
    
    # Создаем mock компоненты
    ingest_pipeline = MockIngestPipeline()
    context_handler = MockContextHandler()
    
    logger.info("Mock компоненты созданы успешно")
    return memory_manager, ingest_pipeline, context_handler, blueprint_tracker


async def demonstrate_basic_usage():
    """Демонстрация базового использования мета-агента."""
    logger.info("=== Демонстрация базового использования ===")
    
    # Создаем компоненты
    memory_manager, ingest_pipeline, context_handler, blueprint_tracker = await create_mock_components()
    
    # Создаем конфигурацию
    config = MetaAgentConfig(
        max_concurrent_tasks=5,
        quality_threshold=0.8,
        enable_auto_scaling=False  # Отключаем для демонстрации
    )
    
    # Создаем мета-агента
    agent = RebeccaMetaAgent(
        memory_manager=memory_manager,
        ingest_pipeline=ingest_pipeline,
        context_handler=context_handler,
        blueprint_tracker=blueprint_tracker,
        config=config
    )
    
    logger.info("Мета-агент создан успешно")
    
    # Проверяем настройку агента
    setup_validation = validate_agent_setup(agent)
    if not setup_validation['setup_valid']:
        logger.error(f"Проблемы с настройкой агента: {setup_validation['issues']}")
        return None
    
    logger.info("Настройка агента проверена и корректна")
    
    # Создаем тестовый файл
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write("""# Пример документа для мета-агента

## Описание
Этот документ создан для демонстрации работы мета-агента Rebecca.

## Содержание
Мета-агент выполняет следующие функции:
- Поглощение источников
- Планирование задач
- Генерация плейбуков
- Координация агентов

## Техническая информация
- Система использует многослойную архитектуру памяти
- Поддерживает различные типы агентов
- Обеспечивает автоматическую оптимизацию

## Заключение
Данная демонстрация показывает основные возможности системы.
""")
        test_file_path = f.name
    
    try:
        # Шаг 1: Поглощение источников
        logger.info("Шаг 1: Поглощение источников...")
        source_ids = await agent.ingest_sources([test_file_path])
        logger.info(f"Обработано источников: {len(source_ids)}")
        
        # Шаг 2: Планирование задачи
        logger.info("Шаг 2: Планирование задачи...")
        requirements = {
            'title': 'Анализ документации',
            'description': 'Проанализировать загруженную документацию и создать структурированный отчет с выделением ключевых концепций и рекомендациями',
            'type': 'analysis',
            'priority': 'medium',
            'metadata': {
                'domain': 'documentation_analysis',
                'complexity': 'medium'
            },
            'success_criteria': [
                'Документация проанализирована',
                'Ключевые концепции выделены',
                'Структурированный отчет создан',
                'Рекомендации сформулированы'
            ]
        }
        
        context = {
            'existing_components': ['document_processor'],
            'required_skills': ['text_analysis', 'documentation'],
            'analysis_depth': 'detailed'
        }
        
        task_plan = await agent.plan_agent(requirements, context)
        logger.info(f"Создан план задачи: {task_plan.task_id}")
        logger.info(f"Сложность: {task_plan.complexity_score:.2f}")
        logger.info(f"Оценка времени: {task_plan.estimated_duration} минут")
        logger.info(f"Требуемые навыки: {[skill.value for skill in task_plan.required_skills]}")
        
        # Шаг 3: Генерация плейбука
        logger.info("Шаг 3: Генерация плейбука...")
        agent_context = {
            'agent_id': 'analysis_agent_001',
            'capabilities': ['text_analysis', 'documentation'],
            'current_workload': 0.2
        }
        
        playbook_steps = await agent.generate_playbook(task_plan, agent_context)
        logger.info(f"Создан плейбук с {len(playbook_steps)} шагами")
        
        # Показываем первые шаги
        for i, step in enumerate(playbook_steps[:3], 1):
            logger.info(f"Шаг {i}: {step.title}")
            logger.info(f"  Тип: {step.action_type}")
            logger.info(f"  Таймаут: {step.timeout_minutes} минут")
        
        # Шаг 4: Получение статуса
        logger.info("Шаг 4: Проверка статуса...")
        system_status = await agent.get_status()
        logger.info(f"Активных агентов: {system_status['active_agents']}")
        logger.info(f"Задач в очереди: {system_status['queued_tasks']}")
        
        # Статус конкретной задачи
        task_status = await agent.get_status(task_plan.task_id)
        logger.info(f"Статус задачи: {task_status['status'].value}")
        
        return {
            'source_ids': source_ids,
            'task_plan': task_plan,
            'playbook_steps_count': len(playbook_steps),
            'system_status': system_status
        }
        
    finally:
        # Удаляем тестовый файл
        Path(test_file_path).unlink(missing_ok=True)
        logger.info("Тестовый файл удален")


async def demonstrate_advanced_features():
    """Демонстрация расширенных возможностей."""
    logger.info("=== Демонстрация расширенных возможностей ===")
    
    # Создаем компоненты с расширенной конфигурацией
    memory_manager, ingest_pipeline, context_handler, blueprint_tracker = await create_mock_components()
    
    # Создаем расширенную конфигурацию
    config = MetaAgentConfig(
        max_concurrent_tasks=10,
        quality_threshold=0.9,
        complexity_weight=0.4,
        priority_weight=0.3,
        dependency_weight=0.3,
        enable_auto_scaling=True,
        enable_learning=True
    )
    
    agent = RebeccaMetaAgent(
        memory_manager=memory_manager,
        ingest_pipeline=ingest_pipeline,
        context_handler=context_handler,
        blueprint_tracker=blueprint_tracker,
        config=config
    )
    
    # Создаем множественные задачи для демонстрации
    tasks = []
    
    # Задача 1: Разработка API
    api_requirements = {
        'title': 'Разработка REST API',
        'description': 'Создать RESTful API для управления пользователями с аутентификацией, авторизацией и CRUD операциями',
        'type': 'development',
        'priority': 'high',
        'success_criteria': [
            'API реализован и протестирован',
            'Аутентификация работает',
            'Документация создана'
        ]
    }
    
    api_context = {
        'existing_components': ['user_database', 'auth_service'],
        'required_skills': ['python', 'fastapi', 'postgresql'],
        'integration_points': ['notification_service'],
        'security_requirements': ['JWT', 'rate limiting']
    }
    
    api_task = await agent.plan_agent(api_requirements, api_context)
    tasks.append(api_task)
    
    # Задача 2: Машинное обучение
    ml_requirements = {
        'title': 'Обучение модели классификации',
        'description': 'Создать и обучить модель машинного обучения для классификации текстовых документов',
        'type': 'data_science',
        'priority': 'medium',
        'success_criteria': [
            'Модель обучена',
            'Точность > 85%',
            'Модель развернута'
        ]
    }
    
    ml_context = {
        'existing_components': ['data_pipeline'],
        'required_skills': ['python', 'scikit-learn', 'numpy'],
        'dataset_info': 'text_classification_dataset.csv',
        'performance_requirements': {'accuracy': '>85%'}
    }
    
    ml_task = await agent.plan_agent(ml_requirements, ml_context)
    tasks.append(ml_task)
    
    logger.info(f"Создано {len(tasks)} планов задач")
    
    # Генерируем плейбуки для всех задач
    playbooks = {}
    for i, task in enumerate(tasks, 1):
        logger.info(f"Генерация плейбука для задачи {i}: {task.title}")
        agent_context = {
            'agent_id': f'agent_{i:03d}',
            'capabilities': [skill.value for skill in task.required_skills],
            'current_workload': 0.1 * i
        }
        
        playbook = await agent.generate_playbook(task, agent_context)
        playbooks[task.task_id] = playbook
        logger.info(f"Плейбук для {task.title}: {len(playbook)} шагов")
    
    # Получаем детальный статус системы
    system_status = await agent.get_status()
    logger.info(f"Общий статус системы:")
    logger.info(f"  Задач создано: {len(agent.task_plans)}")
    logger.info(f"  Назначений агентов: {len(agent.agent_assignments)}")
    logger.info(f"  Плейбуков: {len(agent.playbooks)}")
    
    # Анализируем планы задач
    logger.info("Анализ планов задач:")
    for task in tasks:
        logger.info(f"  {task.title}:")
        logger.info(f"    Сложность: {task.complexity_score:.2f}")
        logger.info(f"    Время: {task.estimated_duration} мин")
        logger.info(f"    Навыки: {[s.value for s in task.required_skills]}")
    
    return {
        'tasks_count': len(tasks),
        'playbooks_count': len(playbooks),
        'system_status': system_status
    }


async def run_comprehensive_demo():
    """Запуск комплексной демонстрации."""
    logger.info("🚀 Запуск комплексной демонстрации мета-агента Rebecca")
    logger.info("=" * 60)
    
    try:
        # Базовая демонстрация
        basic_results = await demonstrate_basic_usage()
        
        if basic_results:
            logger.info("✅ Базовая демонстрация завершена успешно")
        else:
            logger.error("❌ Ошибка в базовой демонстрации")
            return
        
        # Пауза между демонстрациями
        await asyncio.sleep(1)
        
        # Расширенная демонстрация
        advanced_results = await demonstrate_advanced_features()
        
        if advanced_results:
            logger.info("✅ Расширенная демонстрация завершена успешно")
        else:
            logger.error("❌ Ошибка в расширенной демонстрации")
            return
        
        # Итоговая сводка
        logger.info("=" * 60)
        logger.info("📊 ИТОГОВАЯ СВОДКА ДЕМОНСТРАЦИИ")
        logger.info("=" * 60)
        
        logger.info("Базовая демонстрация:")
        if basic_results:
            logger.info(f"  ✓ Источников обработано: {len(basic_results.get('source_ids', []))}")
            logger.info(f"  ✓ План задачи создан: {basic_results.get('task_plan', {}).task_id}")
            logger.info(f"  ✓ Шагов в плейбуке: {basic_results.get('playbook_steps_count', 0)}")
        
        logger.info("Расширенная демонстрация:")
        if advanced_results:
            logger.info(f"  ✓ Задач создано: {advanced_results.get('tasks_count', 0)}")
            logger.info(f"  ✓ Плейбуков сгенерировано: {advanced_results.get('playbooks_count', 0)}")
        
        logger.info("🎉 Демонстрация мета-агента Rebecca завершена успешно!")
        
        return {
            'basic_demo': basic_results,
            'advanced_demo': advanced_results,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка в демонстрации: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def print_demo_header():
    """Выводит заголовок демонстрации."""
    print("\n" + "=" * 60)
    print("🎯 ДЕМОНСТРАЦИЯ МЕТА-АГЕНТА REBECCA")
    print("=" * 60)
    print("Мета-агент для интеллектуального управления")
    print("агентной экосистемой Rebecca Platform")
    print("=" * 60)


def print_demo_footer():
    """Выводит футер демонстрации."""
    print("=" * 60)
    print("📚 Для получения подробной информации:")
    print("  • Документация: src/rebecca/README.md")
    print("  • Примеры: examples/")
    print("  • API: src/rebecca/")
    print("=" * 60)


async def main():
    """Основная функция демонстрации."""
    print_demo_header()
    
    try:
        # Запуск комплексной демонстрации
        results = await run_comprehensive_demo()
        
        if results.get('success'):
            print_demo_footer()
            return 0
        else:
            logger.error("Демонстрация завершена с ошибками")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Демонстрация прервана пользователем")
        return 1
    except Exception as e:
        logger.error(f"Непредвиденная ошибка: {e}")
        return 1


if __name__ == "__main__":
    # Запуск демонстрации
    exit_code = asyncio.run(main())
    sys.exit(exit_code)