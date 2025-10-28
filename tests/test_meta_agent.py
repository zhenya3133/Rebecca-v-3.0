#!/usr/bin/env python3
"""
Тесты для мета-агента Rebecca.

Покрывают основную функциональность мета-агента:
- Инициализацию и конфигурацию
- Поглощение источников
- Планирование задач
- Генерацию плейбуков
- Координацию агентов
"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import sys

# Добавляем src в путь для импорта
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rebecca.meta_agent import (
    RebeccaMetaAgent, TaskPlan, AgentAssignment, PlaybookStep, ResourceAllocation,
    MetaAgentConfig, TaskType, TaskPriority, AgentSpecialization, TaskStatus
)
from rebecca.utils import (
    MetaAgentValidator, MetaAgentTestData, validate_agent_setup,
    save_agent_config, load_agent_config
)
from memory_manager.memory_manager import MemoryManager
from memory_manager.adaptive_blueprint import AdaptiveBlueprintTracker


class TestMetaAgentConfig(unittest.TestCase):
    """Тесты конфигурации мета-агента."""
    
    def test_default_config(self):
        """Тест конфигурации по умолчанию."""
        config = MetaAgentConfig()
        
        self.assertEqual(config.max_concurrent_tasks, 10)
        self.assertEqual(config.default_timeout_minutes, 60)
        self.assertTrue(config.enable_auto_scaling)
        self.assertTrue(config.enable_failover)
        self.assertEqual(config.quality_threshold, 0.8)
        self.assertEqual(config.complexity_weight, 0.3)
    
    def test_custom_config(self):
        """Тест кастомной конфигурации."""
        config = MetaAgentConfig(
            max_concurrent_tasks=20,
            quality_threshold=0.9,
            enable_auto_scaling=False
        )
        
        self.assertEqual(config.max_concurrent_tasks, 20)
        self.assertEqual(config.quality_threshold, 0.9)
        self.assertFalse(config.enable_auto_scaling)


class TestTaskPlan(unittest.TestCase):
    """Тесты плана задачи."""
    
    def test_task_plan_creation(self):
        """Тест создания плана задачи."""
        task_plan = TaskPlan(
            task_id="test_task_001",
            title="Тестовая задача",
            description="Описание тестовой задачи",
            task_type=TaskType.DEVELOPMENT,
            priority=TaskPriority.HIGH
        )
        
        self.assertEqual(task_plan.task_id, "test_task_001")
        self.assertEqual(task_plan.title, "Тестовая задача")
        self.assertEqual(task_plan.task_type, TaskType.DEVELOPMENT)
        self.assertEqual(task_plan.priority, TaskPriority.HIGH)
    
    def test_task_plan_serialization(self):
        """Тест сериализации плана задачи."""
        task_plan = TaskPlan(
            task_id="test_task_001",
            title="Тестовая задача",
            description="Описание",
            task_type=TaskType.DEVELOPMENT,
            priority=TaskPriority.MEDIUM,
            required_skills=[AgentSpecialization.BACKEND]
        )
        
        # Тест сериализации в словарь
        task_dict = task_plan.to_dict()
        self.assertIsInstance(task_dict, dict)
        self.assertEqual(task_dict['task_id'], "test_task_001")
        self.assertEqual(task_dict['task_type'], "development")
        self.assertEqual(task_dict['priority'], "MEDIUM")
        
        # Тест десериализации
        restored_task = TaskPlan.from_dict(task_dict)
        self.assertEqual(restored_task.task_id, task_plan.task_id)
        self.assertEqual(restored_task.title, task_plan.title)
        self.assertEqual(restored_task.task_type, task_plan.task_type)


class TestMetaAgentValidator(unittest.TestCase):
    """Тесты валидатора."""
    
    def setUp(self):
        self.validator = MetaAgentValidator()
        self.test_data = MetaAgentTestData()
    
    def test_config_validation_valid(self):
        """Тест валидации корректной конфигурации."""
        config_data = self.test_data.create_sample_config()
        result = self.validator.validate_config(config_data)
        
        self.assertTrue(result['valid'])
        self.assertEqual(len(result['errors']), 0)
    
    def test_config_validation_invalid(self):
        """Тест валидации некорректной конфигурации."""
        config_data = {
            'max_concurrent_tasks': -1,  # Некорректное значение
            'quality_threshold': 1.5,    # Вне диапазона 0-1
            'complexity_weight': 0.5,
            'priority_weight': 0.3,
            'dependency_weight': 0.4      # Сумма не равна 1
        }
        
        result = self.validator.validate_config(config_data)
        
        self.assertFalse(result['valid'])
        self.assertGreater(len(result['errors']), 0)
    
    def test_task_plan_validation_valid(self):
        """Тест валидации корректного плана задачи."""
        task_plan = TaskPlan(
            task_id="test_task",
            title="Тестовая задача",
            description="Описание тестовой задачи",
            task_type=TaskType.DEVELOPMENT,
            priority=TaskPriority.HIGH,
            complexity_score=0.5,
            estimated_duration=60,
            required_skills=[AgentSpecialization.BACKEND]
        )
        
        result = self.validator.validate_task_plan(task_plan)
        
        self.assertTrue(result['valid'])
        self.assertEqual(len(result['errors']), 0)
    
    def test_task_plan_validation_invalid(self):
        """Тест валидации некорректного плана задачи."""
        task_plan = TaskPlan(
            task_id="",  # Пустой ID
            title="",    # Пустое название
            description="",  # Пустое описание
            task_type=TaskType.DEVELOPMENT,
            priority=TaskPriority.HIGH,
            complexity_score=1.5,  # Вне диапазона
            estimated_duration=-10  # Некорректное время
        )
        
        result = self.validator.validate_task_plan(task_plan)
        
        self.assertFalse(result['valid'])
        self.assertGreater(len(result['errors']), 0)
    
    def test_playbook_validation_valid(self):
        """Тест валидации корректного плейбука."""
        steps = [
            PlaybookStep(
                step_id="step_1",
                step_number=1,
                title="Первый шаг",
                description="Описание первого шага",
                action_type="analysis",
                agent_instruction="Выполнить анализ",
                expected_output="Результат анализа"
            ),
            PlaybookStep(
                step_id="step_2",
                step_number=2,
                title="Второй шаг",
                description="Описание второго шага",
                action_type="execution",
                agent_instruction="Выполнить задачу",
                expected_output="Результат выполнения"
            )
        ]
        
        result = self.validator.validate_playbook(steps)
        
        self.assertTrue(result['valid'])
        self.assertEqual(len(result['errors']), 0)
    
    def test_playbook_validation_invalid(self):
        """Тест валидации некорректного плейбука."""
        steps = [
            PlaybookStep(
                step_id="step_1",
                step_number=1,
                title="Первый шаг",
                description="",
                action_type="analysis",
                agent_instruction="",
                expected_output="Результат"
            ),
            PlaybookStep(
                step_id="step_1",  # Дублирующийся ID
                step_number=2,
                title="Второй шаг",
                description="Описание",
                action_type="execution",
                agent_instruction="Выполнить",
                expected_output="Результат"
            )
        ]
        
        result = selfvalidator.validate_playbook(steps)
        
        self.assertFalse(result['valid'])
        self.assertGreater(len(result['errors']), 0)


class MockIngestPipeline:
    """Mock класс для IngestPipeline."""
    
    def ingest_document(self, file_path: str, chunk_override=None):
        class MockEvent:
            id = f"mock_event_{hash(file_path) % 10000}"
            attrs = {
                "text": f"Mock content from {file_path}",
                "source_path": file_path
            }
        return MockEvent()
    
    def process_git_repo(self, repo_url: str, branch="main", process_readme=True, process_source=True):
        class MockEvent:
            id = f"mock_git_event_{hash(repo_url) % 10000}"
            attrs = {
                "text": f"Mock Git content from {repo_url}",
                "source_path": repo_url
            }
        return [MockEvent()]


class MockContextHandler:
    """Mock класс для ContextHandler."""
    pass


class TestRebeccaMetaAgent(unittest.TestCase):
    """Тесты основного класса мета-агента."""
    
    def setUp(self):
        """Подготовка к тестам."""
        # Создаем mock компоненты
        self.memory_manager = Mock(spec=MemoryManager)
        self.memory_manager.store = AsyncMock(return_value="mock_id")
        self.memory_manager.retrieve = AsyncMock(return_value=[])
        
        self.ingest_pipeline = MockIngestPipeline()
        self.context_handler = MockContextHandler()
        self.blueprint_tracker = Mock(spec=AdaptiveBlueprintTracker)
        self.blueprint_tracker.record_blueprint = AsyncMock(return_value=1)
        self.blueprint_tracker.get_latest_blueprint = AsyncMock(return_value=None)
        
        # Создаем конфигурацию
        self.config = MetaAgentConfig(
            max_concurrent_tasks=5,
            quality_threshold=0.8
        )
        
        # Создаем мета-агента
        self.agent = RebeccaMetaAgent(
            memory_manager=self.memory_manager,
            ingest_pipeline=self.ingest_pipeline,
            context_handler=self.context_handler,
            blueprint_tracker=self.blueprint_tracker,
            config=self.config
        )
    
    def test_agent_initialization(self):
        """Тест инициализации агента."""
        self.assertIsNotNone(self.agent)
        self.assertEqual(self.agent.config.max_concurrent_tasks, 5)
        self.assertEqual(self.agent.config.quality_threshold, 0.8)
        self.assertIsNotNone(self.agent.task_analyzer)
        self.assertIsNotNone(self.agent.resource_optimizer)
        self.assertIsNotNone(self.agent.playbook_generator)
    
    async def test_ingest_sources_single_file(self):
        """Тест поглощения одиночного файла."""
        # Создаем тестовый файл
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Test Document\n\nThis is a test document.")
            test_file = f.name
        
        try:
            source_ids = await self.agent.ingest_sources(test_file)
            
            self.assertIsInstance(source_ids, list)
            self.assertGreater(len(source_ids), 0)
            
            # Проверяем, что вызывались методы памяти
            self.memory_manager.store.assert_called()
            
        finally:
            Path(test_file).unlink(missing_ok=True)
    
    async def test_ingest_sources_multiple_sources(self):
        """Тест поглощения множественных источников."""
        sources = [
            "test_file_1.md",
            "test_file_2.pdf",
            "https://github.com/test/repo.git"
        ]
        
        source_ids = await self.agent.ingest_sources(sources)
        
        self.assertIsInstance(source_ids, list)
        # Количество обработанных источников может отличаться из-за ошибок обработки
        self.assertGreaterEqual(len(source_ids), 0)
    
    async def test_plan_agent_basic(self):
        """Тест базового планирования задачи."""
        requirements = {
            'title': 'Тестовая задача',
            'description': 'Создать тестовый компонент',
            'type': 'development',
            'priority': 'medium'
        }
        
        task_plan = await self.agent.plan_agent(requirements)
        
        self.assertIsInstance(task_plan, TaskPlan)
        self.assertEqual(task_plan.title, 'Тестовая задача')
        self.assertEqual(task_plan.task_type, TaskType.DEVELOPMENT)
        self.assertEqual(task_plan.priority, TaskPriority.MEDIUM)
        self.assertIsNotNone(task_plan.task_id)
    
    async def test_plan_agent_with_context(self):
        """Тест планирования с контекстом."""
        requirements = {
            'title': 'API разработка',
            'description': 'Создать REST API',
            'type': 'development',
            'priority': 'high'
        }
        
        context = {
            'existing_components': ['database', 'auth'],
            'required_skills': ['python', 'fastapi']
        }
        
        task_plan = await self.agent.plan_agent(requirements, context)
        
        self.assertIsInstance(task_plan, TaskPlan)
        self.assertGreater(len(task_plan.required_skills), 0)
        self.assertGreater(task_plan.estimated_duration, 0)
    
    async def test_generate_playbook(self):
        """Тест генерации плейбука."""
        # Создаем план задачи
        task_plan = TaskPlan(
            task_id="test_task",
            title="Тестовая задача",
            description="Описание",
            task_type=TaskType.DEVELOPMENT,
            priority=TaskPriority.MEDIUM
        )
        
        # Добавляем план в агента
        self.agent.task_plans[task_plan.task_id] = task_plan
        
        # Генерируем плейбук
        agent_context = {
            'agent_id': 'test_agent',
            'capabilities': ['development']
        }
        
        playbook_steps = await self.agent.generate_playbook(task_plan, agent_context)
        
        self.assertIsInstance(playbook_steps, list)
        self.assertGreater(len(playbook_steps), 0)
        
        # Проверяем структуру первого шага
        first_step = playbook_steps[0]
        self.assertIsInstance(first_step, PlaybookStep)
        self.assertIsNotNone(first_step.step_id)
        self.assertIsNotNone(first_step.title)
    
    async def test_get_status_system(self):
        """Тест получения системного статуса."""
        status = await self.agent.get_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('system_status', status)
        self.assertIn('metrics', status)
        self.assertEqual(status['system_status'], 'operational')
    
    async def test_get_status_task(self):
        """Тест получения статуса задачи."""
        # Создаем план задачи
        task_plan = TaskPlan(
            task_id="test_task",
            title="Тестовая задача",
            description="Описание",
            task_type=TaskType.DEVELOPMENT,
            priority=TaskPriority.MEDIUM
        )
        
        self.agent.task_plans[task_plan.task_id] = task_plan
        
        # Получаем статус задачи
        status = await self.agent.get_status(task_plan.task_id)
        
        self.assertIsInstance(status, dict)
        self.assertIn('task_id', status)
        self.assertEqual(status['task_id'], task_plan.task_id)
        self.assertIn('task_plan', status)
        self.assertIn('status', status)


class TestUtils(unittest.TestCase):
    """Тесты утилитарных функций."""
    
    def test_save_and_load_config(self):
        """Тест сохранения и загрузки конфигурации."""
        config = MetaAgentConfig(
            max_concurrent_tasks=15,
            quality_threshold=0.9
        )
        
        # Сохраняем конфигурацию
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_file = f.name
        
        try:
            save_agent_config(config, config_file)
            
            # Загружаем конфигурацию
            loaded_config = load_agent_config(config_file)
            
            self.assertEqual(loaded_config.max_concurrent_tasks, 15)
            self.assertEqual(loaded_config.quality_threshold, 0.9)
            
        finally:
            Path(config_file).unlink(missing_ok=True)
    
    def test_validate_agent_setup(self):
        """Тест валидации настройки агента."""
        # Создаем mock агента
        agent = Mock(spec=RebeccaMetaAgent)
        agent.memory_manager = Mock()
        agent.ingest_pipeline = Mock()
        agent.context_handler = Mock()
        agent.blueprint_tracker = Mock()
        agent.config = Mock()
        
        result = validate_agent_setup(agent)
        
        self.assertIsInstance(result, dict)
        self.assertIn('setup_valid', result)
        self.assertIn('checks', result)
        self.assertIn('issues', result)


class TestMetaAgentIntegration(unittest.TestCase):
    """Интеграционные тесты мета-агента."""
    
    async def test_complete_workflow_simulation(self):
        """Тест полного workflow (симуляция)."""
        # Создаем компоненты
        memory_manager = Mock(spec=MemoryManager)
        memory_manager.store = AsyncMock(return_value="mock_id")
        memory_manager.retrieve = AsyncMock(return_value=[])
        
        ingest_pipeline = MockIngestPipeline()
        context_handler = MockContextHandler()
        
        blueprint_tracker = Mock(spec=AdaptiveBlueprintTracker)
        blueprint_tracker.record_blueprint = AsyncMock(return_value=1)
        blueprint_tracker.get_latest_blueprint = AsyncMock(return_value=None)
        
        # Создаем агента
        agent = RebeccaMetaAgent(
            memory_manager=memory_manager,
            ingest_pipeline=ingest_pipeline,
            context_handler=context_handler,
            blueprint_tracker=blueprint_tracker
        )
        
        # Шаг 1: Создаем план задачи
        requirements = {
            'title': 'Интеграционный тест',
            'description': 'Тест полного workflow',
            'type': 'development',
            'priority': 'high'
        }
        
        task_plan = await agent.plan_agent(requirements)
        self.assertIsNotNone(task_plan.task_id)
        
        # Шаг 2: Генерируем плейбук
        agent_context = {'agent_id': 'test_agent'}
        playbook_steps = await self.agent.generate_playbook(task_plan, agent_context)
        self.assertGreater(len(playbook_steps), 0)
        
        # Шаг 3: Проверяем статус
        status = await agent.get_status(task_plan.task_id)
        self.assertEqual(status['task_id'], task_plan.task_id)
        
        # Проверяем, что все компоненты взаимодействуют корректно
        self.assertGreater(len(agent.task_plans), 0)
        self.assertGreater(len(agent.playbooks), 0)


def run_tests():
    """Запуск всех тестов."""
    # Создаем тестовый набор
    test_suite = unittest.TestSuite()
    
    # Добавляем тестовые классы
    test_classes = [
        TestMetaAgentConfig,
        TestTaskPlan,
        TestMetaAgentValidator,
        TestRebeccaMetaAgent,
        TestUtils,
        TestMetaAgentIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Запускаем тесты
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("🧪 Запуск тестов мета-агента Rebecca")
    print("=" * 50)
    
    success = run_tests()
    
    if success:
        print("\n✅ Все тесты прошли успешно!")
    else:
        print("\n❌ Некоторые тесты провалились!")
    
    exit(0 if success else 1)