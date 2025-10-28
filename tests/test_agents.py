"""
Комплексные тесты для системы агентов Rebecca-Platform
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone

import sys
import os

# Добавляем src в путь для импортов
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.multi_agent.base_agent import (
    BaseAgent, AgentType, TaskRequest, TaskResult, TaskStatus,
    AgentCapabilities, AgentStatus, ProgressUpdate, AgentError,
    create_task_request, validate_agent_config
)
from src.multi_agent.agent_factory import AgentFactory


class TestTaskRequest:
    """Тесты для TaskRequest"""
    
    def test_create_task_request(self):
        """Тест создания TaskRequest"""
        task = create_task_request(
            agent_type=AgentType.BACKEND,
            task_type="api_development",
            description="Создать API эндпоинт"
        )
        
        assert task.agent_type == AgentType.BACKEND
        assert task.task_type == "api_development"
        assert task.description == "Создать API эндпоинт"
        assert task.priority == 1
        assert len(task.task_id) > 0
        assert task.created_at is not None
    
    def test_task_request_validation(self):
        """Тест валидации TaskRequest"""
        # Валидная задача
        task = create_task_request(
            agent_type=AgentType.FRONTEND,
            task_type="ui_development",
            description="Создать UI компонент",
            timeout=300
        )
        assert task.timeout == 300
        
        # Негативный тест - отрицательный таймаут
        with pytest.raises(ValueError):
            create_task_request(
                agent_type=AgentType.BACKEND,
                task_type="api_development",
                description="Тест",
                timeout=-1
            )


class TestAgentCapabilities:
    """Тесты для AgentCapabilities"""
    
    def test_create_capabilities(self):
        """Тест создания возможностей агента"""
        capabilities = AgentCapabilities(
            agent_type=AgentType.BACKEND,
            name="Backend Agent",
            description="Backend разработка",
            supported_tasks=["api_development", "database_design"]
        )
        
        assert capabilities.agent_type == AgentType.BACKEND
        assert capabilities.name == "Backend Agent"
        assert "api_development" in capabilities.supported_tasks


class TestBaseAgent:
    """Тесты для базового класса BaseAgent"""
    
    @pytest.fixture
    def mock_memory_manager(self):
        return Mock()
    
    @pytest.fixture
    def mock_context_handler(self):
        return Mock()
    
    @pytest.fixture
    def test_capabilities(self):
        return AgentCapabilities(
            agent_type=AgentType.BACKEND,
            name="Test Backend Agent",
            description="Тестовый агент",
            supported_tasks=["test_task"]
        )
    
    @pytest.fixture
    def test_agent(self, test_capabilities, mock_memory_manager, mock_context_handler):
        """Создание тестового агента"""
        
        class TestAgent(BaseAgent):
            async def execute_task(self, task):
                await asyncio.sleep(0.1)  # Короткая задержка
                return TaskResult(
                    task_id=task.task_id,
                    agent_type=self.agent_type,
                    status=TaskStatus.COMPLETED,
                    started_at=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc),
                    duration=0.1,
                    output={"completed": True}
                )
        
        return TestAgent(
            agent_type=AgentType.BACKEND,
            capabilities=test_capabilities,
            memory_manager=mock_memory_manager,
            context_handler=mock_context_handler
        )
    
    def test_agent_initialization(self, test_agent):
        """Тест инициализации агента"""
        assert test_agent.agent_type == AgentType.BACKEND
        assert test_agent.status.current_status == "idle"
        assert test_agent.status.is_available
        assert len(test_agent.active_tasks) == 0
        assert test_agent.status.completed_tasks == 0
    
    def test_validate_task_success(self, test_agent):
        """Тест успешной валидации задачи"""
        task = create_task_request(
            agent_type=AgentType.BACKEND,
            task_type="test_task",
            description="Тестовая задача"
        )
        
        assert test_agent.validate_task(task) is True
    
    def test_validate_task_wrong_agent_type(self, test_agent):
        """Тест валидации задачи с неправильным типом агента"""
        task = create_task_request(
            agent_type=AgentType.FRONTEND,  # Неправильный тип
            task_type="test_task",
            description="Тестовая задача"
        )
        
        with pytest.raises(Exception):  # TaskValidationError
            test_agent.validate_task(task)
    
    def test_validate_task_unsupported_task_type(self, test_agent):
        """Тест валидации задачи с неподдерживаемым типом"""
        task = create_task_request(
            agent_type=AgentType.BACKEND,
            task_type="unsupported_task",  # Неподдерживаемый тип
            description="Тестовая задача"
        )
        
        with pytest.raises(Exception):  # TaskValidationError
            test_agent.validate_task(task)
    
    @pytest.mark.asyncio
    async def test_execute_task_success(self, test_agent):
        """Тест успешного выполнения задачи"""
        task = create_task_request(
            agent_type=AgentType.BACKEND,
            task_type="test_task",
            description="Тестовая задача"
        )
        
        start_time = time.time()
        result = await test_agent.execute(task)
        end_time = time.time()
        
        assert result.status == TaskStatus.COMPLETED
        assert result.task_id == task.task_id
        assert result.agent_type == AgentType.BACKEND
        assert result.completed_at is not None
        assert result.duration is not None
        assert end_time - start_time >= 0.1  # Минимум время выполнения
        
        # Проверка обновления статуса
        assert test_agent.status.current_status == "idle"
        assert test_agent.status.completed_tasks == 1
    
    @pytest.mark.asyncio
    async def test_execute_task_validation_failure(self, test_agent):
        """Тест выполнения задачи с ошибкой валидации"""
        task = create_task_request(
            agent_type=AgentType.FRONTEND,  # Неправильный тип
            task_type="test_task",
            description="Тестовая задача"
        )
        
        result = await test_agent.execute(task)
        
        assert result.status == TaskStatus.FAILED
        assert len(result.errors) > 0
        assert "не соответствует задаче" in result.errors[0].lower()
    
    def test_get_status(self, test_agent):
        """Тест получения статуса агента"""
        status = test_agent.get_status()
        
        assert isinstance(status, AgentStatus)
        assert status.agent_type == AgentType.BACKEND
        assert status.current_status in ["idle", "busy"]
    
    def test_get_capabilities(self, test_agent):
        """Тест получения возможностей агента"""
        capabilities = test_agent.get_capabilities()
        
        assert isinstance(capabilities, AgentCapabilities)
        assert capabilities.agent_type == AgentType.BACKEND
        assert "test_task" in capabilities.supported_tasks
    
    @pytest.mark.asyncio
    async def test_report_progress(self, test_agent):
        """Тест отчета о прогрессе"""
        progress = ProgressUpdate(
            task_id="test_task_123",
            agent_type=AgentType.BACKEND,
            progress=0.5,
            current_step="Обработка данных",
            message="Половина выполнена"
        )
        
        # Метод не должен выбрасывать исключений
        test_agent.report_progress(progress)
    
    def test_cancel_task_success(self, test_agent):
        """Тест успешной отмены задачи"""
        # Создаем задачу вручную
        task = create_task_request(
            agent_type=AgentType.BACKEND,
            task_type="test_task",
            description="Тестовая задача"
        )
        
        test_agent.active_tasks[task.task_id] = task
        
        # Отменяем задачу
        result = test_agent.cancel_task(task.task_id)
        
        assert result is True
        assert task.task_id not in test_agent.active_tasks
        
        # Проверяем, что задача добавлена в историю
        task_history_ids = [r.task_id for r in test_agent.task_history]
        assert task.task_id in task_history_ids
    
    def test_cancel_task_not_found(self, test_agent):
        """Тест отмены несуществующей задачи"""
        result = test_agent.cancel_task("non_existent_task")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_health_check(self, test_agent):
        """Тест проверки здоровья агента"""
        health = await test_agent.health_check()
        
        assert "agent_type" in health
        assert "status" in health
        assert "checks" in health
        assert health["agent_type"] == "backend"
        assert health["checks"]["memory_manager"] is False  # Mock


class TestAgentFactory:
    """Тесты для AgentFactory"""
    
    @pytest.fixture
    def mock_factory(self):
        with patch('src.multi_agent.agent_factory.yaml.safe_load') as mock_yaml:
            mock_yaml.return_value = {
                'agents': {
                    'backend_agent': {
                        'config_path': 'config/agents/backend_agent.yaml'
                    }
                }
            }
            yield AgentFactory()
    
    def test_factory_initialization(self, mock_factory):
        """Тест инициализации фабрики"""
        assert mock_factory.memory_manager is None
        assert mock_factory.context_handler is None
        assert len(mock_factory.agent_classes) == 5
    
    def test_load_global_config_file_not_found(self):
        """Тест загрузки конфигурации при отсутствии файла"""
        factory = AgentFactory(config_path="non_existent_file.yaml")
        assert factory.global_config == {}
    
    def test_calculate_agent_rating(self, mock_factory):
        """Тест расчета рейтинга агента"""
        # Создаем mock агент
        mock_agent = Mock()
        mock_agent.get_capabilities.return_value = AgentCapabilities(
            agent_type=AgentType.BACKEND,
            name="Backend Agent",
            description="Test",
            supported_tasks=["api_development"],
            max_concurrent_tasks=2
        )
        mock_agent.get_status.return_value = AgentStatus(
            agent_type=AgentType.BACKEND,
            completed_tasks=8,
            failed_tasks=2  # 80% успешности
        )
        mock_agent.active_tasks = {}
        
        task = create_task_request(
            agent_type=AgentType.BACKEND,
            task_type="api_development",
            description="Test task",
            priority=1
        )
        
        rating = mock_factory._calculate_agent_rating(mock_agent, task)
        assert 0 <= rating <= 1
        assert rating > 0  # Должен быть положительный рейтинг
    
    def test_assign_task_no_suitable_agents(self):
        """Тест назначения задачи без подходящих агентов"""
        factory = AgentFactory()
        
        task = create_task_request(
            agent_type=AgentType.BACKEND,
            task_type="api_development",
            description="Test task"
        )
        
        with pytest.raises(ValueError):
            factory.assign_task_to_agent(task)
    
    def test_get_factory_stats_empty_factory(self, mock_factory):
        """Тест получения статистики пустой фабрики"""
        stats = mock_factory.get_factory_stats()
        
        assert stats["total_agents"] == 0
        assert stats["total_active_tasks"] == 0
        assert stats["total_completed_tasks"] == 0
        assert stats["total_failed_tasks"] == 0


class TestUtilityFunctions:
    """Тесты для утилитарных функций"""
    
    def test_validate_agent_config_valid(self):
        """Тест валидации правильной конфигурации"""
        config = {
            "agent_type": "backend",
            "name": "Backend Agent",
            "description": "Test agent",
            "supported_tasks": ["api_development"]
        }
        
        assert validate_agent_config(config) is True
    
    def test_validate_agent_config_invalid(self):
        """Тест валидации неправильной конфигурации"""
        # Отсутствует обязательное поле
        config = {
            "agent_type": "backend",
            "name": "Backend Agent"
            # Отсутствует "description" и "supported_tasks"
        }
        
        assert validate_agent_config(config) is False
    
    def test_validate_agent_config_invalid_agent_type(self):
        """Тест валидации с неправильным типом агента"""
        config = {
            "agent_type": "invalid_type",
            "name": "Backend Agent",
            "description": "Test agent",
            "supported_tasks": ["api_development"]
        }
        
        assert validate_agent_config(config) is False


# =============================================================================
# Интеграционные тесты
# =============================================================================

class TestAgentIntegration:
    """Интеграционные тесты системы агентов"""
    
    @pytest.mark.asyncio
    async def test_agent_communication_with_memory_manager(self):
        """Тест взаимодействия агента с MemoryManager"""
        mock_memory_manager = Mock()
        mock_memory_manager.store = AsyncMock()
        
        class TestAgent(BaseAgent):
            async def execute_task(self, task):
                return TaskResult(
                    task_id=task.task_id,
                    agent_type=self.agent_type,
                    status=TaskStatus.COMPLETED,
                    started_at=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc),
                    duration=0.1,
                    output={"test": True}
                )
        
        agent = TestAgent(
            agent_type=AgentType.BACKEND,
            capabilities=AgentCapabilities(
                agent_type=AgentType.BACKEND,
                name="Test Agent",
                description="Test",
                supported_tasks=["test"]
            ),
            memory_manager=mock_memory_manager
        )
        
        task = create_task_request(
            agent_type=AgentType.BACKEND,
            task_type="test",
            description="Integration test"
        )
        
        result = await agent.execute(task)
        
        # Проверяем, что методы MemoryManager были вызваны
        assert mock_memory_manager.store.call_count >= 2  # task_started, task_completed
        
        # Проверяем аргументы последнего вызова
        last_call = mock_memory_manager.store.call_args
        assert last_call[1]['layer'] in ["SEMANTIC", "CORE"]
    
    @pytest.mark.asyncio
    async def test_agent_factory_lifecycle(self):
        """Тест жизненного цикла фабрики агентов"""
        factory = AgentFactory()
        
        # Создание агента
        backend_agent = factory.create_agent(AgentType.BACKEND, "test-backend-1")
        assert backend_agent is not None
        assert AgentType.BACKEND in factory.agents
        
        # Получение агента
        retrieved_agent = factory.get_agent(AgentType.BACKEND)
        assert retrieved_agent is backend_agent
        
        # Удаление агента
        success = factory.remove_agent(AgentType.BACKEND)
        assert success is True
        assert AgentType.BACKEND not in factory.agents
    
    @pytest.mark.asyncio
    async def test_multiple_agents_concurrent_execution(self):
        """Тест параллельного выполнения задач несколькими агентами"""
        factory = AgentFactory()
        
        # Создаем несколько агентов
        agents = []
        for agent_type in [AgentType.BACKEND, AgentType.FRONTEND]:
            agent = factory.create_agent(agent_type)
            agents.append(agent)
        
        # Создаем задачи для каждого агента
        tasks = []
        for i, agent_type in enumerate([AgentType.BACKEND, AgentType.FRONTEND]):
            task = create_task_request(
                agent_type=agent_type,
                task_type="test_task",
                description=f"Task {i+1}"
            )
            tasks.append(task)
        
        # Выполняем задачи параллельно
        results = await asyncio.gather(*[agent.execute(task) for agent, task in zip(agents, tasks)])
        
        # Проверяем результаты
        assert len(results) == 2
        for result in results:
            assert result.status == TaskStatus.COMPLETED
            assert result.agent_type in [AgentType.BACKEND, AgentType.FRONTEND]


if __name__ == "__main__":
    # Запуск тестов при прямом вызове файла
    pytest.main([__file__, "-v"])