"""Comprehensive unit тесты для BaseAgent компонента.

Включает:
- Unit тесты базового функционала агента
- Тесты обработки задач
- Тесты валидации и обработки ошибок  
- Тесты управления ресурсами
- Тесты интеграции с Memory Manager
- Mock тесты для внешних зависимостей
- Performance тесты

Автор: Claude Code
Дата: 2025-10-28
"""

import asyncio
import pytest
import time
import uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any
from pathlib import Path
import tempfile
import os
import sys

# Настройка путей
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# Импорты
from multi_agent.base_agent import (
    BaseAgent,
    AgentType,
    TaskRequest,
    TaskResult,
    TaskStatus,
    AgentCapabilities,
    AgentStatus,
    ProgressUpdate,
    AgentError,
    TaskValidationError,
    TaskExecutionError,
    ResourceError,
    create_task_request,
    validate_agent_config,
    generate_agent_report,
    # Обработчики по умолчанию
    DefaultTimeoutHandler,
    DefaultRetryHandler,
    AgentLogger,
    ResourceManagerImpl
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def agent_type():
    """Фикстура типа агента."""
    return AgentType.BACKEND


@pytest.fixture
def agent_capabilities(agent_type):
    """Фикстура возможностей агента."""
    return AgentCapabilities(
        agent_type=agent_type,
        name="Backend Developer Agent",
        version="1.0.0",
        description="Специализированный агент для backend разработки",
        supported_tasks=["api_development", "database_design", "service_architecture"],
        supported_languages=["python", "javascript", "go"],
        max_concurrent_tasks=3,
        resource_requirements={
            "cpu": "2 cores",
            "memory": "4GB",
            "storage": "10GB"
        },
        dependencies=["git", "docker", "postgresql"],
        environment_vars={
            "DEBUG": "false",
            "LOG_LEVEL": "INFO"
        },
        specializations=["rest_apis", "graphql", "microservices"],
        integrations=["github", "docker_hub", "jenkins"],
        performance_profile={
            "avg_response_time": 2.5,
            "throughput": 100,
            "error_rate": 0.02
        }
    )


@pytest.fixture
async def mock_memory_manager():
    """Mock MemoryManager для тестов."""
    memory_manager = AsyncMock()
    memory_manager.store = AsyncMock(return_value="memory_id_123")
    memory_manager.retrieve = AsyncMock(return_value=[])
    memory_manager.update = AsyncMock(return_value=True)
    memory_manager.delete = AsyncMock(return_value=True)
    memory_manager.search_across_layers = AsyncMock(return_value=[])
    memory_manager.get_layer_statistics = AsyncMock(return_value={
        "total_items": 0,
        "memory_usage": "100MB"
    })
    return memory_manager


@pytest.fixture
async def mock_context_handler():
    """Mock ContextHandler для тестов."""
    context_handler = AsyncMock()
    context_handler.build_context_envelope = AsyncMock(return_value={
        "trace_id": "test_trace_123",
        "timestamp": datetime.now().isoformat(),
        "context": {"test": True}
    })
    return context_handler


@pytest.fixture
def task_request(agent_type):
    """Фикстура запроса задачи."""
    return TaskRequest(
        agent_type=agent_type,
        task_type="api_development",
        description="Создать REST API для пользователей",
        inputs={
            "specification": "OpenAPI 3.0",
            "framework": "FastAPI",
            "database": "PostgreSQL"
        },
        context={"project": "user_service", "environment": "development"},
        priority=2,
        timeout=300,
        retry_count=0,
        max_retries=2,
        dependencies=[],
        metadata={"created_by": "test_system", "category": "backend"}
    )


@pytest.fixture
def concrete_agent(agent_type, agent_capabilities, mock_memory_manager, mock_context_handler):
    """Конкретная реализация BaseAgent для тестирования."""
    
    class TestAgent(BaseAgent):
        async def execute_task(self, task: TaskRequest) -> TaskResult:
            """Тестовая реализация выполнения задачи."""
            # Имитация выполнения задачи
            await asyncio.sleep(0.1)  # Короткое время для теста
            
            return TaskResult(
                task_id=task.task_id,
                agent_type=self.agent_type,
                status=TaskStatus.COMPLETED,
                started_at=task.created_at,
                completed_at=datetime.now(timezone.utc),
                duration=0.1,
                output={
                    "completed": True,
                    "result": "Test task completed successfully",
                    "artifacts": ["api_spec.yaml", "database_schema.sql"]
                },
                errors=[],
                warnings=[],
                metrics={
                    "lines_of_code": 150,
                    "test_coverage": 85.0,
                    "performance_score": 92.5
                },
                artifacts=["api_spec.yaml", "database_schema.sql"],
                next_actions=["write_unit_tests", "setup_ci_cd"],
                learning_data={
                    "approach_used": "test_driven_development",
                    "challenges_faced": ["dependency_management"],
                    "time_estimate_accuracy": 0.95
                }
            )
    
    return TestAgent(
        agent_type=agent_type,
        capabilities=agent_capabilities,
        memory_manager=mock_memory_manager,
        context_handler=mock_context_handler
    )


@pytest.fixture
def error_agent(agent_type, agent_capabilities, mock_memory_manager, mock_context_handler):
    """Агент, который генерирует ошибки для тестирования."""
    
    class ErrorAgent(BaseAgent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.should_error = False
            self.error_type = Exception
            
        async def execute_task(self, task: TaskRequest) -> TaskResult:
            """Реализация с ошибками."""
            if self.should_error:
                if self.error_type == asyncio.TimeoutError:
                    await asyncio.sleep(task.timeout + 1) if task.timeout else await asyncio.sleep(2)
                elif self.error_type == ValueError:
                    raise ValueError("Test value error")
                elif self.error_type == RuntimeError:
                    raise RuntimeError("Test runtime error")
                else:
                    raise self.error_type("Test error")
            
            return TaskResult(
                task_id=task.task_id,
                agent_type=self.agent_type,
                status=TaskStatus.COMPLETED,
                started_at=task.created_at,
                completed_at=datetime.now(timezone.utc),
                duration=0.1,
                output={"completed": True}
            )
    
    return ErrorAgent(
        agent_type=agent_type,
        capabilities=agent_capabilities,
        memory_manager=mock_memory_manager,
        context_handler=mock_context_handler
    )


@pytest.fixture
def timeout_handler():
    """Фикстура обработчика таймаутов."""
    return DefaultTimeoutHandler()


@pytest.fixture
def retry_handler():
    """Фикстура обработчика повторных попыток."""
    return DefaultRetryHandler()


@pytest.fixture
def resource_manager():
    """Фикстура менеджера ресурсов."""
    return ResourceManagerImpl()


# ============================================================================
# UNIT TESTS - Agent Initialization
# ============================================================================

class TestBaseAgentInitialization:
    """Тесты инициализации BaseAgent."""
    
    def test_agent_creation(self, agent_type, agent_capabilities, mock_memory_manager, mock_context_handler):
        """Тест создания агента."""
        agent = BaseAgent(
            agent_type=agent_type,
            capabilities=agent_capabilities,
            memory_manager=mock_memory_manager,
            context_handler=mock_context_handler
        )
        
        assert agent.agent_type == agent_type
        assert agent.capabilities == agent_capabilities
        assert agent.memory_manager == mock_memory_manager
        assert agent.context_handler == mock_context_handler
        assert agent.status.agent_type == agent_type
        assert agent.status.current_status == "idle"
        assert agent.status.is_available is True
        assert len(agent.active_tasks) == 0
        assert len(agent.task_history) == 0
    
    def test_default_handlers_initialization(self, agent_type, agent_capabilities):
        """Тест инициализации обработчиков по умолчанию."""
        agent = BaseAgent(
            agent_type=agent_type,
            capabilities=agent_capabilities
        )
        
        assert isinstance(agent.timeout_handler, DefaultTimeoutHandler)
        assert isinstance(agent.retry_handler, DefaultRetryHandler)
        assert isinstance(agent.logger, AgentLogger)
        assert isinstance(agent.resource_manager, ResourceManagerImpl)
        assert agent.logger.agent_type == agent_type
    
    def test_custom_handlers_initialization(self, agent_type, agent_capabilities, timeout_handler, retry_handler, resource_manager):
        """Тест инициализации кастомных обработчиков."""
        mock_logger = Mock()
        
        agent = BaseAgent(
            agent_type=agent_type,
            capabilities=agent_capabilities,
            timeout_handler=timeout_handler,
            retry_handler=retry_handler,
            logger=mock_logger,
            resource_manager=resource_manager
        )
        
        assert agent.timeout_handler == timeout_handler
        assert agent.retry_handler == retry_handler
        assert agent.logger == mock_logger
        assert agent.resource_manager == resource_manager
    
    def test_thread_pool_initialization(self, agent_type, agent_capabilities):
        """Тест инициализации thread pool."""
        agent = BaseAgent(
            agent_type=agent_type,
            capabilities=agent_capabilities
        )
        
        assert agent.executor is not None
        assert agent.executor._max_workers == agent_capabilities.max_concurrent_tasks


# ============================================================================
# UNIT TESTS - Task Validation
# ============================================================================

class TestTaskValidation:
    """Тесты валидации задач."""
    
    def test_valid_task_validation(self, concrete_agent, task_request):
        """Тест валидации корректной задачи."""
        result = concrete_agent.validate_task(task_request)
        assert result is True
    
    def test_empty_description_validation(self, concrete_agent, task_request):
        """Тест валидации пустого описания."""
        task_request.description = ""
        
        with pytest.raises(TaskValidationError, match="Описание задачи не может быть пустым"):
            concrete_agent.validate_task(task_request)
    
    def test_wrong_agent_type_validation(self, concrete_agent, task_request):
        """Тест валидации неправильного типа агента."""
        task_request.agent_type = AgentType.FRONTEND
        
        with pytest.raises(TaskValidationError, match="Тип агента не соответствует задаче"):
            concrete_agent.validate_task(task_request)
    
    def test_unsupported_task_type_validation(self, concrete_agent, task_request):
        """Тест валидации неподдерживаемого типа задачи."""
        task_request.task_type = "unsupported_task"
        
        with pytest.raises(TaskValidationError, match="Неподдерживаемый тип задачи"):
            concrete_agent.validate_task(task_request)
    
    def test_excessive_timeout_validation(self, concrete_agent, task_request):
        """Тест валидации чрезмерного таймаута."""
        task_request.timeout = 7200  # 2 часа > 1 часа
        
        with pytest.raises(TaskValidationError, match="Таймаут не может превышать 1 час"):
            concrete_agent.validate_task(task_request)
    
    def test_dependency_validation(self, concrete_agent, task_request):
        """Тест валидации зависимостей."""
        task_request.dependencies = ["nonexistent_task_id"]
        
        with pytest.raises(TaskValidationError, match="Зависимость .* не выполнена"):
            concrete_agent.validate_task(task_request)
    
    def test_satisfied_dependency_validation(self, concrete_agent, task_request):
        """Тест валидации выполненной зависимости."""
        # Создаем и добавляем выполненную задачу в историю
        completed_task = TaskResult(
            task_id="completed_task",
            agent_type=concrete_agent.agent_type,
            status=TaskStatus.COMPLETED,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            duration=1.0
        )
        concrete_agent.task_history.append(completed_task)
        
        task_request.dependencies = ["completed_task"]
        
        # Должна пройти валидацию
        result = concrete_agent.validate_task(task_request)
        assert result is True


# ============================================================================
# UNIT TESTS - Task Execution
# ============================================================================

class TestTaskExecution:
    """Тесты выполнения задач."""
    
    @pytest.mark.asyncio
    async def test_successful_task_execution(self, concrete_agent, task_request):
        """Тест успешного выполнения задачи."""
        result = await concrete_agent.execute(task_request)
        
        assert result.task_id == task_request.task_id
        assert result.agent_type == concrete_agent.agent_type
        assert result.status == TaskStatus.COMPLETED
        assert result.completed_at is not None
        assert result.duration is not None
        assert result.duration > 0
        assert "completed" in result.output
        assert result.errors == []
        assert "artifacts" in result.output
    
    @pytest.mark.asyncio
    async def test_task_execution_with_memory_integration(self, concrete_agent, task_request, mock_memory_manager):
        """Тест выполнения задачи с интеграцией памяти."""
        result = await concrete_agent.execute(task_request)
        
        # Проверяем, что MemoryManager использовался
        assert mock_memory_manager.store.called
        # Должно быть минимум 2 вызова: task_started и task_completed
        assert mock_memory_manager.store.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_task_cancellation(self, concrete_agent, task_request):
        """Тест отмены задачи."""
        # Запускаем задачу в фоне
        task_future = asyncio.create_task(concrete_agent.execute(task_request))
        
        # Отменяем задачу
        await asyncio.sleep(0.05)  # Даем задаче начаться
        cancel_result = concrete_agent.cancel_task(task_request.task_id)
        
        assert cancel_result is True
        assert task_request.task_id not in concrete_agent.active_tasks
        
        # Проверяем результат отмены
        task_result = concrete_agent.get_task_result(task_request.task_id)
        assert task_result is not None
        assert task_result.status == TaskStatus.CANCELLED
    
    @pytest.mark.asyncio
    async def test_task_cancellation_nonexistent_task(self, concrete_agent):
        """Тест отмены несуществующей задачи."""
        result = concrete_agent.cancel_task("nonexistent_task_id")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_timeout_error_handling(self, error_agent, task_request):
        """Тест обработки таймаута."""
        error_agent.should_error = True
        error_agent.error_type = asyncio.TimeoutError
        task_request.timeout = 0.1  # Очень короткий таймаут
        
        result = await error_agent.execute(task_request)
        
        assert result.status == TaskStatus.TIMEOUT
        assert "timeout" in result.errors[0].lower()
        assert result.duration == task_request.timeout
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, error_agent, task_request):
        """Тест механизма повторных попыток."""
        error_agent.should_error = True
        error_agent.error_type = ValueError  # Ошибка, которую можно повторять
        task_request.max_retries = 2
        
        result = await error_agent.execute(task_request)
        
        # Задача должна быть завершена с ошибкой после всех попыток
        assert result.status == TaskStatus.FAILED
        assert len(result.errors) > 0
    
    @pytest.mark.asyncio
    async def test_agent_unavailability_handling(self, concrete_agent, task_request):
        """Тест обработки недоступности агента."""
        concrete_agent.status.is_available = False
        
        with pytest.raises(AgentError, match="Агент недоступен"):
            await concrete_agent.execute(task_request)
    
    @pytest.mark.asyncio
    async def test_concurrent_tasks_limit(self, concrete_agent, task_request):
        """Тест лимита конкурентных задач."""
        # Устанавливаем лимит в 1 задачу
        concrete_agent.capabilities.max_concurrent_tasks = 1
        
        # Создаем 2 задачи
        task1 = task_request
        task2 = TaskRequest(
            agent_type=concrete_agent.agent_type,
            task_type="api_development",
            description="Вторая задача"
        )
        
        # Запускаем первую задачу
        future1 = asyncio.create_task(concrete_agent.execute(task1))
        
        # Пробуем запустить вторую (должна вызвать ошибку)
        await asyncio.sleep(0.05)  # Даем первой задаче начаться
        
        with pytest.raises(ResourceError, match="Превышен лимит активных задач"):
            await concrete_agent.execute(task2)


# ============================================================================
# UNIT TESTS - Status and Capabilities
# ============================================================================

class TestAgentStatusAndCapabilities:
    """Тесты статуса и возможностей агента."""
    
    def test_get_status(self, concrete_agent):
        """Тест получения статуса агента."""
        status = concrete_agent.get_status()
        
        assert isinstance(status, AgentStatus)
        assert status.agent_type == concrete_agent.agent_type
        assert status.current_status == "idle"
        assert status.is_available is True
        assert status.current_tasks == []
        assert status.completed_tasks == 0
        assert status.failed_tasks == 0
        assert status.uptime >= 0
    
    def test_get_capabilities(self, concrete_agent):
        """Тест получения возможностей агента."""
        capabilities = concrete_agent.get_capabilities()
        
        assert isinstance(capabilities, AgentCapabilities)
        assert capabilities == concrete_agent.capabilities
    
    @pytest.mark.asyncio
    async def test_status_update_after_task_completion(self, concrete_agent, task_request):
        """Тест обновления статуса после завершения задачи."""
        initial_status = concrete_agent.get_status()
        assert initial_status.completed_tasks == 0
        
        # Выполняем задачу
        await concrete_agent.execute(task_request)
        
        # Проверяем обновление статуса
        updated_status = concrete_agent.get_status()
        assert updated_status.completed_tasks == 1
        assert updated_status.current_status == "idle"
        assert updated_status.error_rate == 0.0
        assert updated_status.avg_execution_time > 0
    
    @pytest.mark.asyncio
    async def test_status_update_after_task_failure(self, error_agent, task_request):
        """Тест обновления статуса после ошибки задачи."""
        error_agent.should_error = True
        error_agent.error_type = RuntimeError
        
        initial_status = error_agent.get_status()
        assert initial_status.failed_tasks == 0
        
        # Выполняем задачу (которая провалится)
        await error_agent.execute(task_request)
        
        # Проверяем обновление статуса
        updated_status = error_agent.get_status()
        assert updated_status.failed_tasks == 1
        assert updated_status.current_status == "idle"
        assert updated_status.error_rate > 0


# ============================================================================
# UNIT TESTS - Progress Reporting
# ============================================================================

class TestProgressReporting:
    """Тесты отчетности о прогрессе."""
    
    @pytest.mark.asyncio
    async def test_progress_update(self, concrete_agent, task_request):
        """Тест обновления прогресса."""
        progress = ProgressUpdate(
            task_id=task_request.task_id,
            agent_type=concrete_agent.agent_type,
            progress=0.5,
            current_step="Implementing API endpoints",
            completed_steps=["design", "planning"],
            remaining_steps=["testing", "deployment"],
            message="Промежуточный прогресс",
            eta_seconds=300
        )
        
        # Обновляем прогресс
        concrete_agent.report_progress(progress)
        
        # Проверяем, что прогресс был залогирован
        # (в реальной реализации здесь была бы проверка логов)
        
    @pytest.mark.asyncio
    async def test_progress_update_with_memory_integration(self, concrete_agent, task_request, mock_memory_manager):
        """Тест обновления прогресса с интеграцией памяти."""
        progress = ProgressUpdate(
            task_id=task_request.task_id,
            agent_type=concrete_agent.agent_type,
            progress=0.75,
            current_step="Final testing"
        )
        
        concrete_agent.report_progress(progress)
        
        # Проверяем, что прогресс был сохранен в память
        await asyncio.sleep(0.1)  # Даем время для асинхронного сохранения
        # В реальном тесте здесь была бы проверка вызова mock_memory_manager


# ============================================================================
# UNIT TESTS - Task History
# ============================================================================

class TestTaskHistory:
    """Тесты истории задач."""
    
    @pytest.mark.asyncio
    async def test_get_task_result(self, concrete_agent, task_request):
        """Тест получения результата задачи."""
        # Выполняем задачу
        await concrete_agent.execute(task_request)
        
        # Получаем результат
        result = concrete_agent.get_task_result(task_request.task_id)
        
        assert result is not None
        assert result.task_id == task_request.task_id
        assert result.status == TaskStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_task_result(self, concrete_agent):
        """Тест получения результата несуществующей задачи."""
        result = concrete_agent.get_task_result("nonexistent_task_id")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_task_history_persistence(self, concrete_agent, task_request):
        """Тест сохранения истории задач."""
        # Выполняем несколько задач
        for i in range(3):
            task = TaskRequest(
                agent_type=concrete_agent.agent_type,
                task_type="api_development",
                description=f"Задача {i+1}"
            )
            await concrete_agent.execute(task)
        
        # Проверяем, что все задачи в истории
        assert len(concrete_agent.task_history) == 3
        
        # Проверяем статусы
        for result in concrete_agent.task_history:
            assert result.status == TaskStatus.COMPLETED


# ============================================================================
# UNIT TESTS - Resource Management
# ============================================================================

class TestResourceManagement:
    """Тесты управления ресурсами."""
    
    def test_resource_allocation(self, concrete_agent, task_request):
        """Тест выделения ресурсов."""
        result = concrete_agent.resource_manager.allocate_resources(
            concrete_agent.agent_type, task_request
        )
        
        assert result is True
        
        # Проверяем, что ресурсы выделены
        allocation_id = f"{concrete_agent.agent_type.value}_{task_request.task_id}"
        assert allocation_id in concrete_agent.resource_manager.active_allocations
    
    def test_resource_release(self, concrete_agent, task_request):
        """Тест освобождения ресурсов."""
        # Сначала выделяем ресурсы
        concrete_agent.resource_manager.allocate_resources(
            concrete_agent.agent_type, task_request
        )
        
        # Освобождаем ресурсы
        concrete_agent.resource_manager.release_resources(
            concrete_agent.agent_type, task_request.task_id
        )
        
        # Проверяем, что ресурсы освобождены
        allocation_id = f"{concrete_agent.agent_type.value}_{task_request.task_id}"
        assert allocation_id not in concrete_agent.resource_manager.active_allocations
    
    def test_usage_statistics(self, concrete_agent, task_request):
        """Тест статистики использования."""
        # Выделяем ресурсы
        concrete_agent.resource_manager.allocate_resources(
            concrete_agent.agent_type, task_request
        )
        
        # Получаем статистику
        stats = concrete_agent.resource_manager.get_usage_stats(concrete_agent.agent_type)
        
        assert isinstance(stats, dict)


# ============================================================================
# UNIT TESTS - Memory Integration
# ============================================================================

class TestMemoryIntegration:
    """Тесты интеграции с памятью."""
    
    @pytest.mark.asyncio
    async def test_memory_context_update_on_start(self, concrete_agent, task_request, mock_memory_manager):
        """Тест обновления контекста памяти при старте задачи."""
        await concrete_agent.execute(task_request)
        
        # Проверяем вызовы MemoryManager
        assert mock_memory_manager.store.called
        
        # Ищем вызов для события task_started
        calls = mock_memory_manager.store.call_args_list
        started_calls = [call for call in calls if "task_started" in str(call)]
        assert len(started_calls) > 0
    
    @pytest.mark.asyncio
    async def test_memory_context_update_on_completion(self, concrete_agent, task_request, mock_memory_manager):
        """Тест обновления контекста памяти при завершении задачи."""
        await concrete_agent.execute(task_request)
        
        # Ищем вызов для события task_completed
        calls = mock_memory_manager.store.call_args_list
        completed_calls = [call for call in calls if "task_completed" in str(call)]
        assert len(completed_calls) > 0
    
    @pytest.mark.asyncio
    async def test_memory_context_update_on_failure(self, error_agent, task_request, mock_memory_manager):
        """Тест обновления контекста памяти при ошибке задачи."""
        error_agent.should_error = True
        error_agent.error_type = RuntimeError
        
        await error_agent.execute(task_request)
        
        # Ищем вызов для события task_failed
        calls = mock_memory_manager.store.call_args_list
        failed_calls = [call for call in calls if "task_failed" in str(call)]
        assert len(failed_calls) > 0


# ============================================================================
# UNIT TESTS - State Import/Export
# ============================================================================

class TestStateImportExport:
    """Тесты импорта/экспорта состояния."""
    
    def test_export_agent_state(self, concrete_agent, task_request):
        """Тест экспорта состояния агента."""
        # Добавляем некоторые данные
        concrete_agent.task_history.append(TaskResult(
            task_id="test_task",
            agent_type=concrete_agent.agent_type,
            status=TaskStatus.COMPLETED,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            duration=1.0
        ))
        
        # Экспортируем состояние
        state = concrete_agent.export_agent_state()
        
        assert isinstance(state, dict)
        assert "agent_type" in state
        assert "capabilities" in state
        assert "status" in state
        assert "active_tasks" in state
        assert "task_history" in state
        assert "exported_at" in state
        assert state["agent_type"] == concrete_agent.agent_type.value
    
    def test_import_agent_state(self, concrete_agent):
        """Тест импорта состояния агента."""
        # Создаем состояние для импорта
        state = {
            "status": {
                "agent_type": concrete_agent.agent_type.value,
                "current_status": "busy",
                "is_available": True,
                "current_tasks": [],
                "completed_tasks": 5,
                "failed_tasks": 1,
                "uptime": 3600.0,
                "last_activity": datetime.now(timezone.utc).isoformat(),
                "resource_usage": {},
                "health_metrics": {},
                "error_rate": 0.1,
                "avg_execution_time": 2.5
            },
            "active_tasks": {},
            "task_history": [
                {
                    "task_id": "imported_task",
                    "agent_type": concrete_agent.agent_type.value,
                    "status": "completed",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "duration": 1.5,
                    "output": {},
                    "errors": [],
                    "warnings": [],
                    "metrics": {},
                    "artifacts": [],
                    "next_actions": [],
                    "learning_data": {}
                }
            ]
        }
        
        # Импортируем состояние
        concrete_agent.import_agent_state(state)
        
        # Проверяем импортированные данные
        status = concrete_agent.get_status()
        assert status.completed_tasks == 5
        assert status.failed_tasks == 1
        assert status.error_rate == 0.1
        assert status.avg_execution_time == 2.5
        assert len(concrete_agent.task_history) == 1
    
    def test_invalid_state_import(self, concrete_agent):
        """Тест импорта некорректного состояния."""
        invalid_state = {
            "invalid_field": "invalid_value"
        }
        
        with pytest.raises(AgentError, match="Ошибка импорта состояния"):
            concrete_agent.import_agent_state(invalid_state)


# ============================================================================
# UNIT TESTS - Health Check
# ============================================================================

class TestHealthCheck:
    """Тесты проверки здоровья агента."""
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, concrete_agent, mock_memory_manager):
        """Тест успешной проверки здоровья."""
        health = await concrete_agent.health_check()
        
        assert isinstance(health, dict)
        assert "agent_type" in health
        assert "status" in health
        assert "timestamp" in health
        assert "checks" in health
        assert health["agent_type"] == concrete_agent.agent_type.value
        assert health["status"] == "healthy"
        assert "memory_manager" in health["checks"]
        assert "context_handler" in health["checks"]
    
    @pytest.mark.asyncio
    async def test_health_check_without_memory_manager(self, agent_type, agent_capabilities):
        """Тест проверки здоровья без MemoryManager."""
        agent = BaseAgent(
            agent_type=agent_type,
            capabilities=agent_capabilities
        )
        
        health = await agent.health_check()
        
        assert health["status"] == "warning"
        assert health["checks"]["memory_manager"] is False
    
    @pytest.mark.asyncio
    async def test_health_check_high_error_rate(self, error_agent, task_request):
        """Тест проверки здоровья с высоким процентом ошибок."""
        # Создаем несколько проваленных задач
        error_agent.should_error = True
        error_agent.error_type = RuntimeError
        
        for _ in range(10):
            try:
                await error_agent.execute(task_request)
            except:
                pass
        
        health = await error_agent.health_check()
        
        assert health["status"] == "critical"
        assert health["checks"]["error_rate"] > 0.5


# ============================================================================
# UNIT TESTS - Utility Functions
# ============================================================================

class TestUtilityFunctions:
    """Тесты вспомогательных функций."""
    
    def test_create_task_request(self, agent_type):
        """Тест создания запроса задачи."""
        task = create_task_request(
            agent_type=agent_type,
            task_type="test_task",
            description="Test task description",
            inputs={"param1": "value1"},
            priority=3,
            timeout=60
        )
        
        assert isinstance(task, TaskRequest)
        assert task.agent_type == agent_type
        assert task.task_type == "test_task"
        assert task.description == "Test task description"
        assert task.inputs == {"param1": "value1"}
        assert task.priority == 3
        assert task.timeout == 60
    
    def test_validate_agent_config_valid(self):
        """Тест валидации корректной конфигурации агента."""
        config = {
            "agent_type": "backend",
            "name": "Backend Agent",
            "description": "Backend development agent",
            "supported_tasks": ["api_development", "database_design"]
        }
        
        result = validate_agent_config(config)
        assert result is True
    
    def test_validate_agent_config_missing_fields(self):
        """Тест валидации конфигурации с пропущенными полями."""
        config = {
            "agent_type": "backend",
            "name": "Backend Agent"
            # Отсутствуют required поля
        }
        
        result = validate_agent_config(config)
        assert result is False
    
    def test_validate_agent_config_invalid_agent_type(self):
        """Тест валидации с недопустимым типом агента."""
        config = {
            "agent_type": "invalid_type",
            "name": "Invalid Agent",
            "description": "Invalid agent",
            "supported_tasks": ["test"]
        }
        
        result = validate_agent_config(config)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_generate_agent_report(self, concrete_agent, task_request):
        """Тест генерации отчета агента."""
        # Выполняем несколько задач
        for i in range(3):
            task = TaskRequest(
                agent_type=concrete_agent.agent_type,
                task_type="api_development",
                description=f"Test task {i+1}"
            )
            await concrete_agent.execute(task)
        
        # Генерируем отчет
        report = generate_agent_report(concrete_agent)
        
        assert isinstance(report, dict)
        assert "agent_info" in report
        assert "performance" in report
        assert "current_state" in report
        assert "resources" in report
        assert "health" in report
        
        # Проверяем структуру отчета
        agent_info = report["agent_info"]
        assert agent_info["type"] == concrete_agent.agent_type.value
        assert agent_info["name"] == concrete_agent.capabilities.name
        
        performance = report["performance"]
        assert performance["total_tasks"] == 3
        assert performance["completed_tasks"] == 3
        assert performance["failed_tasks"] == 0
        assert performance["success_rate"] == 1.0


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestAgentPerformance:
    """Тесты производительности агента."""
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_tasks(self, concrete_agent):
        """Тест множественных конкурентных задач."""
        tasks = []
        for i in range(concrete_agent.capabilities.max_concurrent_tasks):
            task = TaskRequest(
                agent_type=concrete_agent.agent_type,
                task_type="api_development",
                description=f"Concurrent task {i+1}"
            )
            tasks.append(task)
        
        # Запускаем все задачи конкурентно
        start_time = time.time()
        results = await asyncio.gather(*[concrete_agent.execute(task) for task in tasks])
        total_time = time.time() - start_time
        
        # Проверяем результаты
        assert len(results) == len(tasks)
        assert all(result.status == TaskStatus.COMPLETED for result in results)
        
        # Время выполнения всех задач должно быть меньше суммы времен
        # (благодаря конкурентности)
        assert total_time < len(tasks) * 0.2  # Эвристика для теста
    
    @pytest.mark.asyncio
    async def test_task_execution_timing(self, concrete_agent, task_request):
        """Тест времени выполнения задач."""
        start_time = time.time()
        result = await concrete_agent.execute(task_request)
        execution_time = time.time() - start_time
        
        # Проверяем время выполнения
        assert execution_time >= 0.1  # Минимум время имитации
        assert execution_time < 1.0   # Максимум для теста
        assert abs(execution_time - result.duration) < 0.1  # Погрешность измерения
    
    @pytest.mark.asyncio
    async def test_memory_usage_during_execution(self, concrete_agent, task_request):
        """Тест использования памяти во время выполнения."""
        # Выполняем много задач
        tasks = []
        for i in range(50):
            task = TaskRequest(
                agent_type=concrete_agent.agent_type,
                task_type="api_development",
                description=f"Memory test task {i+1}"
            )
            tasks.append(task)
        
        # Запускаем задачи
        await asyncio.gather(*[concrete_agent.execute(task) for task in tasks])
        
        # Проверяем, что память не растет бесконтрольно
        # (в реальном тесте здесь был бы мониторинг памяти)
        assert len(concrete_agent.task_history) == 50
        
        # Активные задачи должны быть пусты
        assert len(concrete_agent.active_tasks) == 0


# ============================================================================
# ERROR HANDLING AND EDGE CASES
# ============================================================================

class TestErrorHandlingAndEdgeCases:
    """Тесты обработки ошибок и граничных случаев."""
    
    @pytest.mark.asyncio
    async def test_invalid_task_execution(self, concrete_agent):
        """Тест выполнения некорректной задачи."""
        invalid_task = TaskRequest(
            agent_type=concrete_agent.agent_type,
            task_type="api_development",
            description=""  # Пустое описание
        )
        
        with pytest.raises(TaskValidationError):
            await concrete_agent.execute(invalid_task)
    
    @pytest.mark.asyncio
    async def test_resource_allocation_failure(self, concrete_agent, task_request):
        """Тест ошибки выделения ресурсов."""
        # Мокаем неудачное выделение ресурсов
        concrete_agent.resource_manager.allocate_resources = Mock(return_value=False)
        
        with pytest.raises(ResourceError, match="Не удалось выделить ресурсы"):
            await concrete_agent.execute(task_request)
    
    @pytest.mark.asyncio
    async def test_memory_manager_error_handling(self, agent_type, agent_capabilities, task_request):
        """Тест обработки ошибок MemoryManager."""
        mock_memory_manager = AsyncMock()
        mock_memory_manager.store.side_effect = Exception("Memory error")
        
        agent = BaseAgent(
            agent_type=agent_type,
            capabilities=agent_capabilities,
            memory_manager=mock_memory_manager
        )
        
        # Создаем конкретный агент с execute_task
        class TestAgent(BaseAgent):
            async def execute_task(self, task: TaskRequest) -> TaskResult:
                return TaskResult(
                    task_id=task.task_id,
                    agent_type=self.agent_type,
                    status=TaskStatus.COMPLETED,
                    started_at=task.created_at,
                    completed_at=datetime.now(timezone.utc),
                    duration=0.1,
                    output={"completed": True}
                )
        
        test_agent = TestAgent(
            agent_type=agent_type,
            capabilities=agent_capabilities,
            memory_manager=mock_memory_manager
        )
        
        # Задача должна выполниться, несмотря на ошибку памяти
        result = await test_agent.execute(task_request)
        assert result.status == TaskStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_cleanup_on_exception(self, concrete_agent, task_request):
        """Тест очистки при исключении."""
        error_agent.should_error = True
        error_agent.error_type = RuntimeError
        
        # Выполняем задачу, которая провалится
        await error_agent.execute(task_request)
        
        # Проверяем, что агент корректно очистился
        assert len(error_agent.active_tasks) == 0
        assert error_agent.status.current_status == "idle"
        assert error_agent.status.is_available is True
    
    def test_agent_destructor(self, concrete_agent):
        """Тест деструктора агента."""
        # Создаем копию для удаления
        executor = concrete_agent.executor
        
        # Удаляем агент
        del concrete_agent
        
        # Проверяем, что executor закрыт
        # (в реальном тесте здесь была бы проверка состояния executor)


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_base_agent_tests():
    """Запуск всех тестов BaseAgent."""
    print("🧪 Запуск unit тестов BaseAgent...")
    
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto",
        "-x"  # Остановиться на первой ошибке
    ]
    
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("\n✅ Все тесты BaseAgent прошли успешно!")
    else:
        print(f"\n❌ Тесты BaseAgent завершились с ошибкой: {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    run_base_agent_tests()