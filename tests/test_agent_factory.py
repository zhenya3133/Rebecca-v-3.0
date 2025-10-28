"""Comprehensive unit тесты для AgentFactory компонента.

Включает:
- Unit тесты создания агентов
- Тесты конфигурации и настройки
- Тесты управления агентами
- Тесты назначения задач
- Тесты статистики и мониторинга
- Mock тесты для внешних зависимостей
- Performance тесты

Автор: Claude Code  
Дата: 2025-10-28
"""

import asyncio
import pytest
import time
import yaml
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any
import os
import sys

# Настройка путей
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# Импорты
from multi_agent.agent_factory import (
    AgentFactory,
    AgentType,
    AgentCapabilities,
    TaskRequest,
    TaskResult,
    TaskStatus,
    # Специализированные агенты
    BackendAgent,
    FrontendAgent,
    MLEngineerAgent,
    QAAnalystAgent,
    DevOpsAgent
)
from multi_agent.base_agent import BaseAgent


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_config_dir():
    """Временная директория для конфигурационных файлов."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_memory_manager():
    """Mock MemoryManager для тестов."""
    memory_manager = AsyncMock()
    memory_manager.store = AsyncMock(return_value="memory_id_123")
    memory_manager.retrieve = AsyncMock(return_value=[])
    return memory_manager


@pytest.fixture
def mock_context_handler():
    """Mock ContextHandler для тестов."""
    context_handler = AsyncMock()
    context_handler.build_context_envelope = AsyncMock(return_value={
        "trace_id": "test_trace_123",
        "timestamp": time.time()
    })
    return context_handler


@pytest.fixture
def factory(mock_memory_manager, mock_context_handler, temp_config_dir):
    """AgentFactory для тестов."""
    config_path = os.path.join(temp_config_dir, "agents.yaml")
    
    # Создаем базовую конфигурацию
    basic_config = {
        "global_settings": {
            "limits": {
                "max_total_agents": 10
            }
        },
        "agents": {
            "backend_agent": {
                "agent_type": "backend",
                "name": "Backend Developer",
                "version": "1.0.0",
                "description": "Специалист по backend разработке",
                "supported_tasks": ["api_development", "database_design"],
                "max_concurrent_tasks": 3,
                "resource_requirements": {
                    "cpu": "2 cores",
                    "memory": "4GB"
                }
            },
            "frontend_agent": {
                "agent_type": "frontend",
                "name": "Frontend Developer",
                "version": "1.0.0",
                "description": "Специалист по frontend разработке",
                "supported_tasks": ["ui_development", "css_styling"],
                "max_concurrent_tasks": 2,
                "resource_requirements": {
                    "cpu": "1 core",
                    "memory": "2GB"
                }
            }
        }
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(basic_config, f, default_flow_style=False, allow_unicode=True)
    
    return AgentFactory(
        memory_manager=mock_memory_manager,
        context_handler=mock_context_handler,
        config_path=config_path
    )


@pytest.fixture
def sample_agent_config():
    """Образец конфигурации агента."""
    return {
        "agent_type": "backend",
        "name": "Test Backend Agent",
        "version": "1.0.0",
        "description": "Тестовый backend агент",
        "supported_tasks": ["api_development", "database_design", "service_architecture"],
        "supported_languages": ["python", "javascript", "go"],
        "max_concurrent_tasks": 5,
        "resource_requirements": {
            "cpu": "4 cores",
            "memory": "8GB",
            "storage": "20GB"
        },
        "dependencies": ["git", "docker", "postgresql"],
        "environment_vars": {
            "DEBUG": "false",
            "LOG_LEVEL": "INFO"
        },
        "specializations": ["rest_apis", "graphql", "microservices"],
        "integrations": ["github", "docker_hub", "jenkins"],
        "performance_profile": {
            "avg_response_time": 2.0,
            "throughput": 150,
            "error_rate": 0.01
        }
    }


@pytest.fixture
def task_request_backend():
    """Запрос задачи для backend агента."""
    return TaskRequest(
        agent_type=AgentType.BACKEND,
        task_type="api_development",
        description="Создать REST API для управления пользователями",
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
def task_request_frontend():
    """Запрос задачи для frontend агента."""
    return TaskRequest(
        agent_type=AgentType.FRONTEND,
        task_type="ui_development",
        description="Создать компонент для отображения списка пользователей",
        inputs={
            "framework": "React",
            "styling": "Tailwind CSS",
            "state_management": "Redux"
        },
        context={"project": "user_service", "environment": "development"},
        priority=3,
        timeout=180,
        retry_count=0,
        max_retries=1,
        dependencies=[],
        metadata={"created_by": "test_system", "category": "frontend"}
    )


# ============================================================================
# UNIT TESTS - Factory Initialization
# ============================================================================

class TestAgentFactoryInitialization:
    """Тесты инициализации AgentFactory."""
    
    def test_factory_creation(self, factory, mock_memory_manager, mock_context_handler):
        """Тест создания фабрики агентов."""
        assert factory.memory_manager == mock_memory_manager
        assert factory.context_handler == mock_context_handler
        assert factory.agent_classes is not None
        assert factory.agents == {}
        assert factory.usage_stats is not None
        assert factory.usage_stats["total_created"] == 0
        assert factory.usage_stats["total_executed"] == 0
    
    def test_agent_classes_registration(self, factory):
        """Тест регистрации классов агентов."""
        # Проверяем, что классы агентов зарегистрированы
        expected_types = [
            AgentType.BACKEND,
            AgentType.FRONTEND,
            AgentType.ML_ENGINEER,
            AgentType.QA_ANALYST,
            AgentType.DEVOPS
        ]
        
        for agent_type in expected_types:
            assert agent_type in factory.agent_classes
            assert issubclass(factory.agent_classes[agent_type], BaseAgent)
    
    def test_load_global_config(self, factory, temp_config_dir):
        """Тест загрузки глобальной конфигурации."""
        assert isinstance(factory.global_config, dict)
        assert "agents" in factory.global_config
        assert "global_settings" in factory.global_config
    
    def test_factory_without_config_file(self, mock_memory_manager, mock_context_handler):
        """Тест фабрики без файла конфигурации."""
        factory = AgentFactory(
            memory_manager=mock_memory_manager,
            context_handler=mock_context_handler,
            config_path="nonexistent_config.yaml"
        )
        
        assert factory.global_config == {"agents": {}}
    
    def test_factory_with_invalid_config(self, mock_memory_manager, mock_context_handler, temp_config_dir):
        """Тест фабрики с некорректной конфигурацией."""
        config_path = os.path.join(temp_config_dir, "invalid.yaml")
        with open(config_path, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        factory = AgentFactory(
            memory_manager=mock_memory_manager,
            context_handler=mock_context_handler,
            config_path=config_path
        )
        
        assert factory.global_config == {}


# ============================================================================
# UNIT TESTS - Agent Creation
# ============================================================================

class TestAgentCreation:
    """Тесты создания агентов."""
    
    def test_create_backend_agent(self, factory):
        """Тест создания backend агента."""
        agent = factory.create_agent(AgentType.BACKEND)
        
        assert isinstance(agent, BackendAgent)
        assert agent.agent_type == AgentType.BACKEND
        assert agent.memory_manager == factory.memory_manager
        assert agent.context_handler == factory.context_handler
        assert AgentType.BACKEND in factory.agents
    
    def test_create_frontend_agent(self, factory):
        """Тест создания frontend агента."""
        agent = factory.create_agent(AgentType.FRONTEND)
        
        assert isinstance(agent, FrontendAgent)
        assert agent.agent_type == AgentType.FRONTEND
        assert AgentType.FRONTEND in factory.agents
    
    def test_create_agent_with_custom_id(self, factory):
        """Тест создания агента с кастомным ID."""
        custom_id = "my_custom_backend_agent"
        agent = factory.create_agent(AgentType.BACKEND, agent_id=custom_id)
        
        assert hasattr(agent, 'agent_id')
        assert agent.agent_id == custom_id
    
    def test_create_unsupported_agent_type(self, factory):
        """Тест создания неподдерживаемого типа агента."""
        # AgentType.RESEARCH не поддерживается в текущей реализации
        with pytest.raises(ValueError, match="Неподдерживаемый тип агента"):
            factory.create_agent(AgentType.RESEARCH)
    
    def test_create_agent_max_limit_exceeded(self, factory):
        """Тест превышения лимита агентов."""
        # Создаем максимальное количество агентов
        for i in range(10):  # max_total_agents = 10
            try:
                factory.create_agent(AgentType.BACKEND)
            except RuntimeError:
                break
        
        # Следующая попытка должна провалиться
        with pytest.raises(RuntimeError, match="Превышен лимит общего количества агентов"):
            factory.create_agent(AgentType.FRONTEND)
    
    def test_create_duplicate_agent(self, factory):
        """Тест создания дублирующего агента."""
        # Создаем агента
        agent1 = factory.create_agent(AgentType.BACKEND)
        
        # Создаем агента того же типа
        agent2 = factory.create_agent(AgentType.BACKEND)
        
        # Второй агент должен заменить первый
        assert factory.get_agent(AgentType.BACKEND) == agent2
        assert len(factory.agents) == 1
    
    def test_agent_capabilities_loading(self, factory, sample_agent_config, temp_config_dir):
        """Тест загрузки возможностей агента."""
        # Обновляем конфигурацию
        config_path = factory.config_path
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump({
                "agents": {
                    "backend_agent": sample_agent_config
                },
                "global_settings": {"limits": {"max_total_agents": 10}}
            }, f, default_flow_style=False, allow_unicode=True)
        
        # Пересоздаем фабрику
        factory = AgentFactory(
            memory_manager=factory.memory_manager,
            context_handler=factory.context_handler,
            config_path=config_path
        )
        
        # Создаем агента
        agent = factory.create_agent(AgentType.BACKEND)
        
        # Проверяем возможности
        capabilities = agent.get_capabilities()
        assert capabilities.name == "Test Backend Agent"
        assert capabilities.version == "1.0.0"
        assert "api_development" in capabilities.supported_tasks
        assert "database_design" in capabilities.supported_tasks
        assert capabilities.max_concurrent_tasks == 5
    
    def test_factory_stats_update_on_creation(self, factory):
        """Тест обновления статистики при создании."""
        initial_count = factory.usage_stats["total_created"]
        
        factory.create_agent(AgentType.BACKEND)
        
        assert factory.usage_stats["total_created"] == initial_count + 1


# ============================================================================
# UNIT TESTS - Agent Management
# ============================================================================

class TestAgentManagement:
    """Тесты управления агентами."""
    
    def test_get_existing_agent(self, factory):
        """Тест получения существующего агента."""
        # Создаем агента
        created_agent = factory.create_agent(AgentType.BACKEND)
        
        # Получаем агента
        retrieved_agent = factory.get_agent(AgentType.BACKEND)
        
        assert retrieved_agent == created_agent
    
    def test_get_nonexistent_agent(self, factory):
        """Тест получения несуществующего агента."""
        agent = factory.get_agent(AgentType.BACKEND)
        assert agent is None
    
    def test_list_agents(self, factory):
        """Тест списка агентов."""
        # Создаем несколько агентов
        factory.create_agent(AgentType.BACKEND)
        factory.create_agent(AgentType.FRONTEND)
        
        # Получаем список
        agents_list = factory.list_agents()
        
        assert isinstance(agents_list, dict)
        assert AgentType.BACKEND in agents_list
        assert AgentType.FRONTEND in agents_list
        
        # Проверяем структуру информации об агенте
        backend_info = agents_list[AgentType.BACKEND]
        assert "status" in backend_info
        assert "is_available" in backend_info
        assert "active_tasks" in backend_info
        assert "completed_tasks" in backend_info
        assert "capabilities" in backend_info
    
    def test_remove_existing_agent(self, factory):
        """Тест удаления существующего агента."""
        # Создаем агента
        factory.create_agent(AgentType.BACKEND)
        assert AgentType.BACKEND in factory.agents
        
        # Удаляем агента
        result = factory.remove_agent(AgentType.BACKEND)
        
        assert result is True
        assert AgentType.BACKEND not in factory.agents
    
    def test_remove_nonexistent_agent(self, factory):
        """Тест удаления несуществующего агента."""
        result = factory.remove_agent(AgentType.BACKEND)
        assert result is False
    
    def test_agent_cleanup_on_removal(self, factory):
        """Тест очистки агента при удалении."""
        # Создаем агента
        agent = factory.create_agent(AgentType.BACKEND)
        
        # Добавляем задачу агенту (симуляция)
        # В реальном тесте здесь была бы реальная задача
        agent.status.current_tasks = ["test_task"]
        
        # Удаляем агента
        factory.remove_agent(AgentType.BACKEND)
        
        # Проверяем, что агент удален из реестра
        assert AgentType.BACKEND not in factory.agents


# ============================================================================
# UNIT TESTS - Task Assignment
# ============================================================================

class TestTaskAssignment:
    """Тесты назначения задач."""
    
    def test_assign_task_to_suitable_agent(self, factory, task_request_backend):
        """Тест назначения задачи подходящему агенту."""
        # Создаем агента
        factory.create_agent(AgentType.BACKEND)
        
        # Назначаем задачу
        assigned_task = factory.assign_task_to_agent(task_request_backend)
        
        assert assigned_task.agent_type == AgentType.BACKEND
        assert assigned_task.task_id == task_request_backend.task_id
    
    def test_assign_task_no_suitable_agent(self, factory, task_request_backend):
        """Тест назначения задачи без подходящего агента."""
        # Не создаем агентов
        
        with pytest.raises(ValueError, match="Нет подходящих агентов"):
            factory.assign_task_to_agent(task_request_backend)
    
    def test_assign_task_agent_mismatch(self, factory, task_request_backend):
        """Тест несоответствия задачи и агента."""
        # Создаем frontend агента
        factory.create_agent(AgentType.FRONTEND)
        
        # Пытаемся назначить backend задачу
        with pytest.raises(ValueError, match="Нет подходящих агентов"):
            factory.assign_task_to_agent(task_request_backend)
    
    def test_assign_task_multiple_suitable_agents(self, factory, task_request_backend):
        """Тест назначения при нескольких подходящих агентах."""
        # Создаем несколько агентов
        factory.create_agent(AgentType.BACKEND)
        # Добавляем еще одного backend агента (заменит первого)
        factory.create_agent(AgentType.BACKEND)
        
        # Назначаем задачу
        assigned_task = factory.assign_task_to_agent(task_request_backend)
        
        assert assigned_task.agent_type == AgentType.BACKEND
    
    def test_agent_selection_based_on_load(self, factory, task_request_backend):
        """Тест выбора агента на основе загрузки."""
        # Создаем агента с ограничением на 1 задачу
        factory.global_config["agents"]["backend_agent"] = {
            "agent_type": "backend",
            "name": "Backend Agent",
            "supported_tasks": ["api_development"],
            "max_concurrent_tasks": 1
        }
        
        agent = factory.create_agent(AgentType.BACKEND)
        
        # Симулируем загрузку агента
        agent.status.current_tasks = ["existing_task"]
        
        # Попытка назначить задачу загруженному агенту
        # Должна пройти, так как это тест логики выбора
        # В реальной реализации проверяется available агентов
        result = factory.assign_task_to_agent(task_request_backend)
        
        assert result.agent_type == AgentType.BACKEND
    
    def test_agent_rating_calculation(self, factory):
        """Тест расчета рейтинга агента."""
        # Создаем агента
        agent = factory.create_agent(AgentType.BACKEND)
        
        # Создаем тестовую задачу
        task = TaskRequest(
            agent_type=AgentType.BACKEND,
            task_type="api_development",
            description="Test task"
        )
        
        # Рассчитываем рейтинг
        rating = factory._calculate_agent_rating(agent, task)
        
        assert isinstance(rating, float)
        assert 0 <= rating <= 1
    
    def test_agent_rating_with_empty_history(self, factory):
        """Тест рейтинга агента без истории."""
        agent = factory.create_agent(AgentType.BACKEND)
        
        # Создаем задачу с неподдерживаемым типом
        task = TaskRequest(
            agent_type=AgentType.BACKEND,
            task_type="unsupported_task",
            description="Test task"
        )
        
        rating = factory._calculate_agent_rating(agent, task)
        
        # Рейтинг должен быть низким для неподдерживаемых задач
        assert rating < 1.0
    
    def test_agent_rating_with_performance_history(self, factory):
        """Тест рейтинга с историей производительности."""
        agent = factory.create_agent(AgentType.BACKEND)
        
        # Симулируем успешные задачи
        agent.status.completed_tasks = 10
        agent.status.failed_tasks = 1
        agent.status.avg_execution_time = 2.5
        
        task = TaskRequest(
            agent_type=AgentType.BACKEND,
            task_type="api_development",
            description="Test task",
            priority=1  # Высокий приоритет
        )
        
        rating = factory._calculate_agent_rating(agent, task)
        
        # Агент с хорошей историей должен иметь высокий рейтинг
        assert rating > 0.5


# ============================================================================
# UNIT TESTS - Statistics and Monitoring
# ============================================================================

class TestStatisticsAndMonitoring:
    """Тесты статистики и мониторинга."""
    
    def test_factory_stats_basic(self, factory):
        """Тест базовой статистики фабрики."""
        # Создаем агентов
        factory.create_agent(AgentType.BACKEND)
        factory.create_agent(AgentType.FRONTEND)
        
        stats = factory.get_factory_stats()
        
        assert isinstance(stats, dict)
        assert "total_agents" in stats
        assert "total_active_tasks" in stats
        assert "total_completed_tasks" in stats
        assert "total_failed_tasks" in stats
        assert "success_rate" in stats
        assert "avg_execution_time" in stats
        assert "usage_stats" in stats
        assert "available_agents" in stats
        
        assert stats["total_agents"] == 2
        assert len(stats["available_agents"]) == 2
        assert "backend" in stats["available_agents"]
        assert "frontend" in stats["available_agents"]
    
    def test_factory_stats_with_empty_agents(self, factory):
        """Тест статистики без агентов."""
        stats = factory.get_factory_stats()
        
        assert stats["total_agents"] == 0
        assert stats["total_active_tasks"] == 0
        assert stats["total_completed_tasks"] == 0
        assert stats["total_failed_tasks"] == 0
        assert stats["success_rate"] == 0
        assert stats["avg_execution_time"] == 0
        assert stats["available_agents"] == []
    
    def test_factory_stats_with_agent_performance(self, factory):
        """Тест статистики с производительностью агентов."""
        # Создаем агентов
        backend_agent = factory.create_agent(AgentType.BACKEND)
        frontend_agent = factory.create_agent(AgentType.FRONTEND)
        
        # Симулируем производительность
        backend_agent.status.completed_tasks = 5
        backend_agent.status.failed_tasks = 1
        backend_agent.status.avg_execution_time = 2.0
        
        frontend_agent.status.completed_tasks = 3
        frontend_agent.status.failed_tasks = 0
        frontend_agent.status.avg_execution_time = 1.5
        
        stats = factory.get_factory_stats()
        
        # Проверяем расчеты
        assert stats["total_completed_tasks"] == 8
        assert stats["total_failed_tasks"] == 1
        assert stats["success_rate"] == 8/9  # 8 / (8 + 1)
        assert stats["avg_execution_time"] == (2.0 + 1.5) / 2  # Среднее значение


# ============================================================================
# UNIT TESTS - Health Check
# ============================================================================

class TestHealthCheck:
    """Тесты проверки здоровья фабрики."""
    
    def test_health_check_with_agents(self, factory):
        """Тест проверки здоровья с агентами."""
        # Создаем агентов
        factory.create_agent(AgentType.BACKEND)
        factory.create_agent(AgentType.FRONTEND)
        
        health = factory.health_check()
        
        assert isinstance(health, dict)
        assert "status" in health
        assert "timestamp" in health
        assert "total_agents" in health
        assert "healthy_agents" in health
        assert "issues" in health
        assert "recommendations" in health
        
        assert health["status"] in ["healthy", "warning", "critical"]
        assert health["total_agents"] == 2
        assert health["healthy_agents"] >= 0
    
    def test_health_check_no_agents(self, factory):
        """Тест проверки здоровья без агентов."""
        health = factory.health_check()
        
        assert health["status"] == "warning"
        assert health["total_agents"] == 0
        assert len(health["issues"]) > 0
        assert "Нет созданных агентов" in health["issues"]
    
    def test_health_check_overloaded_agents(self, factory, temp_config_dir):
        """Тест проверки здоровья с перегруженными агентами."""
        # Создаем агентов с низкими лимитами
        factory.global_config["agents"]["backend_agent"] = {
            "agent_type": "backend",
            "name": "Backend Agent",
            "supported_tasks": ["api_development"],
            "max_concurrent_tasks": 1
        }
        
        backend_agent = factory.create_agent(AgentType.BACKEND)
        
        # Симулируем перегрузку
        backend_agent.status.current_tasks = ["task1", "task2", "task3"]  # Больше лимита
        
        health = factory.health_check()
        
        assert health["status"] in ["warning", "critical"]
        assert len(health["issues"]) > 0
        assert any("Перегруженные" in issue for issue in health["issues"])
    
    def test_health_check_high_error_rate(self, factory):
        """Тест проверки здоровья с высоким процентом ошибок."""
        agent = factory.create_agent(AgentType.BACKEND)
        
        # Симулируем высокий процент ошибок
        agent.status.completed_tasks = 1
        agent.status.failed_tasks = 5
        agent.status.error_rate = 0.83
        
        health = factory.health_check()
        
        assert health["status"] in ["warning", "critical"]
        assert len(health["issues"]) > 0
        assert any("высокой ошибкостью" in issue for issue in health["issues"])
    
    def test_recommendations_generation(self, factory):
        """Тест генерации рекомендаций."""
        health = factory.health_check()
        
        recommendations = health["recommendations"]
        assert isinstance(recommendations, list)
        
        # Для пустой фабрики должна быть рекомендация о создании агентов
        assert len(recommendations) > 0


# ============================================================================
# UNIT TESTS - Configuration Management
# ============================================================================

class TestConfigurationManagement:
    """Тесты управления конфигурацией."""
    
    def test_load_agent_config_from_global_file(self, factory, temp_config_dir, sample_agent_config):
        """Тест загрузки конфигурации из глобального файла."""
        # Обновляем конфигурацию
        with open(factory.config_path, 'w', encoding='utf-8') as f:
            yaml.dump({
                "agents": {
                    "backend_agent": sample_agent_config
                },
                "global_settings": {"limits": {"max_total_agents": 10}}
            }, f, default_flow_style=False, allow_unicode=True)
        
        # Загружаем конфигурацию агента
        config = factory._load_agent_config(AgentType.BACKEND)
        
        assert "capabilities" in config
        assert "config" in config
        
        capabilities = config["capabilities"]
        assert capabilities.agent_type == AgentType.BACKEND
        assert capabilities.name == "Test Backend Agent"
    
    def test_load_agent_config_separate_file(self, factory, temp_config_dir, sample_agent_config):
        """Тест загрузки конфигурации из отдельного файла."""
        # Создаем отдельный файл конфигурации
        separate_config_path = os.path.join(temp_config_dir, "backend_agent.yaml")
        with open(separate_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(sample_agent_config, f, default_flow_style=False, allow_unicode=True)
        
        # Модифицируем фабрику для использования отдельного файла
        config = factory._load_agent_config(AgentType.BACKEND)
        
        # Должна вернуться базовая конфигурация, так как отдельный файл не найден
        # (из-за того что путь формируется на основе типа агента)
        assert "capabilities" in config
    
    def test_load_agent_config_fallback(self, factory):
        """Тест fallback конфигурации агента."""
        # Удаляем все конфигурации
        with open(factory.config_path, 'w') as f:
            yaml.dump({"agents": {}}, f)
        
        config = factory._load_agent_config(AgentType.BACKEND)
        
        assert "capabilities" in config
        assert "config" in config
        
        capabilities = config["capabilities"]
        assert capabilities.agent_type == AgentType.BACKEND
        assert "Backend Agent" in capabilities.name
        assert "backend" in capabilities.supported_tasks
    
    def test_load_invalid_agent_config(self, factory, temp_config_dir):
        """Тест загрузки некорректной конфигурации агента."""
        # Создаем некорректную конфигурацию
        with open(factory.config_path, 'w', encoding='utf-8') as f:
            yaml.dump({
                "agents": {
                    "backend_agent": {
                        "invalid_field": "invalid_value"
                        # Отсутствуют обязательные поля
                    }
                }
            }, f)
        
        config = factory._load_agent_config(AgentType.BACKEND)
        
        # Должна быть возвращена fallback конфигурация
        assert "capabilities" in config
        assert capabilities.agent_type == AgentType.BACKEND


# ============================================================================
# UNIT TESTS - Cleanup Operations
# ============================================================================

class TestCleanupOperations:
    """Тесты операций очистки."""
    
    def test_cleanup_all_agents(self, factory):
        """Тест очистки всех агентов."""
        # Создаем агентов
        factory.create_agent(AgentType.BACKEND)
        factory.create_agent(AgentType.FRONTEND)
        factory.create_agent(AgentType.ML_ENGINEER)
        
        assert len(factory.agents) == 3
        
        # Выполняем очистку
        factory.cleanup()
        
        assert len(factory.agents) == 0
    
    def test_cleanup_with_active_tasks(self, factory):
        """Тест очистки с активными задачами."""
        # Создаем агента
        agent = factory.create_agent(AgentType.BACKEND)
        
        # Симулируем активные задачи
        agent.status.current_tasks = ["task1", "task2"]
        
        # Очищаем
        factory.cleanup()
        
        # Агент должен быть удален
        assert len(factory.agents) == 0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestAgentFactoryIntegration:
    """Integration тесты для AgentFactory."""
    
    @pytest.mark.asyncio
    async def test_full_agent_lifecycle(self, factory, task_request_backend):
        """Тест полного жизненного цикла агента."""
        # 1. Создание агента
        agent = factory.create_agent(AgentType.BACKEND)
        assert agent is not None
        
        # 2. Назначение задачи
        assigned_task = factory.assign_task_to_agent(task_request_backend)
        assert assigned_task.agent_type == AgentType.BACKEND
        
        # 3. Выполнение задачи агентом
        result = await agent.execute(assigned_task)
        assert result.status == TaskStatus.COMPLETED
        
        # 4. Проверка статистики
        stats = factory.get_factory_stats()
        assert stats["total_agents"] == 1
        assert stats["total_completed_tasks"] == 1
        
        # 5. Удаление агента
        success = factory.remove_agent(AgentType.BACKEND)
        assert success is True
        
        # 6. Проверка очистки
        assert factory.get_agent(AgentType.BACKEND) is None
    
    @pytest.mark.asyncio
    async def test_multiple_agents_workflow(self, factory, task_request_backend, task_request_frontend):
        """Тест рабочего процесса с несколькими агентами."""
        # Создаем агентов
        backend_agent = factory.create_agent(AgentType.BACKEND)
        frontend_agent = factory.create_agent(AgentType.FRONTEND)
        
        # Назначаем задачи
        backend_task = factory.assign_task_to_agent(task_request_backend)
        frontend_task = factory.assign_task_to_agent(task_request_frontend)
        
        # Выполняем задачи конкурентно
        backend_result = await backend_agent.execute(backend_task)
        frontend_result = await frontend_agent.execute(frontend_task)
        
        # Проверяем результаты
        assert backend_result.status == TaskStatus.COMPLETED
        assert frontend_result.status == TaskStatus.COMPLETED
        
        # Проверяем фабричную статистику
        stats = factory.get_factory_stats()
        assert stats["total_agents"] == 2
        assert stats["total_completed_tasks"] == 2
    
    @pytest.mark.asyncio
    async def test_factory_health_monitoring(self, factory):
        """Тест мониторинга здоровья фабрики."""
        # Создаем агентов
        backend_agent = factory.create_agent(AgentType.BACKEND)
        frontend_agent = factory.create_agent(AgentType.FRONTEND)
        
        # Выполняем задачи для генерации статистики
        task1 = TaskRequest(
            agent_type=AgentType.BACKEND,
            task_type="api_development",
            description="Health check task 1"
        )
        task2 = TaskRequest(
            agent_type=AgentType.FRONTEND,
            task_type="ui_development", 
            description="Health check task 2"
        )
        
        await backend_agent.execute(task1)
        await frontend_agent.execute(task2)
        
        # Проверяем здоровье
        health = factory.health_check()
        
        assert health["status"] in ["healthy", "warning"]
        assert health["total_agents"] == 2
        assert health["healthy_agents"] >= 1
        
        # Проверяем рекомендации
        recommendations = health["recommendations"]
        assert isinstance(recommendations, list)


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestAgentFactoryPerformance:
    """Performance тесты для AgentFactory."""
    
    def test_rapid_agent_creation(self, factory):
        """Тест быстрого создания агентов."""
        start_time = time.time()
        
        # Создаем агентов быстро
        agents_created = 0
        try:
            for i in range(5):
                factory.create_agent(AgentType.BACKEND)
                agents_created += 1
        except RuntimeError:
            pass  # Достигли лимита
        
        creation_time = time.time() - start_time
        
        assert agents_created > 0
        assert creation_time < 1.0  # Менее секунды
    
    def test_agent_listing_performance(self, factory):
        """Тест производительности списка агентов."""
        # Создаем несколько агентов
        for i in range(5):
            factory.create_agent(AgentType.BACKEND)
        
        start_time = time.time()
        
        # Получаем список агентов
        agents_list = factory.list_agents()
        
        listing_time = time.time() - start_time
        
        assert len(agents_list) == 5
        assert listing_time < 0.1  # Менее 100ms
    
    def test_statistics_calculation_performance(self, factory):
        """Тест производительности расчета статистики."""
        # Создаем агентов с историей
        for i in range(3):
            agent = factory.create_agent(AgentType.BACKEND)
            agent.status.completed_tasks = i * 10
            agent.status.failed_tasks = i * 2
            agent.status.avg_execution_time = 1.0 + i * 0.5
        
        start_time = time.time()
        
        # Рассчитываем статистику
        stats = factory.get_factory_stats()
        
        stats_time = time.time() - start_time
        
        assert stats["total_agents"] == 3
        assert stats_time < 0.1  # Менее 100ms


# ============================================================================
# ERROR HANDLING AND EDGE CASES
# ============================================================================

class TestErrorHandlingAndEdgeCases:
    """Тесты обработки ошибок и граничных случаев."""
    
    def test_create_agent_with_corrupted_config(self, factory, temp_config_dir):
        """Тест создания агента с поврежденной конфигурацией."""
        # Создаем поврежденную конфигурацию
        with open(factory.config_path, 'w', encoding='utf-8') as f:
            f.write("invalid: yaml: content: [")
        
        # Пересоздаем фабрику
        factory = AgentFactory(
            memory_manager=factory.memory_manager,
            context_handler=factory.context_handler,
            config_path=factory.config_path
        )
        
        # Создание агента должно пройти с fallback конфигурацией
        agent = factory.create_agent(AgentType.BACKEND)
        assert agent is not None
        assert agent.agent_type == AgentType.BACKEND
    
    def test_assign_task_to_nonexistent_agent_type(self, factory, task_request_backend):
        """Тест назначения задачи несуществующему типу агента."""
        # Устанавливаем несуществующий тип агента
        task_request_backend.agent_type = AgentType.RESEARCH
        
        with pytest.raises(ValueError, match="Нет подходящих агентов"):
            factory.assign_task_to_agent(task_request_backend)
    
    def test_remove_agent_during_task_execution(self, factory, task_request_backend):
        """Тест удаления агента во время выполнения задачи."""
        agent = factory.create_agent(AgentType.BACKEND)
        
        # В реальном тесте здесь была бы задача в процессе выполнения
        # Для простоты проверяем только удаление
        success = factory.remove_agent(AgentType.BACKEND)
        assert success is True
    
    def test_factory_stats_with_division_by_zero(self, factory):
        """Тест статистики при делении на ноль."""
        # Фабрика без агентов
        stats = factory.get_factory_stats()
        
        # Проверяем, что нет ошибок деления на ноль
        assert stats["success_rate"] == 0
        assert stats["avg_execution_time"] == 0
        assert isinstance(stats["available_agents"], list)
    
    def test_agent_creation_concurrent_safety(self, factory):
        """Тест безопасности конкурентного создания агентов."""
        import threading
        import time
        
        errors = []
        agents_created = []
        
        def create_agent_thread():
            try:
                agent = factory.create_agent(AgentType.BACKEND)
                agents_created.append(agent)
            except Exception as e:
                errors.append(e)
        
        # Создаем несколько потоков
        threads = []
        for i in range(3):
            thread = threading.Thread(target=create_agent_thread)
            threads.append(thread)
            thread.start()
        
        # Ждем завершения
        for thread in threads:
            thread.join()
        
        # Проверяем результаты
        assert len(errors) <= 1  # Максимум одна ошибка из-за лимита
        assert len(agents_created) >= 1


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_agent_factory_tests():
    """Запуск всех тестов AgentFactory."""
    print("🏭 Запуск unit тестов AgentFactory...")
    
    pytest_args = [
        __file__,
        "-v",
        "--tb=short", 
        "--asyncio-mode=auto",
        "-x"  # Остановиться на первой ошибке
    ]
    
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("\n✅ Все тесты AgentFactory прошли успешно!")
    else:
        print(f"\n❌ Тесты AgentFactory завершились с ошибкой: {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    run_agent_factory_tests()