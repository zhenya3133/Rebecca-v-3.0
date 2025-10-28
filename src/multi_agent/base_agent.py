"""
Базовый класс для всех агентов системы Rebecca-Platform
Реализует архитектуру многоагентной системы с интеграцией памяти и оркестрации
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Protocol
from datetime import datetime, timezone
from enum import Enum
from contextlib import asynccontextmanager
import json
import traceback
from concurrent.futures import ThreadPoolExecutor
import uuid

from pydantic import BaseModel, Field, field_validator
import yaml

from memory_manager.memory_manager import MemoryManager
from orchestrator.main_workflow import ContextHandler


# =============================================================================
# Структуры данных и схемы
# =============================================================================

class TaskStatus(str, Enum):
    """Статусы выполнения задач"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RETRY = "retry"


class AgentType(str, Enum):
    """Типы специализированных агентов"""
    BACKEND = "backend"
    FRONTEND = "frontend"
    ML_ENGINEER = "ml_engineer"
    QA_ANALYST = "qa_analyst"
    DEVOPS = "devops"
    RESEARCH = "research"
    WRITER = "writer"
    COORDINATOR = "coordinator"


class TaskRequest(BaseModel):
    """Запрос на выполнение задачи агентом"""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_type: AgentType
    task_type: str
    description: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    context: Optional[Dict[str, Any]] = None
    priority: int = Field(default=1, ge=1, le=5)  # 1-высокий, 5-низкий
    timeout: Optional[int] = None  # в секундах
    retry_count: int = Field(default=0, ge=0, le=3)
    max_retries: int = Field(default=2, ge=0, le=3)
    dependencies: List[str] = Field(default_factory=list)  # ID зависимых задач
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None

    @field_validator('timeout')
    @classmethod
    def validate_timeout(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Timeout должен быть положительным числом')
        return v


class TaskResult(BaseModel):
    """Результат выполнения задачи агентом"""
    task_id: str
    agent_type: AgentType
    status: TaskStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration: Optional[float] = None  # в секундах
    output: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metrics: Dict[str, Union[int, float, str]] = Field(default_factory=dict)
    artifacts: List[str] = Field(default_factory=list)  # пути к файлам
    next_actions: List[str] = Field(default_factory=list)
    learning_data: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь для сериализации"""
        return self.model_dump(exclude_none=True)


class AgentCapabilities(BaseModel):
    """Возможности агента"""
    agent_type: AgentType
    name: str
    version: str = "1.0.0"
    description: str
    supported_tasks: List[str] = Field(default_factory=list)
    supported_languages: List[str] = Field(default_factory=list)
    max_concurrent_tasks: int = Field(default=1, ge=1, le=10)
    resource_requirements: Dict[str, str] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    environment_vars: Dict[str, str] = Field(default_factory=dict)
    specializations: List[str] = Field(default_factory=list)
    integrations: List[str] = Field(default_factory=list)
    performance_profile: Dict[str, float] = Field(default_factory=dict)


class AgentStatus(BaseModel):
    """Статус агента в системе"""
    agent_type: AgentType
    current_status: str = "idle"
    is_available: bool = True
    current_tasks: List[str] = Field(default_factory=list)
    completed_tasks: int = Field(default=0)
    failed_tasks: int = Field(default=0)
    uptime: float = Field(default=0.0)
    last_activity: Optional[datetime] = None
    resource_usage: Dict[str, float] = Field(default_factory=dict)
    health_metrics: Dict[str, Any] = Field(default_factory=dict)
    error_rate: float = Field(default=0.0)
    avg_execution_time: float = Field(default=0.0)


class ProgressUpdate(BaseModel):
    """Обновление прогресса выполнения задачи"""
    task_id: str
    agent_type: AgentType
    progress: float = Field(ge=0.0, le=1.0)  # 0.0 - 1.0
    current_step: str
    completed_steps: List[str] = Field(default_factory=list)
    remaining_steps: List[str] = Field(default_factory=list)
    message: Optional[str] = None
    eta_seconds: Optional[float] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# Интерфейсы и протоколы
# =============================================================================

class TimeoutHandler(Protocol):
    """Интерфейс для обработки таймаутов"""
    def handle_timeout(self, task: TaskRequest) -> TaskResult:
        ...


class RetryHandler(Protocol):
    """Интерфейс для логики повторных попыток"""
    def should_retry(self, error: Exception, attempt: int) -> bool:
        ...
    
    def get_retry_delay(self, attempt: int) -> float:
        ...


class LoggingHandler(Protocol):
    """Интерфейс для логирования"""
    def log_task_start(self, agent_type: AgentType, task: TaskRequest):
        ...
    
    def log_task_progress(self, agent_type: AgentType, progress: ProgressUpdate):
        ...
    
    def log_task_complete(self, agent_type: AgentType, result: TaskResult):
        ...
    
    def log_error(self, agent_type: AgentType, error: Exception, context: Dict[str, Any]):
        ...


class ResourceManager(Protocol):
    """Интерфейс управления ресурсами"""
    def allocate_resources(self, agent_type: AgentType, task: TaskRequest) -> bool:
        ...
    
    def release_resources(self, agent_type: AgentType, task_id: str):
        ...
    
    def get_usage_stats(self, agent_type: AgentType) -> Dict[str, float]:
        ...


# =============================================================================
# Реализация обработчиков по умолчанию
# =============================================================================

class DefaultTimeoutHandler:
    """Стандартный обработчик таймаутов"""
    
    def handle_timeout(self, task: TaskRequest) -> TaskResult:
        return TaskResult(
            task_id=task.task_id,
            agent_type=task.agent_type,
            status=TaskStatus.TIMEOUT,
            started_at=task.created_at,
            completed_at=datetime.now(timezone.utc),
            duration=task.timeout,
            errors=[f"Задача превысила timeout {task.timeout}s"],
            output={"timeout": True}
        )


class DefaultRetryHandler:
    """Стандартный обработчик повторных попыток"""
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        # Не повторяем попытки для критических ошибок
        if isinstance(error, (SystemExit, KeyboardInterrupt)):
            return False
        # Максимум 3 попытки
        return attempt < 3
    
    def get_retry_delay(self, attempt: int) -> float:
        # Экспоненциальный backoff
        return min(2 ** attempt, 30)  # Максимум 30 секунд


class AgentLogger:
    """Стандартный логгер агента"""
    
    def __init__(self, agent_type: AgentType):
        self.agent_type = agent_type
        self.logger = logging.getLogger(f"agent.{agent_type.value}")
        
    def log_task_start(self, agent_type: AgentType, task: TaskRequest):
        self.logger.info(f"Начинаю выполнение задачи {task.task_id}: {task.description}")
        
    def log_task_progress(self, agent_type: AgentType, progress: ProgressUpdate):
        pct = int(progress.progress * 100)
        self.logger.info(f"Прогресс {progress.task_id}: {pct}% - {progress.current_step}")
        
    def log_task_complete(self, agent_type: AgentType, result: TaskResult):
        self.logger.info(f"Завершена задача {result.task_id} со статусом {result.status}")
        
    def log_error(self, agent_type: AgentType, error: Exception, context: Dict[str, Any]):
        self.logger.error(f"Ошибка в {agent_type.value}: {str(error)}\nКонтекст: {context}")
        self.logger.debug(traceback.format_exc())


class ResourceManagerImpl:
    """Стандартный менеджер ресурсов"""
    
    def __init__(self):
        self.active_allocations: Dict[str, Dict[str, Any]] = {}
        self.usage_stats: Dict[AgentType, Dict[str, float]] = {}
        
    def allocate_resources(self, agent_type: AgentType, task: TaskRequest) -> bool:
        # Простая реализация - всегда разрешаем
        allocation_id = f"{agent_type.value}_{task.task_id}"
        self.active_allocations[allocation_id] = {
            "agent_type": agent_type,
            "task_id": task.task_id,
            "allocated_at": datetime.now(timezone.utc),
            "timeout": task.timeout
        }
        return True
        
    def release_resources(self, agent_type: AgentType, task_id: str):
        allocation_id = f"{agent_type.value}_{task_id}"
        self.active_allocations.pop(allocation_id, None)
        
    def get_usage_stats(self, agent_type: AgentType) -> Dict[str, float]:
        return self.usage_stats.get(agent_type, {})


# =============================================================================
# Исключения агентов
# =============================================================================

class AgentError(Exception):
    """Базовое исключение агента"""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}


class TaskValidationError(AgentError):
    """Ошибка валидации задачи"""
    pass


class TaskExecutionError(AgentError):
    """Ошибка выполнения задачи"""
    pass


class ResourceError(AgentError):
    """Ошибка управления ресурсами"""
    pass


# =============================================================================
# Базовый класс агента
# =============================================================================

class BaseAgent(ABC):
    """
    Абстрактный базовый класс для всех агентов системы Rebecca-Platform
    
    Обеспечивает:
    - Стандартизированный интерфейс выполнения задач
    - Интеграцию с системой памяти
    - Управление таймаутами и повторными попытками
    - Мониторинг и логирование
    - Управление ресурсами
    """
    
    def __init__(
        self,
        agent_type: AgentType,
        capabilities: AgentCapabilities,
        memory_manager: Optional[MemoryManager] = None,
        context_handler: Optional[ContextHandler] = None,
        timeout_handler: Optional[TimeoutHandler] = None,
        retry_handler: Optional[RetryHandler] = None,
        logger: Optional[LoggingHandler] = None,
        resource_manager: Optional[ResourceManager] = None
    ):
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.memory_manager = memory_manager
        self.context_handler = context_handler
        
        # Инициализация обработчиков
        self.timeout_handler = timeout_handler or DefaultTimeoutHandler()
        self.retry_handler = retry_handler or DefaultRetryHandler()
        self.logger = logger or AgentLogger(agent_type)
        self.resource_manager = resource_manager or ResourceManagerImpl()
        
        # Состояние агента
        self.status = AgentStatus(agent_type=agent_type)
        self.active_tasks: Dict[str, TaskRequest] = {}
        self.task_history: List[TaskResult] = []
        
        # Thread pool для выполнения задач
        self.executor = ThreadPoolExecutor(max_workers=capabilities.max_concurrent_tasks)
        
        # Логирование инициализации
        self.logger.log_task_start(agent_type, TaskRequest(
            agent_type=agent_type,
            task_type="init",
            description=f"Инициализация агента {agent_type.value}"
        ))
        
    # =============================================================================
    # Абстрактные методы (должны быть реализованы в наследниках)
    # =============================================================================
    
    @abstractmethod
    async def execute_task(self, task: TaskRequest) -> TaskResult:
        """
        Выполнение задачи агентом
        Должен быть реализован в наследниках
        """
        pass
    
    # =============================================================================
    # Публичные методы интерфейса
    # =============================================================================
    
    async def execute(self, task: TaskRequest) -> TaskResult:
        """
        Главный метод выполнения задачи с полной обработкой ошибок
        """
        start_time = time.time()
        
        try:
            # Валидация задачи
            self.validate_task(task)
            
            # Проверка доступности
            if not self.status.is_available:
                raise AgentError("Агент недоступен")
            
            # Проверка лимитов
            if len(self.active_tasks) >= self.capabilities.max_concurrent_tasks:
                raise ResourceError("Превышен лимит активных задач")
            
            # Резервирование ресурсов
            if not self.resource_manager.allocate_resources(self.agent_type, task):
                raise ResourceError("Не удалось выделить ресурсы")
            
            # Логирование начала
            self.logger.log_task_start(self.agent_type, task)
            self.active_tasks[task.task_id] = task
            self.status.current_status = "busy"
            self.status.current_tasks.append(task.task_id)
            self.status.last_activity = datetime.now(timezone.utc)
            
            # Обновление контекста в памяти
            if self.memory_manager:
                await self._update_memory_context(task, "task_started")
            
            # Выполнение с таймаутом и ретраями
            result = await self._execute_with_retry(task, start_time)
            
            # Логирование завершения
            self.logger.log_task_complete(self.agent_type, result)
            
            # Обновление статистики
            self._update_agent_stats(result)
            
            # Сохранение результата в память
            if self.memory_manager:
                await self._update_memory_context(task, "task_completed", result)
            
            return result
            
        except (TaskValidationError, ResourceError, AgentError) as e:
            # Ожидаемые ошибки - пробрасываем наверх для правильной обработки
            self.logger.log_error(self.agent_type, e, {"task_id": task.task_id, "error_type": "expected"})
            raise
        except Exception as e:
            # Неожиданные ошибки - логируем и возвращаем как TaskResult
            self.logger.log_error(self.agent_type, e, {"task_id": task.task_id, "error_type": "unexpected"})
            
            # Создание результата с ошибкой
            result = TaskResult(
                task_id=task.task_id,
                agent_type=self.agent_type,
                status=TaskStatus.FAILED,
                started_at=datetime.fromtimestamp(start_time, timezone.utc),
                completed_at=datetime.now(timezone.utc),
                duration=time.time() - start_time,
                errors=[str(e)]
            )
            
            self._update_agent_stats(result)
            
            if self.memory_manager:
                await self._update_memory_context(task, "task_failed", result, error=str(e))
            
            return result
            
        finally:
            # Очистка ресурсов
            self._cleanup_task(task.task_id)
    
    def validate_task(self, task: TaskRequest) -> bool:
        """
        Валидация задачи перед выполнением
        """
        # Базовая валидация
        if not task.description:
            raise TaskValidationError("Описание задачи не может быть пустым")
        
        if task.agent_type != self.agent_type:
            raise TaskValidationError(f"Тип агента не соответствует задаче: {task.agent_type} != {self.agent_type}")
        
        # Проверка поддерживаемых типов задач
        if task.task_type not in self.capabilities.supported_tasks:
            raise TaskValidationError(f"Неподдерживаемый тип задачи: {task.task_type}")
        
        # Валидация таймаута
        if task.timeout and task.timeout > 3600:  # Максимум 1 час
            raise TaskValidationError("Таймаут не может превышать 1 час")
        
        # Проверка зависимостей
        for dep_id in task.dependencies:
            if not self._is_dependency_satisfied(dep_id):
                raise TaskValidationError(f"Зависимость {dep_id} не выполнена")
        
        return True
    
    def get_status(self) -> AgentStatus:
        """
        Получение текущего статуса агента
        """
        now = datetime.now(timezone.utc)
        if self.status.last_activity:
            self.status.uptime = (now - self.status.last_activity).total_seconds()
        else:
            self.status.uptime = 0.0
        self.status.current_tasks = list(self.active_tasks.keys())
        return self.status
    
    def get_capabilities(self) -> AgentCapabilities:
        """
        Получение возможностей агента
        """
        return self.capabilities
    
    def report_progress(self, progress: ProgressUpdate) -> None:
        """
        Обновление прогресса выполнения задачи
        """
        self.logger.log_task_progress(self.agent_type, progress)
        
        # Сохранение прогресса в память
        if self.memory_manager:
            asyncio.create_task(self._store_progress_update(progress))
    
    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """
        Получение результата задачи из истории
        """
        for result in self.task_history:
            if result.task_id == task_id:
                return result
        return None
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Отмена активной задачи
        """
        if task_id in self.active_tasks:
            task = self.active_tasks.pop(task_id)
            
            # Создание результата отмены
            result = TaskResult(
                task_id=task_id,
                agent_type=self.agent_type,
                status=TaskStatus.CANCELLED,
                started_at=task.created_at,
                completed_at=datetime.now(timezone.utc),
                errors=["Задача была отменена пользователем"]
            )
            
            self.task_history.append(result)
            self._cleanup_task(task_id)
            return True
        
        return False
    
    # =============================================================================
    # Внутренние методы
    # =============================================================================
    
    async def _execute_with_retry(self, task: TaskRequest, start_time: float) -> TaskResult:
        """
        Выполнение задачи с обработкой таймаутов и повторных попыток
        """
        attempt = 0
        max_attempts = task.max_retries + 1
        
        while attempt < max_attempts:
            try:
                # Создание асинхронной задачи с таймаутом
                if task.timeout:
                    result = await asyncio.wait_for(
                        self.execute_task(task),
                        timeout=task.timeout
                    )
                else:
                    result = await self.execute_task(task)
                
                # Обновление времени завершения
                result.started_at = datetime.fromtimestamp(start_time, timezone.utc)
                result.completed_at = datetime.now(timezone.utc)
                result.duration = result.completed_at.timestamp() - start_time
                result.status = TaskStatus.COMPLETED
                
                # Добавление в историю
                self.task_history.append(result)
                
                return result
                
            except asyncio.TimeoutError:
                if attempt == max_attempts - 1:
                    # Последняя попытка - возвращаем таймаут
                    return self.timeout_handler.handle_timeout(task)
                else:
                    attempt += 1
                    delay = self.retry_handler.get_retry_delay(attempt)
                    await asyncio.sleep(delay)
                    
            except Exception as e:
                if not self.retry_handler.should_retry(e, attempt):
                    raise
                
                if attempt == max_attempts - 1:
                    raise
                
                attempt += 1
                delay = self.retry_handler.get_retry_delay(attempt)
                await asyncio.sleep(delay)
        
        # Если дошли сюда - что-то пошло не так
        raise TaskExecutionError(f"Не удалось выполнить задачу после {max_attempts} попыток")
    
    def _update_agent_stats(self, result: TaskResult) -> None:
        """
        Обновление статистики агента
        """
        # Добавляем результат в историю
        self.task_history.append(result)
        
        # Пересчитываем статистику на основе всей истории
        completed_tasks = [r for r in self.task_history if r.status == TaskStatus.COMPLETED]
        failed_tasks = [r for r in self.task_history if r.status in [TaskStatus.FAILED, TaskStatus.TIMEOUT]]
        
        self.status.completed_tasks = len(completed_tasks)
        self.status.failed_tasks = len(failed_tasks)
        
        # Обновляем статус агента
        if result.status == TaskStatus.COMPLETED:
            self.status.current_status = "idle"
        elif result.status in [TaskStatus.FAILED, TaskStatus.TIMEOUT]:
            self.status.current_status = "idle"
        
        # Обновление среднего времени выполнения
        if completed_tasks:
            total_time = sum(r.duration or 0 for r in completed_tasks)
            self.status.avg_execution_time = total_time / len(completed_tasks)
        
        # Обновление процента ошибок
        total_tasks = len(self.task_history)
        if total_tasks > 0:
            self.status.error_rate = self.status.failed_tasks / total_tasks
        
        # Обновление активных задач
        self.status.current_tasks = [t for t in self.active_tasks.keys()]
    
    def _cleanup_task(self, task_id: str) -> None:
        """
        Очистка после завершения задачи
        """
        # Освобождение ресурсов
        self.resource_manager.release_resources(self.agent_type, task_id)
        
        # Удаление из активных задач
        self.active_tasks.pop(task_id, None)
        
        # Обновление статуса
        if not self.active_tasks:
            self.status.current_status = "idle"
    
    async def _update_memory_context(
        self, 
        task: TaskRequest, 
        event: str, 
        result: Optional[TaskResult] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Обновление контекста в памяти системы
        """
        if not self.memory_manager:
            return
        
        try:
            context_data = {
                "agent_type": self.agent_type.value,
                "task_id": task.task_id,
                "task_type": task.task_type,
                "event": event,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "description": task.description
            }
            
            if result:
                context_data["result"] = result.to_dict()
            
            if error:
                context_data["error"] = error
            
            # Сохранение в семантический слой памяти
            await self.memory_manager.store(
                layer="SEMANTIC",
                key=f"agent_event_{task.task_id}_{event}",
                value=context_data,
                metadata={
                    "type": "agent_event",
                    "agent_type": self.agent_type.value,
                    "task_id": task.task_id,
                    "event": event
                }
            )
            
        except Exception as e:
            self.logger.log_error(self.agent_type, e, {
                "action": "memory_update",
                "task_id": task.task_id,
                "event": event
            })
    
    async def _store_progress_update(self, progress: ProgressUpdate) -> None:
        """
        Сохранение обновления прогресса в память
        """
        if not self.memory_manager:
            return
        
        try:
            await self.memory_manager.store(
                layer="CORE",
                key=f"progress_{progress.task_id}",
                value=progress.model_dump(),
                metadata={
                    "type": "progress_update",
                    "task_id": progress.task_id,
                    "agent_type": self.agent_type.value
                }
            )
            
        except Exception as e:
            self.logger.log_error(self.agent_type, e, {
                "action": "progress_update",
                "task_id": progress.task_id
            })
    
    def _is_dependency_satisfied(self, dependency_id: str) -> bool:
        """
        Проверка выполнения зависимости
        """
        # Проверяем в истории задач
        for result in self.task_history:
            if result.task_id == dependency_id and result.status == TaskStatus.COMPLETED:
                return True
        
        # Проверяем в активных задачах
        if dependency_id in self.active_tasks:
            return False  # Зависимая задача еще выполняется
        
        return False  # Зависимость не найдена или не выполнена
    
    # =============================================================================
    # Factory методы и утилиты
    # =============================================================================
    
    @classmethod
    def create_agent(
        cls,
        agent_type: AgentType,
        memory_manager: Optional[MemoryManager] = None,
        context_handler: Optional[ContextHandler] = None,
        config_path: Optional[str] = None
    ) -> 'BaseAgent':
        """
        Factory метод для создания агента
        """
        # Загрузка конфигурации
        capabilities = cls._load_capabilities(agent_type, config_path)
        
        # Создание экземпляра (должен быть реализован в наследниках)
        # Здесь происходит полиморфизм - вызов конкретного конструктора
        raise NotImplementedError("Factory метод должен быть реализован в наследниках")
    
    @classmethod
    def _load_capabilities(cls, agent_type: AgentType, config_path: Optional[str]) -> AgentCapabilities:
        """
        Загрузка возможностей агента из конфигурации
        """
        if not config_path:
            # Возвращаем базовые возможности
            return AgentCapabilities(
                agent_type=agent_type,
                name=f"{agent_type.value.title()} Agent",
                description=f"Специализированный агент для задач типа {agent_type.value}",
                supported_tasks=[agent_type.value],
                max_concurrent_tasks=1
            )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            agent_config = config.get('agents', {}).get(agent_type.value, {})
            
            return AgentCapabilities(**agent_config)
            
        except Exception:
            # Fallback к базовой конфигурации
            return cls._load_capabilities(agent_type, None)
    
    def export_agent_state(self) -> Dict[str, Any]:
        """
        Экспорт состояния агента для сохранения
        """
        return {
            "agent_type": self.agent_type.value,
            "capabilities": self.capabilities.model_dump(),
            "status": self.status.model_dump(),
            "active_tasks": {tid: task.model_dump() for tid, task in self.active_tasks.items()},
            "task_history": [result.model_dump() for result in self.task_history],
            "exported_at": datetime.now(timezone.utc).isoformat()
        }
    
    def import_agent_state(self, state: Dict[str, Any]) -> None:
        """
        Импорт состояния агента из сохранения
        """
        try:
            self.status = AgentStatus(**state["status"])
            self.active_tasks = {
                tid: TaskRequest(**task) for tid, task in state["active_tasks"].items()
            }
            self.task_history = [
                TaskResult(**result) for result in state["task_history"]
            ]
            
        except Exception as e:
            raise AgentError(f"Ошибка импорта состояния: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Проверка здоровья агента
        """
        health = {
            "agent_type": self.agent_type.value,
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": {
                "memory_manager": self.memory_manager is not None,
                "context_handler": self.context_handler is not None,
                "active_tasks_count": len(self.active_tasks),
                "completed_tasks": self.status.completed_tasks,
                "failed_tasks": self.status.failed_tasks,
                "error_rate": self.status.error_rate
            }
        }
        
        # Проверка критических компонентов
        if not self.memory_manager:
            health["status"] = "warning"
            health["checks"]["memory_manager"] = False
        
        if self.status.error_rate > 0.5:
            health["status"] = "critical"
        
        return health
    
    def __del__(self):
        """
        Очистка ресурсов при удалении агента
        """
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
        except:
            pass


# =============================================================================
# Функции утилиты
# =============================================================================

def create_task_request(
    agent_type: AgentType,
    task_type: str,
    description: str,
    inputs: Optional[Dict[str, Any]] = None,
    priority: int = 1,
    timeout: Optional[int] = None,
    dependencies: Optional[List[str]] = None,
    **kwargs
) -> TaskRequest:
    """
    Вспомогательная функция для создания TaskRequest
    """
    return TaskRequest(
        agent_type=agent_type,
        task_type=task_type,
        description=description,
        inputs=inputs or {},
        priority=priority,
        timeout=timeout,
        dependencies=dependencies or [],
        **kwargs
    )


def validate_agent_config(config: Dict[str, Any]) -> bool:
    """
    Валидация конфигурации агента
    """
    required_fields = ["agent_type", "name", "description", "supported_tasks"]
    
    for field in required_fields:
        if field not in config:
            return False
    
    # Проверка валидности enum значений
    if "agent_type" in config:
        try:
            AgentType(config["agent_type"])
        except ValueError:
            return False
    
    return True


def generate_agent_report(agent: BaseAgent) -> Dict[str, Any]:
    """
    Генерация отчета о работе агента
    """
    status = agent.get_status()
    capabilities = agent.get_capabilities()
    
    # Анализ производительности
    completed_tasks = [r for r in agent.task_history if r.status == TaskStatus.COMPLETED]
    failed_tasks = [r for r in agent.task_history if r.status in [TaskStatus.FAILED, TaskStatus.TIMEOUT]]
    
    # Среднее время выполнения по типам агентов
    agent_type_times = {}
    for task in completed_tasks:
        if task.duration:
            agent_type_key = task.agent_type.value
            if agent_type_key not in agent_type_times:
                agent_type_times[agent_type_key] = []
            agent_type_times[agent_type_key].append(task.duration)
    
    avg_times = {}
    for agent_type, times in agent_type_times.items():
        avg_times[agent_type] = sum(times) / len(times)
    
    return {
        "agent_info": {
            "type": status.agent_type.value,
            "name": capabilities.name,
            "version": capabilities.version,
            "uptime_seconds": status.uptime
        },
        "performance": {
            "total_tasks": len(agent.task_history),
            "completed_tasks": len(completed_tasks),
            "failed_tasks": len(failed_tasks),
            "success_rate": len(completed_tasks) / len(agent.task_history) if agent.task_history else 0,
            "error_rate": status.error_rate,
            "avg_execution_time": status.avg_execution_time,
            "performance_by_task_type": avg_times
        },
        "current_state": {
            "is_available": status.is_available,
            "current_status": status.current_status,
            "active_tasks": len(status.current_tasks),
            "last_activity": status.last_activity.isoformat() if status.last_activity else None
        },
        "resources": status.resource_usage,
        "health": status.health_metrics
    }