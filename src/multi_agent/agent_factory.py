"""
Фабрика создания специализированных агентов для Rebecca-Platform
"""

import os
import yaml
import logging
import asyncio
from typing import Dict, Any, Optional, Type
from pathlib import Path
from datetime import datetime, timezone

from .base_agent import (
    BaseAgent, AgentType, AgentCapabilities, TaskRequest, TaskResult,
    TaskStatus, ProgressUpdate
)
from memory_manager.memory_manager import MemoryManager
from orchestrator.main_workflow import ContextHandler
from rebecca.meta_agent import RebeccaMetaAgent


class AgentFactory:
    """
    Фабрика для создания и управления специализированными агентами
    
    Поддерживает создание агентов разных типов:
    - BackendAgent: для разработки backend API
    - FrontendAgent: для разработки пользовательских интерфейсов
    - MLEngineerAgent: для машинного обучения и AI
    - QAAnalystAgent: для тестирования и контроля качества
    - DevOpsAgent: для DevOps и инфраструктуры
    """
    
    def __init__(
        self,
        memory_manager: Optional[MemoryManager] = None,
        context_handler: Optional[ContextHandler] = None,
        config_path: str = "config/agents/agents.yaml"
    ):
        self.memory_manager = memory_manager
        self.context_handler = context_handler
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        
        # Загрузка конфигурации
        self.global_config = self._load_global_config()
        
        # Регистрация классов агентов
        self.agent_classes: Dict[AgentType, Type[BaseAgent]] = {}
        self._register_agent_classes()
        
        # Созданные агенты
        self.agents: Dict[AgentType, BaseAgent] = {}
        
        # Статистика использования
        self.usage_stats = {
            "total_created": 0,
            "total_executed": 0,
            "success_rate": 0.0,
            "avg_execution_time": 0.0
        }
    
    def _load_global_config(self) -> Dict[str, Any]:
        """Загрузка глобальной конфигурации агентов"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Конфигурационный файл {self.config_path} не найден")
            # Возвращаем базовую конфигурацию если файл не найден
            return {"agents": {}}
        except Exception as e:
            self.logger.error(f"Ошибка загрузки конфигурации: {e}")
            return {}
    
    def _register_agent_classes(self):
        """Регистрация классов агентов"""
        # Будем регистрировать по мере реализации
        self.agent_classes = {}
    
    def create_agent(self, agent_type: AgentType, agent_id: Optional[str] = None) -> BaseAgent:
        """
        Создание агента заданного типа
        
        Args:
            agent_type: Тип агента для создания
            agent_id: Уникальный идентификатор агента (опционально)
            
        Returns:
            Созданный экземпляр агента
        """
        # Проверка поддерживаемого типа
        if agent_type not in self.agent_classes:
            raise ValueError(f"Неподдерживаемый тип агента: {agent_type}")
        
        # Проверка лимитов
        if len(self.agents) >= self.global_config.get('global_settings', {}).get('limits', {}).get('max_total_agents', 10):
            raise RuntimeError("Превышен лимит общего количества агентов")
        
        # Загрузка конфигурации агента
        agent_config = self._load_agent_config(agent_type)
        
        # Создание агента
        agent_class = self.agent_classes[agent_type]
        agent = agent_class(
            agent_type=agent_type,
            capabilities=agent_config['capabilities'],
            memory_manager=self.memory_manager,
            context_handler=self.context_handler
        )
        
        # Установка идентификатора
        if agent_id:
            agent.agent_id = agent_id
        
        # Сохранение в реестре
        self.agents[agent_type] = agent
        
        # Обновление статистики
        self.usage_stats['total_created'] += 1
        
        self.logger.info(f"Создан агент {agent_type.value} с ID {agent_id or 'auto-generated'}")
        
        return agent
    
    def _load_agent_config(self, agent_type: AgentType) -> Dict[str, Any]:
        """Загрузка конфигурации агента"""
        try:
            # Поиск конфигурации в глобальном файле
            agent_config_data = self.global_config.get('agents', {}).get(f'{agent_type.value}_agent', {})
            
            if not agent_config_data:
                # Поиск отдельного файла конфигурации
                config_path = f"config/agents/{agent_type.value}_agent.yaml"
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        agent_config_data = yaml.safe_load(f)
                except FileNotFoundError:
                    # Возвращаем базовую конфигурацию
                    agent_config_data = {}
            
            # Извлечение capabilities
            capabilities_data = agent_config_data.get(agent_type.value, agent_config_data)
            
            # Создание объекта AgentCapabilities
            capabilities = AgentCapabilities(**capabilities_data)
            
            return {
                'capabilities': capabilities,
                'config': agent_config_data
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки конфигурации для {agent_type.value}: {e}")
            # Возвращаем базовую конфигурацию
            return {
                'capabilities': AgentCapabilities(
                    agent_type=agent_type,
                    name=f"{agent_type.value.title()} Agent",
                    description=f"Специализированный агент для {agent_type.value}",
                    supported_tasks=[agent_type.value]
                ),
                'config': {}
            }
    
    def get_agent(self, agent_type: AgentType) -> Optional[BaseAgent]:
        """Получение агента по типу"""
        return self.agents.get(agent_type)
    
    def list_agents(self) -> Dict[AgentType, Dict[str, Any]]:
        """Получение списка всех созданных агентов"""
        result = {}
        for agent_type, agent in self.agents.items():
            result[agent_type] = {
                'status': agent.get_status().current_status,
                'is_available': agent.get_status().is_available,
                'active_tasks': len(agent.active_tasks),
                'completed_tasks': agent.status.completed_tasks,
                'capabilities': agent.get_capabilities().name
            }
        return result
    
    def remove_agent(self, agent_type: AgentType) -> bool:
        """Удаление агента"""
        if agent_type in self.agents:
            # Graceful shutdown
            agent = self.agents[agent_type]
            try:
                # Завершение активных задач
                for task_id in list(agent.active_tasks.keys()):
                    agent.cancel_task(task_id)
                
                # Удаление из реестра
                del self.agents[agent_type]
                
                self.logger.info(f"Удален агент {agent_type.value}")
                return True
                
            except Exception as e:
                self.logger.error(f"Ошибка при удалении агента {agent_type.value}: {e}")
                return False
        
        return False
    
    def assign_task_to_agent(self, task: TaskRequest) -> TaskRequest:
        """
        Назначение задачи подходящему агенту
        
        Автоматически выбирает агента на основе:
        1. Возможностей агента (поддерживает ли задачу)
        2. Текущей загрузки агента
        3. Приоритета агента
        """
        # Поиск подходящих агентов
        suitable_agents = []
        
        for agent_type, agent in self.agents.items():
            capabilities = agent.get_capabilities()
            
            # Проверка поддержки типа задачи
            if task.task_type not in capabilities.supported_tasks:
                continue
            
            # Проверка доступности
            if not agent.get_status().is_available:
                continue
            
            # Проверка лимитов загрузки
            if len(agent.active_tasks) >= capabilities.max_concurrent_tasks:
                continue
            
            # Расчет рейтинга агента
            rating = self._calculate_agent_rating(agent, task)
            suitable_agents.append((rating, agent))
        
        if not suitable_agents:
            raise ValueError(f"Нет подходящих агентов для задачи: {task.task_type}")
        
        # Сортировка по рейтингу (убывание)
        suitable_agents.sort(key=lambda x: x[0], reverse=True)
        
        # Выбор лучшего агента
        _, selected_agent = suitable_agents[0]
        
        # Установка типа агента в задаче
        task.agent_type = selected_agent.agent_type
        
        self.logger.info(f"Задача {task.task_id} назначена агенту {selected_agent.agent_type.value}")
        
        return task
    
    def _calculate_agent_rating(self, agent: BaseAgent, task: TaskRequest) -> float:
        """
        Расчет рейтинга агента для конкретной задачи
        
        Учитывает:
        - Специализацию агента (0-1)
        - Текущую загрузку (0-1)
        - Историю успешности (0-1)
        - Приоритет агента (0-1)
        """
        capabilities = agent.get_capabilities()
        status = agent.get_status()
        
        # Специализация (больше = лучше)
        specialization_score = len([t for t in capabilities.supported_tasks if task.task_type == t]) / max(len(capabilities.supported_tasks), 1)
        
        # Доступность (меньше активных задач = лучше)
        max_tasks = capabilities.max_concurrent_tasks
        availability_score = 1 - (len(agent.active_tasks) / max_tasks) if max_tasks > 0 else 0
        
        # История успешности
        total_tasks = status.completed_tasks + status.failed_tasks
        success_score = status.completed_tasks / total_tasks if total_tasks > 0 else 0.5
        
        # Приоритет (высокий приоритет = лучший рейтинг)
        priority_weights = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4, 5: 0.2}
        priority_score = priority_weights.get(task.priority, 0.5)
        
        # Взвешенный расчет
        rating = (
            specialization_score * 0.4 +
            availability_score * 0.3 +
            success_score * 0.2 +
            priority_score * 0.1
        )
        
        return rating
    
    def get_factory_stats(self) -> Dict[str, Any]:
        """Получение статистики фабрики"""
        total_agents = len(self.agents)
        total_active_tasks = sum(len(agent.active_tasks) for agent in self.agents.values())
        total_completed_tasks = sum(agent.status.completed_tasks for agent in self.agents.values())
        total_failed_tasks = sum(agent.status.failed_tasks for agent in self.agents.values())
        
        success_rate = total_completed_tasks / (total_completed_tasks + total_failed_tasks) if (total_completed_tasks + total_failed_tasks) > 0 else 0
        
        avg_execution_time = sum(agent.status.avg_execution_time for agent in self.agents.values()) / total_agents if total_agents > 0 else 0
        
        return {
            "total_agents": total_agents,
            "total_active_tasks": total_active_tasks,
            "total_completed_tasks": total_completed_tasks,
            "total_failed_tasks": total_failed_tasks,
            "success_rate": success_rate,
            "avg_execution_time": avg_execution_time,
            "usage_stats": self.usage_stats,
            "available_agents": [agent_type.value for agent_type in self.agents.keys() if self.agents[agent_type].get_status().is_available]
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Проверка здоровья фабрики агентов"""
        health_status = "healthy"
        issues = []
        
        # Проверка создания агентов
        if len(self.agents) == 0:
            issues.append("Нет созданных агентов")
            health_status = "warning"
        
        # Проверка загрузки агентов
        overloaded_agents = []
        for agent_type, agent in self.agents.items():
            capabilities = agent.get_capabilities()
            if len(agent.active_tasks) >= capabilities.max_concurrent_tasks:
                overloaded_agents.append(agent_type.value)
        
        if overloaded_agents:
            issues.append(f"Перегруженные агенты: {', '.join(overloaded_agents)}")
            if len(overloaded_agents) > len(self.agents) // 2:
                health_status = "critical"
        
        # Проверка ошибок
        high_error_agents = []
        for agent_type, agent in self.agents.items():
            if agent.get_status().error_rate > 0.5:
                high_error_agents.append(agent_type.value)
        
        if high_error_agents:
            issues.append(f"Агенты с высокой ошибкостью: {', '.join(high_error_agents)}")
            health_status = "warning"
        
        return {
            "status": health_status,
            "timestamp": agent_type.value if 'agent_type' in locals() else "N/A",  # будет перезаписано
            "total_agents": len(self.agents),
            "healthy_agents": len(self.agents) - len(high_error_agents),
            "issues": issues,
            "recommendations": self._generate_recommendations(issues)
        }
    
    def _generate_recommendations(self, issues: list) -> list:
        """Генерация рекомендаций по улучшению"""
        recommendations = []
        
        if not self.agents:
            recommendations.append("Создать хотя бы один агент каждого типа")
        
        if any("Перегруженные" in issue for issue in issues):
            recommendations.append("Увеличить количество агентов или распределить нагрузку")
        
        if any("высокой ошибкостью" in issue for issue in issues):
            recommendations.append("Проанализировать причины ошибок и оптимизировать логику агентов")
        
        return recommendations
    
    def cleanup(self):
        """Очистка ресурсов фабрики"""
        # Удаление всех агентов
        for agent_type in list(self.agents.keys()):
            self.remove_agent(agent_type)
        
        self.logger.info("Фабрика агентов очищена")


# =============================================================================
# Абстрактные классы специализированных агентов
# =============================================================================

class BackendAgent(BaseAgent):
    """Базовый класс для Backend агента"""
    
    async def execute_task(self, task: TaskRequest) -> TaskResult:
        """Реализация выполнения задач backend разработки"""
        # TODO: Реализовать логику выполнения backend задач
        await asyncio.sleep(2)  # Заглушка
        
        return TaskResult(
            task_id=task.task_id,
            agent_type=self.agent_type,
            status=TaskStatus.COMPLETED,
            started_at=task.created_at,
            completed_at=datetime.now(timezone.utc),
            duration=2.0,
            output={"completed": True, "result": "Backend task completed"}
        )


class FrontendAgent(BaseAgent):
    """Базовый класс для Frontend агента"""
    
    async def execute_task(self, task: TaskRequest) -> TaskResult:
        """Реализация выполнения задач frontend разработки"""
        # TODO: Реализовать логику выполнения frontend задач
        await asyncio.sleep(1.5)  # Заглушка
        
        return TaskResult(
            task_id=task.task_id,
            agent_type=self.agent_type,
            status=TaskStatus.COMPLETED,
            started_at=task.created_at,
            completed_at=datetime.now(timezone.utc),
            duration=1.5,
            output={"completed": True, "result": "Frontend task completed"}
        )


class MLEngineerAgent(BaseAgent):
    """Базовый класс для ML Engineer агента"""
    
    async def execute_task(self, task: TaskRequest) -> TaskResult:
        """Реализация выполнения задач ML"""
        # TODO: Реализовать логику выполнения ML задач
        await asyncio.sleep(5)  # Заглушка (ML задачи обычно дольше)
        
        return TaskResult(
            task_id=task.task_id,
            agent_type=self.agent_type,
            status=TaskStatus.COMPLETED,
            started_at=task.created_at,
            completed_at=datetime.now(timezone.utc),
            duration=5.0,
            output={"completed": True, "result": "ML task completed"}
        )


class QAAnalystAgent(BaseAgent):
    """Базовый класс для QA Analyst агента"""
    
    async def execute_task(self, task: TaskRequest) -> TaskResult:
        """Реализация выполнения задач QA"""
        # TODO: Реализовать логику выполнения QA задач
        await asyncio.sleep(3)  # Заглушка
        
        return TaskResult(
            task_id=task.task_id,
            agent_type=self.agent_type,
            status=TaskStatus.COMPLETED,
            started_at=task.created_at,
            completed_at=datetime.now(timezone.utc),
            duration=3.0,
            output={"completed": True, "result": "QA task completed"}
        )


class DevOpsAgent(BaseAgent):
    """Базовый класс для DevOps агента"""
    
    async def execute_task(self, task: TaskRequest) -> TaskResult:
        """Реализация выполнения задач DevOps"""
        # TODO: Реализовать логику выполнения DevOps задач
        await asyncio.sleep(4)  # Заглушка
        
        return TaskResult(
            task_id=task.task_id,
            agent_type=self.agent_type,
            status=TaskStatus.COMPLETED,
            started_at=task.created_at,
            completed_at=datetime.now(timezone.utc),
            duration=4.0,
            output={"completed": True, "result": "DevOps task completed"}
        )


# Обновление регистрации классов в фабрике
AgentFactory._register_agent_classes = lambda self: setattr(
    self, 'agent_classes', {
        AgentType.BACKEND: BackendAgent,
        AgentType.FRONTEND: FrontendAgent,
        AgentType.ML_ENGINEER: MLEngineerAgent,
        AgentType.QA_ANALYST: QAAnalystAgent,
        AgentType.DEVOPS: DevOpsAgent
    }
)