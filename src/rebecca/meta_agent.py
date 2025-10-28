"""
Полноценный мета-агент Rebecca для интеллектуального управления агентной экосистемой.

Обеспечивает:
- Поглощение и анализ различных типов источников
- Планирование задач для специализированных агентов  
- Генерацию инструкций и плейбуков
- Координацию выполнения задач
- Интеграцию с системой памяти и контекстом
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable

from memory_manager.memory_manager_interface import MemoryManager, MemoryLayer, MemoryFilter
from memory_manager.adaptive_blueprint import AdaptiveBlueprintTracker
from ingest.loader import IngestPipeline
from orchestrator.context_handler import ContextHandler
from blueprint_generator.main import run_agent


# Настройка логгера
logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Приоритеты задач."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


class TaskStatus(Enum):
    """Статусы задач."""
    PENDING = "pending"
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentSpecialization(Enum):
    """Специализации агентов."""
    BACKEND = "backend"
    FRONTEND = "frontend"
    MACHINE_LEARNING = "ml"
    QUALITY_ASSURANCE = "qa"
    DEVOPS = "devops"
    RESEARCH = "research"
    DATA_SCIENCE = "data_science"
    SECURITY = "security"
    GENERAL = "general"


class TaskType(Enum):
    """Типы задач."""
    DEVELOPMENT = "development"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    DEPLOYMENT = "deployment"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    OPTIMIZATION = "optimization"
    MONITORING = "monitoring"


@dataclass
class TaskPlan:
    """План выполнения задач."""
    task_id: str
    title: str
    description: str
    task_type: TaskType
    priority: TaskPriority
    complexity_score: float = 0.0
    estimated_duration: int = 0  # в минутах
    required_skills: List[AgentSpecialization] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь."""
        return {
            'task_id': self.task_id,
            'title': self.title,
            'description': self.description,
            'task_type': self.task_type.value,
            'priority': self.priority.name,
            'complexity_score': self.complexity_score,
            'estimated_duration': self.estimated_duration,
            'required_skills': [skill.value for skill in self.required_skills],
            'dependencies': self.dependencies,
            'prerequisites': self.prerequisites,
            'success_criteria': self.success_criteria,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskPlan':
        """Создание из словаря."""
        return cls(
            task_id=data['task_id'],
            title=data['title'],
            description=data['description'],
            task_type=TaskType(data['task_type']),
            priority=TaskPriority[data['priority']],
            complexity_score=data.get('complexity_score', 0.0),
            estimated_duration=data.get('estimated_duration', 0),
            required_skills=[AgentSpecialization(skill) for skill in data.get('required_skills', [])],
            dependencies=data.get('dependencies', []),
            prerequisites=data.get('prerequisites', []),
            success_criteria=data.get('success_criteria', []),
            metadata=data.get('metadata', {}),
            created_at=datetime.fromisoformat(data['created_at'])
        )


@dataclass
class AgentAssignment:
    """Назначение агента на задачу."""
    assignment_id: str
    task_plan: TaskPlan
    agent_type: AgentSpecialization
    agent_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PLANNED
    progress: float = 0.0
    quality_score: float = 0.0
    iterations: int = 0
    context_data: Dict[str, Any] = field(default_factory=dict)
    error_logs: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь."""
        return {
            'assignment_id': self.assignment_id,
            'task_plan': self.task_plan.to_dict(),
            'agent_type': self.agent_type.value,
            'agent_id': self.agent_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'status': self.status.value,
            'progress': self.progress,
            'quality_score': self.quality_score,
            'iterations': self.iterations,
            'context_data': self.context_data,
            'error_logs': self.error_logs
        }


@dataclass
class PlaybookStep:
    """Шаг выполнения в плейбуке."""
    step_id: str
    step_number: int
    title: str
    description: str
    action_type: str
    agent_instruction: str
    expected_output: str
    success_criteria: List[str] = field(default_factory=list)
    timeout_minutes: int = 60
    retry_count: int = 0
    max_retries: int = 3
    conditional_logic: Optional[Dict[str, Any]] = None
    data_flow: Dict[str, str] = field(default_factory=dict)  # input -> output mapping
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь."""
        return {
            'step_id': self.step_id,
            'step_number': self.step_number,
            'title': self.title,
            'description': self.description,
            'action_type': self.action_type,
            'agent_instruction': self.agent_instruction,
            'expected_output': self.expected_output,
            'success_criteria': self.success_criteria,
            'timeout_minutes': self.timeout_minutes,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'conditional_logic': self.conditional_logic,
            'data_flow': self.data_flow
        }


@dataclass
class ResourceAllocation:
    """Выделение ресурсов для выполнения задач."""
    allocation_id: str
    resource_type: str
    resource_id: str
    task_id: str
    agent_id: str
    allocated_at: datetime
    capacity_used: float = 0.0
    efficiency_score: float = 1.0
    cost_estimate: float = 0.0
    utilization_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь."""
        return {
            'allocation_id': self.allocation_id,
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'task_id': self.task_id,
            'agent_id': self.agent_id,
            'allocated_at': self.allocated_at.isoformat(),
            'capacity_used': self.capacity_used,
            'efficiency_score': self.efficiency_score,
            'cost_estimate': self.cost_estimate,
            'utilization_metrics': self.utilization_metrics
        }


@dataclass
class MetaAgentConfig:
    """Конфигурация мета-агента."""
    max_concurrent_tasks: int = 10
    default_timeout_minutes: int = 60
    enable_auto_scaling: bool = True
    enable_failover: bool = True
    quality_threshold: float = 0.8
    complexity_weight: float = 0.3
    priority_weight: float = 0.4
    dependency_weight: float = 0.3
    resource_optimization: bool = True
    enable_learning: bool = True
    memory_retention_days: int = 30
    enable_proactive_planning: bool = True


class TaskAnalyzer:
    """Анализатор задач для определения сложности и требований."""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
    
    async def analyze_task_complexity(self, task_description: str, context: Dict[str, Any] = None) -> float:
        """Анализирует сложность задачи."""
        try:
            # Базовые факторы сложности
            complexity_factors = {
                'word_count': len(task_description.split()) * 0.01,
                'technical_terms': self._count_technical_terms(task_description) * 0.1,
                'integration_points': context.get('integration_count', 0) * 0.2 if context else 0,
                'data_sources': context.get('data_source_count', 0) * 0.15 if context else 0,
                'validation_requirements': context.get('validation_rules', 0) * 0.1 if context else 0
            }
            
            # Анализ на основе истории
            historical_complexity = await self._get_historical_complexity(task_description)
            
            # Семантический анализ
            semantic_complexity = await self._analyze_semantic_complexity(task_description)
            
            # Итоговый расчет сложности
            base_complexity = sum(complexity_factors.values())
            final_complexity = min(1.0, (base_complexity + semantic_complexity + historical_complexity) / 3)
            
            logger.debug(f"Анализ сложности задачи: {final_complexity:.3f}")
            return final_complexity
            
        except Exception as e:
            logger.error(f"Ошибка анализа сложности задачи: {e}")
            return 0.5  # Средняя сложность по умолчанию
    
    def determine_required_skills(self, task_description: str, task_type: TaskType) -> List[AgentSpecialization]:
        """Определяет необходимые навыки для задачи."""
        skill_keywords = {
            AgentSpecialization.BACKEND: [
                'api', 'server', 'база данных', 'database', 'backend', 'python', 'java', 'node.js',
                'микросервис', 'microservice', 'sql', 'orm', 'rest', 'graphql'
            ],
            AgentSpecialization.FRONTEND: [
                'ui', 'ux', 'interface', 'веб', 'web', 'html', 'css', 'javascript', 'react',
                'angular', 'vue', 'frontend', 'пользовательский интерфейс', 'responsive'
            ],
            AgentSpecialization.MACHINE_LEARNING: [
                'ml', 'ai', 'model', 'модель', 'dataset', 'обучение', 'training', 'prediction',
                'classification', 'regression', 'neural', 'algorithm', 'tensorflow', 'pytorch'
            ],
            AgentSpecialization.QA: [
                'test', 'тест', 'quality', 'качество', 'bug', 'issue', 'validation', 'валидация',
                'automation', 'automation', 'coverage', 'integration testing', 'unit testing'
            ],
            AgentSpecialization.DEVOPS: [
                'deploy', 'развертывание', 'docker', 'kubernetes', 'ci/cd', 'pipeline', 'monitoring',
                'production', 'scalability', 'container', 'cloud', 'инфраструктура'
            ],
            AgentSpecialization.SECURITY: [
                'security', 'безопасность', 'authentication', 'authorization', 'encryption',
                'шифрование', 'vulnerability', 'audit', 'compliance', 'gdpr'
            ]
        }
        
        description_lower = task_description.lower()
        required_skills = set()
        
        # Определяем навыки по ключевым словам
        for skill, keywords in skill_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                required_skills.add(skill)
        
        # Добавляем общие навыки по типу задачи
        if task_type == TaskType.DEVELOPMENT:
            required_skills.update([AgentSpecialization.BACKEND, AgentSpecialization.FRONTEND])
        elif task_type == TaskType.DATA_SCIENCE:
            required_skills.update([AgentSpecialization.MACHINE_LEARNING, AgentSpecialization.RESEARCH])
        elif task_type == TaskType.DEPLOYMENT:
            required_skills.add(AgentSpecialization.DEVOPS)
        
        return list(required_skills)
    
    def estimate_duration(self, task_description: str, complexity_score: float, 
                         required_skills: List[AgentSpecialization]) -> int:
        """Оценивает время выполнения задачи."""
        # Базовое время по сложности
        base_duration = {
            TaskPriority.CRITICAL: 30,  # минут
            TaskPriority.HIGH: 60,
            TaskPriority.MEDIUM: 120,
            TaskPriority.LOW: 240,
            TaskPriority.BACKGROUND: 480
        }
        
        # Корректировка по сложности
        complexity_multiplier = 0.5 + (complexity_score * 2)
        
        # Корректировка по количеству навыков
        skill_multiplier = 1.0 + (len(required_skills) - 1) * 0.3
        
        # Базовое время в зависимости от типа задачи
        type_bases = {
            TaskType.DEVELOPMENT: 90,
            TaskType.RESEARCH: 120,
            TaskType.ANALYSIS: 60,
            TaskType.DEPLOYMENT: 45,
            TaskType.TESTING: 30,
            TaskType.DOCUMENTATION: 40,
            TaskType.OPTIMIZATION: 75,
            TaskType.MONITORING: 20
        }
        
        estimated_duration = int(
            type_bases.get(TaskType.DEVELOPMENT, 60) * 
            complexity_multiplier * 
            skill_multiplier
        )
        
        return max(15, estimated_duration)  # Минимум 15 минут
    
    def _count_technical_terms(self, text: str) -> int:
        """Подсчитывает технические термины."""
        technical_terms = {
            'api', 'sdk', 'cli', 'orm', 'crud', 'rest', 'soap', 'graphql', 'json', 'xml',
            'docker', 'kubernetes', 'microservice', 'pipeline', 'ci/cd', 'deployment',
            'database', 'sql', 'nosql', 'redis', 'elasticsearch', 'mongodb', 'postgresql',
            'react', 'angular', 'vue', 'node.js', 'python', 'java', 'go', 'rust',
            'tensorflow', 'pytorch', 'sklearn', 'numpy', 'pandas', 'matplotlib'
        }
        
        text_lower = text.lower()
        return sum(1 for term in technical_terms if term in text_lower)
    
    async def _get_historical_complexity(self, task_description: str) -> float:
        """Получает сложность на основе исторических данных."""
        try:
            # Поиск похожих задач в памяти
            similar_tasks = await self.memory_manager.retrieve(
                MemoryLayer.SEMANTIC,
                {"query": task_description, "similarity_threshold": 0.7}
            )
            
            if similar_tasks:
                avg_complexity = sum(
                    task.metadata.get('complexity_score', 0.5) 
                    for task in similar_tasks
                ) / len(similar_tasks)
                return avg_complexity
            
            return 0.5  # Средняя сложность по умолчанию
            
        except Exception as e:
            logger.warning(f"Не удалось получить историческую сложность: {e}")
            return 0.5
    
    async def _analyze_semantic_complexity(self, task_description: str) -> float:
        """Анализирует семантическую сложность."""
        try:
            # Простой анализ структуры предложений
            sentences = task_description.split('.')
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            
            # Сложность на основе длины предложений
            semantic_complexity = min(1.0, avg_sentence_length / 25)
            
            return semantic_complexity
            
        except Exception as e:
            logger.warning(f"Ошибка семантического анализа: {e}")
            return 0.5


class ResourceOptimizer:
    """Оптимизатор ресурсов для выполнения задач."""
    
    def __init__(self, config: MetaAgentConfig):
        self.config = config
        self.resource_pools: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.utilization_history: deque = deque(maxlen=100)
    
    async def optimize_resource_allocation(self, tasks: List[TaskPlan], 
                                         available_agents: Dict[str, AgentSpecialization]) -> List[ResourceAllocation]:
        """Оптимизирует распределение ресурсов."""
        try:
            allocations = []
            
            # Группируем задачи по типам и приоритетам
            task_groups = self._group_tasks_by_criteria(tasks)
            
            for group_criteria, task_group in task_groups.items():
                group_allocations = await self._allocate_group_resources(
                    task_group, available_agents, group_criteria
                )
                allocations.extend(group_allocations)
            
            # Валидация распределения
            validated_allocations = self._validate_allocations(allocations)
            
            logger.info(f"Оптимизировано распределение {len(validated_allocations)} ресурсов")
            return validated_allocations
            
        except Exception as e:
            logger.error(f"Ошибка оптимизации ресурсов: {e}")
            return []
    
    def _group_tasks_by_criteria(self, tasks: List[TaskPlan]) -> Dict[Tuple, List[TaskPlan]]:
        """Группирует задачи по критериям."""
        groups = defaultdict(list)
        
        for task in tasks:
            # Группировка по типу задачи и приоритету
            group_key = (task.task_type, task.priority)
            groups[group_key].append(task)
        
        # Сортируем внутри групп по сложности
        for group_tasks in groups.values():
            group_tasks.sort(key=lambda t: t.complexity_score, reverse=True)
        
        return dict(groups)
    
    async def _allocate_group_resources(self, tasks: List[TaskPlan], 
                                      available_agents: Dict[str, AgentSpecialization],
                                      group_criteria: Tuple) -> List[ResourceAllocation]:
        """Распределяет ресурсы для группы задач."""
        allocations = []
        
        for task in tasks:
            # Находим подходящих агентов
            suitable_agents = self._find_suitable_agents(task, available_agents)
            
            if suitable_agents:
                # Выбираем лучшего агента
                best_agent = self._select_best_agent(task, suitable_agents)
                
                # Создаем выделение ресурсов
                allocation = ResourceAllocation(
                    allocation_id=str(uuid.uuid4()),
                    resource_type="agent",
                    resource_id=best_agent,
                    task_id=task.task_id,
                    agent_id=best_agent,
                    allocated_at=datetime.now(timezone.utc),
                    capacity_used=self._calculate_capacity_usage(task),
                    cost_estimate=self._estimate_cost(task),
                    utilization_metrics=self._get_utilization_metrics()
                )
                allocations.append(allocation)
        
        return allocations
    
    def _find_suitable_agents(self, task: TaskPlan, 
                            available_agents: Dict[str, AgentSpecialization]) -> List[str]:
        """Находит подходящих агентов для задачи."""
        suitable_agents = []
        
        # Фильтр по требуемым навыкам
        required_skills = set(task.required_skills)
        
        for agent_id, agent_specialization in available_agents.items():
            if agent_specialization in required_skills or agent_specialization == AgentSpecialization.GENERAL:
                suitable_agents.append(agent_id)
        
        return suitable_agents
    
    def _select_best_agent(self, task: TaskPlan, suitable_agents: List[str]) -> str:
        """Выбирает лучшего агента из подходящих."""
        if not suitable_agents:
            return suitable_agents[0] if suitable_agents else "default_agent"
        
        # Простая эвристика: выбираем первого подходящего агента
        # В реальной реализации можно добавить метрики производительности
        return suitable_agents[0]
    
    def _calculate_capacity_usage(self, task: TaskPlan) -> float:
        """Вычисляет использование ресурсов."""
        # Простой расчет на основе сложности
        base_usage = 0.2  # 20% базовое использование
        complexity_factor = task.complexity_score * 0.6  # До 60% дополнительно
        
        return min(1.0, base_usage + complexity_factor)
    
    def _estimate_cost(self, task: TaskPlan) -> float:
        """Оценивает стоимость выполнения задачи."""
        # Базовая стоимость на основе приоритета
        base_costs = {
            TaskPriority.CRITICAL: 100,
            TaskPriority.HIGH: 50,
            TaskPriority.MEDIUM: 25,
            TaskPriority.LOW: 10,
            TaskPriority.BACKGROUND: 5
        }
        
        # Корректировка по сложности и времени
        base_cost = base_costs.get(task.priority, 25)
        complexity_multiplier = 1 + (task.complexity_score * 2)
        duration_multiplier = max(0.5, task.estimated_duration / 60)  # Корректировка по времени
        
        return base_cost * complexity_multiplier * duration_multiplier
    
    def _get_utilization_metrics(self) -> Dict[str, float]:
        """Возвращает метрики использования."""
        return {
            "cpu_usage": 0.5,
            "memory_usage": 0.3,
            "network_io": 0.2,
            "storage_io": 0.1
        }
    
    def _validate_allocations(self, allocations: List[ResourceAllocation]) -> List[ResourceAllocation]:
        """Валидирует распределение ресурсов."""
        validated = []
        
        # Проверяем лимиты одновременных задач
        agent_loads = defaultdict(float)
        
        for allocation in allocations:
            current_load = agent_loads[allocation.agent_id]
            new_load = current_load + allocation.capacity_used
            
            # Проверяем, не превышает ли загрузка 100%
            if new_load <= 1.0:
                agent_loads[allocation.agent_id] = new_load
                validated.append(allocation)
            else:
                logger.warning(f"Отклонено распределение для агента {allocation.agent_id}: "
                             f"превышен лимит нагрузки")
        
        return validated


class PlaybookGenerator:
    """Генератор плейбуков для выполнения задач."""
    
    def __init__(self, memory_manager: MemoryManager, blueprint_tracker: AdaptiveBlueprintTracker):
        self.memory_manager = memory_manager
        self.blueprint_tracker = blueprint_tracker
    
    async def generate_playbook(self, task_plan: TaskPlan, 
                              assignment: AgentAssignment,
                              context_data: Dict[str, Any] = None) -> List[PlaybookStep]:
        """Генерирует плейбук для выполнения задачи."""
        try:
            logger.info(f"Генерируем плейбук для задачи {task_plan.task_id}")
            
            # Получаем исторические данные и лучшие практики
            historical_context = await self._get_historical_context(task_plan)
            best_practices = await self._get_best_practices(task_plan)
            
            # Генерируем базовые шаги
            base_steps = await self._generate_base_steps(task_plan)
            
            # Добавляем проверки качества
            quality_steps = await self._generate_quality_steps(task_plan)
            
            # Объединяем все шаги
            all_steps = base_steps + quality_steps
            
            # Добавляем контекстные шаги
            if context_data:
                context_steps = await self._generate_context_steps(task_plan, context_data)
                all_steps.extend(context_steps)
            
            # Сортируем по номеру шага
            all_steps.sort(key=lambda s: s.step_number)
            
            # Валидируем плейбук
            validated_steps = await self._validate_playbook(all_steps, task_plan)
            
            logger.info(f"Сгенерирован плейбук с {len(validated_steps)} шагами")
            return validated_steps
            
        except Exception as e:
            logger.error(f"Ошибка генерации плейбука: {e}")
            return await self._generate_fallback_playbook(task_plan)
    
    async def _get_historical_context(self, task_plan: TaskPlan) -> Dict[str, Any]:
        """Получает исторический контекст для задачи."""
        try:
            # Поиск похожих выполненных задач
            similar_tasks = await self.memory_manager.retrieve(
                MemoryLayer.EPISODIC,
                f"task type: {task_plan.task_type.value}"
            )
            
            if similar_tasks:
                # Группируем по успешным паттернам
                successful_patterns = []
                for task in similar_tasks[:10]:  # Последние 10 задач
                    if task.metadata.get('status') == 'completed':
                        successful_patterns.append(task.data)
                
                return {'successful_patterns': successful_patterns}
            
            return {}
            
        except Exception as e:
            logger.warning(f"Не удалось получить исторический контекст: {e}")
            return {}
    
    async def _get_best_practices(self, task_plan: TaskPlan) -> List[str]:
        """Получает лучшие практики для типа задачи."""
        try:
            # Поиск в семантической памяти
            practices = await self.memory_manager.retrieve(
                MemoryLayer.SEMANTIC,
                f"best practices {task_plan.task_type.value}"
            )
            
            best_practices = []
            for practice in practices:
                if practice.data.get('domain') == task_plan.task_type.value:
                    best_practices.extend(practice.data.get('practices', []))
            
            return best_practices
            
        except Exception as e:
            logger.warning(f"Не удалось получить лучшие практики: {e}")
            return []
    
    async def _generate_base_steps(self, task_plan: TaskPlan) -> List[PlaybookStep]:
        """Генерирует базовые шаги для задачи."""
        steps = []
        step_counter = 1
        
        # Шаг планирования
        steps.append(PlaybookStep(
            step_id=f"{task_plan.task_id}_planning",
            step_number=step_counter,
            title="Анализ и планирование",
            description=f"Детальный анализ требований для задачи: {task_plan.title}",
            action_type="analysis",
            agent_instruction=f"Проанализируй задачу: {task_plan.description}. "
                            f"Определи ключевые компоненты, зависимости и подход к решению.",
            expected_output="Структурированный план выполнения с выявленными рисками",
            success_criteria=["План создан", "Риски идентифицированы", "Подход определен"],
            timeout_minutes=30
        ))
        step_counter += 1
        
        # Шаг подготовки
        steps.append(PlaybookStep(
            step_id=f"{task_plan.task_id}_preparation",
            step_number=step_counter,
            title="Подготовка окружения",
            description="Подготовка необходимых инструментов и окружения",
            action_type="setup",
            agent_instruction=f"Подготовь окружение для выполнения задачи: {task_plan.title}. "
                            f"Настрой все необходимые инструменты и проверь их работоспособность.",
            expected_output="Настроенное и готовое к работе окружение",
            success_criteria=["Инструменты установлены", "Настройки проверены", "Готовность подтверждена"],
            timeout_minutes=20
        ))
        step_counter += 1
        
        # Основные шаги выполнения
        for i, criterion in enumerate(task_plan.success_criteria):
            steps.append(PlaybookStep(
                step_id=f"{task_plan.task_id}_execution_{i}",
                step_number=step_counter,
                title=f"Выполнение: {criterion}",
                description=f"Реализация критерия успеха: {criterion}",
                action_type="execution",
                agent_instruction=f"Реализуй следующий критерий успеха: {criterion}. "
                                f"Задача: {task_plan.description}",
                expected_output=f"Критерий '{criterion}' реализован и протестирован",
                success_criteria=[criterion],
                timeout_minutes=60
            ))
            step_counter += 1
        
        return steps
    
    async def _generate_quality_steps(self, task_plan: TaskPlan) -> List[PlaybookStep]:
        """Генерирует шаги проверки качества."""
        steps = []
        step_counter = 999  # После основных шагов
        
        # Проверка кода/качества
        steps.append(PlaybookStep(
            step_id=f"{task_plan.task_id}_quality_check",
            step_number=step_counter,
            title="Проверка качества",
            description="Комплексная проверка качества выполненной работы",
            action_type="validation",
            agent_instruction="Проведи комплексную проверку качества выполненной работы. "
                            "Проверь соответствие требованиям, отсутствие ошибок, "
                            "производительность и безопасность.",
            expected_output="Отчет о качестве с выявленными проблемами (если есть)",
            success_criteria=["Качество проверено", "Проблемы выявлены (если есть)", "Рекомендации даны"],
            timeout_minutes=30
        ))
        step_counter += 1
        
        # Документация
        steps.append(PlaybookStep(
            step_id=f"{task_plan.task_id}_documentation",
            step_number=step_counter,
            title="Создание документации",
            description="Создание документации по выполненной работе",
            action_type="documentation",
            agent_instruction="Создай подробную документацию по выполненной работе. "
                            "Включи описание реализации, инструкции по использованию, "
                            "известные ограничения и рекомендации.",
            expected_output="Полная документация по выполненной работе",
            success_criteria=["Документация создана", "Инструкции включены", "Примеры приведены"],
            timeout_minutes=25
        ))
        
        return steps
    
    async def _generate_context_steps(self, task_plan: TaskPlan, 
                                    context_data: Dict[str, Any]) -> List[PlaybookStep]:
        """Генерирует шаги на основе контекста."""
        steps = []
        step_counter = 2000
        
        # Добавляем контекстно-зависимые шаги
        if context_data.get('integration_points'):
            steps.append(PlaybookStep(
                step_id=f"{task_plan.task_id}_integration",
                step_number=step_counter,
                title="Интеграция компонентов",
                description="Интеграция с внешними системами и компонентами",
                action_type="integration",
                agent_instruction="Выполни интеграцию с внешними системами: "
                                f"{context_data['integration_points']}",
                expected_output="Успешно интегрированные компоненты",
                success_criteria=["Интеграция выполнена", "Связи проверены"],
                timeout_minutes=45
            ))
            step_counter += 1
        
        if context_data.get('security_requirements'):
            steps.append(PlaybookStep(
                step_id=f"{task_plan.task_id}_security",
                step_number=step_counter,
                title="Проверка безопасности",
                description="Проверка соблюдения требований безопасности",
                action_type="security_check",
                agent_instruction="Проверь соблюдение требований безопасности: "
                                f"{context_data['security_requirements']}",
                expected_output="Отчет о соответствии требованиям безопасности",
                success_criteria=["Безопасность проверена", "Уязвимости выявлены (если есть)"],
                timeout_minutes=35
            ))
        
        return steps
    
    async def _validate_playbook(self, steps: List[PlaybookStep], 
                               task_plan: TaskPlan) -> List[PlaybookStep]:
        """Валидирует плейбук."""
        # Проверяем уникальность ID шагов
        step_ids = set()
        validated_steps = []
        
        for step in steps:
            if step.step_id not in step_ids:
                step_ids.add(step.step_id)
                validated_steps.append(step)
            else:
                logger.warning(f"Дублирующийся ID шага: {step.step_id}")
        
        # Проверяем, что все критерии успеха покрыты
        task_criteria = set(task_plan.success_criteria)
        step_criteria = set()
        
        for step in validated_steps:
            step_criteria.update(step.success_criteria)
        
        missing_criteria = task_criteria - step_criteria
        if missing_criteria:
            logger.warning(f"Отсутствующие критерии в плейбуке: {missing_criteria}")
        
        return validated_steps
    
    async def _generate_fallback_playbook(self, task_plan: TaskPlan) -> List[PlaybookStep]:
        """Генерирует резервный плейбук."""
        return [
            PlaybookStep(
                step_id=f"{task_plan.task_id}_fallback",
                step_number=1,
                title="Базовое выполнение задачи",
                description=f"Базовое выполнение: {task_plan.title}",
                action_type="basic_execution",
                agent_instruction=f"Выполни задачу: {task_plan.description}",
                expected_output="Результат выполнения задачи",
                success_criteria=["Задача выполнена"],
                timeout_minutes=120
            )
        ]


class RebeccaMetaAgent:
    """
    Полноценный мета-агент Rebecca для интеллектуального управления агентной экосистемой.
    
    Ключевые возможности:
    - Поглощение и анализ источников
    - Планирование и распределение задач
    - Генерация плейбуков
    - Координация агентов
    - Мониторинг и оптимизация
    """
    
    def __init__(self, 
                 memory_manager: MemoryManager,
                 ingest_pipeline: IngestPipeline,
                 context_handler: ContextHandler,
                 blueprint_tracker: AdaptiveBlueprintTracker,
                 config: MetaAgentConfig = None):
        """Инициализирует мета-агента."""
        self.config = config or MetaAgentConfig()
        self.memory_manager = memory_manager
        self.ingest_pipeline = ingest_pipeline
        self.context_handler = context_handler
        self.blueprint_tracker = blueprint_tracker
        
        # Инициализация компонентов
        self.task_analyzer = TaskAnalyzer(memory_manager)
        self.resource_optimizer = ResourceOptimizer(self.config)
        self.playbook_generator = PlaybookGenerator(memory_manager, blueprint_tracker)
        
        # Хранилище данных
        self.task_plans: Dict[str, TaskPlan] = {}
        self.agent_assignments: Dict[str, AgentAssignment] = {}
        self.playbooks: Dict[str, List[PlaybookStep]] = {}
        self.resource_allocations: Dict[str, ResourceAllocation] = {}
        
        # Состояние выполнения
        self.execution_queue: deque = deque()
        self.completed_tasks: List[str] = []
        self.failed_tasks: List[str] = []
        self.active_agents: Dict[str, AgentSpecialization] = {}
        
        # Метрики
        self.performance_metrics: Dict[str, Any] = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'average_quality_score': 0.0,
            'average_completion_time': 0.0,
            'resource_utilization': 0.0
        }
        
        logger.info("RebeccaMetaAgent инициализирован")
    
    async def ingest_sources(self, sources: Union[str, List[str]], 
                           analysis_options: Dict[str, Any] = None) -> List[str]:
        """
        Поглощение и обработка источников.
        
        Args:
            sources: Источники для поглощения (путь к файлу, URL, Git репозиторий)
            analysis_options: Опции анализа источников
            
        Returns:
            Список ID обработанных источников
        """
        try:
            logger.info(f"Начинаем поглощение источников: {sources}")
            
            if isinstance(sources, str):
                sources = [sources]
            
            processed_source_ids = []
            analysis_config = analysis_options or {}
            
            for source in sources:
                try:
                    # Определяем тип источника
                    source_type = self._determine_source_type(source)
                    
                    # Обрабатываем источник
                    if source_type == 'git':
                        events = await self._process_git_source(source, analysis_config)
                    else:
                        events = await self._process_file_source(source, analysis_config)
                    
                    # Анализируем обработанные данные
                    analysis_results = await self._analyze_ingested_content(events, source_type)
                    
                    # Сохраняем анализ в память
                    source_id = await self._store_source_analysis(source, source_type, analysis_results)
                    processed_source_ids.append(source_id)
                    
                    logger.info(f"Источник успешно обработан: {source}")
                    
                except Exception as e:
                    logger.error(f"Ошибка при обработке источника {source}: {e}")
                    continue
            
            # Обновляем blueprint с новой информацией
            await self._update_blueprint_with_sources(processed_source_ids)
            
            logger.info(f"Завершено поглощение {len(processed_source_ids)} источников")
            return processed_source_ids
            
        except Exception as e:
            logger.error(f"Критическая ошибка при поглощении источников: {e}")
            raise
    
    async def plan_agent(self, requirements: Dict[str, Any], 
                        context: Dict[str, Any] = None) -> TaskPlan:
        """
        Планирование задач для агентов.
        
        Args:
            requirements: Требования к задачам
            context: Контекст выполнения
            
        Returns:
            План выполнения задачи
        """
        try:
            logger.info("Начинаем планирование задач для агентов")
            
            # Извлекаем данные из требований
            task_description = requirements.get('description', '')
            task_type = TaskType(requirements.get('type', 'development'))
            priority = TaskPriority(requirements.get('priority', 'medium'))
            context_data = context or {}
            
            # Анализируем сложность задачи
            complexity_score = await self.task_analyzer.analyze_task_complexity(
                task_description, context_data
            )
            
            # Определяем необходимые навыки
            required_skills = self.task_analyzer.determine_required_skills(task_description, task_type)
            
            # Оцениваем время выполнения
            estimated_duration = self.task_analyzer.estimate_duration(
                task_description, complexity_score, required_skills
            )
            
            # Определяем зависимости
            dependencies = await self._identify_dependencies(task_description, context_data)
            prerequisites = await self._identify_prerequisites(task_description, context_data)
            
            # Формируем критерии успеха
            success_criteria = await self._generate_success_criteria(task_description, requirements)
            
            # Создаем план задачи
            task_plan = TaskPlan(
                task_id=str(uuid.uuid4()),
                title=requirements.get('title', task_description[:50] + '...'),
                description=task_description,
                task_type=task_type,
                priority=priority,
                complexity_score=complexity_score,
                estimated_duration=estimated_duration,
                required_skills=required_skills,
                dependencies=dependencies,
                prerequisites=prerequisites,
                success_criteria=success_criteria,
                metadata=requirements.get('metadata', {})
            )
            
            # Сохраняем план в память
            await self._store_task_plan(task_plan)
            
            # Добавляем в очередь выполнения
            self.execution_queue.append(task_plan)
            
            logger.info(f"План создан: {task_plan.task_id}, сложность: {complexity_score:.2f}")
            return task_plan
            
        except Exception as e:
            logger.error(f"Ошибка при планировании задач: {e}")
            raise
    
    async def generate_playbook(self, task_plan: TaskPlan, 
                              agent_context: Dict[str, Any] = None) -> List[PlaybookStep]:
        """
        Создание инструкций для выполнения.
        
        Args:
            task_plan: План выполнения задачи
            agent_context: Контекст агента
            
        Returns:
            Список шагов плейбука
        """
        try:
            logger.info(f"Генерируем плейбук для задачи: {task_plan.task_id}")
            
            # Создаем назначение агента
            assignment = await self._create_agent_assignment(task_plan, agent_context)
            
            # Генерируем плейбук
            playbook_steps = await self.playbook_generator.generate_playbook(
                task_plan, assignment, agent_context
            )
            
            # Сохраняем плейбук
            self.playbooks[task_plan.task_id] = playbook_steps
            
            # Сохраняем в памяти
            await self._store_playbook_in_memory(task_plan.task_id, playbook_steps)
            
            # Обновляем назначение агента
            self.agent_assignments[assignment.assignment_id] = assignment
            
            logger.info(f"Плейбук создан с {len(playbook_steps)} шагами")
            return playbook_steps
            
        except Exception as e:
            logger.error(f"Ошибка генерации плейбука: {e}")
            raise
    
    async def execute_playbook(self, playbook_steps: List[PlaybookStep], 
                             assignment_id: str) -> Dict[str, Any]:
        """
        Выполнение плейбука.
        
        Args:
            playbook_steps: Шаги плейбука
            assignment_id: ID назначения агента
            
        Returns:
            Результаты выполнения
        """
        try:
            logger.info(f"Начинаем выполнение плейбука для назначения: {assignment_id}")
            
            assignment = self.agent_assignments[assignment_id]
            execution_results = {
                'assignment_id': assignment_id,
                'task_id': assignment.task_plan.task_id,
                'start_time': datetime.now(timezone.utc),
                'steps_completed': [],
                'steps_failed': [],
                'overall_status': TaskStatus.IN_PROGRESS,
                'quality_score': 0.0
            }
            
            # Обновляем статус назначения
            assignment.status = TaskStatus.IN_PROGRESS
            assignment.start_time = datetime.now(timezone.utc)
            
            total_steps = len(playbook_steps)
            completed_steps = 0
            
            for step in playbook_steps:
                try:
                    # Выполняем шаг
                    step_result = await self._execute_playbook_step(step, assignment)
                    
                    if step_result['success']:
                        execution_results['steps_completed'].append(step.step_id)
                        completed_steps += 1
                    else:
                        execution_results['steps_failed'].append({
                            'step_id': step.step_id,
                            'error': step_result['error']
                        })
                        
                        # Проверяем, нужно ли повторить
                        if step.retry_count < step.max_retries:
                            step.retry_count += 1
                            logger.info(f"Повторяем шаг {step.step_id} ({step.retry_count}/{step.max_retries})")
                            continue
                        else:
                            logger.error(f"Шаг {step.step_id} не удался после {step.max_retries} попыток")
                    
                    # Обновляем прогресс
                    assignment.progress = (completed_steps / total_steps) * 100
                    
                except Exception as e:
                    logger.error(f"Ошибка при выполнении шага {step.step_id}: {e}")
                    execution_results['steps_failed'].append({
                        'step_id': step.step_id,
                        'error': str(e)
                    })
            
            # Определяем финальный статус
            if len(execution_results['steps_failed']) == 0:
                execution_results['overall_status'] = TaskStatus.COMPLETED
                assignment.status = TaskStatus.COMPLETED
            else:
                execution_results['overall_status'] = TaskStatus.FAILED
                assignment.status = TaskStatus.FAILED
            
            # Завершаем выполнение
            assignment.end_time = datetime.now(timezone.utc)
            assignment.progress = 100.0
            
            # Вычисляем качество
            execution_results['quality_score'] = self._calculate_execution_quality(execution_results)
            assignment.quality_score = execution_results['quality_score']
            
            # Обновляем метрики
            await self._update_performance_metrics(assignment, execution_results)
            
            logger.info(f"Выполнение плейбука завершено, статус: {execution_results['overall_status'].value}")
            return execution_results
            
        except Exception as e:
            logger.error(f"Ошибка при выполнении плейбука: {e}")
            raise
    
    async def get_status(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Получение статуса выполнения.
        
        Args:
            task_id: ID задачи (опционально)
            
        Returns:
            Словарь со статусом
        """
        try:
            if task_id:
                # Статус конкретной задачи
                task_plan = self.task_plans.get(task_id)
                if not task_plan:
                    return {'error': f'Задача {task_id} не найдена'}
                
                # Находим назначения для задачи
                task_assignments = [
                    assignment for assignment in self.agent_assignments.values()
                    if assignment.task_plan.task_id == task_id
                ]
                
                return {
                    'task_id': task_id,
                    'task_plan': task_plan.to_dict(),
                    'assignments': [assignment.to_dict() for assignment in task_assignments],
                    'playbook': self.playbooks.get(task_id, []),
                    'status': self._get_task_status(task_id)
                }
            else:
                # Общий статус системы
                return {
                    'system_status': 'operational',
                    'metrics': self.performance_metrics,
                    'active_agents': len(self.active_agents),
                    'queued_tasks': len(self.execution_queue),
                    'completed_tasks': len(self.completed_tasks),
                    'failed_tasks': len(self.failed_tasks),
                    'blueprint_version': await self.blueprint_tracker.get_latest_blueprint()
                }
                
        except Exception as e:
            logger.error(f"Ошибка получения статуса: {e}")
            return {'error': str(e)}
    
    # Вспомогательные методы
    
    def _determine_source_type(self, source: str) -> str:
        """Определяет тип источника."""
        if any(pattern in source.lower() for pattern in ['github.com', 'gitlab.com', '.git', 'git://']):
            return 'git'
        elif source.startswith('http://') or source.startswith('https://'):
            return 'web'
        else:
            return 'file'
    
    async def _process_git_source(self, source: str, config: Dict[str, Any]) -> List[Any]:
        """Обрабатывает Git источник."""
        try:
            # Обрабатываем репозиторий
            events = self.ingest_pipeline.process_git_repo(
                source,
                branch=config.get('branch', 'main'),
                process_readme=config.get('process_readme', True),
                process_source=config.get('process_source', True)
            )
            return events
        except Exception as e:
            logger.error(f"Ошибка обработки Git источника: {e}")
            return []
    
    async def _process_file_source(self, source: str, config: Dict[str, Any]) -> List[Any]:
        """Обрабатывает файловый источник."""
        try:
            # Обрабатываем документ
            event = self.ingest_pipeline.ingest_document(
                source,
                chunk_override=config.get('chunk_size')
            )
            return [event]
        except Exception as e:
            logger.error(f"Ошибка обработки файла: {e}")
            return []
    
    async def _analyze_ingested_content(self, events: List[Any], source_type: str) -> Dict[str, Any]:
        """Анализирует поглощенное содержимое."""
        try:
            analysis = {
                'source_type': source_type,
                'events_count': len(events),
                'content_summary': '',
                'key_concepts': [],
                'complexity_level': 'medium',
                'recommended_approach': ''
            }
            
            # Анализируем содержимое событий
            for event in events:
                if hasattr(event, 'attrs') and 'text' in event.attrs:
                    content = event.attrs['text']
                    
                    # Извлекаем ключевые концепции (простая реализация)
                    words = content.lower().split()
                    word_freq = defaultdict(int)
                    
                    for word in words:
                        if len(word) > 4:  # Игнорируем короткие слова
                            word_freq[word] += 1
                    
                    # Топ 10 самых частых слов как ключевые концепции
                    analysis['key_concepts'].extend([
                        word for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                    ])
            
            # Удаляем дубликаты
            analysis['key_concepts'] = list(set(analysis['key_concepts']))
            
            return analysis
            
        except Exception as e:
            logger.error(f"Ошибка анализа содержимого: {e}")
            return {}
    
    async def _store_source_analysis(self, source: str, source_type: str, 
                                   analysis: Dict[str, Any]) -> str:
        """Сохраняет анализ источника в память."""
        try:
            source_id = f"source::{source_type}::{hashlib.md5(source.encode()).hexdigest()[:8]}"
            
            await self.memory_manager.store(
                MemoryLayer.VAULT,
                {
                    'source_id': source_id,
                    'source_url': source,
                    'source_type': source_type,
                    'analysis': analysis,
                    'ingested_at': datetime.now(timezone.utc).isoformat()
                },
                metadata={'source_id': source_id, 'type': 'source_analysis'}
            )
            
            return source_id
            
        except Exception as e:
            logger.error(f"Ошибка сохранения анализа источника: {e}")
            return ""
    
    async def _update_blueprint_with_sources(self, source_ids: List[str]) -> None:
        """Обновляет blueprint с новыми источниками."""
        try:
            current_blueprint = await self.blueprint_tracker.get_latest_blueprint()
            
            if current_blueprint:
                # Обновляем существующий blueprint
                updated_blueprint = current_blueprint.blueprint.copy()
                updated_blueprint['sources'] = updated_blueprint.get('sources', []) + source_ids
                updated_blueprint['last_update'] = datetime.now(timezone.utc).isoformat()
                
                await self.blueprint_tracker.record_blueprint(
                    updated_blueprint,
                    change_type='update',
                    change_description=f'Добавлено {len(source_ids)} новых источников'
                )
            else:
                # Создаем новый blueprint
                initial_blueprint = {
                    'sources': source_ids,
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'version': 1
                }
                
                await self.blueprint_tracker.record_blueprint(
                    initial_blueprint,
                    change_type='initial',
                    change_description='Инициализация blueprint с источниками'
                )
                
        except Exception as e:
            logger.error(f"Ошибка обновления blueprint: {e}")
    
    async def _identify_dependencies(self, task_description: str, context: Dict[str, Any]) -> List[str]:
        """Идентифицирует зависимости задачи."""
        dependencies = []
        
        # Простой анализ на основе ключевых слов
        dependency_keywords = {
            'требует api': 'api_integration',
            'зависит от базы данных': 'database_setup',
            'нужна аутентификация': 'auth_setup',
            'требует интеграция': 'integration_service'
        }
        
        description_lower = task_description.lower()
        for keyword, dependency in dependency_keywords.items():
            if keyword in description_lower:
                dependencies.append(dependency)
        
        return dependencies
    
    async def _identify_prerequisites(self, task_description: str, context: Dict[str, Any]) -> List[str]:
        """Идентифицирует предпосылки задачи."""
        prerequisites = []
        
        # Анализ на основе контекста
        if context.get('existing_components'):
            prerequisites.extend(context['existing_components'])
        
        if context.get('required_skills'):
            prerequisites.extend(context['required_skills'])
        
        return list(set(prerequisites))  # Удаляем дубликаты
    
    async def _generate_success_criteria(self, task_description: str, requirements: Dict[str, Any]) -> List[str]:
        """Генерирует критерии успеха."""
        criteria = []
        
        # Базовые критерии
        if ' функциональность ' in task_description.lower():
            criteria.append("Функциональность реализована")
        
        if ' тест ' in task_description.lower() or ' testing ' in task_description.lower():
            criteria.append("Тесты созданы и проходят")
        
        if ' документация ' in task_description.lower() or ' documentation ' in task_description.lower():
            criteria.append("Документация создана")
        
        # Критерии из требований
        if 'success_criteria' in requirements:
            criteria.extend(requirements['success_criteria'])
        
        # Если нет критериев, добавляем базовые
        if not criteria:
            criteria = [
                "Задача выполнена",
                "Код протестирован",
                "Документация обновлена"
            ]
        
        return criteria
    
    async def _store_task_plan(self, task_plan: TaskPlan) -> None:
        """Сохраняет план задачи в память."""
        try:
            # Сохраняем в памяти
            await self.memory_manager.store(
                MemoryLayer.PROCEDURAL,
                {'task_plan': task_plan.to_dict()},
                metadata={'task_id': task_plan.task_id, 'type': 'task_plan'}
            )
            
            # Сохраняем локально
            self.task_plans[task_plan.task_id] = task_plan
            
        except Exception as e:
            logger.error(f"Ошибка сохранения плана задачи: {e}")
    
    async def _create_agent_assignment(self, task_plan: TaskPlan, 
                                     agent_context: Dict[str, Any] = None) -> AgentAssignment:
        """Создает назначение агента."""
        # Выбираем подходящего агента
        agent_id = await self._select_agent_for_task(task_plan, agent_context)
        
        assignment = AgentAssignment(
            assignment_id=str(uuid.uuid4()),
            task_plan=task_plan,
            agent_type=self.active_agents.get(agent_id, AgentSpecialization.GENERAL),
            agent_id=agent_id,
            start_time=datetime.now(timezone.utc),
            context_data=agent_context or {}
        )
        
        return assignment
    
    async def _select_agent_for_task(self, task_plan: TaskPlan, 
                                   agent_context: Dict[str, Any] = None) -> str:
        """Выбирает агента для задачи."""
        # Простая эвристика: выбираем агента с подходящей специализацией
        available_agents = list(self.active_agents.keys())
        
        if not available_agents:
            return "default_agent"
        
        # Ищем агента с подходящей специализацией
        for agent_id in available_agents:
            agent_specialization = self.active_agents[agent_id]
            if agent_specialization in task_plan.required_skills:
                return agent_id
        
        # Если нет подходящего агента, возвращаем первого доступного
        return available_agents[0]
    
    async def _store_playbook_in_memory(self, task_id: str, playbook_steps: List[PlaybookStep]) -> None:
        """Сохраняет плейбук в память."""
        try:
            playbook_data = {
                'task_id': task_id,
                'steps': [step.to_dict() for step in playbook_steps],
                'created_at': datetime.now(timezone.utc).isoformat()
            }
            
            await self.memory_manager.store(
                MemoryLayer.PROCEDURAL,
                playbook_data,
                metadata={'task_id': task_id, 'type': 'playbook'}
            )
            
        except Exception as e:
            logger.error(f"Ошибка сохранения плейбука: {e}")
    
    async def _execute_playbook_step(self, step: PlaybookStep, 
                                   assignment: AgentAssignment) -> Dict[str, Any]:
        """Выполняет отдельный шаг плейбука."""
        try:
            logger.info(f"Выполняем шаг: {step.title}")
            
            # Симуляция выполнения шага
            # В реальной реализации здесь был бы вызов соответствующего агента
            
            await asyncio.sleep(1)  # Симуляция выполнения
            
            # Простая проверка успешности (в реальности - результат от агента)
            success = step.retry_count == 0 or step.retry_count > 0  # Условная логика
            
            if success:
                logger.info(f"Шаг {step.step_id} выполнен успешно")
            else:
                logger.warning(f"Шаг {step.step_id} завершился с ошибкой")
            
            return {
                'success': success,
                'output': f"Результат выполнения шага {step.title}",
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Ошибка выполнения шага {step.step_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def _calculate_execution_quality(self, execution_results: Dict[str, Any]) -> float:
        """Вычисляет качество выполнения."""
        total_steps = len(execution_results['steps_completed']) + len(execution_results['steps_failed'])
        
        if total_steps == 0:
            return 0.0
        
        # Базовая оценка качества на основе количества успешных шагов
        success_rate = len(execution_results['steps_completed']) / total_steps
        
        # Корректировка на основе критичности ошибок
        error_penalty = len(execution_results['steps_failed']) * 0.1
        
        quality_score = max(0.0, success_rate - error_penalty)
        
        return min(1.0, quality_score)
    
    async def _update_performance_metrics(self, assignment: AgentAssignment, 
                                        execution_results: Dict[str, Any]) -> None:
        """Обновляет метрики производительности."""
        try:
            # Обновляем счетчики
            if execution_results['overall_status'] == TaskStatus.COMPLETED:
                self.performance_metrics['tasks_completed'] += 1
                self.completed_tasks.append(assignment.task_plan.task_id)
            else:
                self.performance_metrics['tasks_failed'] += 1
                self.failed_tasks.append(assignment.task_plan.task_id)
            
            # Вычисляем время выполнения
            if assignment.end_time:
                duration = (assignment.end_time - assignment.start_time).total_seconds() / 60  # минуты
                
                # Обновляем среднее время
                completed_count = self.performance_metrics['tasks_completed']
                current_avg = self.performance_metrics['average_completion_time']
                
                self.performance_metrics['average_completion_time'] = (
                    (current_avg * (completed_count - 1) + duration) / completed_count
                )
            
            # Обновляем среднее качество
            quality_score = execution_results['quality_score']
            completed_count = self.performance_metrics['tasks_completed']
            
            if completed_count > 0:
                current_avg_quality = self.performance_metrics['average_quality_score']
                self.performance_metrics['average_quality_score'] = (
                    (current_avg_quality * (completed_count - 1) + quality_score) / completed_count
                )
            
            logger.debug("Метрики производительности обновлены")
            
        except Exception as e:
            logger.error(f"Ошибка обновления метрик: {e}")
    
    def _get_task_status(self, task_id: str) -> TaskStatus:
        """Получает статус задачи."""
        # Ищем назначения для задачи
        task_assignments = [
            assignment for assignment in self.agent_assignments.values()
            if assignment.task_plan.task_id == task_id
        ]
        
        if not task_assignments:
            return TaskStatus.PENDING
        
        # Определяем статус на основе назначений
        statuses = [assignment.status for assignment in task_assignments]
        
        if TaskStatus.FAILED in statuses:
            return TaskStatus.FAILED
        elif TaskStatus.IN_PROGRESS in statuses:
            return TaskStatus.IN_PROGRESS
        elif all(status == TaskStatus.COMPLETED for status in statuses):
            return TaskStatus.COMPLETED
        else:
            return TaskStatus.PLANNED


# Фабрика методы
class RebeccaMetaAgentFactory:
    """Фабрика для создания мета-агента Rebecca."""
    
    @staticmethod
    def create_basic_agent(memory_manager: MemoryManager,
                         ingest_pipeline: IngestPipeline,
                         context_handler: ContextHandler) -> RebeccaMetaAgent:
        """Создает базового мета-агента с настройками по умолчанию."""
        blueprint_tracker = AdaptiveBlueprintTracker(memory_manager.semantic)
        config = MetaAgentConfig()
        
        return RebeccaMetaAgent(
            memory_manager=memory_manager,
            ingest_pipeline=ingest_pipeline,
            context_handler=context_handler,
            blueprint_tracker=blueprint_tracker,
            config=config
        )
    
    @staticmethod
    def create_advanced_agent(memory_manager: MemoryManager,
                            ingest_pipeline: IngestPipeline,
                            context_handler: ContextHandler,
                            blueprint_tracker: AdaptiveBlueprintTracker,
                            config: MetaAgentConfig) -> RebeccaMetaAgent:
        """Создает расширенного мета-агента с полной конфигурацией."""
        return RebeccaMetaAgent(
            memory_manager=memory_manager,
            ingest_pipeline=ingest_pipeline,
            context_handler=context_handler,
            blueprint_tracker=blueprint_tracker,
            config=config
        )
    
    @staticmethod
    def create_production_agent(config_path: str) -> RebeccaMetaAgent:
        """Создает production мета-агента из конфигурации."""
        # Загрузка конфигурации
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        config = MetaAgentConfig(**config_data)
        
        # Создание компонентов (требует настройки)
        memory_manager = MemoryManager()
        ingest_pipeline = IngestPipeline(memory_manager, None, None, None, None, None, None)
        context_handler = ContextHandler()
        blueprint_tracker = AdaptiveBlueprintTracker(memory_manager.semantic)
        
        return RebeccaMetaAgent(
            memory_manager=memory_manager,
            ingest_pipeline=ingest_pipeline,
            context_handler=context_handler,
            blueprint_tracker=blueprint_tracker,
            config=config
        )


# Экспорт основных классов
__all__ = [
    "RebeccaMetaAgent",
    "TaskPlan",
    "AgentAssignment",
    "PlaybookStep", 
    "ResourceAllocation",
    "MetaAgentConfig",
    "RebeccaMetaAgentFactory"
]