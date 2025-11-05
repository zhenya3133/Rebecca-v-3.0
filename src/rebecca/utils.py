"""
Утилиты и примеры использования мета-агента Rebecca.

Предоставляет дополнительные функции для работы с мета-агентом, включая:
- Валидацию конфигураций
- Примеры использования
- Тестовые данные
- Вспомогательные функции
"""

import asyncio
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .meta_agent import (
    RebeccaMetaAgent, TaskPlan, AgentAssignment, PlaybookStep, ResourceAllocation,
    MetaAgentConfig, TaskType, TaskPriority, AgentSpecialization, TaskStatus,
    RebeccaMetaAgentFactory
)


# Настройка логгера
logger = logging.getLogger(__name__)


class MetaAgentValidator:
    """Валидатор конфигураций мета-агента."""
    
    @staticmethod
    def validate_config(config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Валидирует конфигурацию мета-агента."""
        errors = []
        warnings = []
        
        # Проверяем обязательные поля
        required_fields = ['max_concurrent_tasks', 'default_timeout_minutes']
        for field in required_fields:
            if field not in config_data:
                errors.append(f"Отсутствует обязательное поле: {field}")
        
        # Проверяем типы данных
        if 'max_concurrent_tasks' in config_data:
            if not isinstance(config_data['max_concurrent_tasks'], int) or config_data['max_concurrent_tasks'] <= 0:
                errors.append("max_concurrent_tasks должно быть положительным целым числом")
        
        if 'quality_threshold' in config_data:
            quality = config_data['quality_threshold']
            if not isinstance(quality, (int, float)) or not 0 <= quality <= 1:
                errors.append("quality_threshold должно быть числом в диапазоне 0-1")
        
        # Проверяем веса
        weight_fields = ['complexity_weight', 'priority_weight', 'dependency_weight']
        total_weight = sum(config_data.get(field, 0) for field in weight_fields)
        
        if abs(total_weight - 1.0) > 0.01:  # Допускаем небольшую погрешность
            warnings.append(f"Сумма весов {total_weight:.3f} не равна 1.0")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    @staticmethod
    def validate_task_plan(task_plan: TaskPlan) -> Dict[str, Any]:
        """Валидирует план задачи."""
        errors = []
        warnings = []
        
        # Проверяем обязательные поля
        if not task_plan.title or len(task_plan.title.strip()) == 0:
            errors.append("Название задачи не может быть пустым")
        
        if not task_plan.description or len(task_plan.description.strip()) == 0:
            errors.append("Описание задачи не может быть пустым")
        
        if not task_plan.required_skills:
            warnings.append("Не указаны требуемые навыки для задачи")
        
        if task_plan.complexity_score < 0 or task_plan.complexity_score > 1:
            errors.append("Сложность задачи должна быть в диапазоне 0-1")
        
        if task_plan.estimated_duration <= 0:
            errors.append("Время выполнения задачи должно быть положительным")
        
        # Проверяем зависимости
        for dep_id in task_plan.dependencies:
            if not dep_id or not isinstance(dep_id, str):
                errors.append(f"Некорректная зависимость: {dep_id}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    @staticmethod
    def validate_playbook(playbook_steps: List[PlaybookStep]) -> Dict[str, Any]:
        """Валидирует плейбук."""
        errors = []
        warnings = []
        
        if not playbook_steps:
            errors.append("Плейбук не может быть пустым")
            return {'valid': False, 'errors': errors, 'warnings': warnings}
        
        # Проверяем уникальность ID шагов
        step_ids = set()
        step_numbers = set()
        
        for step in playbook_steps:
            if step.step_id in step_ids:
                errors.append(f"Дублирующийся ID шага: {step.step_id}")
            step_ids.add(step.step_id)
            
            if step.step_number in step_numbers:
                errors.append(f"Дублирующийся номер шага: {step.step_number}")
            step_numbers.add(step.step_number)
            
            # Проверяем обязательные поля
            if not step.title:
                errors.append(f"Шаг {step.step_id} не имеет названия")
            
            if not step.agent_instruction:
                errors.append(f"Шаг {step.step_id} не имеет инструкций для агента")
            
            if step.timeout_minutes <= 0:
                errors.append(f"Шаг {step.step_id} имеет некорректный таймаут")
        
        # Проверяем последовательность номеров
        expected_numbers = set(range(1, len(playbook_steps) + 1))
        if step_numbers != expected_numbers:
            warnings.append("Номера шагов не следуют последовательно")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }


class MetaAgentTestData:
    """Генератор тестовых данных для мета-агента."""
    
    @staticmethod
    def create_sample_requirements() -> Dict[str, Any]:
        """Создает пример требований для задачи."""
        return {
            'title': 'Разработка API для управления пользователями',
            'description': 'Создать RESTful API для полного CRUD функционала управления пользователями с аутентификацией и авторизацией',
            'type': 'development',
            'priority': 'high',
            'metadata': {
                'domain': 'backend',
                'complexity': 'medium',
                'estimated_scope': '2 недели'
            },
            'success_criteria': [
                'API полностью функционален',
                'Реализована аутентификация',
                'Написаны тесты с покрытием > 80%',
                'Создана документация API'
            ]
        }
    
    @staticmethod
    def create_sample_context() -> Dict[str, Any]:
        """Создает пример контекста выполнения."""
        return {
            'existing_components': ['auth_service', 'database_schema'],
            'required_skills': ['python', 'postgresql', 'docker'],
            'integration_points': ['payment_system', 'notification_service'],
            'security_requirements': ['JWT authentication', 'Rate limiting', 'Input validation'],
            'performance_requirements': {
                'max_response_time': '200ms',
                'throughput': '1000 req/min'
            }
        }
    
    @staticmethod
    def create_sample_config() -> Dict[str, Any]:
        """Создает пример конфигурации."""
        return {
            'max_concurrent_tasks': 10,
            'default_timeout_minutes': 60,
            'enable_auto_scaling': True,
            'enable_failover': True,
            'quality_threshold': 0.8,
            'complexity_weight': 0.3,
            'priority_weight': 0.4,
            'dependency_weight': 0.3,
            'resource_optimization': True,
            'enable_learning': True,
            'memory_retention_days': 30,
            'enable_proactive_planning': True
        }
    
    @staticmethod
    def create_sample_agent_assignments() -> List[Dict[str, Any]]:
        """Создает примеры назначений агентов."""
        return [
            {
                'agent_type': 'backend',
                'agent_id': 'agent_backend_001',
                'specialization': AgentSpecialization.BACKEND
            },
            {
                'agent_type': 'qa',
                'agent_id': 'agent_qa_001',
                'specialization': AgentSpecialization.QUALITY_ASSURANCE
            },
            {
                'agent_type': 'devops',
                'agent_id': 'agent_devops_001',
                'specialization': AgentSpecialization.DEVOPS
            }
        ]


class MetaAgentDemo:
    """Демонстрация возможностей мета-агента."""
    
    def __init__(self, agent: RebeccaMetaAgent):
        self.agent = agent
        self.validator = MetaAgentValidator()
        self.test_data = MetaAgentTestData()
    
    async def run_full_demo(self) -> Dict[str, Any]:
        """Запускает полную демонстрацию мета-агента."""
        logger.info("Начинаем полную демонстрацию мета-агента Rebecca")
        
        demo_results = {
            'demo_id': str(int(time.time())),
            'start_time': time.time(),
            'steps_completed': [],
            'errors': [],
            'metrics': {}
        }
        
        try:
            # Шаг 1: Создание конфигурации
            config_result = await self._demo_config_creation()
            demo_results['steps_completed'].append(config_result)
            
            # Шаг 2: Поглощение источников
            sources_result = await self._demo_source_ingestion()
            demo_results['steps_completed'].append(sources_result)
            
            # Шаг 3: Планирование задач
            planning_result = await self._demo_task_planning()
            demo_results['steps_completed'].append(planning_result)
            
            # Шаг 4: Генерация плейбука
            playbook_result = await self._demo_playbook_generation()
            demo_results['steps_completed'].append(playbook_result)
            
            # Шаг 5: Получение статуса
            status_result = await self._demo_status_check()
            demo_results['steps_completed'].append(status_result)
            
            demo_results['end_time'] = time.time()
            demo_results['duration'] = demo_results['end_time'] - demo_results['start_time']
            
            logger.info(f"Демонстрация завершена успешно за {demo_results['duration']:.2f} секунд")
            
        except Exception as e:
            error_msg = f"Ошибка в демонстрации: {str(e)}"
            logger.error(error_msg)
            demo_results['errors'].append(error_msg)
            demo_results['end_time'] = time.time()
            demo_results['duration'] = demo_results['end_time'] - demo_results['start_time']
        
        return demo_results
    
    async def _demo_config_creation(self) -> Dict[str, Any]:
        """Демонстрация создания конфигурации."""
        logger.info("Шаг 1: Создание конфигурации")
        
        config_data = self.test_data.create_sample_config()
        validation_result = self.validator.validate_config(config_data)
        
        return {
            'step': 'config_creation',
            'success': validation_result['valid'],
            'config_data': config_data,
            'validation': validation_result
        }
    
    async def _demo_source_ingestion(self) -> Dict[str, Any]:
        """Демонстрация поглощения источников."""
        logger.info("Шаг 2: Поглощение источников")
        
        # Создаем тестовый файл
        test_file_path = await self._create_test_file()
        
        try:
            # Поглощаем источник
            source_ids = await self.agent.ingest_sources(
                [test_file_path],
                analysis_options={'chunk_size': 500}
            )
            
            return {
                'step': 'source_ingestion',
                'success': len(source_ids) > 0,
                'source_ids': source_ids,
                'processed_files': 1
            }
            
        finally:
            # Удаляем тестовый файл
            if Path(test_file_path).exists():
                Path(test_file_path).unlink()
    
    async def _demo_task_planning(self) -> Dict[str, Any]:
        """Демонстрация планирования задач."""
        logger.info("Шаг 3: Планирование задач")
        
        requirements = self.test_data.create_sample_requirements()
        context = self.test_data.create_sample_context()
        
        # Создаем план задачи
        task_plan = await self.agent.plan_agent(requirements, context)
        
        # Валидируем план
        validation_result = self.validator.validate_task_plan(task_plan)
        
        return {
            'step': 'task_planning',
            'success': validation_result['valid'],
            'task_plan_id': task_plan.task_id,
            'validation': validation_result,
            'task_details': {
                'complexity_score': task_plan.complexity_score,
                'estimated_duration': task_plan.estimated_duration,
                'required_skills': [skill.value for skill in task_plan.required_skills]
            }
        }
    
    async def _demo_playbook_generation(self) -> Dict[str, Any]:
        """Демонстрация генерации плейбука."""
        logger.info("Шаг 4: Генерация плейбука")
        
        # Получаем последний созданный план задачи
        if not self.agent.task_plans:
            raise Exception("Нет доступных планов задач для генерации плейбука")
        
        last_task_id = list(self.agent.task_plans.keys())[-1]
        task_plan = self.agent.task_plans[last_task_id]
        
        # Создаем контекст для агента
        agent_context = {
            'agent_id': 'demo_agent_001',
            'capabilities': ['backend_development', 'api_design'],
            'current_workload': 0.3
        }
        
        # Генерируем плейбук
        playbook_steps = await self.agent.generate_playbook(task_plan, agent_context)
        
        # Валидируем плейбук
        validation_result = self.validator.validate_playbook(playbook_steps)
        
        return {
            'step': 'playbook_generation',
            'success': validation_result['valid'],
            'task_id': task_plan.task_id,
            'playbook_steps_count': len(playbook_steps),
            'validation': validation_result,
            'sample_steps': [
                {
                    'step_id': step.step_id,
                    'title': step.title,
                    'action_type': step.action_type
                } for step in playbook_steps[:3]  # Первые 3 шага
            ]
        }
    
    async def _demo_status_check(self) -> Dict[str, Any]:
        """Демонстрация проверки статуса."""
        logger.info("Шаг 5: Проверка статуса")
        
        # Получаем общий статус
        system_status = await self.agent.get_status()
        
        # Получаем статус конкретной задачи (если есть)
        task_status = None
        if self.agent.task_plans:
            task_id = list(self.agent.task_plans.keys())[-1]
            task_status = await self.agent.get_status(task_id)
        
        return {
            'step': 'status_check',
            'success': True,
            'system_status': system_status,
            'task_status': task_status
        }
    
    async def _create_test_file(self) -> str:
        """Создает тестовый файл для демонстрации."""
        test_content = """
# Тестовый документ для мета-агента Rebecca

## Описание
Этот документ создан для демонстрации возможностей поглощения источников мета-агентом.

## Содержание
Мета-агент Rebecca предназначен для:
- Анализа и обработки различных типов источников
- Планирования задач для специализированных агентов
- Генерации плейбуков выполнения
- Координации агентной экосистемы

## Технические детали
Система использует:
- Многослойную архитектуру памяти
- Адаптивное отслеживание изменений
- Интеллектуальное распределение ресурсов

## Заключение
Данный файл служит для тестирования функциональности мета-агента.
"""
        
        # Создаем временный файл
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(test_content)
            return f.name


async def run_example_workflow():
    """Запускает пример полного workflow мета-агента."""
    
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    # Создаем mock компоненты для демонстрации
    class MockMemoryManager:
        async def store(self, layer, data, metadata=None):
            return "mock_id"
        
        async def retrieve(self, layer, query, filters=None):
            return []
    
    class MockIngestPipeline:
        def ingest_document(self, file_path, chunk_override=None):
            class MockEvent:
                id = "mock_event_id"
                attrs = {"text": "Mock document content"}
            return MockEvent()
        
        def process_git_repo(self, repo_url, branch="main", process_readme=True, process_source=True):
            return []
    
    class MockContextHandler:
        pass
    
    class MockBlueprintTracker:
        async def record_blueprint(self, blueprint, metadata=None, change_type="update", change_description=""):
            return 1
        
        async def get_latest_blueprint(self):
            class MockBlueprint:
                version = 1
                blueprint = {"sources": []}
            return MockBlueprint()
    
    # Создаем мета-агента
    memory_manager = MockMemoryManager()
    ingest_pipeline = MockIngestPipeline()
    context_handler = MockContextHandler()
    blueprint_tracker = MockBlueprintTracker()
    
    agent = RebeccaMetaAgent(
        memory_manager=memory_manager,
        ingest_pipeline=ingest_pipeline,
        context_handler=context_handler,
        blueprint_tracker=blueprint_tracker
    )
    
    # Создаем демо и запускаем
    demo = MetaAgentDemo(agent)
    results = await demo.run_full_demo()
    
    print("\n=== Результаты демонстрации ===")
    print(f"Демо ID: {results['demo_id']}")
    print(f"Продолжительность: {results['duration']:.2f} секунд")
    print(f"Шагов выполнено: {len(results['steps_completed'])}")
    print(f"Ошибок: {len(results['errors'])}")
    
    if results['errors']:
        print("\nОшибки:")
        for error in results['errors']:
            print(f"  - {error}")
    
    print("\nВыполненные шаги:")
    for step in results['steps_completed']:
        status = "✓" if step['success'] else "✗"
        print(f"  {status} {step['step']}")
    
    return results


# Утилитные функции

def save_agent_config(config: MetaAgentConfig, filepath: str):
    """Сохраняет конфигурацию агента в файл."""
    config_dict = asdict(config)
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_agent_config(filepath: str) -> MetaAgentConfig:
    """Загружает конфигурацию агента из файла."""
    with open(filepath, 'r') as f:
        config_dict = json.load(f)
    return MetaAgentConfig(**config_dict)


def create_sample_blueprint() -> Dict[str, Any]:
    """Создает пример blueprint архитектуры."""
    return {
        "version": 1,
        "components": {
            "ingest_pipeline": {
                "status": "active",
                "last_update": time.time(),
                "supported_formats": ["pdf", "markdown", "git", "json"]
            },
            "memory_manager": {
                "status": "active",
                "layers": ["core", "episodic", "semantic", "procedural", "vault", "security"],
                "total_items": 0
            },
            "meta_agent": {
                "status": "active",
                "agents_count": 0,
                "active_tasks": 0
            }
        },
        "integrations": {
            "orchestrator": "connected",
            "blueprint_tracker": "connected"
        },
        "metadata": {
            "created_at": time.time(),
            "last_modified": time.time(),
            "author": "rebecca_system"
        }
    }


def validate_agent_setup(agent: RebeccaMetaAgent) -> Dict[str, Any]:
    """Проверяет корректность настройки агента."""
    checks = {
        'components_loaded': True,
        'memory_manager_ok': hasattr(agent, 'memory_manager'),
        'ingest_pipeline_ok': hasattr(agent, 'ingest_pipeline'),
        'context_handler_ok': hasattr(agent, 'context_handler'),
        'blueprint_tracker_ok': hasattr(agent, 'blueprint_tracker'),
        'config_valid': agent.config is not None
    }
    
    all_ok = all(checks.values())
    
    return {
        'setup_valid': all_ok,
        'checks': checks,
        'issues': [k for k, v in checks.items() if not v]
    }


class OfflineLLMStub:
    """Deterministic LLM stub for offline mode testing."""
    
    @staticmethod
    def generate_response(prompt: str, max_tokens: int = 100, temperature: float = 0.0) -> str:
        """
        Generate a deterministic response based on the prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens (ignored in offline mode)
            temperature: Temperature parameter (ignored in offline mode)
            
        Returns:
            Deterministic response based on prompt hash
        """
        import hashlib
        
        hash_val = int(hashlib.sha256(prompt.encode()).hexdigest()[:16], 16)
        
        templates = [
            "Based on the provided information, I can help you with that task. The key considerations are outlined in the context.",
            "After analyzing the request, here is a structured approach: First, we need to understand the requirements. Second, implement the solution. Third, validate the results.",
            "The system has processed your request. In offline mode, responses are generated deterministically without external API calls.",
            "This is a deterministic response generated from the input prompt. The content is predictable and reproducible.",
            "According to the information provided, the recommended approach involves careful planning and systematic execution.",
        ]
        
        response = templates[hash_val % len(templates)]
        
        if "code" in prompt.lower() or "implement" in prompt.lower():
            response += " Here's a code-based solution approach."
        elif "analyze" in prompt.lower() or "review" in prompt.lower():
            response += " A thorough analysis reveals important insights."
        elif "test" in prompt.lower():
            response += " Comprehensive testing should be performed."
        
        return response
    
    @staticmethod
    def generate_embedding(text: str, model: str = "default") -> List[float]:
        """
        Generate a deterministic embedding for the text.
        
        Args:
            text: Input text
            model: Model name (ignored in offline mode)
            
        Returns:
            Deterministic embedding vector
        """
        import hashlib
        
        hash_obj = hashlib.sha256(text.encode('utf-8'))
        hash_bytes = hash_obj.digest()
        
        embedding_size = 384
        embedding = []
        
        for i in range(embedding_size):
            byte_idx = i % len(hash_bytes)
            value = (hash_bytes[byte_idx] / 255.0) * 2 - 1
            embedding.append(value)
        
        return embedding
    
    @staticmethod
    def score_relevance(query: str, document: str) -> float:
        """
        Score the relevance between query and document.
        
        Args:
            query: Query text
            document: Document text
            
        Returns:
            Relevance score between 0 and 1
        """
        import hashlib
        
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        
        if not query_words:
            return 0.5
        
        overlap = len(query_words & doc_words) / len(query_words)
        
        hash_val = int(hashlib.md5((query + document).encode()).hexdigest()[:8], 16)
        hash_component = (hash_val % 100) / 1000
        
        return min(1.0, 0.3 + overlap * 0.7 + hash_component)


# Экспорт утилит
__all__ = [
    "MetaAgentValidator",
    "MetaAgentTestData", 
    "MetaAgentDemo",
    "run_example_workflow",
    "save_agent_config",
    "load_agent_config",
    "create_sample_blueprint",
    "validate_agent_setup",
    "OfflineLLMStub"
]