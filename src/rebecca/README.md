# Rebecca Meta-Agent

Полноценный мета-агент для интеллектуального управления агентной экосистемой Rebecca Platform.

## Обзор

Мета-агент Rebecca обеспечивает координацию между специализированными агентами для решения сложных задач. Он анализирует входные данные, планирует выполнение задач, распределяет ресурсы и генерирует инструкции для эффективной работы агентной системы.

### Ключевые возможности

- **Поглощение и обработка источников**: Автоматическое извлечение и анализ информации из различных источников (файлы, Git-репозитории, веб-ресурсы)
- **Интеллектуальное планирование**: Анализ задач и создание оптимальных планов выполнения
- **Генерация плейбуков**: Создание детальных инструкций для агентов
- **Координация агентов**: Распределение задач по специализации агентов
- **Мониторинг и оптимизация**: Отслеживание производительности и оптимизация ресурсов

## Архитектура

### Основные компоненты

1. **RebeccaMetaAgent** - Основной класс мета-агента
2. **TaskAnalyzer** - Анализатор сложности и требований задач
3. **ResourceOptimizer** - Оптимизатор распределения ресурсов
4. **PlaybookGenerator** - Генератор инструкций выполнения
5. **AdaptiveBlueprintTracker** - Отслеживание изменений архитектуры

### Интеграция с системой

Мета-агент интегрируется со всеми основными компонентами платформы:

- **MemoryManager** - Многослойная система памяти (6 слоев)
- **IngestPipeline** - Обработка документов, аудио, видео, изображений, Git
- **Orchestrator ContextHandler** - Управление контекстом
- **BlueprintTracker** - Отслеживание архитектурных изменений

## Установка и настройка

### Базовое использование

```python
from src.rebecca import RebeccaMetaAgent, MetaAgentConfig
from src.memory_manager import MemoryManager
from src.ingest import IngestPipeline
from src.orchestrator import ContextHandler
from src.memory_manager.adaptive_blueprint import AdaptiveBlueprintTracker

# Создание компонентов
memory_manager = MemoryManager()
ingest_pipeline = IngestPipeline(memory_manager, dao, bm25, vec, graph_idx, graph_view, object_store)
context_handler = ContextHandler()
blueprint_tracker = AdaptiveBlueprintTracker(memory_manager.semantic)

# Создание мета-агента
agent = RebeccaMetaAgent(
    memory_manager=memory_manager,
    ingest_pipeline=ingest_pipeline,
    context_handler=context_handler,
    blueprint_tracker=blueprint_tracker
)

# Или с кастомной конфигурацией
config = MetaAgentConfig(
    max_concurrent_tasks=15,
    quality_threshold=0.9,
    enable_auto_scaling=True
)

agent = RebeccaMetaAgent(
    memory_manager=memory_manager,
    ingest_pipeline=ingest_pipeline,
    context_handler=context_handler,
    blueprint_tracker=blueprint_tracker,
    config=config
)
```

### Использование фабрики

```python
from src.rebecca import RebeccaMetaAgentFactory

# Базовая конфигурация
agent = RebeccaMetaAgentFactory.create_basic_agent(
    memory_manager=memory_manager,
    ingest_pipeline=ingest_pipeline,
    context_handler=context_handler
)

# Расширенная конфигурация
agent = RebeccaMetaAgentFactory.create_advanced_agent(
    memory_manager=memory_manager,
    ingest_pipeline=ingest_pipeline,
    context_handler=context_handler,
    blueprint_tracker=blueprint_tracker,
    config=config
)
```

## Основные интерфейсы

### 1. ingest_sources() - Поглощение источников

Обрабатывает различные типы источников и извлекает из них информацию.

```python
# Поглощение одиночного источника
source_ids = await agent.ingest_sources("path/to/document.pdf")

# Поглощение нескольких источников
sources = [
    "path/to/document1.pdf",
    "path/to/document2.md",
    "https://github.com/user/repo.git"
]
source_ids = await agent.ingest_sources(sources, {
    'chunk_size': 1000,
    'branch': 'main',
    'process_readme': True
})

# Поглощение с дополнительными опциями
source_ids = await agent.ingest_sources(
    sources,
    analysis_options={
        'extract_metadata': True,
        'generate_summary': True,
        'identify_topics': True
    }
)
```

**Параметры:**
- `sources`: Список источников или один источник
- `analysis_options`: Дополнительные опции анализа

**Возвращает:** Список ID обработанных источников

### 2. plan_agent() - Планирование задач

Создает детальный план выполнения задачи.

```python
# Базовое планирование
requirements = {
    'title': 'Разработка API',
    'description': 'Создать RESTful API для управления пользователями',
    'type': 'development',
    'priority': 'high'
}

task_plan = await agent.plan_agent(requirements)

# Планирование с контекстом
context = {
    'existing_components': ['auth_service'],
    'required_skills': ['python', 'postgresql'],
    'integration_points': ['payment_system'],
    'security_requirements': ['JWT auth', 'rate limiting']
}

task_plan = await agent.plan_agent(requirements, context)
```

**Параметры:**
- `requirements`: Требования к задаче
- `context`: Дополнительный контекст выполнения

**Возвращает:** Объект TaskPlan с детальным планом

### 3. generate_playbook() - Генерация плейбука

Создает пошаговые инструкции для выполнения задачи.

```python
# Генерация плейбука
playbook_steps = await agent.generate_playbook(task_plan)

# Генерация с контекстом агента
agent_context = {
    'agent_id': 'backend_agent_001',
    'capabilities': ['python', 'api_design'],
    'current_workload': 0.3
}

playbook_steps = await agent.generate_playbook(task_plan, agent_context)
```

**Параметры:**
- `task_plan`: План задачи
- `agent_context`: Контекст агента (опционально)

**Возвращает:** Список объектов PlaybookStep

### 4. execute_playbook() - Выполнение плейбука

Запускает выполнение созданного плейбука.

```python
# Выполнение плейбука
execution_results = await agent.execute_playbook(playbook_steps, assignment_id)

# Проверка результатов
if execution_results['overall_status'] == TaskStatus.COMPLETED:
    print("Задача выполнена успешно!")
    print(f"Качество выполнения: {execution_results['quality_score']:.2f}")
else:
    print("Обнаружены ошибки в выполнении:")
    for failed_step in execution_results['steps_failed']:
        print(f"- {failed_step['step_id']}: {failed_step['error']}")
```

## Структуры данных

### TaskPlan (План задачи)

```python
@dataclass
class TaskPlan:
    task_id: str                           # Уникальный ID задачи
    title: str                            # Название задачи
    description: str                      # Подробное описание
    task_type: TaskType                   # Тип задачи
    priority: TaskPriority                # Приоритет
    complexity_score: float = 0.0         # Оценка сложности (0-1)
    estimated_duration: int = 0           # Оценка времени (минуты)
    required_skills: List[AgentSpecialization]  # Требуемые навыки
    dependencies: List[str] = field(default_factory=list)    # Зависимости
    prerequisites: List[str] = field(default_factory=list)  # Предпосылки
    success_criteria: List[str] = field(default_factory=list) # Критерии успеха
    metadata: Dict[str, Any] = field(default_factory=dict)   # Метаданные
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
```

### AgentAssignment (Назначение агента)

```python
@dataclass
class AgentAssignment:
    assignment_id: str          # Уникальный ID назначения
    task_plan: TaskPlan         # План задачи
    agent_type: AgentSpecialization  # Специализация агента
    agent_id: str               # ID агента
    start_time: datetime        # Время начала
    end_time: Optional[datetime] = None     # Время окончания
    status: TaskStatus = TaskStatus.PLANNED # Статус выполнения
    progress: float = 0.0       # Прогресс (0-100%)
    quality_score: float = 0.0  # Оценка качества (0-1)
    iterations: int = 0         # Количество итераций
    context_data: Dict[str, Any] = field(default_factory=dict)  # Контекст
    error_logs: List[str] = field(default_factory=list)         # Логи ошибок
```

### PlaybookStep (Шаг плейбука)

```python
@dataclass
class PlaybookStep:
    step_id: str                    # Уникальный ID шага
    step_number: int                # Номер шага
    title: str                      # Название шага
    description: str                # Описание шага
    action_type: str                # Тип действия
    agent_instruction: str          # Инструкция для агента
    expected_output: str            # Ожидаемый результат
    success_criteria: List[str] = field(default_factory=list)  # Критерии успеха
    timeout_minutes: int = 60       # Таймаут (минуты)
    retry_count: int = 0            # Количество повторов
    max_retries: int = 3            # Максимум повторов
    conditional_logic: Optional[Dict[str, Any]] = None  # Условная логика
    data_flow: Dict[str, str] = field(default_factory=dict)  # Поток данных
```

### ResourceAllocation (Выделение ресурсов)

```python
@dataclass
class ResourceAllocation:
    allocation_id: str              # Уникальный ID выделения
    resource_type: str              # Тип ресурса
    resource_id: str                # ID ресурса
    task_id: str                    # ID задачи
    agent_id: str                   # ID агента
    allocated_at: datetime          # Время выделения
    capacity_used: float = 0.0      # Использование ресурса (0-1)
    efficiency_score: float = 1.0   # Оценка эффективности
    cost_estimate: float = 0.0      # Оценка стоимости
    utilization_metrics: Dict[str, float] = field(default_factory=dict)  # Метрики
```

## Конфигурация

### MetaAgentConfig

```python
@dataclass
class MetaAgentConfig:
    max_concurrent_tasks: int = 10                    # Макс. одновременных задач
    default_timeout_minutes: int = 60                 # Таймаут по умолчанию
    enable_auto_scaling: bool = True                  # Автомасштабирование
    enable_failover: bool = True                      # Резервирование
    quality_threshold: float = 0.8                    # Порог качества
    complexity_weight: float = 0.3                    # Вес сложности
    priority_weight: float = 0.4                      # Вес приоритета
    dependency_weight: float = 0.3                    # Вес зависимостей
    resource_optimization: bool = True                # Оптимизация ресурсов
    enable_learning: bool = True                      # Обучение
    memory_retention_days: int = 30                   # Хранение памяти
    enable_proactive_planning: bool = True            # Проактивное планирование
```

### Примеры конфигурации

```python
# Конфигурация для высокой нагрузки
high_load_config = MetaAgentConfig(
    max_concurrent_tasks=50,
    enable_auto_scaling=True,
    resource_optimization=True
)

# Конфигурация для качества
quality_config = MetaAgentConfig(
    quality_threshold=0.95,
    enable_failover=True,
    enable_learning=True
)

# Конфигурация для быстрого выполнения
fast_config = MetaAgentConfig(
    default_timeout_minutes=30,
    enable_proactive_planning=True,
    memory_retention_days=7
)
```

## Мониторинг и статус

### Получение статуса системы

```python
# Общий статус системы
system_status = await agent.get_status()
print(f"Активных агентов: {system_status['active_agents']}")
print(f"Задач в очереди: {system_status['queued_tasks']}")
print(f"Выполнено: {system_status['completed_tasks']}")
print(f"Ошибок: {system_status['failed_tasks']}")
```

### Статус конкретной задачи

```python
# Статус задачи
task_status = await agent.get_status(task_id)
print(f"Статус задачи: {task_status['status']}")
print(f"План задачи: {task_status['task_plan']['title']}")

# Подробности назначений
for assignment in task_status['assignments']:
    print(f"Агент: {assignment['agent_id']}")
    print(f"Прогресс: {assignment['progress']:.1f}%")
    print(f"Качество: {assignment['quality_score']:.2f}")
```

## Примеры использования

### Полный workflow

```python
import asyncio
from src.rebecca import RebeccaMetaAgent, MetaAgentConfig

async def complete_workflow_example():
    # 1. Инициализация агента
    agent = RebeccaMetaAgent(...)
    
    # 2. Поглощение источников
    source_ids = await agent.ingest_sources([
        "documentation/api_spec.md",
        "https://github.com/company/repo.git"
    ])
    
    # 3. Планирование задачи
    requirements = {
        'title': 'Разработка нового API эндпоинта',
        'description': 'Создать эндпоинт для управления профилями пользователей',
        'type': 'development',
        'priority': 'high',
        'success_criteria': [
            'Эндпоинт реализован',
            'Тесты написаны',
            'Документация создана'
        ]
    }
    
    context = {
        'existing_components': ['user_database', 'auth_service'],
        'required_skills': ['python', 'fastapi', 'postgresql'],
        'integration_points': ['notification_service']
    }
    
    task_plan = await agent.plan_agent(requirements, context)
    
    # 4. Генерация плейбука
    playbook_steps = await agent.generate_playbook(task_plan)
    
    # 5. Выполнение (симуляция)
    # В реальной системе здесь был бы вызов агентов
    
    # 6. Получение статуса
    status = await agent.get_status(task_plan.task_id)
    
    return {
        'source_ids': source_ids,
        'task_plan': task_plan,
        'playbook_steps': len(playbook_steps),
        'status': status
    }

# Запуск примера
results = asyncio.run(complete_workflow_example())
```

### Обработка ошибок

```python
async def robust_usage_example():
    try:
        # Попытка поглотить источники
        source_ids = await agent.ingest_sources(sources)
        
        # Планирование с валидацией
        requirements = validate_requirements(raw_requirements)
        task_plan = await agent.plan_agent(requirements)
        
        # Генерация плейбука
        playbook_steps = await agent.generate_playbook(task_plan)
        
        # Проверка плейбука
        validator = MetaAgentValidator()
        validation_result = validator.validate_playbook(playbook_steps)
        
        if not validation_result['valid']:
            print("Проблемы в плейбуке:")
            for error in validation_result['errors']:
                print(f"  - {error}")
            return None
        
        # Выполнение
        assignment_id = create_assignment(task_plan)
        results = await agent.execute_playbook(playbook_steps, assignment_id)
        
        # Проверка результатов
        if results['overall_status'] == TaskStatus.COMPLETED:
            print("Задача выполнена успешно!")
            return results
        else:
            print("Обнаружены ошибки:")
            for error in results.get('errors', []):
                print(f"  - {error}")
            return results
            
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        return None
```

## Утилиты и инструменты

### Валидация

```python
from src.rebecca.utils import MetaAgentValidator

validator = MetaAgentValidator()

# Валидация конфигурации
config_validation = validator.validate_config(config_dict)
if not config_validation['valid']:
    print("Ошибки конфигурации:")
    for error in config_validation['errors']:
        print(f"  - {error}")

# Валидация плана задачи
task_validation = validator.validate_task_plan(task_plan)
if task_validation['warnings']:
    print("Предупреждения:")
    for warning in task_validation['warnings']:
        print(f"  - {warning}")
```

### Тестовые данные

```python
from src.rebecca.utils import MetaAgentTestData

test_data = MetaAgentTestData()

# Создание примеров данных
sample_requirements = test_data.create_sample_requirements()
sample_context = test_data.create_sample_context()
sample_config = test_data.create_sample_config()
```

### Демонстрация

```python
from src.rebecca.utils import MetaAgentDemo

demo = MetaAgentDemo(agent)
results = await demo.run_full_demo()

print(f"Демонстрация завершена за {results['duration']:.2f} секунд")
print(f"Выполнено шагов: {len(results['steps_completed'])}")
```

## Лучшие практики

### 1. Управление ресурсами

```python
# Используйте оптимизацию ресурсов для больших проектов
config = MetaAgentConfig(
    max_concurrent_tasks=20,
    resource_optimization=True,
    enable_auto_scaling=True
)

# Мониторьте использование ресурсов
resource_metrics = await agent.get_resource_metrics()
if resource_metrics['utilization'] > 0.8:
    print("Высокая нагрузка, рассмотрите увеличение ресурсов")
```

### 2. Обеспечение качества

```python
# Установите высокий порог качества для критичных задач
config = MetaAgentConfig(quality_threshold=0.9)

# Реализуйте многоэтапную проверку качества
quality_steps = [
    "Код-ревью",
    "Автоматические тесты", 
    "Интеграционные тесты",
    "Производительность"
]
```

### 3. Обработка ошибок

```python
# Всегда проверяйте результаты выполнения
try:
    results = await agent.execute_playbook(playbook, assignment_id)
    
    if results['overall_status'] == TaskStatus.FAILED:
        # Логирование и анализ ошибок
        for failed_step in results['steps_failed']:
            logger.error(f"Шаг {failed_step['step_id']} провалился: {failed_step['error']}")
        
        # Возможность повторного выполнения
        if should_retry(results):
            results = await agent.execute_playbook(playbook, assignment_id)
            
except Exception as e:
    logger.critical(f"Критическая ошибка выполнения: {e}")
    # Уведомления, алерты, откат изменений
```

### 4. Мониторинг производительности

```python
# Регулярно проверяйте метрики
async def monitor_performance():
    while True:
        status = await agent.get_status()
        
        metrics = status['metrics']
        print(f"Выполнено задач: {metrics['tasks_completed']}")
        print(f"Среднее качество: {metrics['average_quality_score']:.2f}")
        print(f"Среднее время: {metrics['average_completion_time']:.1f} мин")
        
        await asyncio.sleep(60)  # Проверка каждую минуту
```

## Интеграция с агентами

### Поддерживаемые специализации агентов

- **Backend** - Серверная разработка, API, базы данных
- **Frontend** - Пользовательские интерфейсы, веб-разработка
- **Machine Learning** - ML модели, анализ данных, алгоритмы
- **Quality Assurance** - Тестирование, валидация, контроль качества
- **DevOps** - Развертывание, инфраструктура, CI/CD
- **Research** - Исследования, анализ, изучение технологий
- **Security** - Безопасность, аудит, соответствие стандартам

### Регистрация агентов

```python
# Добавление агентов в систему
agent.register_agent('backend_agent_001', AgentSpecialization.BACKEND)
agent.register_agent('frontend_agent_001', AgentSpecialization.FRONTEND)
agent.register_agent('ml_agent_001', AgentSpecialization.MACHINE_LEARNING)

# Проверка доступных агентов
available_agents = agent.get_available_agents()
for agent_id, specialization in available_agents.items():
    print(f"Агент {agent_id}: {specialization.value}")
```

## Производительность и масштабирование

### Оптимизация для больших нагрузок

```python
# Настройка для высокопроизводительной работы
production_config = MetaAgentConfig(
    max_concurrent_tasks=100,
    enable_auto_scaling=True,
    resource_optimization=True,
    memory_retention_days=90,
    enable_learning=True
)

# Использование кэширования
agent.enable_caching()
agent.set_cache_size(10000)
agent.set_cache_ttl(3600)  # 1 час
```

### Мониторинг производительности

```python
# Получение метрик производительности
metrics = await agent.get_performance_metrics()
print(f"Запросов в секунду: {metrics['requests_per_second']}")
print(f"Среднее время отклика: {metrics['average_response_time']:.2f}ms")
print(f"Использование памяти: {metrics['memory_usage']:.1f}%")
```

## Расширение функциональности

### Добавление новых типов задач

```python
class CustomTaskType(TaskType):
    BLOCKCHAIN = "blockchain"
    IOT = "iot"
    AR_VR = "ar_vr"

# Интеграция в анализатор задач
class ExtendedTaskAnalyzer(TaskAnalyzer):
    async def analyze_blockchain_complexity(self, description: str) -> float:
        # Специализированный анализ для блокчейн задач
        pass
```

### Кастомные оптимизаторы ресурсов

```python
class GPUResourceOptimizer(ResourceOptimizer):
    async def optimize_gpu_allocation(self, tasks: List[TaskPlan]) -> List[ResourceAllocation]:
        # Специализированная оптимизация для GPU задач
        pass
```

## Заключение

Мета-агент Rebecca представляет собой мощную и гибкую систему для управления агентной экосистемой. Он обеспечивает:

- **Интеллектуальное планирование** на основе анализа требований и контекста
- **Автоматическую координацию** между специализированными агентами
- **Адаптивную оптимизацию** ресурсов и производительности
- **Комплексный мониторинг** качества и прогресса выполнения задач
- **Масштабируемую архитектуру** для работы с большими проектами

Система готова к продуктивному использованию и может быть легко расширена под специфические требования проекта.

---

**Версия:** 1.0.0  
**Последнее обновление:** 2025-10-28  
**Автор:** Rebecca Platform Team