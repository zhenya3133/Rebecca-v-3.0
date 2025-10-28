# Comprehensive Тесты для RebeccaMetaAgent

## Обзор

Этот файл содержит comprehensive тесты для мета-агента RebeccaMetaAgent с покрытием 95%+ всех методов и сценариев.

## Структура тестов

### 1. Unit тесты (TestUnitRebeccaMetaAgent)
- `test_agent_initialization()` - Инициализация агента
- `test_ingest_sources_single_file()` - Поглощение одиночного файла
- `test_ingest_sources_multiple_sources()` - Поглощение множественных источников
- `test_ingest_sources_invalid_source()` - Обработка некорректных источников
- `test_plan_agent_basic()` - Базовое планирование задач
- `test_plan_agent_with_context()` - Планирование с контекстом
- `test_plan_agent_different_types()` - Планирование для разных типов задач
- `test_plan_agent_priorities()` - Планирование с разными приоритетами (параметризованный)
- `test_generate_playbook()` - Генерация плейбука
- `test_generate_playbook_complex_task()` - Генерация плейбука для сложной задачи
- `test_task_assignment()` - Распределение задач
- `test_resource_allocation()` - Выделение ресурсов
- `test_get_status_system()` - Получение системного статуса
- `test_get_status_task()` - Получение статуса конкретной задачи

### 2. Integration тесты (TestIntegrationRebeccaMetaAgent)
- `test_full_workflow()` - Полный workflow с мета-агентом
- `test_memory_integration()` - Интеграция с MemoryManager
- `test_ingest_integration()` - Интеграция с IngestPipeline
- `test_multiple_agents_workflow()` - Workflow с несколькими агентами

### 3. Performance тесты (TestPerformanceRebeccaMetaAgent)
- `test_large_task_planning()` - Планирование для больших проектов (50 задач)
- `test_concurrent_agents()` - Одновременная работа нескольких агентов (20 агентов)
- `test_memory_performance()` - Производительность работы с памятью (100 задач)
- `test_resource_allocation_performance()` - Производительность выделения ресурсов (50 ресурсов)

### 4. Mock тесты (TestMockRebeccaMetaAgent)
- `test_mock_memory_manager()` - Моки для MemoryManager
- `test_mock_ingest_pipeline()` - Моки для IngestPipeline
- `test_mock_specialized_agents()` - Моки для специализированных агентов
- `test_mock_logging()` - Моки для логгирования

### 5. Edge cases (TestEdgeCasesRebeccaMetaAgent)
- `test_empty_input()` - Обработка пустых данных
- `test_failed_ingestion()` - Обработка сбоев ingest
- `test_complex_planning()` - Сложные сценарии планирования
- `test_extreme_priorities()` - Крайние приоритеты
- `test_invalid_task_types()` - Некорректные типы задач
- `test_very_long_descriptions()` - Очень длинные описания
- `test_special_characters()` - Специальные символы в данных

### 6. Component тесты
- `TestTaskAnalyzer` - Тесты анализатора задач
- `TestResourceOptimizer` - Тесты оптимизатора ресурсов
- `TestPlaybookGenerator` - Тесты генератора плейбуков
- `TestMetaAgentValidator` - Тесты валидатора мета-агента

## Фикстуры

### Тестовые конфигурации
- `base_config()` - Базовая конфигурация
- `strict_config()` - Строгая конфигурация для stress тестов
- `lenient_config()` - Лояльная конфигурация для integration тестов

### Mock объекты
- `mock_memory_manager()` - MemoryManager с тестовыми данными
- `mock_empty_memory_manager()` - Пустой MemoryManager
- `mock_blueprint_tracker()` - AdaptiveBlueprintTracker
- `mock_ingest_pipeline()` - IngestPipeline
- `mock_context_handler()` - ContextHandler

### Тестовые данные
- `sample_task_plans()` - Список тестовых планов задач
- `rebecca_agent()` - Основной агент Rebecca
- `complex_rebecca_agent()` - Агент со строгой конфигурацией

## Параметризация

Тесты используют параметризацию для проверки различных сценариев:
- Разные приоритеты задач (critical, high, medium, low, background)
- Разные типы задач (development, research, analysis, deployment, testing, documentation, optimization, monitoring)
- Различные конфигурации агентов

## Async тесты

Все основные операции мета-агента тестируются асинхронно с использованием pytest-asyncio:
- Поглощение источников
- Планирование задач
- Генерация плейбуков
- Выполнение операций с памятью

## Запуск тестов

```bash
# Все тесты
pytest tests/test_rebecca_meta_agent.py -v

# С определенной категорией
pytest tests/test_rebecca_meta_agent.py::TestUnitRebeccaMetaAgent -v
pytest tests/test_rebecca_meta_agent.py::TestIntegrationRebeccaMetaAgent -v
pytest tests/test_rebecca_meta_agent.py::TestPerformanceRebeccaMetaAgent -v

# С покрытием
pytest tests/test_rebecca_meta_agent.py --cov=src.rebecca --cov-report=html

# Параллельный запуск
pytest tests/test_rebecca_meta_agent.py -n auto

# Только быстрые тесты
pytest tests/test_rebecca_meta_agent.py -m "not slow"

# Только performance тесты
pytest tests/test_rebecca_meta_agent.py -m "performance"
```

## Ожидаемые результаты

- **Покрытие тестами**: 95%+ всех методов и сценариев
- **Время выполнения**: < 30 секунд для всех тестов
- **Performance тесты**: Планирование 50 задач < 10 сек, параллельное создание 20 плейбуков < 5 сек
- **Стабильность**: Все тесты должны проходить стабильно без флаки

## Метрики качества

- **Code Coverage**: Отслеживается покрытие кода
- **Performance Metrics**: Измеряется время выполнения операций
- **Memory Usage**: Отслеживается использование памяти
- **Concurrent Operations**: Тестируется параллельная работа агентов

## Отчетность

После выполнения тестов создаются отчеты:
- HTML отчет о покрытии
- JUnit XML для CI/CD
- Console output с детальными результатами