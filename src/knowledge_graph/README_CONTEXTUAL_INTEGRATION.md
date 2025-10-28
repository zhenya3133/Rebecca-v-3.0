# Contextual Knowledge Integration - Руководство пользователя

## Обзор

Модуль Contextual Knowledge Integration предоставляет мощную систему для контекстуальной интеграции знаний в Rebecca-Platform. Система автоматически анализирует задачи, извлекает релевантные знания и обогащает решения контекстуальной информацией.

## 🚀 Быстрый старт

### Базовое использование

```python
from knowledge_graph import create_context_engine, ContextRequest, KnowledgeDomain

# Инициализация
context_engine = await create_context_engine(memory_manager)

# Создание запроса контекста
request = ContextRequest(
    current_task=your_task,
    target_domains=[KnowledgeDomain.PSYCHOLOGY, KnowledgeDomain.MEDICINE],
    reasoning_depth=2,
    cross_domain_links=True
)

# Получение обогащенного контекста
context_result = await context_engine.process_context_request(request)
print(f"Найдено концептов: {len(context_result.relevant_concepts)}")
```

### Интеграция с агентом

```python
from knowledge_graph import ContextAwareAgentFactory

factory = ContextAwareAgentFactory(memory_manager, context_engine)
agent = factory.create_agent(AgentType.RESEARCH, capabilities)

# Автоматическое обогащение контекстом
result = await agent.execute_task_with_context(task)
print(f"Уверенность: {result.metrics.get('context_confidence', 0):.2%}")
```

## 📚 Основные компоненты

### 1. ContextEngine
Главный движок системы, координирующий все компоненты:

- **Dynamic Context Building** - построение контекста на основе задач
- **Knowledge Retrieval** - извлечение релевантных знаний
- **Multi-hop Reasoning** - рассуждения через связанные концепты
- **Temporal Validation** - проверка актуальности знаний
- **Cross-domain Linking** - междоменные связи

### 2. ContextAwareAgent
Расширенный BaseAgent с поддержкой контекста:

```python
# Автоматический анализ потребности в контексте
# Контекстуальное обогащение задач
# Адаптивная настройка параметров
# Обучение на основе результатов
```

### 3. Специализированные компоненты

#### DynamicContextBuilder
Анализирует контекст задачи и строит релевантный контекст.

#### ContextAwareRetriever
Улучшенный поиск знаний с учетом контекста.

#### MultiHopReasoningEngine
Построение цепочек рассуждений через связанные концепты.

#### TemporalValidationEngine
Проверка актуальности и согласованности знаний во времени.

#### CrossDomainLinkingEngine
Поиск и анализ связей между различными доменами знаний.

## 🎯 Психологический домен

### Специализированные примеры

Система включает готовые примеры для психологических задач:

```python
from knowledge_graph import PsychologyContextExamples

examples = PsychologyContextExamples(context_engine)

# Когнитивная оценка
await examples.example_1_cognitive_assessment_analysis()

# Планирование терапии  
await examples.example_2_therapy_session_planning()

# Оценка развития ребенка
await examples.example_3_child_development_assessment()
```

### База знаний психологии

```python
from knowledge_graph import PsychologyKnowledgeBase

kb = PsychologyKnowledgeBase()

# Информация о концепте
concept_info = kb.get_concept_info("cognitive_assessment")
related_concepts = kb.get_related_concepts("anxiety_disorders")
domains = kb.get_domains("child_development")
```

### Шаблоны задач

```python
from knowledge_graph import PsychologyTaskTemplates

templates = PsychologyTaskTemplates()
template = templates.get_template("cognitive_assessment")
customized_task = templates.customize_task(template, custom_values)
```

## 🔧 Конфигурация

### Настройка агента

```python
agent_config = {
    "auto_enrich_context": True,
    "reasoning_depth": 3,
    "freshness_threshold": 0.8,
    "cross_domain_links": True,
    "temporal_validation": True,
    "learning_enabled": True
}

agent.update_context_config(agent_config)
```

### Доменные специализации

```python
# Агент автоматически определяет специализации
specializations = agent.domain_specializations
# Результат: [KnowledgeDomain.PSYCHOLOGY, KnowledgeDomain.MEDICINE]
```

## 📊 Результаты и метрики

### Статистика выполнения

```python
# Получение статистики агента
capabilities = agent.get_context_capabilities()
print(f"Успешность обогащения: {capabilities['enrichment_success_rate']:.2%}")
print(f"Средняя уверенность: {capabilities['average_confidence']:.2%}")

# Статистика движка
stats = context_engine.get_statistics()
print(f"Всего запросов: {stats['total_requests']}")
print(f"Попадания в кэш: {stats['cache_hits']}")
```

### Метрики качества

- **Контекстуальная релевантность**: точность извлечения релевантных знаний
- **Временная актуальность**: процент актуальных знаний
- **Междоменные связи**: количество найденных связей между доменами
- **Уверенность контекста**: общая оценка качества контекстного анализа

## 🎭 Примеры использования

### Пример 1: Анализ когнитивной оценки

```python
task = TaskRequest(
    agent_type=AgentType.RESEARCH,
    task_type="cognitive_assessment",
    description="Провести когнитивную оценку пациента с подозрением на болезнь Альцгеймера",
    inputs={
        "patient_age": 68,
        "assessment_tools": ["MMSE", "MoCA"],
        "focus_areas": ["memory", "attention", "executive_function"]
    }
)

result = await agent.execute_task_with_context(task)

# Результат содержит:
# - Обогащенный контекст
# - Релевантные психологические концепты
# - Цепочки рассуждений
# - Междоменные связи с медициной
# - Временные инсайты
```

### Пример 2: Междоменный анализ

```python
# Задача, требующая знаний из нескольких доменов
request = ContextRequest(
    current_task=task,
    target_domains=[
        KnowledgeDomain.PSYCHOLOGY,
        KnowledgeDomain.MEDICINE, 
        KnowledgeDomain.EDUCATION
    ],
    cross_domain_links=True,
    reasoning_depth=3
)

result = await context_engine.process_context_request(request)

# Система найдет связи между:
# - Психологией и медициной (нейропсихология, клиническая психология)
# - Психологией и образованием (педагогическая психология, теории обучения)
# - Всеми тремя доменами (развивающая психология)
```

### Пример 3: Временная валидация

```python
# Система автоматически проверяет актуальность знаний
temporal_insights = result.temporal_insights
print(f"Согласованность: {temporal_insights['consistency_score']:.2%}")
print(f"Валидных единиц: {temporal_insights['valid_units']}")
print(f"Устаревших единиц: {temporal_insights['expired_units']}")

# Рекомендации по обновлению
for recommendation in temporal_insights['recommendations']:
    print(f"Рекомендация: {recommendation}")
```

## 🛠️ Расширение функциональности

### Добавление нового домена

```python
# 1. Определение нового домена
class KnowledgeDomain(str, Enum):
    NEW_DOMAIN = "new_domain"

# 2. Добавление специфической логики
class NewDomainLinker(CrossDomainLinkingEngine):
    async def find_domain_connections(self, domain1, domain2):
        # Специфическая логика для нового домена
        pass

# 3. Регистрация в системе
context_engine.cross_domain_linker = NewDomainLinker(memory_manager)
```

### Кастомные типы рассуждений

```python
class ReasoningHop(str, Enum):
    CUSTOM_REASONING = "custom_reasoning"

# Добавление в MultiHopReasoningEngine
async def custom_reasoning_chain(self, concept_id: str):
    # Реализация кастомного типа рассуждений
    pass
```

## 📈 Производительность

### Рекомендации по оптимизации

1. **Кэширование**: Система автоматически кэширует результаты
2. **Параллельная обработка**: Компоненты выполняются параллельно
3. **Ограничение глубины**: Настройте `reasoning_depth` согласно потребностям
4. **Доменная фильтрация**: Используйте только необходимые домены

### Мониторинг производительности

```python
# Встроенная статистика
health_check = await context_engine.health_check()
print(f"Статус: {health_check['status']}")
print(f"Активных компонентов: {sum(health_check['components'].values())}")

# Кастомные метрики
start_time = time.time()
result = await context_engine.process_context_request(request)
processing_time = time.time() - start_time
print(f"Время обработки: {processing_time:.3f}s")
```

## 🐛 Отладка и устранение неисправностей

### Логирование

```python
import logging

# Включение подробного логирования
logging.getLogger("knowledge_graph.context_engine").setLevel(logging.DEBUG)
logging.getLogger("knowledge_graph.agent_integration").setLevel(logging.INFO)
```

### Типичные проблемы

1. **Низкая уверенность контекста**
   - Увеличьте `reasoning_depth`
   - Проверьте релевантность доменов
   - Добавьте больше контекстной информации в задачу

2. **Мало междоменных связей**
   - Включите `cross_domain_links=True`
   - Проверьте, что задача действительно затрагивает несколько доменов
   - Расширьте список `target_domains`

3. **Устаревшие знания**
   - Включите `temporal_validation=True`
   - Уменьшите `freshness_threshold`
   - Обновите базу знаний в MemoryManager

### Тестирование компонентов

```python
# Тестирование отдельных компонентов
from knowledge_graph import (
    DynamicContextBuilder,
    ContextAwareRetriever,
    MultiHopReasoningEngine
)

# Тест построителя контекста
builder = DynamicContextBuilder(memory_manager)
context = await builder.build_dynamic_context(request)

# Тест retriever
retriever = ContextAwareRetriever(memory_manager)
knowledge = await retriever.retrieve_relevant_knowledge(context, domains)

# Тест рассуждений
reasoning = MultiHopReasoningEngine(memory_manager)
relations = await reasoning.multi_hop_reasoning(concepts, depth=2)
```

## 📚 Дополнительные ресурсы

- **Полный API**: См. документацию в `context_engine.py`
- **Примеры психологии**: `psychology_examples.py`
- **Интеграция агентов**: `agent_integration.py`
- **Отчет о реализации**: `/workspace/reports/contextual_integration_implementation.md`

## 🎯 Заключение

Система Contextual Knowledge Integration предоставляет мощные инструменты для интеллектуального анализа задач и обогащения решений контекстуальными знаниями. Особенно эффективна для психологических и междисциплинарных задач.

Для начала работы рекомендуется:
1. Запустить примеры из `psychology_examples.py`
2. Интегрировать `ContextAwareAgent` в существующие агенты
3. Настроить доменные специализации согласно вашим потребностям
4. Мониторить метрики производительности и качества

---

**Версия**: 1.0.0  
**Последнее обновление**: 28.10.2025  
**Документация**: Rebecca-Platform Development Team
