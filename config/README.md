# Rebecca-Platform Configuration Guide

## Подготовка окружения

### Production режим

1. Установите необходимые хранилища и сервисы LLM:
   - Postgres (`docker run -p 5432:5432 postgres`)
   - MongoDB (при необходимости)
   - Qdrant (`docker run -p 6333:6333 qdrant/qdrant`)
   - Ollama (`ollama serve`), OpenAI, OpenRouter или локальный llama.cpp

2. Скопируйте `config/config.yaml` и обновите параметры:
   - Хосты, порты и креды для Postgres/Mongo/Qdrant
   - API ключи и базовые URL для LLM провайдеров
   - Список агентов и выбранные LLM

3. Перезапустите платформу. В логах появятся сообщения autodetect:
   - Если сервис не доступен, выводится предупреждение, но система продолжает работу.

### Offline режим (для тестирования)

Для запуска без внешних зависимостей используйте offline mode:

```bash
# Активация через переменные окружения
export REBECCA_OFFLINE_MODE=1
# или
export REBECCA_TEST_MODE=1
```

**В offline mode автоматически используются:**
- In-memory векторное хранилище вместо Qdrant/ChromaDB/Weaviate
- Детерминированные hash-based embeddings без загрузки моделей
- Rule-based NLP без spaCy моделей
- Stub LLM responses без обращения к внешним API

**Применение offline mode:**
- ✅ CI/CD пайплайны без доступа к внешним сервисам
- ✅ Unit и integration тесты с детерминированным поведением
- ✅ Разработка без установки тяжелых зависимостей
- ❌ Не для production использования

## Быстрая настройка

### Выбор LLM для агента
```yaml
agents:
  content_agent:
    role: content
    llm: creative  # ссылается на llm_adapters.creative
```

### Замена storage
```yaml
storage:
  primary:
    type: mongo
    host: localhost
    port: 27017
```

### Добавление ingest-пайплайна
```yaml
ingest:
  pipeline_map:
    csv: csv_parser
```

Реализуйте `csv_parser` по аналогии с имеющимися пайплайнами и зарегистрируйте его в модуле ingest.

## CI/CD и Docker

- Пример `docker-compose.yml`:
```yaml
version: "3.9"
services:
  api:
    build: .
    environment:
      CONFIG_PATH: /app/config/config.yaml
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - qdrant

  postgres:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"

  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
```

- Для CI используйте шаги:
  1. `pip install -r src/requirements.txt`
  2. `python -m pytest src/tests`
  3. При необходимости `python -m pytest tests`

## Расширение

- Multi-agent режим: добавляйте новые записи в `agents` и соответствующие роли/policies.
- Мультимодальные данные: расширяйте `pipeline_map` и создавайте новые parser-модули.
- Ограничить функционал: закомментируйте секции storage/LLM, которые не нужны.
