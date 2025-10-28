# DROId Freeze Installation Manual

## Requirements
- Python 3.11
- Node.js 18+ (для UI дев сервера)
- Docker (опционально)

## Быстрый старт (Mock Mode)
1. Клонируйте репозиторий и checkout freeze тег/ветку.
2. Выберите один из вариантов запуска в мок-режиме:
   - Windows: powershell -ExecutionPolicy Bypass -File install/setup_mock.ps1
   - Unix: bash install/setup_mock.sh
   - Docker: docker compose -f docker/docker-compose.mock.yml up
3. API доступен на http://localhost:8000. UI dev сервер (при запуске) — http://localhost:5173.

## Mock инфраструктура
- Все хранилища in-memory, persistent storage не требуется.
- Chat/Voice используют заглушки, ответы приходят синхронно.
- Документы хранятся в объектном сторе in-memory.

## Проверка
- python -m pytest tests/test_core_connection.py — smoke проверки чата, загрузок и core настроек.

## Переход на production
1. Обновите `config/core.yaml` с реальными endpoint/token и параметрами LLM/STT/TTS.
2. Добавьте `.env` с секретами и подгрузите переменные окружения.
3. Подключите реальные DAO/ObjectStore/VectorStore в `src/api.py` вместо `InMemory*` реализаций.
4. Настройте подключение к Rebecca Core/агентам через обновление `core.yaml` и при необходимости дополните `config/providers.yaml`.
5. Интегрируйте ingest pipeline с S3/Postgres/Qdrant и др. сервисами (обновите соответствующие конфиги/клиенты).

## Полезные команды
- uvicorn src.api:app --reload — локальный запуск API.
- npm install && npm run dev -- --host (frontend) — UI дев сервер.
- python -m pytest — полный прогон тестов.

## Дальнейшие шаги
- См. README.md и CORE_AUDIT.md для архитектуры и обязанностей агентов.
- Подготовьте целевой CI/CD pipeline для production на основе freeze версии.
