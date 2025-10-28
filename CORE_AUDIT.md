# Core Audit – DROId Freeze (18.10.2025)

## Системные контуры
- **API (`src/api.py`)** – FastAPI-приложение с REST/WebSocket чатами, загрузкой документов, health-check и core-settings. По умолчанию поднимает in-memory DAO/ObjectStore/VectorStore и использует `RebeccaCoreAdapter`.
- **Core Adapter (`src/core_adapter/`)** – слой интеграции с Rebecca Core (Context, Memory, Event bridge). Конфигурация загружается из `config/core.yaml`, поддерживает hot-reload и сохранение.
- **Orchestrator (`src/orchestrator/`)** – главный workflow, который управляет стадиями: исследование → документация → код → QA → SecOps → Deployment → Ops.
- **Memory (`src/memory_manager/`)** – многоуровневая память (core/episodic/semantic/procedural/vault/security) + `AdaptiveBlueprintTracker` для артефактов архитектуры.
- **Ingest (`src/ingest/`)** – загрузка документов, связывание с blueprint-трекером, генерация событий в in-memory DAO.
- **Frontend (`frontend/`)** – React + TypeScript: настройки ядра, загрузка документов, чат-панель с голосовыми заглушками.

## Агентный состав
- `architect`, `research_scout`, `knowledge_curator`, `blueprint_generator`, `qa_guardian`, `sec_ops`, `deployment_ops`, `ops_commander` – настроены через `config/config.yaml` и промпты в `config/prompts/`.
- Каждый агент имеет dedicated модуль `src/<agent>/main.py` и следует протоколу orchestrator.

## Конфигурация и окружение
- **Ядро**: `config/core.yaml` – endpoint, токены, LLM/STT/TTS, таймауты. Сохраняется через API `PUT /core-settings`.
- **Mock режим**: активен по умолчанию, InMemory реализации подключены в API. Для docker-старта используйте `docker/docker-compose.mock.yml` (переменная `MOCK_MODE=true`).
- **Скрипты**: `install/setup_mock.ps1` и `install/setup_mock.sh` – локальный запуск API (uvicorn) в mock-режиме.

## Тестирование
- `tests/test_core_connection.py` – smoke сценарии: запуск пайплайна, health, core-settings, загрузка документов, чат + voice заглушки.
- Дополнительно присутствуют тесты на память и агентов: `tests/test_agent_army.py`, `tests/test_memory_persistence.py`.

## Состояние freeze
- Репозиторий готов к тегированию (`freeze-2025-10-18`).
- Mock запуск подтверждён (инструкции в `install/manual.md`).
- Готовы шаги по переходу в production (README.md, manual).
