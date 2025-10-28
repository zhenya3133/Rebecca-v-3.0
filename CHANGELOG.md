## [freeze-2025-10-18] - 2025-10-18

### Added
- Расширенный набор агентов (architect, research_scout, knowledge_curator, blueprint_generator, qa_guardian, sec_ops, deployment_ops, ops_commander) с промптами и оркестратором.
- RebeccaCoreAdapter с конфигом `config/core.yaml`, hot-reload и тестами подключения.
- API эндпоинты: health, core-settings (GET/PUT), documents/upload, chat (REST + WebSocket), voice (STT/TTS).
- React UI: CoreSettings страница с формами, UploadDropzone для документов, ChatPanel с голосовыми заглушками.
- Mock docker-compose (`docker/docker-compose.mock.yml`) и установочные скрипты (`install/setup_mock.ps1`, `install/setup_mock.sh`).
- Installation manual (`install/manual.md`) с инструкциями mock → production.
- Документация README.md и CORE_AUDIT.md обновлена под freeze-состояние.

### Fixed
- Blueprint generator записывает артефакты в AdaptiveBlueprintTracker и память.
- Тесты покрывают документ-ингест, чат и voice заглушки, настройки ядра.

### Notes
- Смоук-тесты: `python -m pytest tests/test_core_connection.py`.
- Mock режим активен по умолчанию; реальные сервисы подключаются через обновление конфигов.
