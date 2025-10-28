# Rebecca-Platform Test Guide

## Overview

На текущем этапе поддерживаются smoke-проверки API и гибридного ретривера. Ночные регрессии и инкрементальные сценарии добавляются на следующих этапах roadmap.

## Smoke Suite
- Тесты отмечены маркером `@pytest.mark.smoke` и расположены в:
  - `tests/test_health.py`
  - `tests/test_run.py`
  - `tests/test_core_connection.py`
  - `src/retrieval/test_hybrid_retriever.py`
- Запуск локально: `python -m pytest -m smoke -q`
- Скрипты быстрого прогона:
  - Windows: `powershell -File scripts/smoke.ps1`
  - Linux/macOS: `bash scripts/smoke.sh`

## CI Pipeline
- Workflow `.github/workflows/tests.yml` выполняет:
  1. `scripts/bootstrap.ps1 -DryRun`
  2. `python -m pytest -m smoke -q` на Windows и Ubuntu раннерах
- Дополнительные маркеры (integration/nightly) будут добавлены вместе с расширением тестового покрытия.
