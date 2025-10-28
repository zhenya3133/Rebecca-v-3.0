# Руководство запуска Rebecca-Platform

## Требования
- Python 3.11+, Git, Docker
- Настроенный `.env` на основе [.env.example](.env.example)
- Конфиги [config/config.yaml](config/config.yaml) и [config/core.yaml](config/core.yaml)

## Подготовка
1. Скопируй `.env.example` → `.env`, проверь `REBECCA_API_TOKEN`, `REBECCA_CONFIG`, `REBECCA_CORE_CONFIG`.
2. Запусти `scripts/bootstrap.ps1 -DryRun` (PowerShell) для проверки зависимостей.

## Локальный запуск API

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r src\requirements.txt
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

### Linux/macOS (bash)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/requirements.txt
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

Для защищённых эндпоинтов используй заголовок `Authorization: Bearer $REBECCA_API_TOKEN`.

## Smoke-запросы
- Health: `curl http://127.0.0.1:8000/health`
- Run: `curl -H "Authorization: Bearer $REBECCA_API_TOKEN" -H "Content-Type: application/json" -d '{"input_data":"hello"}' http://127.0.0.1:8000/run`
- Upload: `curl -H "Authorization: Bearer $REBECCA_API_TOKEN" -F "file=@docs/ARCHITECTURE.md" http://127.0.0.1:8000/documents/upload`

## Тесты
- Smoke: `python -m pytest tests/test_health.py tests/test_run.py tests/test_core_connection.py src/retrieval/test_hybrid_retriever.py`
- Полный прогон: `python -m pytest -q`

## Docker compose
```bash
docker compose -f docker/docker-compose.local.yml up -d --build
docker compose -f docker/docker-compose.local.yml down
```

## CI
Пайплайн описан в [.github/workflows/tests.yml](.github/workflows/tests.yml) и запускает bootstrap DryRun плюс smoke-тесты на Windows и Linux.
