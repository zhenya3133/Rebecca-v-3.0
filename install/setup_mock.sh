#!/bin/bash
set -euo pipefail

python -m pip install -r src/requirements.txt
uvicorn src.api:app --host 0.0.0.0 --port 8000
