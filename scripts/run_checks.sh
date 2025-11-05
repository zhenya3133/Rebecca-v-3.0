#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if command -v python >/dev/null 2>&1; then
  PYTHON="python"
else
  PYTHON="python3"
fi

"$PYTHON" -m ruff check src tests
"$PYTHON" -m black --check src tests
"$PYTHON" -m mypy src
"$PYTHON" -m pytest "$@"
