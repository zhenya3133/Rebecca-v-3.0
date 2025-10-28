#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  . "$ROOT_DIR/.env"
  set +a
fi

BASE_URL="${1:-}"
TOKEN="${2:-}"

if [[ -z "$BASE_URL" ]]; then
  HOST="${API_HOST:-127.0.0.1}"
  PORT="${API_PORT:-8000}"
  BASE_URL="http://$HOST:$PORT"
fi

if [[ -z "$TOKEN" ]]; then
  TOKEN="${REBECCA_API_TOKEN:-}"
fi

if [[ -z "$TOKEN" ]]; then
  echo "REBECCA_API_TOKEN is not set" >&2
  exit 1
fi

echo "Running smoke checks against $BASE_URL"

curl -fsSL "$BASE_URL/health" >/dev/null
echo "Health check passed"

curl -fsSL \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"input_data":"smoke-check"}' \
  "$BASE_URL/run" >/dev/null
echo "Run endpoint succeeded"

echo "Smoke pipeline completed"
