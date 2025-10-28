Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Write-Host 'Installing dependencies...'
python -m pip install -r src/requirements.txt | Write-Output

Write-Host 'Starting Rebecca API in mock mode...'
uvicorn src.api:app --host 0.0.0.0 --port 8000
