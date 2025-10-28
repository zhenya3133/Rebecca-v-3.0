param(
  [string]$BaseUrl,
  [string]$Token
)

$ErrorActionPreference = "Stop"

function Load-DotEnv {
  if (-not (Test-Path ".\.env")) { return }
  Get-Content ".\.env" | Where-Object { $_ -match '^\s*[^#].*=' } | ForEach-Object {
    $k, $v = $_.Split('=', 2)
    if ($k -and $v) { $env:$k = $v }
  }
}

function Resolve-BaseUrl {
  param([string]$Url)
  if ($Url) { return $Url }
  $host = if ($env:API_HOST) { $env:API_HOST } else { "127.0.0.1" }
  $port = if ($env:API_PORT) { $env:API_PORT } else { "8000" }
  return "http://$host:$port"
}

function Resolve-Token {
  param([string]$Explicit)
  if ($Explicit) { return $Explicit }
  if ($env:REBECCA_API_TOKEN) { return $env:REBECCA_API_TOKEN }
  throw "REBECCA_API_TOKEN is not set."
}

Load-DotEnv
$resolvedBase = Resolve-BaseUrl -Url $BaseUrl
$resolvedToken = Resolve-Token -Explicit $Token

Write-Host "Running smoke checks against $resolvedBase" -ForegroundColor Cyan

$headers = @{ Authorization = "Bearer $resolvedToken" }

Invoke-WebRequest -Uri "$resolvedBase/health" -Method GET | Out-Null
Write-Host "Health check passed" -ForegroundColor Green

$body = @{ input_data = "smoke-check" } | ConvertTo-Json
Invoke-WebRequest -Uri "$resolvedBase/run" -Method POST -Headers $headers -ContentType "application/json" -Body $body | Out-Null
Write-Host "Run endpoint succeeded" -ForegroundColor Green

Write-Host "Smoke pipeline completed" -ForegroundColor Green
