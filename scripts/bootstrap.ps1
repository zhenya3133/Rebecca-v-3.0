param(
  [switch]$DryRun,
  [switch]$Force
)

$ErrorActionPreference = "Stop"
$global:BootstrapFailed = $false

function Step($m){ Write-Host "==> $m" -ForegroundColor Cyan }
function Info($m){ Write-Host "    $m" }
function Warn($m){ Write-Host "!!  $m" -ForegroundColor Yellow }
function Ok($m){ Write-Host "[OK] $m" -ForegroundColor Green }

# 0) Проверка места запуска
if (-not (Test-Path ".\.git") -and -not $Force) {
  Warn "Запусти скрипт из корня репозитория (где .git)"
  exit 1
}

# 1) Загрузка .env (если есть)
if (Test-Path ".\.env") {
  Step "Загружаю .env"
  (Get-Content .\.env) | Where-Object { $_ -match '^\s*[^#].*=' } | ForEach-Object {
    $k,$v = $_.Split('=',2); if($k -and $v){ Set-Item -Path Env:$k -Value $v }
  }
} else { Warn ".env не найден. Создай из .env.example" }

# 1a) Core config path sanity
$repo = (Get-Location).Path
if (-not $env:REBECCA_CORE_CONFIG) { $env:REBECCA_CORE_CONFIG = "config/core.yaml" }
$corePath = if ([System.IO.Path]::IsPathRooted($env:REBECCA_CORE_CONFIG)) {
  $env:REBECCA_CORE_CONFIG
} else {
  Join-Path $repo $env:REBECCA_CORE_CONFIG
}
if (-not (Test-Path $corePath)) { Warn "REBECCA_CORE_CONFIG не найден по пути $corePath" } else { Info "Core config: $corePath" }

# 2) PYTHONPATH (для import src.*)
$src  = Join-Path $repo "src"
$env:PYTHONPATH = $src

# 3) Node deps (если есть)
if (Test-Path ".\package.json") {
  Step "Node зависимости"
  if (Test-Path ".\pnpm-lock.yaml") {
    if ($DryRun) { Info "[DryRun] pnpm i" } else { pnpm i }
  } elseif (Test-Path ".\yarn.lock") {
    if ($DryRun) { Info "[DryRun] yarn install --frozen-lockfile" } else { yarn install --frozen-lockfile }
  } else {
    if ($DryRun) { Info "[DryRun] npm ci" } else { npm ci }
  }
} else { Info "package.json не найден — пропускаю Node" }

# 4) Python deps (если есть)
if ((Test-Path ".\pyproject.toml") -or (Test-Path ".\requirements.txt") -or (Test-Path ".\src\requirements.txt")) {
  Step "Python зависимости"
  if (-not (Test-Path ".\.venv")) {
    if ($DryRun) { Info "[DryRun] python -m venv .venv" } else { python -m venv .venv }
  }
  if (-not $DryRun) { & ".\.venv\Scripts\Activate.ps1" }
  if (Test-Path ".\src\requirements.txt") {
    if ($DryRun) { Info "[DryRun] pip install -r src\requirements.txt" } else { pip install -U pip; pip install -r src\requirements.txt }
  } elseif (Test-Path ".\requirements.txt") {
    if ($DryRun) { Info "[DryRun] pip install -r requirements.txt" } else { pip install -U pip; pip install -r requirements.txt }
  } else {
    Info "requirements не найдены — пропускаю установку"
  }
} else { Info "Python манифесты не найдены — пропускаю Python" }

# 5) DB dry-run (без записи)
if ($env:DB_TYPE -and $env:DB_HOST -and $env:DB_PORT) {
  Step "DB dry-run ($($env:DB_TYPE))"
  if ($env:DB_TYPE -eq "postgres") {
    $ok = (Test-NetConnection -ComputerName $env:DB_HOST -Port ([int]$env:DB_PORT)).TcpTestSucceeded
    if ($ok) { Ok "DB reachable $($env:DB_HOST):$($env:DB_PORT)" } else { Warn "DB not reachable" }
  } else { Info "Поддержан только postgres на этом шаге" }
} else { Info "DB_* не заданы — пропускаю" }

# 6) Smoke-тесты
Step "Smoke-тесты"
if ($DryRun) {
  Info "[DryRun] python -m pytest -m smoke -q"
} else {
  if (Test-Path ".\tests") {
    try { python -m pytest -m smoke -q } catch { Warn "pytest завершился с ошибкой"; $global:BootstrapFailed = $true; if(-not $Force){ throw } }
  } else { Info "Папка tests не найдена — пропускаю" }
}

if ($global:BootstrapFailed) {
  Warn "Bootstrap завершён с предупреждениями"
} else {
  Ok "Bootstrap завершён успешно"
}
Info "Для безопасной проверки: .\scripts\bootstrap.ps1 -DryRun"