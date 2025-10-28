param(
    [switch]$Docker,
    [switch]$RebuildDocker,
    [switch]$SkipFrontend
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$runtimeFile = Join-Path $scriptDir ".runtime-state.json"

function Write-RuntimeState {
    param(
        [string]$Mode,
        [Nullable[int]]$BackendPid,
        [Nullable[int]]$FrontendPid
    )

    $payload = [ordered]@{
        mode        = $Mode
        backendPid  = $BackendPid
        frontendPid = $FrontendPid
        updatedAt   = (Get-Date).ToString("o")
    }

    $payload | ConvertTo-Json | Set-Content -Path $runtimeFile -Encoding UTF8
}

function Ensure-Venv {
    $venvPath = Join-Path $scriptDir ".venv"
    $pythonExe = Join-Path $venvPath "Scripts\python.exe"

    if (-not (Test-Path $venvPath)) {
        Write-Host "Creating virtual environment (.venv)..."
        python -m venv $venvPath | Out-Null
    }

    if (-not (Test-Path $pythonExe)) {
        throw "Unable to locate python.exe in $venvPath"
    }

    Write-Host "Ensuring backend dependencies..."
    & $pythonExe -m pip install --upgrade pip | Out-Null
    & $pythonExe -m pip install -r (Join-Path $scriptDir "src\requirements.txt") | Out-Null

    return $pythonExe
}

function Ensure-EnvFile {
    $envPath = Join-Path $scriptDir ".env"
    if (Test-Path $envPath) { return }

    $defaultEnv = @"
# Local defaults created by start_rebecca.ps1
LLM_PROVIDER=ollama
LLM_MODEL=llama3:8b
OLLAMA_BASE_URL=http://localhost:11434

API_TOKEN=local-dev
AUTH_TOKEN=local-dev
REBECCA_API_TOKEN=local-dev

LLM_DEFAULT=ollama:llama3:8b
OLLAMA_MODELS=llama3:8b,qwen2.5:7b-instruct
OPENAI_MODELS=
"@

    Write-Host "Creating .env with default values..."
    $defaultEnv | Set-Content -Path $envPath -Encoding UTF8
}

if ($Docker) {
    try {
        docker version | Out-Null
    } catch {
        $dockerExe = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
        if (Test-Path $dockerExe) {
            Write-Host "Starting Docker Desktop..."
            Start-Process $dockerExe | Out-Null
            Start-Sleep -Seconds 8
        } else {
            throw "Docker CLI is not available and Docker Desktop was not found."
        }
    }

    $compose = "docker\docker-compose.mock.yml"
    if ($RebuildDocker) {
        docker compose -f $compose down --remove-orphans
        docker compose -f $compose up -d --build
    } else {
        docker compose -f $compose up -d
    }

    Write-RuntimeState -Mode "docker" -BackendPid $null -FrontendPid $null
    Write-Host "`nRebecca (docker mock) is running. Visit http://localhost:8000/docs"
    return
}

# Local mode
$pythonExe = Ensure-Venv
Ensure-EnvFile

$originalPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = if ($originalPythonPath) {
    "$scriptDir\src;$originalPythonPath"
} else {
    "$scriptDir\src"
}

$backendArgs = @(
    "-m", "uvicorn", "api:app",
    "--app-dir", "src",
    "--host", "0.0.0.0",
    "--port", "8000",
    "--reload",
    "--reload-dir", "src",
    "--reload-dir", "config"
)

Write-Host "Starting backend (uvicorn)..."
$backendProcess = Start-Process -FilePath $pythonExe -ArgumentList $backendArgs -WorkingDirectory $scriptDir -PassThru

$frontendProcess = $null
if (-not $SkipFrontend) {
    Write-Host "Starting frontend (pnpm dev --host)..."
    $frontendProcess = Start-Process -FilePath "pnpm" -ArgumentList "--dir", "frontend", "dev", "--", "--host" -WorkingDirectory $scriptDir -PassThru
}

$env:PYTHONPATH = $originalPythonPath

$frontendPid = $null
if ($frontendProcess) { $frontendPid = $frontendProcess.Id }

Write-RuntimeState -Mode "local" -BackendPid $backendProcess.Id -FrontendPid $frontendPid

Write-Host "`nRebecca (local) is running. Backend: http://localhost:8000/docs"
if (-not $SkipFrontend) {
    Write-Host "Frontend: http://localhost:5173"
} else {
    Write-Host "Frontend was not started (use -SkipFrontend)."
}
