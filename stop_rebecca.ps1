$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir
$runtimeFile = Join-Path $scriptDir ".runtime-state.json"

function Stop-ProcessSafe {
    param([int]$ProcessId, [string]$Name)

    try {
        $proc = Get-Process -Id $ProcessId -ErrorAction Stop
        Write-Host "Stopping process $Name (PID $ProcessId)..."
        $proc | Stop-Process -Force
    } catch {
        Write-Host "Process $Name (PID $ProcessId) is already stopped." -ForegroundColor Yellow
    }
}

if (-not (Test-Path $runtimeFile)) {
    Write-Host "Runtime state not found. Stopping docker mock by default..." -ForegroundColor Yellow
    docker compose -f "docker\docker-compose.mock.yml" down --remove-orphans
    return
}

$state = Get-Content $runtimeFile | ConvertFrom-Json

switch ($state.mode) {
    "docker" {
        $compose = "docker\docker-compose.mock.yml"
        docker compose -f $compose down --remove-orphans
        Write-Host "Rebecca (docker mock) stopped."
    }
    "local" {
        if ($state.backendPid) { Stop-ProcessSafe -ProcessId $state.backendPid -Name "uvicorn" }
        if ($state.frontendPid) { Stop-ProcessSafe -ProcessId $state.frontendPid -Name "pnpm dev" }
        Write-Host "Rebecca (local) stopped."
    }
    default {
        Write-Host "Unknown mode in runtime state. Attempting to stop docker..." -ForegroundColor Yellow
        docker compose -f "docker\docker-compose.mock.yml" down --remove-orphans
    }
}

Remove-Item $runtimeFile -ErrorAction SilentlyContinue
