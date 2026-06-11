# Start the full_chain_lab Docker environment (PowerShell)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path "$ScriptDir\..\..\.."

Set-Location $ScriptDir

if (-not (Test-Path "$ScriptDir\private\.env")) {
    Write-Warning "private\.env not found. Copying from .env.example (no real secrets)."
    Copy-Item "$ScriptDir\private\goal_secret.env.example" "$ScriptDir\private\.env"
}

Write-Host "[INFO] Starting full_chain_lab environment..."
docker compose -f "$ScriptDir\docker-compose.yml" up -d --build

Write-Host "[INFO] Waiting for services to be ready..."
Start-Sleep -Seconds 5

Write-Host "[INFO] Lab started. Run check_lab.ps1 to verify topology."
