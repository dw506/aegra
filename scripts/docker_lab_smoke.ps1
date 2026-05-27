param(
    [switch]$KeepRunning
)

$ErrorActionPreference = "Stop"

$targets = @(
    "http://target-web-1:80/",
    "http://target-web-2:80/",
    "http://target-web-3:80/",
    "http://internal-service:80/"
)

function Assert-LastExitCode {
    param(
        [string]$Action
    )

    if ($LASTEXITCODE -ne 0) {
        throw "$Action failed with exit code $LASTEXITCODE"
    }
}

function Wait-HttpReady {
    param(
        [string]$Url
    )

    for ($attempt = 1; $attempt -le 30; $attempt++) {
        $previousErrorActionPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = "SilentlyContinue"
            $response = & curl.exe --noproxy "*" -fsS $Url 2>$null
            $exitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $previousErrorActionPreference
        }

        if ($exitCode -eq 0) {
            $response | Out-Host
            return
        }
        Start-Sleep -Seconds 1
    }

    throw "Timed out waiting for $Url"
}

docker compose build
Assert-LastExitCode "docker compose build"
docker compose up -d
Assert-LastExitCode "docker compose up"

try {
    Wait-HttpReady "http://127.0.0.1:8000/health"
    Wait-HttpReady "http://127.0.0.1:8000/ready"

    foreach ($target in $targets) {
        Write-Host "Running orchestrator smoke against $target"
        docker compose run --rm `
            -e AEGRA_RUN_VULHUB_ORCHESTRATOR_SMOKE=1 `
            -e AEGRA_VULHUB_BASE_URL=$target `
            aegra python -m pytest tests/test_vulhub_orchestrator_smoke.py::test_vulhub_orchestrator_cycle_builds_runtime_and_minimal_kg_chain -q
        Assert-LastExitCode "orchestrator smoke against $target"
    }
}
finally {
    if (-not $KeepRunning) {
        docker compose down
    }
}
