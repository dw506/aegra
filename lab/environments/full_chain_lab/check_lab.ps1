# Verify full_chain_lab topology correctness (PowerShell)
$Pass = 0
$Fail = 0

function Check($Name, $Condition) {
    if ($Condition) {
        Write-Host "[PASS] $Name" -ForegroundColor Green
        $script:Pass++
    } else {
        Write-Host "[FAIL] $Name" -ForegroundColor Red
        $script:Fail++
    }
}

function Test-ContainerTcp($Container, $TargetHost, $Port, $TimeoutSeconds = 3) {
    docker exec $Container python -c "import socket, sys; s=socket.socket(); s.settimeout(float(sys.argv[3])); s.connect((sys.argv[1], int(sys.argv[2]))); s.close()" $TargetHost $Port $TimeoutSeconds 2>$null
    return $LASTEXITCODE -eq 0
}

# 1. aegra-api health
try {
    $r = docker exec aegra-api curl -sf http://localhost:8000/health 2>&1
    Check "aegra-api health" ($LASTEXITCODE -eq 0)
} catch { Check "aegra-api health" $false }

# 2. mcp-tools running
$running = docker inspect mcp-tools --format '{{.State.Running}}' 2>$null
Check "mcp-tools running" ($running -eq "true")

# 3. DMZ reachable from mcp-tools
Check "dmz_net reachable from mcp-tools" (Test-ContainerTcp "mcp-tools" "10.20.0.10" 8080 3)

# 4. internal_net NOT reachable from mcp-tools
Check "internal_net isolated from mcp-tools" (-not (Test-ContainerTcp "mcp-tools" "10.30.0.11" 8080 2))

# 5. pivot-ssh sees both networks
$dmzOk = Test-ContainerTcp "pivot-ssh" "10.20.0.10" 8080 2
$intOk = Test-ContainerTcp "pivot-ssh" "10.30.0.11" 8080 2
Check "pivot-ssh bridges dmz and internal" ($dmzOk -and $intOk)

# 6. internal services not reachable from host
try {
    $r = Invoke-WebRequest -Uri "http://10.30.0.11:8080" -TimeoutSec 2 -ErrorAction Stop
    Check "internal services not host-accessible" $false
} catch {
    Check "internal services not host-accessible" $true
}

Write-Host ""
Write-Host "Results: $Pass passed, $Fail failed"
exit $Fail
