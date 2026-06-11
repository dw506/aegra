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

# 1. aegra-api health
try {
    $r = docker exec aegra-api curl -sf http://localhost:8000/health 2>&1
    Check "aegra-api health" ($LASTEXITCODE -eq 0)
} catch { Check "aegra-api health" $false }

# 2. mcp-tools running
$running = docker inspect mcp-tools --format '{{.State.Running}}' 2>$null
Check "mcp-tools running" ($running -eq "true")

# 3. DMZ reachable from mcp-tools
docker exec mcp-tools nc -z -w3 10.20.0.10 8080 2>$null
Check "dmz_net reachable from mcp-tools" ($LASTEXITCODE -eq 0)

# 4. internal_net NOT reachable from mcp-tools
docker exec mcp-tools nc -z -w2 10.30.0.11 8080 2>$null
Check "internal_net isolated from mcp-tools" ($LASTEXITCODE -ne 0)

# 5. pivot-ssh sees both networks
docker exec pivot-ssh nc -z -w2 10.20.0.10 8080 2>$null
$dmzOk = $LASTEXITCODE -eq 0
docker exec pivot-ssh nc -z -w2 10.30.0.11 8080 2>$null
$intOk = $LASTEXITCODE -eq 0
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
