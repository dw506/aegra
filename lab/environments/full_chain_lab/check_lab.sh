#!/usr/bin/env bash
# Verify full_chain_lab topology correctness
set -euo pipefail

PASS=0
FAIL=0

check() {
  local name="$1"
  local result="$2"
  if [[ "$result" == "ok" ]]; then
    echo "[PASS] $name"
    ((PASS++))
  else
    echo "[FAIL] $name: $result"
    ((FAIL++))
  fi
}

check_tcp() {
  local container="$1"
  local host="$2"
  local port="$3"
  local timeout="${4:-3}"
  docker exec "$container" python -c 'import socket, sys; s=socket.socket(); s.settimeout(float(sys.argv[3])); s.connect((sys.argv[1], int(sys.argv[2]))); s.close()' "$host" "$port" "$timeout" >/dev/null 2>&1
}

# 1. aegra-api health
if docker exec aegra-api curl -sf http://localhost:8000/health > /dev/null 2>&1; then
  check "aegra-api health" "ok"
else
  check "aegra-api health" "not reachable"
fi

# 2. mcp-tools running
if docker inspect mcp-tools --format '{{.State.Running}}' 2>/dev/null | grep -q "true"; then
  check "mcp-tools running" "ok"
else
  check "mcp-tools running" "not running"
fi

# 3. DMZ reachable from mcp-tools (nmap probe via docker exec)
if check_tcp mcp-tools 10.20.0.10 8080 3; then
  check "dmz_net reachable from mcp-tools" "ok"
else
  check "dmz_net reachable from mcp-tools" "not reachable"
fi

# 4. internal_net NOT directly reachable from mcp-tools
if check_tcp mcp-tools 10.30.0.11 8080 2; then
  check "internal_net isolated from mcp-tools" "FAIL - internal is reachable without pivot!"
else
  check "internal_net isolated from mcp-tools" "ok"
fi

# 5. pivot-ssh can see both networks, including the internal database
if check_tcp pivot-ssh 10.20.0.10 8080 2 && \
   check_tcp pivot-ssh 10.30.0.11 8080 2 && \
   check_tcp pivot-ssh 10.30.0.12 5432 2; then
  check "pivot-ssh bridges dmz and internal" "ok"
else
  check "pivot-ssh bridges dmz and internal" "one or both networks unreachable from pivot-ssh"
fi

# 6. internal postgres is initialized with the loot table
if docker exec internal-db-mock pg_isready -U dbadmin -d corp >/dev/null 2>&1 && \
   docker exec internal-db-mock psql -U dbadmin -d corp -Atc "select count(*) from loot_records" 2>/dev/null | grep -qx "1"; then
  check "internal-db-mock loot table initialized" "ok"
else
  check "internal-db-mock loot table initialized" "not ready or loot row missing"
fi

# 7. internal services not directly reachable from host
if curl -sf --max-time 2 http://10.30.0.11:8080 > /dev/null 2>&1; then
  check "internal services not host-accessible" "FAIL - accessible from host!"
else
  check "internal services not host-accessible" "ok"
fi

echo ""
echo "Results: $PASS passed, $FAIL failed"
[[ $FAIL -eq 0 ]]
