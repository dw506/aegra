#!/usr/bin/env bash
# Start the full_chain_lab Docker environment
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../" && pwd)"

cd "$SCRIPT_DIR"

# Check that private secrets are configured
if [[ ! -f "$SCRIPT_DIR/private/.env" ]]; then
  echo "[WARN] private/.env not found. Copying from .env.example (no real secrets)."
  cp "$SCRIPT_DIR/private/goal_secret.env.example" "$SCRIPT_DIR/private/.env"
fi

echo "[INFO] Starting full_chain_lab environment..."
docker compose -f "$SCRIPT_DIR/docker-compose.yml" up -d --build

echo "[INFO] Waiting for services to be ready..."
sleep 5

echo "[INFO] Lab started. Run check_lab.sh to verify topology."
