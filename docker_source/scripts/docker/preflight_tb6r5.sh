#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT_DIR}/scripts/docker/.env.tb6r5.example}"
COMPOSE_FILE="${COMPOSE_FILE:-${ROOT_DIR}/scripts/docker/compose.tb6r5.yml}"

CHECKPOINT_DIR="$(
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
  printf '%s' "${CHECKPOINT_DIR:-checkpoints/pi05_tb6r5/tb6r5_lora_3k/1500}"
)"
CHECKPOINT_PATH="${ROOT_DIR}/${CHECKPOINT_DIR}"

failures=0

check() {
  local name="$1"
  shift
  if "$@" >/tmp/preflight_tb6r5.out 2>&1; then
    echo "[OK] ${name}"
  else
    echo "[FAIL] ${name}"
    sed 's/^/  /' /tmp/preflight_tb6r5.out
    failures=$((failures + 1))
  fi
}

check "docker client" docker --version
check "docker compose plugin" docker compose version
check "docker buildx plugin" docker buildx version
check "docker daemon" docker info
check "nvidia container runtime" nvidia-container-runtime --version
check "docker can run containers" docker run --rm hello-world

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/tmp/preflight_tb6r5.out 2>&1; then
  echo "[OK] nvidia-smi"
else
  echo "[WARN] nvidia-smi"
  if [ -s /tmp/preflight_tb6r5.out ]; then
    sed 's/^/  /' /tmp/preflight_tb6r5.out
  else
    echo "  nvidia-smi is not available or no GPU is mounted yet."
  fi
fi

for required in model.safetensors metadata.pt assets/norm_stats.json; do
  if [ -f "${CHECKPOINT_PATH}/${required}" ]; then
    echo "[OK] checkpoint ${required}"
  else
    echo "[FAIL] checkpoint ${required}"
    echo "  Missing: ${CHECKPOINT_PATH}/${required}"
    failures=$((failures + 1))
  fi
done

check "compose config" docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" config

rm -f /tmp/preflight_tb6r5.out

if [ "${failures}" -gt 0 ]; then
  echo "Preflight finished with ${failures} failure(s)."
  exit 1
fi

echo "Preflight passed."
