#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/.env.orin.tb6r5}"
COMPOSE_FILE="${COMPOSE_FILE:-${SCRIPT_DIR}/compose.orin.tb6r5.yml}"

if [[ ! -r "${ENV_FILE}" ]]; then
  echo "[FAIL] missing environment file: ${ENV_FILE}" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

POLICY_PORT="${POLICY_PORT:-8002}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_tb6r5}"
CHECKPOINT_HOST_DIR="${CHECKPOINT_HOST_DIR:-/home/ai/pi0.5/pi05_infer/policy}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/home/test/.cache/openpi}"
ORIN_BASE_IMAGE="${ORIN_BASE_IMAGE:-nvcr.io/nvidia/pytorch:25.04-py3-igpu}"

failures=0

pass() { printf '[OK] %s\n' "$1"; }
warn() { printf '[WARN] %s\n' "$1"; }
fail() { printf '[FAIL] %s\n' "$1"; failures=$((failures + 1)); }

if [[ "$(uname -m)" == "aarch64" ]]; then
  pass "host architecture aarch64"
else
  fail "host architecture is $(uname -m), expected aarch64"
fi

if [[ -r /etc/nv_tegra_release ]] && grep -q 'R36' /etc/nv_tegra_release; then
  pass "Jetson L4T R36 host"
else
  fail "expected a Jetson L4T R36 host"
fi

if [[ "${POLICY_CONFIG}" == "pi05_tb6r5" ]]; then
  pass "policy config ${POLICY_CONFIG}"
else
  warn "policy config is ${POLICY_CONFIG}; checkpoint was prepared for pi05_tb6r5"
fi

for required in model.safetensors metadata.pt assets/norm_stats.json; do
  if [[ -r "${CHECKPOINT_HOST_DIR}/${required}" ]]; then
    pass "checkpoint ${required}"
  else
    fail "missing checkpoint file ${CHECKPOINT_HOST_DIR}/${required}"
  fi
done

if [[ "${CHECKPOINT_HOST_DIR}" == /home/* ]]; then
  pass "checkpoint is on /home NVMe"
else
  warn "checkpoint is not under /home: ${CHECKPOINT_HOST_DIR}"
fi

if [[ "${OPENPI_DATA_HOME}" == /home/* ]]; then
  pass "OpenPI cache is on /home NVMe"
else
  fail "OpenPI cache must be under /home, not ${OPENPI_DATA_HOME}"
fi

port_valid=true
if [[ ! "${POLICY_PORT}" =~ ^[0-9]+$ ]] || (( POLICY_PORT < 1 || POLICY_PORT > 65535 )); then
  fail "POLICY_PORT must be an integer from 1 to 65535, got ${POLICY_PORT@Q}"
  port_valid=false
fi

if [[ "${port_valid}" == true ]] && ss -H -ltn | awk '{print $4}' | grep -Eq "(^|:)${POLICY_PORT}$"; then
  fail "TCP port ${POLICY_PORT} is already listening; stop the existing native/container policy server first"
elif [[ "${port_valid}" == true ]]; then
  pass "TCP port ${POLICY_PORT} is free"
else
  warn "skipping listener check because POLICY_PORT is invalid"
fi

if command -v docker >/dev/null 2>&1; then
  pass "docker client $(docker --version | head -n1)"
else
  fail "docker client is missing"
fi

if docker compose version >/dev/null 2>&1; then
  pass "docker compose plugin"
else
  fail "docker compose plugin is unavailable"
fi

if docker buildx version >/dev/null 2>&1; then
  pass "docker buildx plugin"
else
  fail "docker buildx plugin is unavailable"
fi

if command -v nvidia-container-runtime >/dev/null 2>&1; then
  pass "NVIDIA container runtime"
else
  fail "nvidia-container-runtime is missing"
fi

if docker buildx imagetools inspect "${ORIN_BASE_IMAGE}" >/dev/null 2>&1; then
  pass "pinned base image ARM64 manifest is reachable"
else
  fail "cannot inspect pinned base image ${ORIN_BASE_IMAGE}"
fi

if docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" config >/dev/null 2>&1; then
  pass "compose configuration"
else
  fail "compose configuration is invalid"
fi

if docker info >/dev/null 2>&1; then
  pass "docker daemon access"
  docker_root="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || true)"
  if [[ "${docker_root}" == /home/* ]]; then
    pass "Docker data root is on /home (${docker_root})"
  else
    warn "Docker data root is ${docker_root:-unknown}; the Orin root filesystem is nearly full"
  fi
else
  fail "cannot access Docker daemon as $(id -un)"
  echo "       Grant access, then reconnect: sudo usermod -aG docker $(id -un)"
fi

echo
df -h / /home 2>/dev/null || true

if (( failures > 0 )); then
  echo "Preflight finished with ${failures} failure(s)." >&2
  exit 1
fi

echo "Preflight passed."
