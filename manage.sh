#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${ROOT_DIR}/runtime/openpi/scripts/docker/compose.orin.tb6r5.yml"
ENV_FILE="${ROOT_DIR}/runtime/openpi/scripts/docker/.env.orin.tb6r5"
POLICY_DIR="${POLICY_DIR:-${CHECKPOINT_HOST_DIR:-/home/ai/pi0.5/model/pytorch/pp_white_red/19999}}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-600}"
TB6R5_PACKAGE_ROOT="${TB6R5_PACKAGE_ROOT:-${ROOT_DIR}/../tb6r5_policy_infer}"
OPENPI_HOST_PYTHON="${OPENPI_HOST_PYTHON:-/home/ai/condaenv/openpi_orin/bin/python3}"

docker_compose() {
  sudo docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" "$@"
}

run_host_bridge() {
  TB6R5_PACKAGE_ROOT="${TB6R5_PACKAGE_ROOT}" \
  PYTHONPATH="${TB6R5_PACKAGE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${OPENPI_HOST_PYTHON}" "${ROOT_DIR}/run_pi05_tb6r5.py" "$@"
}

require_runtime() {
  test -f "${COMPOSE_FILE}" || { echo "Missing ${COMPOSE_FILE}" >&2; exit 1; }
  test -f "${ENV_FILE}" || { echo "Missing ${ENV_FILE}" >&2; exit 1; }
  mkdir -p "${ROOT_DIR}/cache/openpi" "${POLICY_DIR}/assets"
  sudo docker image inspect openpi_tb6r5_policy_server:orin-jp62 >/dev/null
}

require_checkpoint() {
  local missing=0
  for relative in model.safetensors assets/norm_stats.json; do
    if [[ -f "${POLICY_DIR}/${relative}" ]]; then
      echo "[OK] ${POLICY_DIR}/${relative}"
    else
      echo "[MISSING] ${POLICY_DIR}/${relative}" >&2
      missing=1
    fi
  done
  [[ "${missing}" -eq 0 ]] || exit 1

  "${OPENPI_HOST_PYTHON}" - "${POLICY_DIR}/model.safetensors" <<'PY'
import json
from pathlib import Path
import struct
import sys

path = Path(sys.argv[1])
with path.open("rb") as handle:
    header_size = struct.unpack("<Q", handle.read(8))[0]
    header = json.loads(handle.read(header_size))
offsets = [value["data_offsets"] for key, value in header.items() if key != "__metadata__"]
expected_size = 8 + header_size + max(end for _, end in offsets)
actual_size = path.stat().st_size
if actual_size != expected_size:
    raise SystemExit(
        f"[INVALID] truncated/corrupt safetensors: {path} "
        f"(actual={actual_size}, expected={expected_size})"
    )
print(f"[OK] safetensors structure and size: {path} ({actual_size} bytes)")
PY
}

command="${1:-status}"
case "${command}" in
  check)
    require_runtime
    require_checkpoint
    run_host_bridge --local-check --no-camera
    echo "[OK] local runtime is ready"
    ;;
  start)
    require_runtime
    require_checkpoint
    docker_compose up -d --wait --wait-timeout "${WAIT_TIMEOUT}"
    ;;
  stop)
    require_runtime
    docker_compose stop
    ;;
  placeholder-start)
    require_runtime
    docker_compose --profile placeholder up -d --wait --wait-timeout "${WAIT_TIMEOUT}" \
      tb6r5_placeholder_policy_server
    ;;
  placeholder-stop)
    require_runtime
    docker_compose --profile placeholder stop tb6r5_placeholder_policy_server
    ;;
  placeholder-logs)
    require_runtime
    docker_compose --profile placeholder logs --tail 200 -f tb6r5_placeholder_policy_server
    ;;
  restart)
    require_runtime
    require_checkpoint
    docker_compose restart
    docker_compose up -d --wait --wait-timeout "${WAIT_TIMEOUT}"
    ;;
  down)
    require_runtime
    docker_compose down
    ;;
  status)
    require_runtime
    docker_compose ps
    curl --noproxy '*' --fail --silent --show-error --max-time 3 \
      http://127.0.0.1:8002/healthz || true
    echo
    ;;
  logs)
    require_runtime
    docker_compose logs --tail 200 -f tb6r5_policy_server
    ;;
  *)
    echo "Usage: $0 {check|start|stop|restart|down|status|logs|placeholder-start|placeholder-stop|placeholder-logs}" >&2
    exit 2
    ;;
esac
