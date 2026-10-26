#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/.env.orin.tb6r5}"
COMPOSE_FILE="${COMPOSE_FILE:-${SCRIPT_DIR}/compose.orin.tb6r5.yml}"

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

compose=(docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}")

"${compose[@]}" ps
timeout_seconds="${VERIFY_TIMEOUT_SECONDS:-600}"
ready=false
for ((elapsed = 0; elapsed < timeout_seconds; elapsed += 5)); do
  if python3 - "${POLICY_PORT:-8002}" <<'PY'
import socket
import sys

try:
    with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=2):
        pass
except OSError:
    raise SystemExit(1)
PY
  then
    ready=true
    break
  fi

  if ! "${compose[@]}" ps --status running -q tb6r5_policy_server | grep -q .; then
    echo "Policy-server container stopped before becoming ready." >&2
    "${compose[@]}" ps >&2 || true
    "${compose[@]}" logs --tail=120 tb6r5_policy_server >&2 || true
    exit 1
  fi
  sleep 5
done

if [[ "${ready}" != true ]]; then
  echo "Policy server did not open port ${POLICY_PORT:-8002} within ${timeout_seconds}s." >&2
  "${compose[@]}" logs --tail=120 tb6r5_policy_server >&2 || true
  exit 1
fi

"${compose[@]}" exec -T tb6r5_policy_server python - <<'PY'
import jax
import platform
import torch
import torchvision
import transformers
from torchvision.ops import nms

print("machine:", platform.machine())
print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("torch CUDA available:", torch.cuda.is_available())
print("torch device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("torchvision:", torchvision.__version__)
print("jax:", jax.__version__, jax.default_backend(), jax.devices())
print("transformers:", transformers.__version__)

assert platform.machine() == "aarch64"
assert torch.version.cuda is not None
assert torch.cuda.is_available()
nms(torch.tensor([[0.0, 0.0, 1.0, 1.0]]), torch.tensor([1.0]), 0.5)
assert jax.default_backend() == "cpu"
assert transformers.__version__ == "4.53.2"
PY

python3 - "${POLICY_PORT:-8002}" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.create_connection(("127.0.0.1", port), timeout=5):
    print(f"policy server TCP port {port}: reachable")
PY

"${compose[@]}" logs --tail=80 tb6r5_policy_server
