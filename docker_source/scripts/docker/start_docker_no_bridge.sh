#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="${DOCKERD_LOG_FILE:-/tmp/dockerd.log}"
DATA_ROOT="${DOCKERD_DATA_ROOT:-/var/lib/docker-vfs}"

if docker info >/dev/null 2>&1; then
  echo "Docker daemon is already running."
  docker info --format 'Server={{.ServerVersion}} Storage={{.Driver}} DefaultRuntime={{.DefaultRuntime}}'
  exit 0
fi

mkdir -p /var/run "${DATA_ROOT}"

# Some hosted/containerized machines do not allow Docker to create iptables
# NAT chains. The TB6R5 compose file uses host networking, so a no-bridge
# daemon is enough for building and running this policy server. vfs avoids
# overlay mount setup in restricted environments, at the cost of slower builds.
nohup dockerd \
  --host=unix:///var/run/docker.sock \
  --data-root="${DATA_ROOT}" \
  --storage-driver=vfs \
  --iptables=false \
  --bridge=none \
  --ip-forward=false \
  --ip-masq=false \
  >"${LOG_FILE}" 2>&1 &

for _ in $(seq 1 20); do
  if docker info >/dev/null 2>&1; then
    echo "Docker daemon started."
    docker info --format 'Server={{.ServerVersion}} Storage={{.Driver}} DockerRoot={{.DockerRootDir}} DefaultRuntime={{.DefaultRuntime}}'
    exit 0
  fi
  sleep 1
done

echo "Docker daemon did not become ready. Last log lines:"
tail -120 "${LOG_FILE}" || true
exit 1
