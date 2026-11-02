#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${RECORDER_PYTHON:-/home/ai/condaenv/tb6r5/bin/python}"
SOCKET="${PI05_RECORDER_SOCKET:-/tmp/pi05_lerobot_recorder.sock}"
DEFAULT_PARENT="${LEROBOT_RECORD_ROOT:-/home/ai/pi0.5/data/lerobotdata}"
OUT="${1:-$DEFAULT_PARENT/pi05_episode_$(date +%Y%m%d_%H%M%S)}"

if [[ -e "$OUT" ]] && [[ -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Recording path is not empty: $OUT" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUT")"
echo "[RECORD] output: $OUT"
echo "[RECORD] start or keep inference running in another terminal"
echo "[RECORD] press Ctrl+C here to save one episode"
exec "$PY" -u "$ROOT/rtc_stage/lerobot_recorder.py" \
  --root "$OUT" \
  --fps "${FPS:-30}" \
  --episode-frames 1000000 \
  --socket "$SOCKET"
