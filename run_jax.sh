#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-/home/ai/condaenv/tb6r5/bin/python}"
MODEL_CONFIG="${MODEL_CONFIG:-auto}"
JAX_CHECKPOINT="${JAX_CHECKPOINT:-/home/ai/pi0.5/model/jax/pi05_pp_white_red_h50/19999}"
ROBOT_IP="${ROBOT_IP:-192.168.2.151}"
POLICY_PORT="${POLICY_PORT:-8003}"
CAMERA_IP="${CAMERA_IP:-$ROBOT_IP}"
CAMERA_PORT="${CAMERA_PORT:-6888}"
CAMERA_URLS="${CAMERA_URLS:-realsense_0=http://${CAMERA_IP}:${CAMERA_PORT}/RsCameraSensor/0/0/color,realsense_1=http://${CAMERA_IP}:${CAMERA_PORT}/RsCameraSensor/1/0/color}"
CAMERA_MODE="${CAMERA_MODE:-http}"
CAMERA_TOPIC_PORT="${CAMERA_TOPIC_PORT:-19091}"
CAMERA_TOPIC_SENSORS="${CAMERA_TOPIC_SENSORS:-realsense_0=RsCameraSensor#0#0,realsense_1=RsCameraSensor#1#0}"
TASK="${TASK:-Pick up the red block and place it in the box on the left}"

export MODEL_CONFIG JAX_CHECKPOINT

if [[ "${REAL:-0}" == 1 ]]; then
  MAX_CALLS="${MAX_INFERENCE_CALLS:-0}"
else
  MAX_CALLS="${MAX_INFERENCE_CALLS:-3}"
fi

args=(
  --robot-ip "$ROBOT_IP"
  --policy-port "$POLICY_PORT"
  --no-show-camera
  --fps "${FPS:-30}"
  --n-action-steps "${N_ACTION_STEPS:-50}"
  --max-inference-calls "$MAX_CALLS"
  --joint-step-max-rad "${JOINT_STEP_MAX_RAD:-0.1}"
  --ee-step-max-m "${EE_STEP_MAX_M:-0.1}"
  --rotation-step-max-rad "${ROTATION_STEP_MAX_RAD:-0.3}"
  --task "$TASK"
)

case "$CAMERA_MODE" in
  sdk|topic)
    args+=(--camera-topic-sensors "$CAMERA_TOPIC_SENSORS" --camera-topic-port "$CAMERA_TOPIC_PORT")
    if [[ "${CAMERA_TOPIC_FALLBACK_HTTP:-0}" == 1 ]]; then
      args+=(--camera-topic-fallback-http)
    fi
    ;;
  http)
    args+=(--camera-urls "$CAMERA_URLS")
    ;;
  *)
    echo "CAMERA_MODE must be sdk/topic or http" >&2
    exit 2
    ;;
esac

if [[ "${REAL:-0}" != 1 ]]; then
  args+=(--dry-run --print-rpc)
else
  args+=(--confirm-hardware "${CONFIRM_HARDWARE:-TB6R5_REAL_INFERENCE}")
fi

exec "$PY" "$ROOT/rtc_stage/run_pi05_tb6r5.py" "${args[@]}" "$@"
