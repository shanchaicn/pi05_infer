#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-/home/ai/condaenv/tb6r5/bin/python}"
RUNNER="${JAX_RTC_RUNNER:-$ROOT/run_jax_rtc.sh}"
ROBOT_IP="${ROBOT_IP:-192.168.2.151}"
RPC_PORT="${RPC_PORT:-5868}"
STAGE_CALLS="${STAGE_MAX_INFERENCE_CALLS:-120}"
HOME_TARGET="${HOME_JOINTTARGET:-act_vla_0}"
HOME_SPEED="${HOME_SPEED:-v60}"
HOME_TIMEOUT="${HOME_TIMEOUT_S:-120}"
PAUSE_S="${BETWEEN_STAGE_PAUSE_S:-2}"
KEEP_ENABLED="${KEEP_ENABLED_BETWEEN_STAGES:-1}"
INITIAL_HOME="${INITIAL_HOME:-1}"
PRECOMPILE_RTC="${PRECOMPILE_RTC:-1}"
TASK_COMPLETE_TCP="${TASK_COMPLETE_TCP:--0.190,-0.651,0.230}"
TASK_COMPLETE_RADIUS_M="${TASK_COMPLETE_RADIUS_M:-0.060}"
TASK_COMPLETE_GRIPPER_OPEN_M="${TASK_COMPLETE_GRIPPER_OPEN_M:-0.040}"
TASK_COMPLETE_HOLD_STEPS="${TASK_COMPLETE_HOLD_STEPS:-5}"
TASK_COMPLETE_MIN_STEPS="${TASK_COMPLETE_MIN_STEPS:-30}"
TASK_COMPLETE_GRACE_STEPS="${TASK_COMPLETE_GRACE_STEPS:-30}"
SEQUENCE_LOG="${SEQUENCE_LOG:-}"
if [[ "${REAL:-0}" == 1 && -z "$SEQUENCE_LOG" ]]; then
  mkdir -p "$ROOT/logs"
  SEQUENCE_LOG="$ROOT/logs/${0##*/}_$(date +%Y%m%d_%H%M%S).log"
fi
[[ -z "$SEQUENCE_LOG" ]] || echo "[SEQUENCE] log: $SEQUENCE_LOG"

TASK_1="${TASK_1:-Pick up the red block and place it in the box on the left}"
TASK_2="${TASK_2:-Pick up the white block and place it in the box on the left}"
TASK_3="${TASK_3:-Pick up the red block and place it in the box on the left}"
tasks=("$TASK_1" "$TASK_2" "$TASK_3")
colors=(red white red)
stage_calls=(
  "${STAGE_1_MAX_INFERENCE_CALLS:-$STAGE_CALLS}"
  "${STAGE_2_MAX_INFERENCE_CALLS:-$STAGE_CALLS}"
  "${STAGE_3_MAX_INFERENCE_CALLS:-$STAGE_CALLS}"
)
[[ "$KEEP_ENABLED" == 0 || "$KEEP_ENABLED" == 1 ]] || {
  echo "KEEP_ENABLED_BETWEEN_STAGES must be 0 or 1" >&2
  exit 2
}
[[ "$INITIAL_HOME" == 0 || "$INITIAL_HOME" == 1 ]] || {
  echo "INITIAL_HOME must be 0 or 1" >&2
  exit 2
}
[[ "$PRECOMPILE_RTC" == 0 || "$PRECOMPILE_RTC" == 1 ]] || {
  echo "PRECOMPILE_RTC must be 0 or 1" >&2
  exit 2
}
for calls in "${stage_calls[@]}"; do
  [[ "$calls" =~ ^[1-9][0-9]*$ ]] || {
    echo "Each stage MAX_INFERENCE_CALLS must be a positive integer" >&2
    exit 2
  }
done

if [[ "${REAL:-0}" == 1 ]]; then
  home_mode=()
else
  home_mode=(--dry-run)
fi

# Compile the final JAX RTC static shape before any robot movement.  The three
# real stages still perform their short safety warmup, but hit this cache.
if [[ "${REAL:-0}" == 1 && "$PRECOMPILE_RTC" == 1 ]]; then
  echo "[SEQUENCE] precompiling RTC before robot motion"
  PI05_RECORDER_SOCKET="/tmp/pi05_precompile_no_recorder.sock" \
  MAX_INFERENCE_CALLS=1 \
  REAL=0 \
  "$RUNNER" --rtc-warmup-only --no-camera "$@"
fi

# A previous aborted run can leave the TCP beside the box.  Always establish
# the same trained start pose before stage 1 so task completion cannot trigger
# from a previous stage's endpoint.  This initial move disconnects normally;
# keep-enabled handoff is used only between successful sequence stages.
if [[ "$INITIAL_HOME" == 1 ]]; then
  echo "[SEQUENCE] initializing at $HOME_TARGET ($HOME_SPEED)"
  PYTHONPATH="/home/ai/pi0.5/tb6r5_policy_infer${PYTHONPATH:+:$PYTHONPATH}" \
  "$PY" "$ROOT/move_to_named_jointtarget.py" \
    --robot-ip "$ROBOT_IP" \
    --rpc-port "$RPC_PORT" \
    --jointtarget "$HOME_TARGET" \
    --speed "$HOME_SPEED" \
    --timeout-s "$HOME_TIMEOUT" \
    "${home_mode[@]}"
  sleep "$PAUSE_S"
fi

for index in 0 1 2; do
  stage=$((index + 1))
  echo "[SEQUENCE] stage $stage/3: ${colors[$index]} block"
  echo "[SEQUENCE] task: ${tasks[$index]}"

  runner_handoff=()
  home_handoff=()
  if [[ "${REAL:-0}" == 1 && "$KEEP_ENABLED" == 1 ]]; then
    runner_handoff+=(--keep-robot-enabled)
    (( index > 0 )) && runner_handoff+=(--reuse-robot-enabled)
    home_handoff+=(--reuse-enabled)
    (( index < 2 )) && home_handoff+=(--keep-enabled)
  fi

  # Keep live output and inspect the exact child status. The vendor Topic SDK
  # can abort with 134 during interpreter finalization after a fully completed,
  # disconnected run. Only that narrowly verified case may continue.
  stage_log="$(mktemp "/tmp/pi05_rtc_stage_${stage}_XXXXXX.log")"
  set +e
  TASK="${tasks[$index]}" \
  MAX_INFERENCE_CALLS="${stage_calls[$index]}" \
  REAL="${REAL:-0}" \
  "$RUNNER" \
    --task-complete-tcp="$TASK_COMPLETE_TCP" \
    --task-complete-radius-m "$TASK_COMPLETE_RADIUS_M" \
    --task-complete-gripper-open-m "$TASK_COMPLETE_GRIPPER_OPEN_M" \
    --task-complete-hold-steps "$TASK_COMPLETE_HOLD_STEPS" \
    --task-complete-min-steps "$TASK_COMPLETE_MIN_STEPS" \
    --task-complete-grace-steps "$TASK_COMPLETE_GRACE_STEPS" \
    "${runner_handoff[@]}" \
    "$@" 2>&1 | tee "$stage_log"
  runner_status=${PIPESTATUS[0]}
  set -e
  if [[ -n "$SEQUENCE_LOG" ]]; then
    {
      echo "===== stage $stage: ${colors[$index]} ====="
      cat "$stage_log"
    } >> "$SEQUENCE_LOG"
  fi

  if (( runner_status != 0 )); then
    if (( runner_status == 134 )) \
      && grep -Eq '^\[OK\] RTC completed .*task completion detected' "$stage_log" \
      && grep -q '^TB6-R5 disconnected' "$stage_log"; then
      echo "[SEQUENCE][WARNING] Ignoring vendor Topic SDK shutdown abort after verified completion and robot disconnect" >&2
    else
      echo "[SEQUENCE][ABORT] stage $stage runner exited with status $runner_status" >&2
      rm -f "$stage_log"
      exit "$runner_status"
    fi
  fi
  rm -f "$stage_log"

  echo "[SEQUENCE] stage $stage completed; returning to $HOME_TARGET"
  PYTHONPATH="/home/ai/pi0.5/tb6r5_policy_infer${PYTHONPATH:+:$PYTHONPATH}" \
  "$PY" "$ROOT/move_to_named_jointtarget.py" \
    --robot-ip "$ROBOT_IP" \
    --rpc-port "$RPC_PORT" \
    --jointtarget "$HOME_TARGET" \
    --speed "$HOME_SPEED" \
    --timeout-s "$HOME_TIMEOUT" \
    "${home_handoff[@]}" \
    "${home_mode[@]}"

  if (( index < 2 )); then
    sleep "$PAUSE_S"
  fi
done

echo "[SEQUENCE] all red/white/red stages completed"
