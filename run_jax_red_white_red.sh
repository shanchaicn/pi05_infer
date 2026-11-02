#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-/home/ai/condaenv/tb6r5/bin/python}"
RUNNER="${JAX_RUNNER:-$ROOT/run_jax.sh}"
ROBOT_IP="${ROBOT_IP:-192.168.2.151}"
RPC_PORT="${RPC_PORT:-5868}"
ACTION_HORIZON="${ACTION_HORIZON:-${N_ACTION_STEPS:-25}}"
STAGE_CALLS="${STAGE_MAX_INFERENCE_CALLS:-120}"
HOME_TARGET="${HOME_JOINTTARGET:-act_vla_0}"
HOME_SPEED="${HOME_SPEED:-v50}"
HOME_TIMEOUT="${HOME_TIMEOUT_S:-120}"
PAUSE_S="${BETWEEN_STAGE_PAUSE_S:-2}"
KEEP_ENABLED="${KEEP_ENABLED_BETWEEN_STAGES:-1}"
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
[[ "$ACTION_HORIZON" =~ ^[1-9][0-9]*$ ]] && (( ACTION_HORIZON <= 50 )) || {
  echo "ACTION_HORIZON must be an integer in 1..50 for the H50 checkpoint" >&2
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

for index in 0 1 2; do
  stage=$((index + 1))
  echo "[SEQUENCE] non-RTC stage $stage/3: ${colors[$index]} block; execution horizon=$ACTION_HORIZON"
  echo "[SEQUENCE] task: ${tasks[$index]}"

  runner_handoff=()
  home_handoff=()
  if [[ "${REAL:-0}" == 1 && "$KEEP_ENABLED" == 1 ]]; then
    runner_handoff+=(--keep-robot-enabled)
    (( index > 0 )) && runner_handoff+=(--reuse-robot-enabled)
    home_handoff+=(--reuse-enabled)
    (( index < 2 )) && home_handoff+=(--keep-enabled)
  fi

  # Keep live output and only tolerate the known vendor Topic SDK exit abort
  # after both task completion and robot disconnect are present in the log.
  stage_log="$(mktemp "/tmp/pi05_jax_stage_${stage}_XXXXXX.log")"
  set +e
  TASK="${tasks[$index]}" \
  N_ACTION_STEPS="$ACTION_HORIZON" \
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
      && grep -Eq '^\[OK\] Completed .*task completion detected' "$stage_log" \
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

echo "[SEQUENCE] all non-RTC red/white/red stages completed"
