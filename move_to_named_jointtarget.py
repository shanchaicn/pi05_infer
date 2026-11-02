#!/usr/bin/env python3
"""Move TB6-R5 to a controller-side named joint target after an RTC stage."""
from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default="192.168.2.151")
    parser.add_argument("--rpc-port", type=int, default=5868)
    parser.add_argument("--jointtarget", default="act_vla_0")
    parser.add_argument("--speed", default="v30")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse-enabled", action="store_true")
    parser.add_argument("--keep-enabled", action="store_true")
    args = parser.parse_args()

    if not args.jointtarget or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.jointtarget):
        parser.error("--jointtarget contains unsupported characters")
    if not args.speed or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in args.speed):
        parser.error("--speed contains unsupported characters")
    if args.timeout_s <= 0:
        parser.error("--timeout-s must be positive")

    arm_inner = f"MoveAbsJ --jointtarget={args.jointtarget} --speed={args.speed}"
    exact = f"{{{arm_inner}||NotRunExecute}}"
    if args.dry_run:
        print(f"[SEQUENCE][DRY-RUN] would send: {exact}", flush=True)
        return 0

    from tb6r5_policy_infer.hardware.tb6r5 import TB6R5Interface

    robot = TB6R5Interface(
        args.robot_ip,
        args.rpc_port,
        enable_topic=True,
        print_rpc=True,
        g_model=3,
        cd_version=45,
        subloop=0,
        reuse_enabled=args.reuse_enabled,
    )
    move_completed = False
    try:
        robot.connect()
        print(f"[SEQUENCE] returning with: {exact}", flush=True)
        ok = robot.send_dual_model(
            arm_inner,
            "NotRunExecute",
            timeout_ms=int(args.timeout_s * 1000),
            sleep_s=0.1,
            log_kind="sequence MoveAbsJ",
        )
        if not ok:
            raise RuntimeError("controller rejected sequence MoveAbsJ")
        # The synchronous RPC normally returns after MoveAbsJ. Confirm that
        # feedback is stationary before the next policy process reconnects.
        time.sleep(0.2)
        if not robot._wait_motion_settled(args.timeout_s, target_q=None):
            raise RuntimeError(f"robot did not settle within {args.timeout_s:g}s")
        move_completed = True
        print(f"[SEQUENCE] reached named joint target {args.jointtarget}", flush=True)
        return 0
    finally:
        robot.disconnect(disable=not (args.keep_enabled and move_completed))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"[SEQUENCE][ABORT] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
