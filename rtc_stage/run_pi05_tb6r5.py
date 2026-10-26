#!/usr/bin/env python3
"""Compatibility entry point for the local TB6-R5 OpenPI host backend."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "tb6r5_policy_infer"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from openpi_runner import *  # noqa: E402,F403
from openpi_runner import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
