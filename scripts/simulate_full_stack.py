#!/usr/bin/env python3
"""Run the in-process web/edge/vision simulation from a source checkout."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from iot_servo_tracker.sim.full_stack import main  # noqa: E402


if __name__ == "__main__":
    main()
