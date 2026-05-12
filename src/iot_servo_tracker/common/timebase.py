"""Time helpers.

All runtime packets use microseconds so edge, server, and web processes can
compare timestamps without floating-point drift.
"""

from __future__ import annotations

import time


def now_us() -> int:
    """Return a monotonic timestamp in microseconds."""

    return time.monotonic_ns() // 1_000


def wall_clock_cmd_id(prefix: str = "cmd") -> str:
    """Create a short human-readable command id."""

    stamp_ms = int(time.time() * 1_000)
    return f"{prefix}-{stamp_ms}"
