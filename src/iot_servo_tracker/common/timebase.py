"""Time helpers.

All runtime packets use microseconds so edge, server, and web processes can
compare timestamps without floating-point drift.
"""

from __future__ import annotations

import time
import uuid
from threading import Lock


_now_lock = Lock()
_last_now_us = 0


def now_us() -> int:
    """Return a strictly increasing monotonic timestamp in microseconds."""

    global _last_now_us
    current = time.monotonic_ns() // 1_000
    with _now_lock:
        if current <= _last_now_us:
            current = _last_now_us + 1
        _last_now_us = current
        return current


def wall_clock_cmd_id(prefix: str = "cmd") -> str:
    """Create a short human-readable command id."""

    stamp_ms = int(time.time() * 1_000)
    suffix = uuid.uuid4().hex[:8]
    return f"{prefix}-{stamp_ms}-{suffix}"
