"""Timestamped circular buffers for frames and detection history."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Generic, TypeVar

from iot_servo_tracker.common.packets import BBox, TrackingResult

T = TypeVar("T")


@dataclass(frozen=True)
class TimedRecord(Generic[T]):
    ts_us: int
    value: T


class RingBuffer(Generic[T]):
    def __init__(self, maxlen: int = 60) -> None:
        self.records: deque[TimedRecord[T]] = deque(maxlen=maxlen)

    def append(self, ts_us: int, value: T) -> None:
        self.records.append(TimedRecord(ts_us=ts_us, value=value))

    def nearest(self, ts_us: int) -> TimedRecord[T] | None:
        if not self.records:
            return None
        return min(self.records, key=lambda item: abs(item.ts_us - ts_us))

    def latest(self) -> TimedRecord[T] | None:
        if not self.records:
            return None
        return self.records[-1]

    def __len__(self) -> int:
        return len(self.records)


class DetectionHistory:
    def __init__(self, maxlen: int = 10) -> None:
        self.results: deque[TrackingResult] = deque(maxlen=maxlen)

    def append(self, result: TrackingResult) -> None:
        if result.bbox is not None:
            self.results.append(result)

    def estimate_current(self, result: TrackingResult, now_us: int) -> BBox | None:
        """Approximate current bbox using the last two valid detections."""

        if result.bbox is None:
            return None
        if len(self.results) < 2:
            return result.bbox
        prev, latest = self.results[-2], self.results[-1]
        if prev.bbox is None or latest.bbox is None:
            return result.bbox
        dt_history_s = max((latest.ts_resp - prev.ts_resp) / 1_000_000.0, 1e-3)
        dx = latest.bbox.center[0] - prev.bbox.center[0]
        dy = latest.bbox.center[1] - prev.bbox.center[1]
        vx = dx / dt_history_s
        vy = dy / dt_history_s
        delay_s = max((now_us - result.ts_req) / 1_000_000.0, 0.0)
        return result.bbox.shifted(vx * delay_s, vy * delay_s)
