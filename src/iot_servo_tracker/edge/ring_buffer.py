"""Timestamped circular buffers for frames and detection history."""

from __future__ import annotations

import math
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
    def __init__(self, maxlen: int = 30, ema_alpha: float = 0.8) -> None:
        self.results: deque[TrackingResult] = deque(maxlen=maxlen)
        self.ema_alpha = ema_alpha
        self.ema_bbox: BBox | None = None
        self.ema_vx: float = 0.0
        self.ema_vy: float = 0.0
        self.raw_last_bbox: BBox | None = None
        self.raw_last_ts: int = 0

    def append(self, result: TrackingResult) -> None:
        if result.bbox is None:
            self.results.append(result)
            return

        dt = 0.0
        if self.raw_last_ts > 0:
            dt = max((result.ts_resp - self.raw_last_ts) / 1_000_000.0, 1e-3)

        # 1. Update Velocity EMA using raw coordinates
        if self.raw_last_bbox is not None and dt > 0:
            raw_vx = (result.bbox.center[0] - self.raw_last_bbox.center[0]) / dt
            raw_vy = (result.bbox.center[1] - self.raw_last_bbox.center[1]) / dt

            # Time-aware EMA for velocity (tau = 0.08s, gives ~0.33 alpha at 30fps).
            # Long gaps should trust the newest observed motion instead of coasting stale speed.
            v_alpha = 1.0 if dt > 0.2 else 1.0 - math.exp(-dt / 0.08)
            if self.ema_vx == 0.0 and self.ema_vy == 0.0:
                self.ema_vx = raw_vx
                self.ema_vy = raw_vy
            else:
                self.ema_vx = self.ema_vx * (1.0 - v_alpha) + raw_vx * v_alpha
                self.ema_vy = self.ema_vy * (1.0 - v_alpha) + raw_vy * v_alpha

        # 2. Update Coordinate EMA
        if self.ema_bbox is None or self.raw_last_bbox is None or dt > 0.2:
            self.ema_bbox = result.bbox
        else:
            # Time-aware EMA for coordinates (tau = 0.02s, gives ~0.80 alpha at 30fps)
            alpha = 1.0 - math.exp(-dt / 0.02)
            alpha = min(max(alpha, 0.0), 1.0)
            
            x1 = alpha * result.bbox.x1 + (1.0 - alpha) * self.ema_bbox.x1
            y1 = alpha * result.bbox.y1 + (1.0 - alpha) * self.ema_bbox.y1
            x2 = alpha * result.bbox.x2 + (1.0 - alpha) * self.ema_bbox.x2
            y2 = alpha * result.bbox.y2 + (1.0 - alpha) * self.ema_bbox.y2
            self.ema_bbox = BBox(x1, y1, x2, y2)
            
        self.raw_last_bbox = result.bbox
        self.raw_last_ts = result.ts_resp
            
        smoothed_result = TrackingResult(
            packet=result.packet,
            ts_req=result.ts_req,
            ts_resp=result.ts_resp,
            bbox=self.ema_bbox,
            confidence=result.confidence,
            track_id=result.track_id,
            query=result.query,
        )
        self.results.append(smoothed_result)

    def estimate_current(self, result: TrackingResult, now_us: int) -> BBox | None:
        """Approximate current bbox using the last two valid detections."""

        if result.bbox is None:
            return None
        if not self.results:
            return result.bbox

        latest = self.results[-1]
        if len(self.results) < 2:
            return latest.bbox
            
        prev = self.results[-2]
        if prev.bbox is None or latest.bbox is None:
            return latest.bbox
        vx, vy = self._estimate_velocity()
        delay_s = max((now_us - result.ts_req) / 1_000_000.0, 0.0)
        return latest.bbox.shifted(vx * delay_s, vy * delay_s)

    def _estimate_velocity(self) -> tuple[float, float]:
        """Return the exponential moving average (EMA) of velocity to prevent sudden direction change overshoots."""
        return self.ema_vx, self.ema_vy

    def predict_trajectory(self, now_us: int, friction: float = 1.5) -> BBox | None:
        """Predict the target's current position using smooth coasting extrapolation."""
        if len(self.results) < 2:
            return None
        latest = self.results[-1]
        if latest.bbox is None:
            return None

        vx, vy = self._estimate_velocity()

        dt_s = max((now_us - latest.ts_resp) / 1_000_000.0, 0.0)

        if friction > 0:
            decay = 1.0 - math.exp(-friction * dt_s)
            shift_x = (vx / friction) * decay
            shift_y = (vy / friction) * decay
        else:
            shift_x = vx * dt_s
            shift_y = vy * dt_s

        return latest.bbox.shifted(shift_x, shift_y)
