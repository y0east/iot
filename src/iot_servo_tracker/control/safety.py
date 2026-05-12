"""Sensor validation and communication delay safety gates."""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass
from enum import Enum

from iot_servo_tracker.common.config import SafetyConfig
from iot_servo_tracker.common.packets import BBox, SensorSample


class ValidationCategory(str, Enum):
    OK = "OK"
    MISSING = "MISSING"
    SIMILAR_TARGET = "SIMILAR_TARGET"
    OCCLUSION = "OCCLUSION"
    SENSOR_UNAVAILABLE = "SENSOR_UNAVAILABLE"


@dataclass(frozen=True)
class ValidationResult:
    category: ValidationCategory
    safe_hold: bool
    consecutive_hits: int
    reason: str


class SensorValidator:
    def __init__(self, config: SafetyConfig) -> None:
        self.config = config
        self.prev_bbox: BBox | None = None
        self.prev_sample: SensorSample | None = None
        self._hit_count = 0

    def evaluate(self, bbox: BBox | None, sample: SensorSample) -> ValidationResult:
        category = ValidationCategory.OK
        reason = "vision and sensors are consistent"

        if bbox is None:
            category = ValidationCategory.MISSING
            reason = "vision result is missing"
        elif self.prev_bbox is not None and self.prev_sample is not None:
            pixel_jump = _pixel_jump(bbox, self.prev_bbox)
            tof_delta = _delta(sample.tof_mm, self.prev_sample.tof_mm)
            ultrasonic_delta = _delta(sample.ultrasonic_mm, self.prev_sample.ultrasonic_mm)

            if (
                pixel_jump > self.config.pixel_jump_threshold
                and tof_delta is not None
                and tof_delta < self.config.tof_delta_threshold_mm
            ):
                category = ValidationCategory.SIMILAR_TARGET
                reason = "vision center jumped but ToF distance barely changed"
            elif (
                ultrasonic_delta is not None
                and ultrasonic_delta > self.config.ultrasonic_jump_threshold_mm
            ):
                category = ValidationCategory.OCCLUSION
                reason = "ultrasonic distance changed abruptly"
            elif sample.tof_mm is None and sample.ultrasonic_mm is None:
                category = ValidationCategory.SENSOR_UNAVAILABLE
                reason = "no distance sensor sample is available"

        if category in {
            ValidationCategory.MISSING,
            ValidationCategory.SIMILAR_TARGET,
            ValidationCategory.OCCLUSION,
        }:
            self._hit_count += 1
        else:
            self._hit_count = 0

        self.prev_bbox = bbox
        self.prev_sample = sample
        return ValidationResult(
            category=category,
            safe_hold=self._hit_count >= self.config.consecutive_frames,
            consecutive_hits=self._hit_count,
            reason=reason,
        )


class DelayStats:
    def __init__(self, window: int = 30, default_threshold_ms: float = 250.0) -> None:
        self.samples_ms: deque[float] = deque(maxlen=window)
        self.default_threshold_ms = default_threshold_ms

    def update(self, rtt_ms: float) -> None:
        if rtt_ms >= 0:
            self.samples_ms.append(rtt_ms)

    @property
    def threshold_ms(self) -> float:
        if len(self.samples_ms) < 5:
            return self.default_threshold_ms
        mean = statistics.fmean(self.samples_ms)
        stdev = statistics.pstdev(self.samples_ms)
        return mean + 3.0 * stdev

    def is_delayed(self, rtt_ms: float) -> bool:
        self.update(rtt_ms)
        return rtt_ms > self.threshold_ms


def dynamic_timeout_s(
    loss_velocity_px_s: float,
    config: SafetyConfig,
    coefficient: float = 200.0,
    epsilon: float = 1e-3,
) -> float:
    timeout = coefficient / (abs(loss_velocity_px_s) + epsilon)
    return min(config.timeout_max_s, max(config.timeout_min_s, timeout))


def _pixel_jump(current: BBox, previous: BBox) -> float:
    x0, y0 = previous.center
    x1, y1 = current.center
    return math.hypot(x1 - x0, y1 - y0)


def _delta(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None:
        return None
    return abs(current - previous)
