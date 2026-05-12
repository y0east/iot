"""Sensor validation and communication delay safety gates."""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass
from enum import Enum

from iot_servo_tracker.common.config import CameraConfig, SafetyConfig
from iot_servo_tracker.common.packets import BBox, SensorSample


class ValidationCategory(str, Enum):
    OK = "OK"
    MISSING = "MISSING"
    SIMILAR_TARGET = "SIMILAR_TARGET"
    OCCLUSION = "OCCLUSION"
    SENSOR_UNAVAILABLE = "SENSOR_UNAVAILABLE"
    LIMIT_SWITCH = "LIMIT_SWITCH"


@dataclass(frozen=True)
class ValidationResult:
    category: ValidationCategory
    safe_hold: bool
    consecutive_hits: int
    reason: str


class SensorValidator:
    def __init__(self, config: SafetyConfig, camera: CameraConfig | None = None) -> None:
        self.config = config
        self.camera = camera
        self.prev_bbox: BBox | None = None
        self.prev_sample: SensorSample | None = None
        self._hit_count = 0

    def evaluate(self, bbox: BBox | None, sample: SensorSample) -> ValidationResult:
        category = ValidationCategory.OK
        reason = "vision and sensors are consistent"
        required_hits = self.config.consecutive_frames

        if sample.limit_switch_active:
            category = ValidationCategory.LIMIT_SWITCH
            reason = "limit switch is active"
            required_hits = 1
        elif bbox is None:
            category = ValidationCategory.MISSING
            reason = "vision result is missing"
        elif self.prev_bbox is not None and self.prev_sample is not None:
            pixel_jump = _pixel_jump(bbox, self.prev_bbox)
            tof_delta = _delta(sample.tof_mm, self.prev_sample.tof_mm)
            ultrasonic_drop = _drop(sample.ultrasonic_mm, self.prev_sample.ultrasonic_mm)

            if (
                pixel_jump > self.config.pixel_jump_threshold
                and tof_delta is not None
                and tof_delta < self.config.tof_delta_threshold_mm
            ):
                category = ValidationCategory.SIMILAR_TARGET
                reason = "vision center jumped but ToF distance barely changed"
                if not self._is_central(bbox):
                    required_hits = self.config.consecutive_frames + 1
            elif (
                ultrasonic_drop is not None
                and ultrasonic_drop > self.config.ultrasonic_jump_threshold_mm
            ):
                category = ValidationCategory.OCCLUSION
                reason = "ultrasonic distance dropped abruptly"
            elif sample.tof_mm is None and sample.ultrasonic_mm is None:
                category = ValidationCategory.SENSOR_UNAVAILABLE
                reason = "no distance sensor sample is available"

        if category in {
            ValidationCategory.MISSING,
            ValidationCategory.SIMILAR_TARGET,
            ValidationCategory.OCCLUSION,
            ValidationCategory.LIMIT_SWITCH,
        }:
            self._hit_count += 1
        else:
            self._hit_count = 0

        self.prev_bbox = bbox
        self.prev_sample = sample
        return ValidationResult(
            category=category,
            safe_hold=self._hit_count >= required_hits,
            consecutive_hits=self._hit_count,
            reason=reason,
        )

    def _is_central(self, bbox: BBox) -> bool:
        if self.camera is None:
            return True
        x, y = bbox.center
        half_w = self.camera.width * self.config.central_region_ratio / 2.0
        half_h = self.camera.height * self.config.central_region_ratio / 2.0
        return (
            abs(x - self.camera.width / 2.0) <= half_w
            and abs(y - self.camera.height / 2.0) <= half_h
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
        delayed = rtt_ms > self.threshold_ms
        self.update(rtt_ms)
        return delayed


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


def _drop(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None:
        return None
    return previous - current
