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
    BBOX_ABSORPTION = "BBOX_ABSORPTION"
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
        sensors_unavailable = sample.tof_mm is None and sample.ultrasonic_mm is None

        if sample.limit_switch_active:
            category = ValidationCategory.LIMIT_SWITCH
            reason = "limit switch is active"
            required_hits = 1
        elif bbox is None:
            if sensors_unavailable:
                category = ValidationCategory.MISSING
                reason = "vision result is missing (sensors unavailable but bypassed)"
            else:
                category = ValidationCategory.MISSING
                reason = "vision result is missing"
        elif sensors_unavailable and self.config.pixel_jump_threshold < 900.0:
            category = ValidationCategory.SENSOR_UNAVAILABLE
            reason = "no distance sensor sample is available"
        elif self.prev_bbox is not None and self.prev_sample is not None:
            pixel_jump = _pixel_jump(bbox, self.prev_bbox)
            tof_delta = _delta(sample.tof_mm, self.prev_sample.tof_mm)
            ultrasonic_drop = _drop(sample.ultrasonic_mm, self.prev_sample.ultrasonic_mm)
            bbox_absorption_reason = self._bbox_absorption_reason(bbox, self.prev_bbox)

            if bbox_absorption_reason is not None:
                category = ValidationCategory.BBOX_ABSORPTION
                reason = bbox_absorption_reason
            elif (
                pixel_jump > self.config.pixel_jump_threshold
                and tof_delta is not None
                and tof_delta < self.config.tof_delta_threshold_mm
            ):
                category = ValidationCategory.SIMILAR_TARGET
                reason = "vision center jumped but ToF distance barely changed"
            elif (
                ultrasonic_drop is not None
                and ultrasonic_drop > self.config.ultrasonic_jump_threshold_mm
            ):
                category = ValidationCategory.OCCLUSION
                reason = "ultrasonic distance dropped abruptly"
        if category in {
            ValidationCategory.MISSING,
            ValidationCategory.SIMILAR_TARGET,
            ValidationCategory.BBOX_ABSORPTION,
            ValidationCategory.OCCLUSION,
            ValidationCategory.SENSOR_UNAVAILABLE,
            ValidationCategory.LIMIT_SWITCH,
        }:
            self._hit_count += 1
        else:
            self._hit_count = 0

        safe_hold = self._hit_count >= required_hits
        if category is ValidationCategory.OK or (
            category is ValidationCategory.SENSOR_UNAVAILABLE and not safe_hold
        ):
            self.prev_bbox = bbox
            self.prev_sample = sample
        return ValidationResult(
            category=category,
            safe_hold=safe_hold,
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

    def _bbox_absorption_reason(self, bbox: BBox, previous: BBox) -> str | None:
        frame_area = None
        if self.camera is not None and self.camera.width > 0 and self.camera.height > 0:
            frame_area = self.camera.width * self.camera.height
        if frame_area and bbox.area / frame_area > self.config.bbox_frame_area_threshold:
            return "vision bbox is too large for the expected target"

        if previous.area > 0:
            area_growth = bbox.area / previous.area
            if area_growth > self.config.bbox_area_growth_threshold:
                return "vision bbox grew too much between frames"
            if area_growth < self.config.bbox_area_shrink_threshold:
                return "vision bbox shrank too much between frames"

        aspect_change = _aspect_ratio_change(bbox, previous)
        if aspect_change > self.config.bbox_aspect_ratio_change_threshold:
            return "vision bbox aspect ratio changed too much"
        return None


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
        return max(self.default_threshold_ms, mean + 3.0 * stdev)

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


def _aspect_ratio_change(current: BBox, previous: BBox) -> float:
    current_width = max(current.x2 - current.x1, 1e-6)
    current_height = max(current.y2 - current.y1, 1e-6)
    previous_width = max(previous.x2 - previous.x1, 1e-6)
    previous_height = max(previous.y2 - previous.y1, 1e-6)
    current_ratio = current_width / current_height
    previous_ratio = previous_width / previous_height
    return max(current_ratio / previous_ratio, previous_ratio / current_ratio)
