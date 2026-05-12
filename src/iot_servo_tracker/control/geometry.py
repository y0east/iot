"""Geometry and servo mapping helpers."""

from __future__ import annotations

import math

from iot_servo_tracker.common.config import AxisLimit, CameraConfig
from iot_servo_tracker.common.packets import BBox


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def focal_length_px(size_px: int, fov_deg: float) -> float:
    return (size_px / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def pixel_error_to_angle_deg(
    bbox: BBox,
    camera: CameraConfig,
) -> tuple[float, float]:
    """Convert a detection center into yaw/pitch angle error in degrees."""

    x, y = bbox.center
    cx = camera.width / 2.0
    cy = camera.height / 2.0
    fx = focal_length_px(camera.width, camera.horizontal_fov_deg)
    fy = focal_length_px(camera.height, camera.vertical_fov_deg)
    yaw = math.degrees(math.atan((x - cx) / fx))
    pitch = math.degrees(math.atan((cy - y) / fy))
    return yaw, pitch


def angle_to_pwm_us(angle_deg: float, limit: AxisLimit) -> int:
    bounded = clamp(angle_deg, limit.min_deg, limit.max_deg)
    ratio = (bounded - limit.min_deg) / (limit.max_deg - limit.min_deg)
    pwm = limit.pwm_min_us + ratio * (limit.pwm_max_us - limit.pwm_min_us)
    return int(round(pwm))


def move_toward(current: float, target: float, max_delta: float) -> float:
    if abs(target - current) <= max_delta:
        return target
    return current + math.copysign(max_delta, target - current)
