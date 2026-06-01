"""PD servo controller with speed and acceleration limiting."""

from __future__ import annotations

from dataclasses import dataclass

from iot_servo_tracker.common.config import AppConfig
from iot_servo_tracker.common.packets import BBox
from iot_servo_tracker.control.geometry import (
    angle_to_pwm_us,
    clamp,
    move_toward,
    pixel_error_to_angle_deg,
)


@dataclass(frozen=True)
class ServoCommand:
    pan_deg: float
    tilt_deg: float
    pan_pwm_us: int
    tilt_pwm_us: int
    pan_omega_deg_s: float
    tilt_omega_deg_s: float


@dataclass
class ControllerState:
    pan_deg: float = 0.0
    tilt_deg: float = 0.0
    pan_omega_deg_s: float = 0.0
    tilt_omega_deg_s: float = 0.0
    prev_yaw_error_deg: float = 0.0
    prev_pitch_error_deg: float = 0.0
    filtered_yaw_derivative: float = 0.0
    filtered_pitch_derivative: float = 0.0


class PDServoController:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.max_speed_deg_s = config.control.max_speed_deg_s
        self.state = ControllerState(
            pan_deg=config.control.pan.center_deg,
            tilt_deg=config.control.tilt.center_deg,
        )

    def set_max_speed_limit(self, max_speed_deg_s: float | None) -> None:
        if max_speed_deg_s is None:
            self.max_speed_deg_s = self.config.control.max_speed_deg_s
            return
        self.max_speed_deg_s = clamp(
            max_speed_deg_s,
            1.0,
            self.config.control.max_speed_deg_s,
        )

    def reset_history(self) -> None:
        """Reset PID error history to prevent derivative kick (sudden jerks) when tracking resumes."""
        self.state.prev_yaw_error_deg = 0.0
        self.state.prev_pitch_error_deg = 0.0
        self.state.filtered_yaw_derivative = 0.0
        self.state.filtered_pitch_derivative = 0.0

    def update(self, bbox: BBox, dt_s: float) -> ServoCommand:
        dt_s = max(dt_s, 1e-3)
        yaw_error, pitch_error = pixel_error_to_angle_deg(bbox, self.config.camera)

        pan_cmd = self._axis_omega(
            error=yaw_error,
            prev_error=self.state.prev_yaw_error_deg,
            filtered_derivative=self.state.filtered_yaw_derivative,
            kp=self.config.control.pan_kp,
            kd=self.config.control.pan_kd,
            dt_s=dt_s,
        )
        tilt_cmd = self._axis_omega(
            error=pitch_error,
            prev_error=self.state.prev_pitch_error_deg,
            filtered_derivative=self.state.filtered_pitch_derivative,
            kp=self.config.control.tilt_kp,
            kd=self.config.control.tilt_kd,
            dt_s=dt_s,
        )

        self.state.filtered_yaw_derivative = pan_cmd[1]
        self.state.filtered_pitch_derivative = tilt_cmd[1]
        self.state.prev_yaw_error_deg = yaw_error
        self.state.prev_pitch_error_deg = pitch_error
        return self._integrate(pan_cmd[0], tilt_cmd[0], dt_s)

    def command_current(self) -> ServoCommand:
        return self._command()

    def scan_pan_step(self, target_pan_deg: float, speed_deg_s: float, dt_s: float) -> ServoCommand:
        dt_s = max(dt_s, 1e-3)
        previous_pan = self.state.pan_deg
        bounded_target = clamp(
            target_pan_deg,
            self.config.control.pan.min_deg,
            self.config.control.pan.max_deg,
        )
        max_delta = abs(speed_deg_s) * dt_s
        self.state.pan_deg = move_toward(previous_pan, bounded_target, max_delta)
        self.state.tilt_deg = move_toward(
            self.state.tilt_deg,
            self.config.control.tilt.center_deg,
            max_delta,
        )
        self.state.pan_omega_deg_s = (self.state.pan_deg - previous_pan) / dt_s
        if self.state.pan_deg == bounded_target:
            self.state.pan_omega_deg_s = 0.0
        return self._command()

    def soft_stop(self, dt_s: float) -> ServoCommand:
        return self._integrate(0.0, 0.0, max(dt_s, 1e-3))

    def center_step(self, dt_s: float) -> ServoCommand:
        self.soft_stop(dt_s)
        max_delta = self.max_speed_deg_s * max(dt_s, 1e-3)
        self.state.pan_deg = move_toward(
            self.state.pan_deg, self.config.control.pan.center_deg, max_delta
        )
        self.state.tilt_deg = move_toward(
            self.state.tilt_deg, self.config.control.tilt.center_deg, max_delta
        )
        if self.state.pan_deg == self.config.control.pan.center_deg:
            self.state.pan_omega_deg_s = 0.0
        if self.state.tilt_deg == self.config.control.tilt.center_deg:
            self.state.tilt_omega_deg_s = 0.0
        return self._command()

    def drive_to_absolute(self, target_pan_deg: float, target_tilt_deg: float, dt_s: float) -> ServoCommand:
        self.soft_stop(dt_s)
        max_delta = self.max_speed_deg_s * max(dt_s, 1e-3)
        self.state.pan_deg = move_toward(
            self.state.pan_deg, target_pan_deg, max_delta
        )
        self.state.tilt_deg = move_toward(
            self.state.tilt_deg, target_tilt_deg, max_delta
        )
        if self.state.pan_deg == target_pan_deg:
            self.state.pan_omega_deg_s = 0.0
        if self.state.tilt_deg == target_tilt_deg:
            self.state.tilt_omega_deg_s = 0.0
        return self._command()

    def is_centered(self, tolerance_deg: float = 0.5) -> bool:
        return (
            abs(self.state.pan_deg - self.config.control.pan.center_deg) <= tolerance_deg
            and abs(self.state.tilt_deg - self.config.control.tilt.center_deg) <= tolerance_deg
        )

    def _axis_omega(
        self,
        error: float,
        prev_error: float,
        filtered_derivative: float,
        kp: float,
        kd: float,
        dt_s: float,
    ) -> tuple[float, float]:
        if abs(error) < self.config.control.deadband_deg:
            return 0.0, 0.0
        raw_derivative = (error - prev_error) / dt_s
        gamma = self.config.control.derivative_filter_gamma
        derivative = gamma * filtered_derivative + (1.0 - gamma) * raw_derivative
        omega = kp * error + kd * derivative
        omega = clamp(
            omega,
            -self.max_speed_deg_s,
            self.max_speed_deg_s,
        )
        return omega, derivative

    def _integrate(self, pan_omega_cmd: float, tilt_omega_cmd: float, dt_s: float) -> ServoCommand:
        max_delta = self.config.control.max_accel_deg_s2 * dt_s
        self.state.pan_omega_deg_s += clamp(
            pan_omega_cmd - self.state.pan_omega_deg_s,
            -max_delta,
            max_delta,
        )
        self.state.tilt_omega_deg_s += clamp(
            tilt_omega_cmd - self.state.tilt_omega_deg_s,
            -max_delta,
            max_delta,
        )
        self.state.pan_deg = clamp(
            self.state.pan_deg + self.state.pan_omega_deg_s * dt_s,
            self.config.control.pan.min_deg,
            self.config.control.pan.max_deg,
        )
        self.state.tilt_deg = clamp(
            self.state.tilt_deg + self.state.tilt_omega_deg_s * dt_s,
            self.config.control.tilt.min_deg,
            self.config.control.tilt.max_deg,
        )
        return self._command()

    def _command(self) -> ServoCommand:
        return ServoCommand(
            pan_deg=self.state.pan_deg,
            tilt_deg=self.state.tilt_deg,
            pan_pwm_us=angle_to_pwm_us(self.state.pan_deg, self.config.control.pan),
            tilt_pwm_us=angle_to_pwm_us(self.state.tilt_deg, self.config.control.tilt),
            pan_omega_deg_s=self.state.pan_omega_deg_s,
            tilt_omega_deg_s=self.state.tilt_omega_deg_s,
        )
