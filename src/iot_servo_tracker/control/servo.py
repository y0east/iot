"""Servo driver boundary.

The simulated driver is used by tests and desktop development. A Raspberry Pi
deployment can implement the same interface with PCA9685 or hardware PWM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from iot_servo_tracker.common.config import AxisLimit
from iot_servo_tracker.control.pd_controller import ServoCommand


class ServoDriver(Protocol):
    def apply(self, command: ServoCommand) -> None:
        """Apply the newest pan/tilt command."""


@dataclass
class SimulatedServoDriver:
    last_command: ServoCommand | None = None
    applied_count: int = 0

    def apply(self, command: ServoCommand) -> None:
        self.last_command = command
        self.applied_count += 1


class Pca9685ServoDriver:
    """PCA9685 servo driver using Adafruit ServoKit."""

    def __init__(
        self,
        pan_limit: AxisLimit,
        tilt_limit: AxisLimit,
        pan_channel: int = 0,
        tilt_channel: int = 1,
        channels: int = 16,
    ) -> None:
        try:
            from adafruit_servokit import ServoKit
        except ImportError as exc:
            raise RuntimeError("Install adafruit-circuitpython-servokit on Raspberry Pi") from exc
        self.kit = ServoKit(channels=channels)
        self.pan_limit = pan_limit
        self.tilt_limit = tilt_limit
        self.pan_channel = pan_channel
        self.tilt_channel = tilt_channel
        self.kit.servo[pan_channel].set_pulse_width_range(
            pan_limit.pwm_min_us,
            pan_limit.pwm_max_us,
        )
        self.kit.servo[tilt_channel].set_pulse_width_range(
            tilt_limit.pwm_min_us,
            tilt_limit.pwm_max_us,
        )

    def apply(self, command: ServoCommand) -> None:
        self.kit.servo[self.pan_channel].angle = _servo_angle(
            command.pan_deg,
            self.pan_limit,
        )
        self.kit.servo[self.tilt_channel].angle = _servo_angle(
            command.tilt_deg,
            self.tilt_limit,
        )


def _servo_angle(angle_deg: float, limit: AxisLimit) -> float:
    ratio = (angle_deg - limit.min_deg) / (limit.max_deg - limit.min_deg)
    return min(180.0, max(0.0, ratio * 180.0))


class DirectGpioServoDriver:
    """Direct GPIO servo driver using gpiozero (for Raspberry Pi direct connection)."""

    def __init__(
        self,
        pan_limit: AxisLimit,
        tilt_limit: AxisLimit,
        pan_pin: int = 12,
        tilt_pin: int = 19,
    ) -> None:
        try:
            from gpiozero import AngularServo
        except ImportError as exc:
            raise RuntimeError("Install gpiozero on Raspberry Pi (pip install gpiozero)") from exc

        self.pan_servo = AngularServo(
            pan_pin,
            min_angle=pan_limit.min_deg,
            max_angle=pan_limit.max_deg,
            min_pulse_width=pan_limit.pwm_min_us / 1000000.0,
            max_pulse_width=pan_limit.pwm_max_us / 1000000.0,
        )
        self.tilt_servo = AngularServo(
            tilt_pin,
            min_angle=tilt_limit.min_deg,
            max_angle=tilt_limit.max_deg,
            min_pulse_width=tilt_limit.pwm_min_us / 1000000.0,
            max_pulse_width=tilt_limit.pwm_max_us / 1000000.0,
        )
        self._last_pan_deg = None
        self._last_tilt_deg = None

    def apply(self, command: ServoCommand) -> None:
        # pigpio가 안 먹히는 환경(라즈베리파이 5 등)을 위해 데드밴드를 0.01도로 아주 작게 줄이거나 없앱니다.
        # 확확 돌아가는 현상을 없애고 최대한 부드럽게 따라가도록 원복합니다.
        if self._last_pan_deg is None or abs(self._last_pan_deg - command.pan_deg) > 0.01:
            self.pan_servo.angle = command.pan_deg
            self._last_pan_deg = command.pan_deg
            
        if self._last_tilt_deg is None or abs(self._last_tilt_deg - command.tilt_deg) > 0.01:
            self.tilt_servo.angle = command.tilt_deg
            self._last_tilt_deg = command.tilt_deg


class NativeSysfsServoDriver:
    """Zero-dependency hardware PWM driver using Linux sysfs.
    Requires `dtoverlay=pwm-2chan` in /boot/config.txt.
    Maps PWM0 to GPIO 12 and PWM1 to GPIO 13 on Raspberry Pi 4.
    """

    def __init__(self, pan_limit: AxisLimit, tilt_limit: AxisLimit, pwmchip: int = 0) -> None:
        import os
        import time
        self.pan_limit = pan_limit
        self.tilt_limit = tilt_limit
        self.pwmchip = pwmchip
        self.os = os
        self.time = time
        self._last_pan_ns: int | None = None
        self._last_tilt_ns: int | None = None
        self._last_write_s = 0.0
        self._deadband_ns = 1_000
        self._min_write_interval_s = 0.0

        self._init_pwm(0)
        self._init_pwm(1)

    def _init_pwm(self, channel: int) -> None:
        base = f"/sys/class/pwm/pwmchip{self.pwmchip}"
        if not self.os.path.exists(base):
            raise RuntimeError(
                f"{base} does not exist. You MUST enable 'dtoverlay=pwm-2chan' in /boot/firmware/config.txt and reboot!"
            )

        pwm_path = f"{base}/pwm{channel}"
        if not self.os.path.exists(pwm_path):
            with open(f"{base}/export", "w") as f:
                f.write(str(channel))
                
        # 50 Hz = 20ms = 20,000,000 ns
        with open(f"{pwm_path}/period", "w") as f:
            f.write("20000000")
            
        with open(f"{pwm_path}/enable", "w") as f:
            f.write("1")

    def _angle_to_ns(self, angle_deg: float, limit: AxisLimit) -> int:
        ratio = (angle_deg - limit.min_deg) / (limit.max_deg - limit.min_deg)
        ratio = min(1.0, max(0.0, ratio))
        us = limit.pwm_min_us + ratio * (limit.pwm_max_us - limit.pwm_min_us)
        return int(us * 1000)

    def apply(self, command: ServoCommand) -> None:
        pan_ns = self._angle_to_ns(command.pan_deg, self.pan_limit)
        tilt_ns = self._angle_to_ns(command.tilt_deg, self.tilt_limit)
        now_s = self.time.monotonic()
        pan_changed = self._duty_changed(pan_ns, self._last_pan_ns)
        tilt_changed = self._duty_changed(tilt_ns, self._last_tilt_ns)
        if not pan_changed and not tilt_changed:
            return
        if now_s - self._last_write_s < self._min_write_interval_s:
            return

        try:
            if pan_changed:
                with open(f"/sys/class/pwm/pwmchip{self.pwmchip}/pwm0/duty_cycle", "w") as f:
                    f.write(str(pan_ns))
                self._last_pan_ns = pan_ns
            if tilt_changed:
                with open(f"/sys/class/pwm/pwmchip{self.pwmchip}/pwm1/duty_cycle", "w") as f:
                    f.write(str(tilt_ns))
                self._last_tilt_ns = tilt_ns
            self._last_write_s = now_s
        except OSError:
            pass # Ignore occasional permission/write errors if the kernel is busy

    def _duty_changed(self, current_ns: int, previous_ns: int | None) -> bool:
        return previous_ns is None or abs(current_ns - previous_ns) >= self._deadband_ns
