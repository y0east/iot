"""Servo driver boundary.

The simulated driver is used by tests and desktop development. A Raspberry Pi
deployment can implement the same interface with PCA9685 or hardware PWM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

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
    """Placeholder for Raspberry Pi deployment.

    Keep hardware imports out of module import time so tests and local
    development can run on non-Pi machines.
    """

    def __init__(self, pan_channel: int = 0, tilt_channel: int = 1) -> None:
        self.pan_channel = pan_channel
        self.tilt_channel = tilt_channel
        raise NotImplementedError(
            "Install and wire the PCA9685 library, then implement apply()."
        )

    def apply(self, command: ServoCommand) -> None:
        raise NotImplementedError
