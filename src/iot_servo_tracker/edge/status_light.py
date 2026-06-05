"""RGB status-light boundary for Raspberry Pi edge state feedback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from iot_servo_tracker.control.states import SystemState


@dataclass(frozen=True)
class RgbColor:
    red: bool = False
    green: bool = False
    blue: bool = False


OFF = RgbColor()
RED = RgbColor(red=True)
GREEN = RgbColor(green=True)
BLUE = RgbColor(blue=True)
YELLOW = RgbColor(red=True, green=True)
CYAN = RgbColor(green=True, blue=True)
WHITE = RgbColor(red=True, green=True, blue=True)


STATE_COLORS: dict[SystemState, RgbColor] = {
    SystemState.IDLE: OFF,
    SystemState.SCAN: BLUE,
    SystemState.DELAY_COMPENSATION: CYAN,
    SystemState.TRACKING: GREEN,
    SystemState.SAFE_HOLD: RED,
    SystemState.LIMITED_RESCAN: YELLOW,
    SystemState.CENTERING: BLUE,
    SystemState.ERROR: RED,
}


class StatusLight(Protocol):
    def set_state(self, state: SystemState) -> None:
        """Reflect the current runtime state."""

    def off(self) -> None:
        """Turn every channel off."""


@dataclass
class SimulatedStatusLight:
    last_state: SystemState | None = None
    last_color: RgbColor = OFF

    def set_state(self, state: SystemState) -> None:
        self.last_state = state
        self.last_color = color_for_state(state)

    def off(self) -> None:
        self.last_color = OFF


class NullStatusLight:
    def set_state(self, state: SystemState) -> None:
        return None

    def off(self) -> None:
        return None


class RaspberryPiRgbStatusLight:
    """GPIO RGB LED driver.

    Defaults are for a common-cathode RGB LED. Set ``active_low`` to true for a
    common-anode LED where driving a GPIO low turns that channel on.
    """

    def __init__(
        self,
        red_pin: int = 5,
        green_pin: int = 6,
        blue_pin: int = 23,
        active_low: bool = False,
    ) -> None:
        try:
            import RPi.GPIO as GPIO
        except ImportError as exc:
            raise RuntimeError("Install RPi.GPIO on Raspberry Pi") from exc
        self.GPIO = GPIO
        self.red_pin = red_pin
        self.green_pin = green_pin
        self.blue_pin = blue_pin
        self.active_low = active_low
        GPIO.setmode(GPIO.BCM)
        for pin in (red_pin, green_pin, blue_pin):
            GPIO.setup(pin, GPIO.OUT)
        self.off()

    def set_state(self, state: SystemState) -> None:
        self.set_color(color_for_state(state))

    def set_color(self, color: RgbColor) -> None:
        self._write(self.red_pin, color.red)
        self._write(self.green_pin, color.green)
        self._write(self.blue_pin, color.blue)

    def off(self) -> None:
        self.set_color(OFF)

    def _write(self, pin: int, on: bool) -> None:
        if self.active_low:
            value = self.GPIO.LOW if on else self.GPIO.HIGH
        else:
            value = self.GPIO.HIGH if on else self.GPIO.LOW
        self.GPIO.output(pin, value)


def color_for_state(state: SystemState) -> RgbColor:
    return STATE_COLORS.get(state, WHITE)
