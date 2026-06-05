"""Distance sensor reader boundaries."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from iot_servo_tracker.common.packets import SensorSample
from iot_servo_tracker.common.timebase import now_us


class SensorReader(Protocol):
    def read(self) -> SensorSample:
        """Return the newest distance and safety-input sample."""


@dataclass
class SimulatedSensorReader:
    tof_mm: float = 620.0
    ultrasonic_mm: float = 650.0
    infrared_active: bool = False

    def read(self) -> SensorSample:
        return SensorSample(
            ts=now_us(),
            tof_mm=self.tof_mm,
            ultrasonic_mm=self.ultrasonic_mm,
            infrared_active=self.infrared_active,
        )


class RaspberryPiSensorReader:
    """VL53L0X, HC-SR04 ultrasonic, IR obstacle, and limit-switch reader."""

    def __init__(
        self,
        trig_pin: int = 22,
        echo_pin: int = 27,
        infrared_pin: int = 17,
        limit_pin: int = 25,
        infrared_active_low: bool = True,
        echo_timeout_s: float = 0.03,
    ) -> None:
        try:
            import RPi.GPIO as GPIO
            import adafruit_vl53l0x
            import board
            import busio
        except ImportError as exc:
            raise RuntimeError(
                "Install RPi.GPIO and adafruit-circuitpython-vl53l0x on Raspberry Pi"
            ) from exc
        self.GPIO = GPIO
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        self.infrared_pin = infrared_pin
        self.limit_pin = limit_pin
        self.infrared_active_low = infrared_active_low
        self.echo_timeout_s = echo_timeout_s
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(trig_pin, GPIO.OUT)
        GPIO.setup(echo_pin, GPIO.IN)
        GPIO.setup(
            infrared_pin,
            GPIO.IN,
            pull_up_down=GPIO.PUD_UP if infrared_active_low else GPIO.PUD_DOWN,
        )
        GPIO.setup(limit_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.output(trig_pin, False)
        i2c = busio.I2C(board.SCL, board.SDA)
        self.tof = adafruit_vl53l0x.VL53L0X(i2c)

    def read(self) -> SensorSample:
        tof_mm = float(self.tof.range)
        ultrasonic_mm = self._read_ultrasonic_mm()
        infrared_active = self._input_active(self.infrared_pin, self.infrared_active_low)
        limit_active = self.GPIO.input(self.limit_pin) == self.GPIO.LOW
        return SensorSample(
            ts=now_us(),
            tof_mm=tof_mm,
            ultrasonic_mm=ultrasonic_mm,
            infrared_active=infrared_active,
            limit_switch_active=limit_active,
        )

    def _input_active(self, pin: int, active_low: bool) -> bool:
        active_value = self.GPIO.LOW if active_low else self.GPIO.HIGH
        return self.GPIO.input(pin) == active_value

    def _read_ultrasonic_mm(self) -> float | None:
        gpio = self.GPIO
        gpio.output(self.trig_pin, False)
        time.sleep(0.000002)
        gpio.output(self.trig_pin, True)
        time.sleep(0.00001)
        gpio.output(self.trig_pin, False)

        deadline = time.monotonic() + self.echo_timeout_s
        while gpio.input(self.echo_pin) == gpio.LOW:
            if time.monotonic() > deadline:
                return None
        pulse_start = time.monotonic()
        while gpio.input(self.echo_pin) == gpio.HIGH:
            if time.monotonic() > deadline:
                return None
        pulse_duration = time.monotonic() - pulse_start
        return pulse_duration * 343_000.0 / 2.0
