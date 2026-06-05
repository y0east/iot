import sys
import types
import unittest
from unittest.mock import patch

from iot_servo_tracker.edge.sensors import RaspberryPiSensorReader


class FakeGpio(types.ModuleType):
    BCM = "BCM"
    OUT = "OUT"
    IN = "IN"
    PUD_UP = "PUD_UP"
    PUD_DOWN = "PUD_DOWN"
    HIGH = 1
    LOW = 0

    def __init__(self) -> None:
        super().__init__("RPi.GPIO")
        self.mode = None
        self.setup_calls = []
        self.output_calls = []
        self.inputs = {}

    def setmode(self, mode) -> None:
        self.mode = mode

    def setup(self, pin, mode, pull_up_down=None) -> None:
        self.setup_calls.append((pin, mode, pull_up_down))

    def output(self, pin, value) -> None:
        self.output_calls.append((pin, value))

    def input(self, pin):
        return self.inputs.get(pin, self.HIGH)


class FakeVl53l0x(types.ModuleType):
    class VL53L0X:
        range = 620

        def __init__(self, i2c) -> None:
            self.i2c = i2c


class EdgeSensorTests(unittest.TestCase):
    def test_reader_uses_configured_ultrasonic_and_infrared_pins(self) -> None:
        gpio = FakeGpio()
        rpi = types.ModuleType("RPi")
        rpi.GPIO = gpio
        board = types.ModuleType("board")
        board.SCL = object()
        board.SDA = object()
        busio = types.ModuleType("busio")
        busio.I2C = lambda scl, sda: (scl, sda)

        modules = {
            "RPi": rpi,
            "RPi.GPIO": gpio,
            "adafruit_vl53l0x": FakeVl53l0x("adafruit_vl53l0x"),
            "board": board,
            "busio": busio,
        }
        with patch.dict(sys.modules, modules):
            reader = RaspberryPiSensorReader(
                trig_pin=22,
                echo_pin=27,
                infrared_pin=17,
                limit_pin=25,
                infrared_active_low=True,
            )
            reader._read_ultrasonic_mm = lambda: 650.0
            gpio.inputs[17] = gpio.LOW
            gpio.inputs[25] = gpio.HIGH

            sample = reader.read()

        self.assertIn((22, gpio.OUT, None), gpio.setup_calls)
        self.assertIn((27, gpio.IN, None), gpio.setup_calls)
        self.assertIn((17, gpio.IN, gpio.PUD_UP), gpio.setup_calls)
        self.assertTrue(sample.infrared_active)
        self.assertFalse(sample.limit_switch_active)
        self.assertEqual(sample.tof_mm, 620.0)
        self.assertEqual(sample.ultrasonic_mm, 650.0)

    def test_reader_still_reads_ultrasonic_and_infrared_without_tof_library(self) -> None:
        gpio = FakeGpio()
        rpi = types.ModuleType("RPi")
        rpi.GPIO = gpio
        modules = {
            "RPi": rpi,
            "RPi.GPIO": gpio,
            "adafruit_vl53l0x": None,
            "board": None,
            "busio": None,
        }
        with patch.dict(sys.modules, modules):
            reader = RaspberryPiSensorReader(
                trig_pin=22,
                echo_pin=27,
                infrared_pin=17,
                limit_pin=25,
                infrared_active_low=True,
            )
            reader._read_ultrasonic_mm = lambda: 640.0
            gpio.inputs[17] = gpio.LOW
            gpio.inputs[25] = gpio.HIGH

            sample = reader.read()

        self.assertIsNone(sample.tof_mm)
        self.assertEqual(sample.ultrasonic_mm, 640.0)
        self.assertTrue(sample.infrared_active)


if __name__ == "__main__":
    unittest.main()
