import tempfile
import unittest
from pathlib import Path

from iot_servo_tracker.common.config import load_config


class ConfigTests(unittest.TestCase):
    def test_hardware_config_loads_sensor_and_rgb_pins(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.toml"
            path.write_text(
                "\n".join(
                    [
                        "[hardware]",
                        "ultrasonic_trig_pin = 22",
                        "ultrasonic_echo_pin = 27",
                        "infrared_pin = 17",
                        "infrared_active_low = true",
                        "rgb_red_pin = 5",
                        "rgb_green_pin = 6",
                        "rgb_blue_pin = 23",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(path)

        self.assertEqual(config.hardware.ultrasonic_trig_pin, 22)
        self.assertEqual(config.hardware.ultrasonic_echo_pin, 27)
        self.assertEqual(config.hardware.infrared_pin, 17)
        self.assertTrue(config.hardware.infrared_active_low)
        self.assertEqual(config.hardware.rgb_red_pin, 5)
        self.assertEqual(config.hardware.rgb_green_pin, 6)
        self.assertEqual(config.hardware.rgb_blue_pin, 23)


if __name__ == "__main__":
    unittest.main()
