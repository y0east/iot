import unittest

from iot_servo_tracker.common.config import SafetyConfig
from iot_servo_tracker.common.packets import BBox, SensorSample
from iot_servo_tracker.control.safety import DelayStats, SensorValidator, ValidationCategory


class SafetyTests(unittest.TestCase):
    def test_similar_target_requires_consecutive_hits(self) -> None:
        config = SafetyConfig(consecutive_frames=2, pixel_jump_threshold=50, tof_delta_threshold_mm=20)
        validator = SensorValidator(config)
        validator.evaluate(BBox(100, 100, 150, 150), SensorSample(ts=1, tof_mm=500))
        first = validator.evaluate(BBox(250, 100, 300, 150), SensorSample(ts=2, tof_mm=505))
        second = validator.evaluate(BBox(400, 100, 450, 150), SensorSample(ts=3, tof_mm=510))
        self.assertEqual(first.category, ValidationCategory.SIMILAR_TARGET)
        self.assertFalse(first.safe_hold)
        self.assertTrue(second.safe_hold)

    def test_delay_threshold_uses_default_until_warm(self) -> None:
        stats = DelayStats(default_threshold_ms=100)
        self.assertFalse(stats.is_delayed(90))
        self.assertTrue(stats.is_delayed(120))


if __name__ == "__main__":
    unittest.main()
