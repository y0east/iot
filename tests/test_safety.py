import unittest

from iot_servo_tracker.common.config import CameraConfig, SafetyConfig
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

    def test_stable_similar_target_keeps_original_baseline_until_safe_hold(self) -> None:
        config = SafetyConfig(consecutive_frames=2, pixel_jump_threshold=50, tof_delta_threshold_mm=20)
        validator = SensorValidator(config)
        validator.evaluate(BBox(100, 100, 150, 150), SensorSample(ts=1, tof_mm=500))
        first = validator.evaluate(BBox(250, 100, 300, 150), SensorSample(ts=2, tof_mm=505))
        second = validator.evaluate(BBox(250, 100, 300, 150), SensorSample(ts=3, tof_mm=505))

        self.assertEqual(first.category, ValidationCategory.SIMILAR_TARGET)
        self.assertEqual(second.category, ValidationCategory.SIMILAR_TARGET)
        self.assertTrue(second.safe_hold)

    def test_delay_threshold_uses_default_until_warm(self) -> None:
        stats = DelayStats(default_threshold_ms=100)
        self.assertFalse(stats.is_delayed(90))
        self.assertTrue(stats.is_delayed(120))

    def test_delay_outlier_is_checked_before_stats_update(self) -> None:
        stats = DelayStats(default_threshold_ms=250)
        for _ in range(5):
            self.assertFalse(stats.is_delayed(20))
        self.assertTrue(stats.is_delayed(1000))

    def test_ultrasonic_only_drop_counts_as_occlusion(self) -> None:
        config = SafetyConfig(consecutive_frames=1, ultrasonic_jump_threshold_mm=100)
        validator = SensorValidator(config)
        validator.evaluate(BBox(0, 0, 10, 10), SensorSample(ts=1, ultrasonic_mm=200))
        farther = validator.evaluate(BBox(0, 0, 10, 10), SensorSample(ts=2, ultrasonic_mm=400))
        self.assertFalse(farther.safe_hold)
        closer = validator.evaluate(BBox(0, 0, 10, 10), SensorSample(ts=3, ultrasonic_mm=100))
        self.assertTrue(closer.safe_hold)

    def test_sensor_unavailable_counts_as_safe_hold_hit(self) -> None:
        config = SafetyConfig(consecutive_frames=2)
        validator = SensorValidator(config)

        first = validator.evaluate(BBox(0, 0, 10, 10), SensorSample.empty())
        second = validator.evaluate(BBox(0, 0, 10, 10), SensorSample.empty())

        self.assertEqual(first.category, ValidationCategory.SENSOR_UNAVAILABLE)
        self.assertFalse(first.safe_hold)
        self.assertTrue(second.safe_hold)

    def test_large_background_bbox_counts_as_absorption(self) -> None:
        config = SafetyConfig(consecutive_frames=2, bbox_frame_area_threshold=0.20)
        validator = SensorValidator(config, CameraConfig(width=640, height=480))
        validator.evaluate(BBox(300, 220, 340, 260), SensorSample(ts=1, tof_mm=500))
        first = validator.evaluate(BBox(100, 80, 600, 460), SensorSample(ts=2, tof_mm=505))
        second = validator.evaluate(BBox(100, 80, 600, 460), SensorSample(ts=3, tof_mm=505))

        self.assertEqual(first.category, ValidationCategory.BBOX_ABSORPTION)
        self.assertFalse(first.safe_hold)
        self.assertTrue(second.safe_hold)

    def test_area_growth_absorption_keeps_original_baseline_until_safe_hold(self) -> None:
        config = SafetyConfig(
            consecutive_frames=2,
            bbox_area_growth_threshold=2.0,
            bbox_frame_area_threshold=0.35,
        )
        validator = SensorValidator(config, CameraConfig(width=640, height=480))
        validator.evaluate(BBox(300, 220, 340, 260), SensorSample(ts=1, tof_mm=500))
        first = validator.evaluate(BBox(260, 180, 420, 340), SensorSample(ts=2, tof_mm=505))
        second = validator.evaluate(BBox(260, 180, 420, 340), SensorSample(ts=3, tof_mm=505))

        self.assertEqual(first.category, ValidationCategory.BBOX_ABSORPTION)
        self.assertEqual(second.category, ValidationCategory.BBOX_ABSORPTION)
        self.assertTrue(second.safe_hold)


if __name__ == "__main__":
    unittest.main()
