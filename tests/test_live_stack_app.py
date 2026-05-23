import unittest
from unittest.mock import patch

from iot_servo_tracker.common.config import AppConfig
from iot_servo_tracker.web import live_stack_app


class FakeCamera:
    """Unit-test stand-in only; runtime paths open OpenCvCamera."""

    def __init__(self, *args, **kwargs) -> None:
        self.closed = False

    def read_jpeg(self) -> bytes:
        return b"fake-webcam-jpeg"

    def close(self) -> None:
        self.closed = True


class LiveStackAppTests(unittest.TestCase):
    def test_live_stack_uses_webcam_comms_edge_vision_and_status_return(self) -> None:
        with (
            patch.object(live_stack_app, "OpenCvCamera", FakeCamera),
            patch.object(
                live_stack_app,
                "_annotate_frame_jpeg",
                lambda frame, bbox, label: b"annotated-" + label.encode("utf-8"),
            ),
        ):
            records = live_stack_app.run_live_stack_validation(
                AppConfig(),
                live_stack_app.LiveStackOptions(
                    query="red cup",
                    frames=15,
                    sleep_s=0.0,
                    approximate_tof_mm=610.0,
                    approximate_ultrasonic_mm=640.0,
                ),
            )

        self.assertTrue(records)
        self.assertEqual(records[0].web_target, "red cup")
        self.assertEqual(records[0].vision_mode, "scripted")
        self.assertTrue(any(record.frame_sent for record in records))
        self.assertTrue(any(record.vision_processed for record in records))
        self.assertTrue(any(record.web_view == "TRACKING" for record in records))
        self.assertEqual(records[-1].mqtt_commands, 1)
        self.assertGreater(records[-1].mqtt_statuses, 1)
        self.assertEqual(records[-1].tof_mm, 610.0)
        self.assertEqual(records[-1].ultrasonic_mm, 640.0)
        self.assertTrue(records[-1].annotated_jpeg.startswith(b"annotated-"))

    def test_approximate_sensor_sample_is_available_and_non_blocking(self) -> None:
        sample = live_stack_app._approximate_sensor_sample(
            live_stack_app.LiveStackOptions(
                approximate_tof_mm=700.0,
                approximate_ultrasonic_mm=710.0,
            )
        )

        self.assertEqual(sample.tof_mm, 700.0)
        self.assertEqual(sample.ultrasonic_mm, 710.0)
        self.assertFalse(sample.limit_switch_active)


if __name__ == "__main__":
    unittest.main()
