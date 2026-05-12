import unittest

from iot_servo_tracker.common.packets import BBox, TrackingResult
from iot_servo_tracker.edge.ring_buffer import DetectionHistory, RingBuffer


class RingBufferTests(unittest.TestCase):
    def test_nearest_record(self) -> None:
        buffer = RingBuffer[str](maxlen=3)
        buffer.append(100, "a")
        buffer.append(200, "b")
        buffer.append(300, "c")
        self.assertEqual(buffer.nearest(240).value, "b")

    def test_detection_history_compensates_forward(self) -> None:
        history = DetectionHistory()
        history.append(
            TrackingResult("tracking_result", 0, 1_000_000, BBox(0, 0, 10, 10), 0.8, 1)
        )
        history.append(
            TrackingResult("tracking_result", 0, 2_000_000, BBox(10, 0, 20, 10), 0.8, 1)
        )
        current = TrackingResult("tracking_result", 2_000_000, 2_000_000, BBox(10, 0, 20, 10), 0.8, 1)
        estimated = history.estimate_current(current, now_us=3_000_000)
        self.assertGreater(estimated.center[0], current.bbox.center[0])


if __name__ == "__main__":
    unittest.main()
