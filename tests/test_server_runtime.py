import unittest

from iot_servo_tracker.common.config import AppConfig
from iot_servo_tracker.common.packets import TrackingResult
from iot_servo_tracker.server.runtime import VisionRuntime


class RecordingPipeline:
    def __init__(self) -> None:
        self.frame_bytes = b""

    def process_frame(
        self,
        ts_req: int,
        query: str,
        frame_bytes: bytes = b"",
        frame_index: int = 0,
    ) -> TrackingResult:
        self.frame_bytes = frame_bytes
        return TrackingResult.empty(ts_req=ts_req, query=query)


class VisionRuntimeTests(unittest.TestCase):
    def test_passes_frame_bytes_to_pipeline(self) -> None:
        pipeline = RecordingPipeline()
        runtime = VisionRuntime(AppConfig(), pipeline=pipeline)
        runtime.process(ts_req=1, query="red cup", frame_bytes=b"jpeg-bytes")
        self.assertEqual(pipeline.frame_bytes, b"jpeg-bytes")


if __name__ == "__main__":
    unittest.main()
