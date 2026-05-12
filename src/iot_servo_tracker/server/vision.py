"""Vision pipeline boundaries.

The simulated implementation keeps the project runnable without GPU models.
Replace the detector/tracker methods with WeDetect and YOLO26 integrations on
the RTX server.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from iot_servo_tracker.common.config import CameraConfig
from iot_servo_tracker.common.packets import BBox, TrackingResult
from iot_servo_tracker.common.timebase import now_us


class VisionPipeline(Protocol):
    def process_frame(self, ts_req: int, query: str, frame_index: int = 0) -> TrackingResult:
        """Return a tracking result for the given frame timestamp."""


@dataclass
class SimulatedVisionPipeline:
    camera: CameraConfig
    confidence: float = 0.86

    def process_frame(self, ts_req: int, query: str, frame_index: int = 0) -> TrackingResult:
        width = 80.0
        height = 60.0
        cx = self.camera.width / 2 + 120 * math.sin(frame_index / 20.0)
        cy = self.camera.height / 2 + 35 * math.sin(frame_index / 33.0)
        bbox = BBox(cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)
        return TrackingResult(
            packet="tracking_result",
            ts_req=ts_req,
            ts_resp=now_us(),
            bbox=bbox,
            confidence=self.confidence,
            track_id=1,
            query=query,
        )


class WeDetectYoloPipeline:
    """Production boundary for RTX-server inference.

    Expected responsibilities:
    - WeDetect open-vocabulary initial detection from the natural language query.
    - YOLO26 + BoT-SORT or ByteTrack update after target lock.
    - Redetect on edge SAFE_HOLD recovery requests.
    """

    def __init__(self) -> None:
        raise NotImplementedError("Connect WeDetect and YOLO26 models here.")

    def process_frame(self, ts_req: int, query: str, frame_index: int = 0) -> TrackingResult:
        raise NotImplementedError
