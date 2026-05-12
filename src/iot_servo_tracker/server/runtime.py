"""Server-side frame processing runtime."""

from __future__ import annotations

from dataclasses import dataclass

from iot_servo_tracker.common.config import AppConfig
from iot_servo_tracker.common.packets import TrackingResult
from iot_servo_tracker.server.vision import SimulatedVisionPipeline, VisionPipeline


@dataclass
class VisionRuntime:
    config: AppConfig
    pipeline: VisionPipeline | None = None
    frame_index: int = 0

    def __post_init__(self) -> None:
        if self.pipeline is None:
            self.pipeline = SimulatedVisionPipeline(self.config.camera)

    def process(
        self,
        ts_req: int,
        query: str,
        frame_bytes: bytes = b"",
        redetect: bool = False,
    ) -> TrackingResult:
        assert self.pipeline is not None
        if redetect and hasattr(self.pipeline, "redetect"):
            result = self.pipeline.redetect(frame_bytes, query, ts_req)  # type: ignore[attr-defined]
            self.frame_index += 1
            return result
        result = self.pipeline.process_frame(
            ts_req=ts_req,
            query=query,
            frame_bytes=frame_bytes,
            frame_index=self.frame_index,
        )
        self.frame_index += 1
        return result
