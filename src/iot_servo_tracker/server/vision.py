"""Vision pipeline boundaries for simulation and RTX-server inference."""

from __future__ import annotations

import math
import base64
import json
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from iot_servo_tracker.common.config import CameraConfig
from iot_servo_tracker.common.packets import BBox, TrackingResult
from iot_servo_tracker.common.timebase import now_us


class VisionPipeline(Protocol):
    def process_frame(
        self,
        ts_req: int,
        query: str,
        frame_bytes: bytes = b"",
        frame_index: int = 0,
    ) -> TrackingResult:
        """Return a tracking result for the given frame timestamp."""


@dataclass
class SimulatedVisionPipeline:
    camera: CameraConfig
    confidence: float = 0.86

    def process_frame(
        self,
        ts_req: int,
        query: str,
        frame_bytes: bytes = b"",
        frame_index: int = 0,
    ) -> TrackingResult:
        del frame_bytes
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


class WeDetectClient(Protocol):
    def detect(self, frame_bytes: bytes, query: str, ts_req: int) -> TrackingResult:
        """Run open-vocabulary detection for initial lock or redetection."""


@dataclass
class HttpWeDetectClient:
    endpoint: str

    def detect(self, frame_bytes: bytes, query: str, ts_req: int) -> TrackingResult:
        if not self.endpoint:
            raise RuntimeError("wedetect_endpoint is required for production inference")
        payload = json.dumps(
            {
                "query": query,
                "ts_req": ts_req,
                "image_b64": base64.b64encode(frame_bytes).decode("ascii"),
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
        bbox = data.get("bbox")
        return TrackingResult(
            packet="tracking_result",
            ts_req=ts_req,
            ts_resp=now_us(),
            bbox=BBox(*bbox) if bbox else None,
            confidence=float(data.get("confidence", 0.0)),
            track_id=data.get("track_id"),
            query=query,
        )


class WeDetectYoloPipeline:
    """Production inference pipeline for WeDetect lock-on and YOLO tracking."""

    def __init__(
        self,
        wedetect_client: WeDetectClient,
        yolo_model: str = "yolo26n.pt",
        tracker: str = "bytetrack.yaml",
        confidence_threshold: float = 0.25,
    ) -> None:
        try:
            import cv2
            import numpy as np
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Install production dependencies: opencv-python, numpy, ultralytics"
            ) from exc
        self.cv2 = cv2
        self.np = np
        self.yolo = YOLO(yolo_model)
        self.tracker = tracker
        self.confidence_threshold = confidence_threshold
        self.wedetect_client = wedetect_client
        self.locked_bbox: BBox | None = None
        self.locked_track_id: int | None = None

    def process_frame(
        self,
        ts_req: int,
        query: str,
        frame_bytes: bytes = b"",
        frame_index: int = 0,
    ) -> TrackingResult:
        del frame_index
        if self.locked_bbox is None:
            result = self.wedetect_client.detect(frame_bytes, query, ts_req)
            if result.bbox is not None and result.confidence >= self.confidence_threshold:
                self.locked_bbox = result.bbox
                self.locked_track_id = result.track_id
            return result

        frame = self._decode_frame(frame_bytes)
        results = self.yolo.track(
            frame,
            persist=True,
            tracker=self.tracker,
            conf=self.confidence_threshold,
            verbose=False,
        )
        best = self._select_box(results)
        if best is None:
            return TrackingResult.empty(ts_req=ts_req, query=query)
        bbox, confidence, track_id = best
        self.locked_bbox = bbox
        self.locked_track_id = track_id
        return TrackingResult(
            packet="tracking_result",
            ts_req=ts_req,
            ts_resp=now_us(),
            bbox=bbox,
            confidence=confidence,
            track_id=track_id,
            query=query,
        )

    def redetect(self, frame_bytes: bytes, query: str, ts_req: int) -> TrackingResult:
        self.locked_bbox = None
        self.locked_track_id = None
        return self.process_frame(ts_req=ts_req, query=query, frame_bytes=frame_bytes)

    def _decode_frame(self, frame_bytes: bytes):
        array = self.np.frombuffer(frame_bytes, dtype=self.np.uint8)
        frame = self.cv2.imdecode(array, self.cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("failed to decode frame bytes")
        return frame

    def _select_box(self, results) -> tuple[BBox, float, int | None] | None:
        if not results:
            return None
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return None
        candidates = []
        for index, xyxy in enumerate(boxes.xyxy.cpu().tolist()):
            confidence = float(boxes.conf[index].cpu().item()) if boxes.conf is not None else 0.0
            track_id = None
            if boxes.id is not None:
                track_id = int(boxes.id[index].cpu().item())
            bbox = BBox(*map(float, xyxy))
            candidates.append((bbox, confidence, track_id))
        if not candidates:
            return None
        if self.locked_track_id is not None:
            for candidate in candidates:
                if candidate[2] == self.locked_track_id:
                    return candidate
        return min(candidates, key=lambda item: _center_distance(item[0], self.locked_bbox))


def _center_distance(a: BBox, b: BBox | None) -> float:
    if b is None:
        return 0.0
    ax, ay = a.center
    bx, by = b.center
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
