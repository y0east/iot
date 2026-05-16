"""Vision pipeline boundaries for simulation and RTX-server inference."""

from __future__ import annotations

import importlib
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from iot_servo_tracker.common.config import CameraConfig, ServerConfig
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
class HuggingFaceWeDetectRefClient:
    """Run WeDetect-Ref from Hugging Face using a local module or script adapter.

    The official WeDetect-Ref model is paired with a WeDetect-Uni proposal checkpoint.
    This client downloads both artifacts with ``huggingface_hub.snapshot_download`` and
    delegates inference to a project-local adapter because the official repository
    exposes WeDetect-Ref through its own script/module code.
    """

    ref_repo_id: str = "fushh7/WeDetect-Ref-2B"
    uni_repo_id: str = "fushh7/WeDetect"
    uni_filename: str = "wedetect_base_uni.pth"
    cache_dir: str = ""
    ref_model_dir: str = ""
    uni_checkpoint: str = ""
    module: str = ""
    script: str = ""
    device: str = "cuda:0"
    timeout_s: float = 30.0

    def detect(self, frame_bytes: bytes, query: str, ts_req: int) -> TrackingResult:
        ref_model_dir, uni_checkpoint = self.resolve_artifacts()
        if self.module:
            return self._detect_with_module(
                frame_bytes,
                query,
                ts_req,
                ref_model_dir,
                uni_checkpoint,
            )
        if self.script:
            return self._detect_with_script(
                frame_bytes,
                query,
                ts_req,
                ref_model_dir,
                uni_checkpoint,
            )
        raise RuntimeError(
            "WeDetect-Ref requires wedetect_ref_module or wedetect_ref_script after "
            "downloading Hugging Face artifacts"
        )

    def resolve_artifacts(self) -> tuple[str, str]:
        ref_model_dir = self.ref_model_dir
        uni_checkpoint = self.uni_checkpoint
        if ref_model_dir and uni_checkpoint:
            return ref_model_dir, uni_checkpoint

        try:
            from huggingface_hub import hf_hub_download, snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "Install huggingface-hub or provide wedetect_ref_model_dir and "
                "wedetect_uni_checkpoint explicitly"
            ) from exc

        cache_dir = self.cache_dir or None
        if not ref_model_dir:
            ref_model_dir = snapshot_download(repo_id=self.ref_repo_id, cache_dir=cache_dir)
        if not uni_checkpoint:
            uni_checkpoint = hf_hub_download(
                repo_id=self.uni_repo_id,
                filename=self.uni_filename,
                cache_dir=cache_dir,
            )
        return ref_model_dir, uni_checkpoint

    def _detect_with_module(
        self,
        frame_bytes: bytes,
        query: str,
        ts_req: int,
        ref_model_dir: str,
        uni_checkpoint: str,
    ) -> TrackingResult:
        module_name, _, function_name = self.module.partition(":")
        function_name = function_name or "detect"
        module = importlib.import_module(module_name)
        detector = getattr(module, function_name)
        result = detector(
            frame_bytes=frame_bytes,
            query=query,
            ts_req=ts_req,
            wedetect_ref_model_dir=ref_model_dir,
            wedetect_uni_checkpoint=uni_checkpoint,
            device=self.device,
        )
        return _tracking_result_from_payload(result, query=query, ts_req=ts_req)

    def _detect_with_script(
        self,
        frame_bytes: bytes,
        query: str,
        ts_req: int,
        ref_model_dir: str,
        uni_checkpoint: str,
    ) -> TrackingResult:
        suffix = _image_suffix(frame_bytes)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fp:
            fp.write(frame_bytes)
            image_path = fp.name
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    self.script,
                    "--wedetect_ref_checkpoint",
                    ref_model_dir,
                    "--wedetect_uni_checkpoint",
                    uni_checkpoint,
                    "--image",
                    image_path,
                    "--query",
                    query,
                    "--device",
                    self.device,
                    "--json",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_s,
                check=False,
            )
        finally:
            Path(image_path).unlink(missing_ok=True)
        if completed.returncode != 0:
            error = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"WeDetect-Ref script failed: {error}")
        data = json.loads(completed.stdout.decode("utf-8"))
        return _tracking_result_from_payload(data, query=query, ts_req=ts_req)


class WeDetectYoloPipeline:
    """Production inference pipeline for WeDetect lock-on and YOLO tracking."""

    def __init__(
        self,
        wedetect_client: WeDetectClient,
        yolo_model: str = "yolo26n.pt",
        tracker: str = "bytetrack.yaml",
        confidence_threshold: float = 0.25,
        yolo_lost_frames: int = 3,
        yolo_suspect_frames: int = 2,
        yolo_max_center_jump_px: float = 120.0,
        yolo_max_area_growth_ratio: float = 4.0,
        yolo_max_frame_area_ratio: float = 0.35,
        yolo_max_aspect_ratio_change: float = 3.0,
        yolo_min_iou_on_id_change: float = 0.10,
        camera: CameraConfig | None = None,
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
        self.yolo_lost_frames = max(1, yolo_lost_frames)
        self.yolo_suspect_frames = max(1, yolo_suspect_frames)
        self.yolo_max_center_jump_px = yolo_max_center_jump_px
        self.yolo_max_area_growth_ratio = yolo_max_area_growth_ratio
        self.yolo_max_frame_area_ratio = yolo_max_frame_area_ratio
        self.yolo_max_aspect_ratio_change = yolo_max_aspect_ratio_change
        self.yolo_min_iou_on_id_change = yolo_min_iou_on_id_change
        self.camera = camera or CameraConfig()
        self.wedetect_client = wedetect_client
        self.locked_bbox: BBox | None = None
        self.locked_track_id: int | None = None
        self.redetect_reference_bbox: BBox | None = None
        self.yolo_lost_count = 0
        self.yolo_suspect_count = 0
        self.last_yolo_reject_reason = ""

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
                if self.redetect_reference_bbox is not None:
                    reason = self._candidate_rejection_reason(
                        result.bbox,
                        result.confidence,
                        result.track_id,
                        previous_bbox=self.redetect_reference_bbox,
                    )
                    if reason is not None:
                        self.last_yolo_reject_reason = reason
                        return TrackingResult.empty(ts_req=ts_req, query=query)
                self.locked_bbox = result.bbox
                self.locked_track_id = result.track_id
                self.redetect_reference_bbox = None
                self.yolo_lost_count = 0
                self.yolo_suspect_count = 0
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
            if self.last_yolo_reject_reason:
                return self._handle_suspect_yolo_candidate(frame_bytes, query, ts_req)
            return self._handle_yolo_loss(frame_bytes, query, ts_req)
        bbox, confidence, track_id = best
        self.locked_bbox = bbox
        self.locked_track_id = track_id
        self.yolo_lost_count = 0
        self.yolo_suspect_count = 0
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
        self.redetect_reference_bbox = self.locked_bbox
        self.locked_bbox = None
        self.locked_track_id = None
        self.yolo_lost_count = 0
        self.yolo_suspect_count = 0
        return self.process_frame(ts_req=ts_req, query=query, frame_bytes=frame_bytes)

    def _decode_frame(self, frame_bytes: bytes):
        array = self.np.frombuffer(frame_bytes, dtype=self.np.uint8)
        frame = self.cv2.imdecode(array, self.cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("failed to decode frame bytes")
        return frame

    def _select_box(self, results) -> tuple[BBox, float, int | None] | None:
        self.last_yolo_reject_reason = ""
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
                raw_track_id = boxes.id[index].cpu().item()
                if raw_track_id is not None:
                    track_id = int(raw_track_id)
            bbox = BBox(*map(float, xyxy))
            candidates.append((bbox, confidence, track_id))
        if not candidates:
            return None

        valid_candidates = []
        rejection_reasons = []
        for candidate in candidates:
            bbox, confidence, track_id = candidate
            reason = self._candidate_rejection_reason(bbox, confidence, track_id)
            if reason is None:
                valid_candidates.append(candidate)
            else:
                rejection_reasons.append(reason)

        if not valid_candidates:
            self.last_yolo_reject_reason = rejection_reasons[0] if rejection_reasons else ""
            return None

        if self.locked_track_id is not None:
            for candidate in valid_candidates:
                if candidate[2] == self.locked_track_id:
                    return candidate
        return min(valid_candidates, key=lambda item: _center_distance(item[0], self.locked_bbox))

    def _handle_yolo_loss(
        self,
        frame_bytes: bytes,
        query: str,
        ts_req: int,
    ) -> TrackingResult:
        self.yolo_lost_count += 1
        self.yolo_suspect_count = 0
        if self.yolo_lost_count >= self.yolo_lost_frames:
            return self._run_wedetect_redetection(frame_bytes, query, ts_req)
        return TrackingResult.empty(ts_req=ts_req, query=query)

    def _handle_suspect_yolo_candidate(
        self,
        frame_bytes: bytes,
        query: str,
        ts_req: int,
    ) -> TrackingResult:
        self.yolo_suspect_count += 1
        self.yolo_lost_count = 0
        if self.yolo_suspect_count >= self.yolo_suspect_frames:
            return self._run_wedetect_redetection(frame_bytes, query, ts_req)
        return TrackingResult.empty(ts_req=ts_req, query=query)

    def _run_wedetect_redetection(
        self,
        frame_bytes: bytes,
        query: str,
        ts_req: int,
    ) -> TrackingResult:
        previous_bbox = self.locked_bbox
        self.locked_bbox = None
        self.locked_track_id = None
        result = self.wedetect_client.detect(frame_bytes, query, ts_req)
        if result.bbox is not None and result.confidence >= self.confidence_threshold:
            reason = self._candidate_rejection_reason(
                result.bbox,
                result.confidence,
                result.track_id,
                previous_bbox=previous_bbox,
            )
            if reason is None:
                self.locked_bbox = result.bbox
                self.locked_track_id = result.track_id
                self.redetect_reference_bbox = None
                self.yolo_lost_count = 0
                self.yolo_suspect_count = 0
                return result
            self.last_yolo_reject_reason = reason
            self.redetect_reference_bbox = previous_bbox
        elif previous_bbox is not None:
            self.redetect_reference_bbox = previous_bbox
        return TrackingResult.empty(ts_req=ts_req, query=query)

    def _candidate_rejection_reason(
        self,
        bbox: BBox,
        confidence: float,
        track_id: int | None,
        previous_bbox: BBox | None = None,
    ) -> str | None:
        previous = previous_bbox or self.locked_bbox
        if confidence < self.confidence_threshold:
            return "YOLO candidate confidence is below threshold"
        if previous is None:
            return None
        if bbox.area <= 0:
            return "YOLO candidate bbox is empty"

        frame_area = self.camera.width * self.camera.height
        if frame_area > 0 and bbox.area / frame_area > self.yolo_max_frame_area_ratio:
            return "YOLO candidate bbox is too large for the target"
        if previous.area > 0 and bbox.area / previous.area > self.yolo_max_area_growth_ratio:
            return "YOLO candidate bbox grew too much"
        if _aspect_ratio_change(bbox, previous) > self.yolo_max_aspect_ratio_change:
            return "YOLO candidate aspect ratio changed too much"
        if _center_distance(bbox, previous) > self.yolo_max_center_jump_px:
            return "YOLO candidate center jumped too far"
        if (
            self.locked_track_id is not None
            and track_id is not None
            and track_id != self.locked_track_id
            and _iou(bbox, previous) < self.yolo_min_iou_on_id_change
        ):
            return "YOLO candidate track id changed without enough overlap"
        return None


def build_wedetect_client(config: ServerConfig) -> WeDetectClient:
    return HuggingFaceWeDetectRefClient(
        ref_repo_id=config.wedetect_ref_repo_id,
        uni_repo_id=config.wedetect_uni_repo_id,
        uni_filename=config.wedetect_uni_filename,
        cache_dir=config.wedetect_cache_dir,
        ref_model_dir=config.wedetect_ref_model_dir,
        uni_checkpoint=config.wedetect_uni_checkpoint,
        module=config.wedetect_ref_module,
        script=config.wedetect_ref_script,
        device=config.wedetect_device,
    )


def _tracking_result_from_payload(
    payload: TrackingResult | dict[str, Any] | None,
    query: str,
    ts_req: int,
) -> TrackingResult:
    if isinstance(payload, TrackingResult):
        return payload
    if payload is None:
        return TrackingResult.empty(ts_req=ts_req, query=query)
    bbox_payload = payload.get("bbox")
    bbox = None
    if isinstance(bbox_payload, dict):
        bbox = BBox(**bbox_payload)
    elif bbox_payload is not None:
        bbox = BBox(*bbox_payload)
    track_id = payload.get("track_id")
    return TrackingResult(
        packet=str(payload.get("packet", "tracking_result")),
        ts_req=int(payload.get("ts_req", ts_req)),
        ts_resp=int(payload.get("ts_resp", now_us())),
        bbox=bbox,
        confidence=float(payload.get("confidence", 0.0)),
        track_id=int(track_id) if track_id is not None else None,
        query=str(payload.get("query", query)),
    )


def _center_distance(a: BBox, b: BBox | None) -> float:
    if b is None:
        return 0.0
    ax, ay = a.center
    bx, by = b.center
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _image_suffix(frame_bytes: bytes) -> str:
    if frame_bytes.startswith(b"\xff\xd8"):
        return ".jpg"
    if frame_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    return ".bin"


def _aspect_ratio_change(current: BBox, previous: BBox) -> float:
    current_width = max(current.x2 - current.x1, 1e-6)
    current_height = max(current.y2 - current.y1, 1e-6)
    previous_width = max(previous.x2 - previous.x1, 1e-6)
    previous_height = max(previous.y2 - previous.y1, 1e-6)
    current_ratio = current_width / current_height
    previous_ratio = previous_width / previous_height
    return max(current_ratio / previous_ratio, previous_ratio / current_ratio)


def _iou(a: BBox, b: BBox) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.area + b.area - intersection
    if union <= 0:
        return 0.0
    return intersection / union
