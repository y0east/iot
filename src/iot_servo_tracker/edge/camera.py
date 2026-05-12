"""Camera source adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class CameraSource(Protocol):
    def read_jpeg(self) -> bytes:
        """Return one encoded frame."""

    def close(self) -> None:
        """Release camera resources."""


@dataclass
class SimulatedCamera:
    payload: bytes = b"simulated-frame"

    def read_jpeg(self) -> bytes:
        return self.payload

    def close(self) -> None:
        return None


class OpenCvCamera:
    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Install opencv-python to use the camera") from exc

        self.cv2 = cv2
        self.capture = cv2.VideoCapture(device_index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.capture.isOpened():
            raise RuntimeError(f"failed to open camera index {device_index}")

    def read_jpeg(self) -> bytes:
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError("failed to read camera frame")
        ok, encoded = self.cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("failed to encode camera frame")
        return encoded.tobytes()

    def close(self) -> None:
        self.capture.release()
