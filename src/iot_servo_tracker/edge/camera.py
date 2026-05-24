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
            import threading
            import time
        except ImportError as exc:
            raise RuntimeError("Install opencv-python to use the camera") from exc

        self.cv2 = cv2
        self.capture = cv2.VideoCapture(device_index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.capture.isOpened():
            raise RuntimeError(f"failed to open camera index {device_index}")

        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True

        # 버퍼에 쌓인 오래된 프레임을 버리고 항상 최신 프레임만 유지하기 위한 백그라운드 스레드
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

        # 첫 프레임이 들어올 때까지 잠시 대기
        time.sleep(0.5)

    def _capture_loop(self):
        import time
        while self.running:
            ok, frame = self.capture.read()
            if ok:
                with self.lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.01)

    def read_jpeg(self) -> bytes:
        with self.lock:
            frame = self.latest_frame
        if frame is None:
            raise RuntimeError("failed to read camera frame (no frames arrived yet)")
            
        ok, encoded = self.cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("failed to encode camera frame")
        return encoded.tobytes()

    def close(self) -> None:
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        self.capture.release()
