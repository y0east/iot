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

        # 첫 프레임이 들어올 때까지 대기 (최대 5초)
        start_wait = time.time()
        while self.latest_frame is None and time.time() - start_wait < 5.0:
            time.sleep(0.1)

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


class RpiCamVidCamera:
    """Robust camera source for modern Raspberry Pi OS using rpicam-vid."""

    def __init__(self, width: int = 640, height: int = 480) -> None:
        import subprocess
        import threading
        import time

        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True

        cmd = [
            "rpicam-vid",
            "-t", "0",
            "--codec", "mjpeg",
            "--width", str(width),
            "--height", str(height),
            "--framerate", "15",
            "--inline",
            "-o", "-",
        ]
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except FileNotFoundError as exc:
            raise RuntimeError("rpicam-vid not found. Ensure you are on a compatible Raspberry Pi OS.") from exc

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

        # Wait for first frame (up to 5 seconds)
        start_wait = time.time()
        while self.latest_frame is None and time.time() - start_wait < 5.0:
            time.sleep(0.1)

    def _capture_loop(self):
        buffer = b""
        while self.running and self.proc.poll() is None:
            chunk = self.proc.stdout.read(4096)
            if not chunk:
                break
            buffer += chunk

            # Find JPEG start (0xff 0xd8) and end (0xff 0xd9)
            start = buffer.find(b"\xff\xd8")
            if start != -1:
                end = buffer.find(b"\xff\xd9", start)
                if end != -1:
                    frame = buffer[start : end + 2]
                    with self.lock:
                        self.latest_frame = frame
                    buffer = buffer[end + 2 :]
            
            # Prevent buffer from growing infinitely if stream is corrupted
            if len(buffer) > 2 * 1024 * 1024:
                buffer = b""

    def read_jpeg(self) -> bytes:
        with self.lock:
            frame = self.latest_frame
        if frame is None:
            raise RuntimeError("failed to read camera frame (rpicam-vid produced no frames)")
        return frame

    def close(self) -> None:
        self.running = False
        if hasattr(self, 'proc'):
            self.proc.terminate()
            self.proc.wait(timeout=2.0)
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
