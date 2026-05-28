"""Background MJPEG streaming server for the vision server."""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

_FRAME_LOCK = threading.Lock()
_LATEST_RAW_JPEG: bytes | None = None
_LATEST_BBOX: tuple[float, float, float, float] | None = None
_LATEST_LABEL: str = ""


class MJPEGStreamHandler(BaseHTTPRequestHandler):
    """Serve MJPEG stream, decoding and annotating on-the-fly per client."""

    def do_GET(self):
        if self.path != "/stream.mjpg":
            self.send_error(404, "File Not Found")
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        try:
            import cv2
            import numpy as np
        except ImportError:
            cv2 = None
            np = None

        last_processed = -1

        try:
            while True:
                with _FRAME_LOCK:
                    raw_jpeg = _LATEST_RAW_JPEG
                    bbox = _LATEST_BBOX
                    label = _LATEST_LABEL

                if raw_jpeg is None or id(raw_jpeg) == last_processed:
                    time.sleep(0.03)
                    continue

                last_processed = id(raw_jpeg)
                out_jpeg = raw_jpeg

                if cv2 is not None and np is not None and bbox is not None:
                    array = np.frombuffer(raw_jpeg, dtype=np.uint8)
                    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
                    if frame is not None:
                        x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                        if label:
                            cv2.putText(
                                frame,
                                label,
                                (x1, max(24, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                (0, 220, 0),
                                2,
                                cv2.LINE_AA,
                            )
                        ok, encoded = cv2.imencode(".jpg", frame)
                        if ok:
                            out_jpeg = encoded.tobytes()

                header = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(out_jpeg)).encode() + b"\r\n"
                    b"\r\n"
                )
                self.wfile.write(header)
                self.wfile.write(out_jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                time.sleep(0.03)  # Rate limit to ~30 FPS per client
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            pass

    def log_message(self, format, *args):
        """Suppress per-request log spam."""
        pass


class BackgroundMjpegServer:
    """Non-blocking MJPEG server that accepts frames from the vision loop."""

    def __init__(self, port: int = 8000):
        self.port = port
        self.server = None
        self.thread = None

    def start(self):
        self.server = ThreadingHTTPServer(("0.0.0.0", self.port), MJPEGStreamHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()

    def update_frame(self, raw_jpeg: bytes, bbox: Any = None, label: str = ""):
        global _LATEST_RAW_JPEG, _LATEST_BBOX, _LATEST_LABEL
        tup = (bbox.x1, bbox.y1, bbox.x2, bbox.y2) if bbox is not None else None
        with _FRAME_LOCK:
            _LATEST_RAW_JPEG = raw_jpeg
            _LATEST_BBOX = tup
            _LATEST_LABEL = label
