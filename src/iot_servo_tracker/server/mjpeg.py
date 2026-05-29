"""Background MJPEG streaming server for the vision server."""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

_FRAME_LOCK = threading.Lock()
_LATEST_RAW_JPEG: bytes | None = None
_LATEST_STREAM_JPEG: bytes | None = None
_LATEST_BBOX: tuple[float, float, float, float] | None = None
_LATEST_LABEL: str = ""
_LATEST_FRAME_VERSION = 0
_SMOOTH_CROP_BOX: tuple[float, float, float, float] | None = None


class MJPEGStreamHandler(BaseHTTPRequestHandler):
    """Serve the latest pre-rendered MJPEG frame to each client."""

    def do_GET(self):
        if self.path != "/stream.mjpg":
            self.send_error(404, "File Not Found")
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        last_processed = -1

        try:
            while True:
                with _FRAME_LOCK:
                    out_jpeg = _LATEST_STREAM_JPEG
                    version = _LATEST_FRAME_VERSION

                if out_jpeg is None or version == last_processed:
                    time.sleep(0.03)
                    continue

                last_processed = version

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
        global _LATEST_RAW_JPEG, _LATEST_STREAM_JPEG, _LATEST_BBOX, _LATEST_LABEL
        global _LATEST_FRAME_VERSION
        tup = (bbox.x1, bbox.y1, bbox.x2, bbox.y2) if bbox is not None else None
        stream_jpeg = _annotate_frame_jpeg(raw_jpeg, tup, label)
        with _FRAME_LOCK:
            _LATEST_RAW_JPEG = raw_jpeg
            _LATEST_STREAM_JPEG = stream_jpeg
            _LATEST_BBOX = tup
            _LATEST_LABEL = label
            _LATEST_FRAME_VERSION += 1

    def update_raw_jpeg(self, raw_jpeg: bytes):
        global _LATEST_RAW_JPEG, _LATEST_STREAM_JPEG, _LATEST_FRAME_VERSION
        with _FRAME_LOCK:
            bbox = _LATEST_BBOX
            label = _LATEST_LABEL
        stream_jpeg = _annotate_frame_jpeg(raw_jpeg, bbox, label)
        with _FRAME_LOCK:
            _LATEST_RAW_JPEG = raw_jpeg
            _LATEST_STREAM_JPEG = stream_jpeg
            _LATEST_FRAME_VERSION += 1

    def update_bbox(self, bbox: Any = None, label: str = ""):
        global _LATEST_STREAM_JPEG, _LATEST_BBOX, _LATEST_LABEL, _LATEST_FRAME_VERSION
        tup = (bbox.x1, bbox.y1, bbox.x2, bbox.y2) if bbox is not None else None
        with _FRAME_LOCK:
            raw_jpeg = _LATEST_RAW_JPEG
        stream_jpeg = _annotate_frame_jpeg(raw_jpeg, tup, label) if raw_jpeg is not None else None
        with _FRAME_LOCK:
            _LATEST_BBOX = tup
            _LATEST_LABEL = label
            if stream_jpeg is not None:
                _LATEST_STREAM_JPEG = stream_jpeg
                _LATEST_FRAME_VERSION += 1


def _annotate_frame_jpeg(
    raw_jpeg: bytes,
    bbox: tuple[float, float, float, float] | None,
    label: str = "",
) -> bytes:
    global _SMOOTH_CROP_BOX
    try:
        import cv2
        import numpy as np
    except ImportError:
        return raw_jpeg

    array = np.frombuffer(raw_jpeg, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        return raw_jpeg

    h, w = frame.shape[:2]
    
    # Target crop box calculation
    if bbox is None:
        # 줌아웃 타겟: 전체 화면
        target_crop = (0.0, 0.0, float(w), float(h))
    else:
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        cx, cy = x1 + bw / 2.0, y1 + bh / 2.0
        
        # 1.5배 마진 (안정적인 뷰)
        margin = 1.5
        crop_w = bw * margin
        crop_h = bh * margin
        
        # 화면 밖으로 나가지 않도록 조정, 너무 작아지지 않도록 최소 크기(원래 화면의 20%) 지정
        crop_w = max(crop_w, w * 0.2)
        crop_h = max(crop_h, h * 0.2)
        
        # 종횡비를 원본 해상도(w:h)와 맞춤
        aspect_ratio = w / h
        if crop_w / crop_h > aspect_ratio:
            crop_h = crop_w / aspect_ratio
        else:
            crop_w = crop_h * aspect_ratio
            
        cx1 = max(0, min(w - crop_w, cx - crop_w / 2.0))
        cy1 = max(0, min(h - crop_h, cy - crop_h / 2.0))
        cx2 = min(w, cx1 + crop_w)
        cy2 = min(h, cy1 + crop_h)
        target_crop = (float(cx1), float(cy1), float(cx2), float(cy2))

    # EMA 스무딩 적용 (알파값이 작을수록 부드러움)
    alpha = 0.1
    if _SMOOTH_CROP_BOX is None:
        _SMOOTH_CROP_BOX = target_crop
    else:
        scx1, scy1, scx2, scy2 = _SMOOTH_CROP_BOX
        tcx1, tcy1, tcx2, tcy2 = target_crop
        
        scx1 += (tcx1 - scx1) * alpha
        scy1 += (tcy1 - scy1) * alpha
        scx2 += (tcx2 - scx2) * alpha
        scy2 += (tcy2 - scy2) * alpha
        _SMOOTH_CROP_BOX = (scx1, scy1, scx2, scy2)

    # 크롭 영역 정수로 변환
    ix1, iy1, ix2, iy2 = [int(v) for v in _SMOOTH_CROP_BOX]
    ix1, iy1 = max(0, ix1), max(0, iy1)
    ix2, iy2 = min(w, ix2), min(h, iy2)
    
    # 줌 인/아웃 크롭 및 리사이즈
    cropped_frame = frame[iy1:iy2, ix1:ix2]
    if cropped_frame.size == 0:
        cropped_frame = frame
    else:
        cropped_frame = cv2.resize(cropped_frame, (w, h), interpolation=cv2.INTER_LINEAR)

    # BBox 그리기 (크롭된 화면 기준 좌표로 변환)
    if bbox is not None:
        scale_x = w / (ix2 - ix1)
        scale_y = h / (iy2 - iy1)
        bx1 = int((bbox[0] - ix1) * scale_x)
        by1 = int((bbox[1] - iy1) * scale_y)
        bx2 = int((bbox[2] - ix1) * scale_x)
        by2 = int((bbox[3] - iy1) * scale_y)
        
        cv2.rectangle(cropped_frame, (bx1, by1), (bx2, by2), (0, 220, 0), 2)
        if label:
            cv2.putText(
                cropped_frame,
                label,
                (bx1, max(24, by1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 220, 0),
                2,
                cv2.LINE_AA,
            )

    # 원본 이미지를 보여줘야 할 정도로 완전히 줌아웃 된 상태이고, bbox도 없으면 그대로 반환
    if bbox is None and ix1 == 0 and iy1 == 0 and ix2 == w and iy2 == h:
        return raw_jpeg

    ok, encoded = cv2.imencode(".jpg", cropped_frame)
    return encoded.tobytes() if ok else raw_jpeg
