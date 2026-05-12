"""ZMQ multipart packet helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from iot_servo_tracker.common.packets import TrackingResult


@dataclass(frozen=True)
class MultipartFrame:
    header: dict[str, Any]
    payload: bytes

    def encode(self) -> list[bytes]:
        return [
            json.dumps(self.header, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            self.payload,
        ]

    @classmethod
    def decode(cls, parts: list[bytes]) -> "MultipartFrame":
        if len(parts) != 2:
            raise ValueError("expected two ZMQ parts: JSON header and binary payload")
        return cls(header=json.loads(parts[0].decode("utf-8")), payload=parts[1])


def require_zmq():
    try:
        import zmq
    except ImportError as exc:
        raise RuntimeError("Install optional dependency: pip install '.[edge,server]'") from exc
    return zmq


class ZmqEdgeTransport:
    """Edge-side frame sender and inference-result receiver."""

    def __init__(self, frame_endpoint: str, result_endpoint: str) -> None:
        zmq = require_zmq()
        self.zmq = zmq
        self.context = zmq.Context.instance()
        self.frame_socket = self.context.socket(zmq.PUSH)
        self.frame_socket.connect(frame_endpoint)
        self.result_socket = self.context.socket(zmq.PULL)
        self.result_socket.connect(result_endpoint)

    def send_frame(self, ts_req: int, query: str, frame_bytes: bytes, frame_index: int) -> None:
        frame = MultipartFrame(
            header={
                "packet": "frame",
                "ts_req": ts_req,
                "query": query,
                "frame_index": frame_index,
            },
            payload=frame_bytes,
        )
        self.frame_socket.send_multipart(frame.encode())

    def recv_result(self, timeout_ms: int = 0) -> TrackingResult | None:
        poller = self.zmq.Poller()
        poller.register(self.result_socket, self.zmq.POLLIN)
        events = dict(poller.poll(timeout_ms))
        if self.result_socket not in events:
            return None
        parts = self.result_socket.recv_multipart()
        header = MultipartFrame.decode(parts).header
        return TrackingResult.from_json(header["result"])

    def close(self) -> None:
        self.frame_socket.close(linger=0)
        self.result_socket.close(linger=0)


class ZmqVisionTransport:
    """Server-side frame receiver and inference-result sender."""

    def __init__(self, frame_endpoint: str, result_endpoint: str) -> None:
        zmq = require_zmq()
        self.zmq = zmq
        self.context = zmq.Context.instance()
        self.frame_socket = self.context.socket(zmq.PULL)
        self.frame_socket.bind(frame_endpoint)
        self.result_socket = self.context.socket(zmq.PUSH)
        self.result_socket.bind(result_endpoint)

    def recv_frame(self, timeout_ms: int = 1000) -> tuple[dict[str, Any], bytes] | None:
        poller = self.zmq.Poller()
        poller.register(self.frame_socket, self.zmq.POLLIN)
        events = dict(poller.poll(timeout_ms))
        if self.frame_socket not in events:
            return None
        frame = MultipartFrame.decode(self.frame_socket.recv_multipart())
        return frame.header, frame.payload

    def send_result(self, result: TrackingResult) -> None:
        frame = MultipartFrame(header={"packet": "tracking_result", "result": result.to_json()}, payload=b"")
        self.result_socket.send_multipart(frame.encode())

    def close(self) -> None:
        self.frame_socket.close(linger=0)
        self.result_socket.close(linger=0)
