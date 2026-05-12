"""ZMQ multipart packet helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


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
