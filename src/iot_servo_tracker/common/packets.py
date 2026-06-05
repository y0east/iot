"""JSON packet contracts shared by web, edge, and server processes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from iot_servo_tracker.common.timebase import now_us, wall_clock_cmd_id


class CommandType(str, Enum):
    TRACK = "TRACK"
    STOP = "STOP"
    CENTER = "CENTER"
    REDETECT = "REDETECT"


class PacketError(ValueError):
    """Raised when a packet cannot be validated."""


@dataclass(frozen=True)
class CommandPacket:
    packet: str
    cmd_id: str
    cmd_type: CommandType
    query: str = ""
    scan_range_deg: float = 45.0
    max_speed_deg_s: float = 20.0
    ts: int = field(default_factory=now_us)

    @classmethod
    def create(
        cls,
        cmd_type: CommandType | str,
        query: str = "",
        scan_range_deg: float = 45.0,
        max_speed_deg_s: float = 20.0,
    ) -> "CommandPacket":
        cmd = CommandType(cmd_type)
        return cls(
            packet="web_command",
            cmd_id=wall_clock_cmd_id(cmd.value.lower()),
            cmd_type=cmd,
            query=query.strip(),
            scan_range_deg=scan_range_deg,
            max_speed_deg_s=max_speed_deg_s,
        ).validate()

    def validate(self, max_query_len: int = 120) -> "CommandPacket":
        if self.packet != "web_command":
            raise PacketError(f"unexpected packet type: {self.packet}")
        if self.cmd_type == CommandType.TRACK and not self.query.strip():
            raise PacketError("TRACK command requires a non-empty query")
        if len(self.query) > max_query_len:
            raise PacketError("query is too long")
        if self.scan_range_deg <= 0 or self.scan_range_deg > 90:
            raise PacketError("scan_range_deg must be in (0, 90]")
        if self.max_speed_deg_s <= 0 or self.max_speed_deg_s > 120:
            raise PacketError("max_speed_deg_s must be in (0, 120]")
        return self

    def to_json(self) -> str:
        payload = asdict(self)
        payload["cmd_type"] = self.cmd_type.value
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str | bytes) -> "CommandPacket":
        data = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        data["cmd_type"] = CommandType(data["cmd_type"])
        return cls(**data).validate()


@dataclass(frozen=True)
class StatusAck:
    packet: str = "status_ack"
    cmd_id: str = ""
    ack: bool = True
    system_state: str = "IDLE"
    pan_deg: float = 0.0
    tilt_deg: float = 0.0
    rtt_ms: float = 0.0
    confidence: float = 0.0
    message: str = ""
    ts: int = field(default_factory=now_us)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str | bytes) -> "StatusAck":
        data = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        return cls(**data)


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def shifted(self, dx: float, dy: float) -> "BBox":
        return BBox(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy)


@dataclass(frozen=True)
class TrackingResult:
    packet: str
    ts_req: int
    ts_resp: int
    bbox: BBox | None
    confidence: float = 0.0
    track_id: int | None = None
    query: str = ""

    @classmethod
    def empty(cls, ts_req: int, query: str = "") -> "TrackingResult":
        ts = now_us()
        return cls(
            packet="tracking_result",
            ts_req=ts_req,
            ts_resp=ts,
            bbox=None,
            confidence=0.0,
            track_id=None,
            query=query,
        )

    def to_json(self) -> str:
        payload: dict[str, Any] = asdict(self)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str | bytes) -> "TrackingResult":
        data = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        if data.get("bbox") is not None:
            data["bbox"] = BBox(**data["bbox"])
        return cls(**data)


@dataclass(frozen=True)
class SensorSample:
    ts: int
    tof_mm: float | None = None
    ultrasonic_mm: float | None = None
    infrared_active: bool = False
    limit_switch_active: bool = False

    @classmethod
    def empty(cls) -> "SensorSample":
        return cls(ts=now_us())

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))
