"""Configuration dataclasses and TOML loading."""

from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AxisLimit:
    min_deg: float
    max_deg: float
    center_deg: float = 0.0
    pwm_min_us: int = 1000
    pwm_max_us: int = 2000


@dataclass(frozen=True)
class CameraConfig:
    width: int = 640
    height: int = 480
    horizontal_fov_deg: float = 62.0
    vertical_fov_deg: float = 48.0


@dataclass(frozen=True)
class ControlConfig:
    deadband_deg: float = 0.6
    max_speed_deg_s: float = 35.0
    max_accel_deg_s2: float = 120.0
    derivative_filter_gamma: float = 0.65
    pan_kp: float = 1.4
    pan_kd: float = 0.05
    tilt_kp: float = 1.1
    tilt_kd: float = 0.04
    pan: AxisLimit = field(default_factory=lambda: AxisLimit(-60.0, 60.0))
    tilt: AxisLimit = field(default_factory=lambda: AxisLimit(-30.0, 45.0))


@dataclass(frozen=True)
class SafetyConfig:
    pixel_jump_threshold: float = 80.0
    tof_delta_threshold_mm: float = 40.0
    ultrasonic_jump_threshold_mm: float = 120.0
    consecutive_frames: int = 3
    timeout_min_s: float = 1.5
    timeout_max_s: float = 5.0


@dataclass(frozen=True)
class ScanConfig:
    range_deg: float = 45.0
    speed_deg_s: float = 10.0
    confidence_threshold: float = 0.50
    confirmation_frames: int = 3
    passes: int = 2


@dataclass(frozen=True)
class MqttConfig:
    host: str = "127.0.0.1"
    port: int = 1883
    command_topic: str = "iot_servo_tracker/command"
    status_topic: str = "iot_servo_tracker/status"


@dataclass(frozen=True)
class ZmqConfig:
    frame_endpoint: str = "tcp://0.0.0.0:5555"
    result_endpoint: str = "tcp://0.0.0.0:5556"


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)
    mqtt: MqttConfig = field(default_factory=MqttConfig)
    zmq: ZmqConfig = field(default_factory=ZmqConfig)


def _axis(data: dict[str, Any], default: AxisLimit) -> AxisLimit:
    merged = {**default.__dict__, **data}
    return AxisLimit(**merged)


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        return AppConfig()
    config_path = Path(path)
    if not config_path.exists():
        return AppConfig()

    raw = _read_toml(config_path)

    camera = CameraConfig(**raw.get("camera", {}))

    default_control = ControlConfig()
    control_raw = raw.get("control", {})
    control = ControlConfig(
        deadband_deg=control_raw.get("deadband_deg", default_control.deadband_deg),
        max_speed_deg_s=control_raw.get("max_speed_deg_s", default_control.max_speed_deg_s),
        max_accel_deg_s2=control_raw.get(
            "max_accel_deg_s2", default_control.max_accel_deg_s2
        ),
        derivative_filter_gamma=control_raw.get(
            "derivative_filter_gamma", default_control.derivative_filter_gamma
        ),
        pan_kp=control_raw.get("pan_kp", default_control.pan_kp),
        pan_kd=control_raw.get("pan_kd", default_control.pan_kd),
        tilt_kp=control_raw.get("tilt_kp", default_control.tilt_kp),
        tilt_kd=control_raw.get("tilt_kd", default_control.tilt_kd),
        pan=_axis(control_raw.get("pan", {}), default_control.pan),
        tilt=_axis(control_raw.get("tilt", {}), default_control.tilt),
    )

    return AppConfig(
        camera=camera,
        control=control,
        safety=SafetyConfig(**raw.get("safety", {})),
        scan=ScanConfig(**raw.get("scan", {})),
        mqtt=MqttConfig(**raw.get("mqtt", {})),
        zmq=ZmqConfig(**raw.get("zmq", {})),
    )


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib  # type: ignore[import-not-found]

        with path.open("rb") as fp:
            return tomllib.load(fp)
    except ModuleNotFoundError:
        try:
            import tomli  # type: ignore[import-not-found]

            with path.open("rb") as fp:
                return tomli.load(fp)
        except ModuleNotFoundError:
            return _read_simple_toml(path)


def _read_simple_toml(path: Path) -> dict[str, Any]:
    """Parse the simple settings file format without external dependencies."""

    parser = configparser.ConfigParser()
    parser.read(path)
    data: dict[str, Any] = {}
    for section in parser.sections():
        target = data
        for part in section.split("."):
            target = target.setdefault(part, {})
        for key, value in parser.items(section):
            target[key] = _parse_scalar(value)
    return data


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
