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
    pixel_jump_threshold: float = 180.0
    tof_delta_threshold_mm: float = 40.0
    ultrasonic_jump_threshold_mm: float = 120.0
    bbox_area_growth_threshold: float = 16.0
    bbox_frame_area_threshold: float = 0.85
    bbox_aspect_ratio_change_threshold: float = 8.0
    consecutive_frames: int = 45
    recovery_confirm_frames: int = 5
    default_ping_threshold_ms: float = 250.0
    timeout_min_s: float = 3.5
    timeout_max_s: float = 5.0
    safe_hold_rescan_delay_s: float = 2.5
    limited_rescan_range_deg: float = 10.0
    central_region_ratio: float = 0.60


@dataclass(frozen=True)
class ScanConfig:
    range_deg: float = 45.0
    speed_deg_s: float = 10.0
    confidence_threshold: float = 0.50
    confirmation_frames: int = 3
    passes: int = 2
    max_center_distance_ratio: float = 0.45
    min_box_area_ratio: float = 0.002


@dataclass(frozen=True)
class MqttConfig:
    host: str = "127.0.0.1"
    port: int = 1883
    command_topic: str = "iot_servo_tracker/command"
    status_topic: str = "iot_servo_tracker/status"


@dataclass(frozen=True)
class ZmqConfig:
    frame_bind_endpoint: str = "tcp://0.0.0.0:5555"
    result_bind_endpoint: str = "tcp://0.0.0.0:5556"
    frame_connect_endpoint: str = "tcp://127.0.0.1:5555"
    result_connect_endpoint: str = "tcp://127.0.0.1:5556"
    frame_snd_hwm: int = 1
    frame_rcv_hwm: int = 1
    result_snd_hwm: int = 1
    result_rcv_hwm: int = 1


@dataclass(frozen=True)
class WebConfig:
    processed_stream_url: str = ""


@dataclass(frozen=True)
class ServerConfig:
    wedetect_ref_repo_id: str = "fushh7/WeDetect-Ref-2B"
    wedetect_uni_repo_id: str = "fushh7/WeDetect"
    wedetect_uni_filename: str = "wedetect_base_uni.pth"
    wedetect_cache_dir: str = ""
    wedetect_ref_model_dir: str = ""
    wedetect_uni_checkpoint: str = ""
    wedetect_ref_module: str = ""
    wedetect_ref_script: str = ""
    wedetect_device: str = "cuda:0"
    yolo_lost_frames: int = 30
    yolo_suspect_frames: int = 5
    yolo_max_center_jump_px: float = 160.0
    yolo_max_area_growth_ratio: float = 16.0
    yolo_max_frame_area_ratio: float = 0.85
    yolo_max_aspect_ratio_change: float = 8.0
    yolo_min_iou_on_id_change: float = 0.01
    yolo_model: str = "yolo26n.pt"
    tracker: str = "botsort.yaml"
    confidence_threshold: float = 0.25
    wedetect_confidence_threshold: float = 0.45


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)
    mqtt: MqttConfig = field(default_factory=MqttConfig)
    zmq: ZmqConfig = field(default_factory=ZmqConfig)
    web: WebConfig = field(default_factory=WebConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


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

    default_zmq = ZmqConfig()
    zmq_raw = raw.get("zmq", {})
    zmq = ZmqConfig(
        frame_bind_endpoint=zmq_raw.get(
            "frame_bind_endpoint",
            zmq_raw.get("frame_endpoint", default_zmq.frame_bind_endpoint),
        ),
        result_bind_endpoint=zmq_raw.get(
            "result_bind_endpoint",
            zmq_raw.get("result_endpoint", default_zmq.result_bind_endpoint),
        ),
        frame_connect_endpoint=zmq_raw.get(
            "frame_connect_endpoint",
            zmq_raw.get("frame_endpoint", default_zmq.frame_connect_endpoint),
        ),
        result_connect_endpoint=zmq_raw.get(
            "result_connect_endpoint",
            zmq_raw.get("result_endpoint", default_zmq.result_connect_endpoint),
        ),
        frame_snd_hwm=zmq_raw.get("frame_snd_hwm", default_zmq.frame_snd_hwm),
        frame_rcv_hwm=zmq_raw.get("frame_rcv_hwm", default_zmq.frame_rcv_hwm),
        result_snd_hwm=zmq_raw.get("result_snd_hwm", default_zmq.result_snd_hwm),
        result_rcv_hwm=zmq_raw.get("result_rcv_hwm", default_zmq.result_rcv_hwm),
    )

    return AppConfig(
        camera=camera,
        control=control,
        safety=SafetyConfig(**raw.get("safety", {})),
        scan=ScanConfig(**raw.get("scan", {})),
        mqtt=MqttConfig(**raw.get("mqtt", {})),
        zmq=zmq,
        web=WebConfig(**raw.get("web", {})),
        server=ServerConfig(**raw.get("server", {})),
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
