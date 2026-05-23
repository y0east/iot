"""Streamlit live validation for webcam, web commands, comms, edge, and vision."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from iot_servo_tracker.common.config import AppConfig, load_config
from iot_servo_tracker.common.packets import BBox, SensorSample, TrackingResult
from iot_servo_tracker.common.timebase import now_us
from iot_servo_tracker.comms.mqtt import InMemoryMqttBus, MqttTopics
from iot_servo_tracker.control.servo import SimulatedServoDriver
from iot_servo_tracker.edge.camera import OpenCvCamera
from iot_servo_tracker.edge.runtime import EdgeRuntime
from iot_servo_tracker.server.main import process_frame_safely
from iot_servo_tracker.server.runtime import VisionRuntime
from iot_servo_tracker.sim.full_stack import (
    InMemoryZmqBus,
    InMemoryZmqEdgeTransport,
    InMemoryZmqVisionTransport,
    SimulatedMqttEdgeBridge,
    SimulatedWebClient,
)
from iot_servo_tracker.sim.offline import ScriptedVisionPipeline
from iot_servo_tracker.web.vision_validation_app import (
    DEFAULT_WEDETECT_MODULE,
    VisionValidationOptions,
    _annotate_frame_jpeg,
    build_validation_pipeline,
)


@dataclass(frozen=True)
class LiveStackOptions:
    query: str = "person"
    frames: int = 120
    camera_index: int = 0
    dt_s: float = 0.033
    sleep_s: float = 0.03
    approximate_tof_mm: float = 620.0
    approximate_ultrasonic_mm: float = 650.0
    production: bool = False
    preflight: bool = True
    device: str = "cuda:0"
    yolo_model: str = ""
    tracker: str = ""
    confidence_threshold: float | None = None
    wedetect_repo: str = ""
    wedetect_ref_module: str = DEFAULT_WEDETECT_MODULE
    wedetect_ref_script: str = ""
    wedetect_ref_model_dir: str = ""
    wedetect_uni_checkpoint: str = ""
    wedetect_cache_dir: str = ""
    wedetect_attn: str = "sdpa"
    wedetect_dtype: str = "auto"
    wedetect_num_proposals: int = 100
    wedetect_score_threshold: float = -1.0


@dataclass(frozen=True)
class LiveStackFrame:
    frame_index: int
    web_view: str
    web_target: str
    edge_state: str
    query: str
    redetect: bool
    vision_mode: str
    inference_source: str
    frame_sent: bool
    vision_processed: bool
    bbox: tuple[float, float, float, float] | None
    confidence: float
    track_id: int | None
    pan_deg: float
    tilt_deg: float
    rtt_ms: float
    tof_mm: float | None
    ultrasonic_mm: float | None
    mqtt_commands: int
    mqtt_statuses: int
    message: str
    annotated_jpeg: bytes

    @property
    def detected(self) -> bool:
        return self.bbox is not None


FrameCallback = Callable[[LiveStackFrame], None]


def run_live_stack_validation(
    config: AppConfig,
    options: LiveStackOptions,
    on_frame: FrameCallback | None = None,
) -> list[LiveStackFrame]:
    """Run the complete local web-to-vision-to-web loop on webcam frames."""

    mqtt_bus = InMemoryMqttBus()
    topics = MqttTopics.from_config(config.mqtt)
    edge = EdgeRuntime(config=config, servo=SimulatedServoDriver())
    bridge = SimulatedMqttEdgeBridge(mqtt_bus, topics, edge)
    web = SimulatedWebClient(mqtt_bus, topics)
    zmq_bus = InMemoryZmqBus()
    edge_transport = InMemoryZmqEdgeTransport(zmq_bus, config.zmq.frame_snd_hwm)
    vision_transport = InMemoryZmqVisionTransport(zmq_bus, config.zmq.frame_rcv_hwm)
    vision = VisionRuntime(config, pipeline=_build_live_vision_pipeline(config, options))
    camera = OpenCvCamera(
        options.camera_index,
        width=config.camera.width,
        height=config.camera.height,
    )

    frames: list[LiveStackFrame] = []
    frame_index = 0
    web.start_tracking(
        options.query,
        scan_range_deg=config.scan.range_deg,
        max_speed_deg_s=config.control.max_speed_deg_s,
    )

    try:
        for _ in range(max(1, options.frames)):
            raw_frame = camera.read_jpeg()
            ts_req = now_us()
            edge.capture_frame(raw_frame, ts_us=ts_req)
            query, redetect = edge.next_frame_request()

            frame_sent = False
            if query:
                frame_sent = edge_transport.send_frame(
                    ts_req=ts_req,
                    query=query,
                    frame_bytes=raw_frame,
                    frame_index=frame_index,
                    redetect=redetect,
                )
                if frame_sent:
                    frame_index += 1

            vision_processed = _process_one_live_vision_frame(vision_transport, vision)
            result = edge_transport.recv_result()
            sensor = _approximate_sensor_sample(options)
            if result is not None:
                status = edge.handle_tracking_result(
                    result,
                    sensor,
                    dt_s=options.dt_s,
                    received_ts_us=now_us(),
                )
            else:
                status = edge.control_step(dt_s=options.dt_s, sensor_sample=sensor)
            bridge.publish_status(status)

            source = _inference_source(vision, options.production)
            bbox = result.bbox if result is not None else None
            confidence = result.confidence if result is not None else status.confidence
            label = _live_frame_label(web.view_state, status.system_state, source, result)
            record = LiveStackFrame(
                frame_index=len(frames),
                web_view=web.view_state,
                web_target=web.target,
                edge_state=status.system_state,
                query=query or web.target,
                redetect=redetect,
                vision_mode="production" if options.production else "scripted",
                inference_source=source,
                frame_sent=frame_sent,
                vision_processed=vision_processed,
                bbox=_bbox_tuple(bbox),
                confidence=round(confidence, 4),
                track_id=result.track_id if result is not None else None,
                pan_deg=round(status.pan_deg, 3),
                tilt_deg=round(status.tilt_deg, 3),
                rtt_ms=round(status.rtt_ms, 3),
                tof_mm=sensor.tof_mm,
                ultrasonic_mm=sensor.ultrasonic_mm,
                mqtt_commands=len(web.commands),
                mqtt_statuses=len(web.statuses),
                message=status.message,
                annotated_jpeg=_annotate_frame_jpeg(raw_frame, bbox, label),
            )
            frames.append(record)
            if on_frame is not None:
                on_frame(record)
            if options.sleep_s > 0:
                time.sleep(options.sleep_s)
    finally:
        camera.close()
    return frames


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError("Install optional dependency: pip install '.[web,server]'") from exc

    st.set_page_config(page_title="Live Stack Validation", layout="wide")
    st.title("Live Stack Validation")
    st.caption(
        "Web command, in-memory MQTT, edge runtime, in-memory ZMQ, "
        "vision, status, and webcam overlay"
    )

    with st.sidebar.form("live_stack_form"):
        config_path = st.text_input("Config", value="config/settings.toml")
        query = st.text_input("Target", value="person")
        camera_index = st.number_input("Camera index", min_value=0, value=0, step=1)
        frames = st.slider("Frames", min_value=5, max_value=600, value=120, step=5)
        sleep_s = st.slider("Frame delay", 0.0, 0.20, 0.03, 0.01)
        tof_mm = st.number_input("Approx ToF mm", min_value=1.0, value=620.0, step=10.0)
        ultrasonic_mm = st.number_input(
            "Approx ultrasonic mm",
            min_value=1.0,
            value=650.0,
            step=10.0,
        )
        production = st.checkbox("Real WeDetect + YOLO")
        preflight = st.checkbox("Preflight", value=True)
        device = st.text_input("Device", value="cuda:0")
        yolo_model = st.text_input("YOLO model", value="")
        tracker = st.text_input("Tracker", value="")
        confidence = st.slider("Confidence", 0.05, 0.90, 0.25, 0.05)
        with st.expander("WeDetect"):
            wedetect_repo = st.text_input("WEDETECT_REPO", value="")
            wedetect_ref_module = st.text_input("Ref module", value=DEFAULT_WEDETECT_MODULE)
            wedetect_ref_script = st.text_input("Ref script", value="")
            wedetect_ref_model_dir = st.text_input("Ref model dir", value="")
            wedetect_uni_checkpoint = st.text_input("Uni checkpoint", value="")
            wedetect_cache_dir = st.text_input("Cache dir", value="")
            wedetect_attn = st.text_input("Attention", value="sdpa")
            wedetect_dtype = st.selectbox(
                "DType",
                ("auto", "float16", "bfloat16", "float32"),
            )
            wedetect_num_proposals = st.number_input(
                "Proposals",
                min_value=1,
                max_value=500,
                value=100,
                step=10,
            )
            wedetect_score_threshold = st.number_input(
                "Score threshold",
                value=-1.0,
                step=0.05,
            )
        submitted = st.form_submit_button("Run live stack")

    if not submitted:
        records = st.session_state.get("live_stack_records", [])
        if records:
            _render_records(st, records)
        else:
            st.info("Run live stack to start webcam capture and communication validation.")
        return

    try:
        config = _load_streamlit_config(config_path)
        options = LiveStackOptions(
            query=query,
            frames=int(frames),
            camera_index=int(camera_index),
            sleep_s=float(sleep_s),
            approximate_tof_mm=float(tof_mm),
            approximate_ultrasonic_mm=float(ultrasonic_mm),
            production=production,
            preflight=preflight,
            device=device,
            yolo_model=yolo_model,
            tracker=tracker,
            confidence_threshold=float(confidence),
            wedetect_repo=wedetect_repo,
            wedetect_ref_module=wedetect_ref_module,
            wedetect_ref_script=wedetect_ref_script,
            wedetect_ref_model_dir=wedetect_ref_model_dir,
            wedetect_uni_checkpoint=wedetect_uni_checkpoint,
            wedetect_cache_dir=wedetect_cache_dir,
            wedetect_attn=wedetect_attn,
            wedetect_dtype=wedetect_dtype,
            wedetect_num_proposals=int(wedetect_num_proposals),
            wedetect_score_threshold=float(wedetect_score_threshold),
        )
        live_slot = st.empty()
        metric_slot = st.empty()
        table_slot = st.empty()
        records: list[LiveStackFrame] = []

        def update(record: LiveStackFrame) -> None:
            records.append(record)
            _render_live_frame(st, live_slot, metric_slot, table_slot, records)

        with st.spinner("Running full local stack"):
            run_live_stack_validation(config, options, on_frame=update)
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        return

    st.session_state.live_stack_records = records


def _build_live_vision_pipeline(config: AppConfig, options: LiveStackOptions):
    if not options.production:
        return ScriptedVisionPipeline(config.camera)
    validation_options = VisionValidationOptions(
        query=options.query,
        frames=options.frames,
        camera_index=options.camera_index,
        device=options.device,
        yolo_model=options.yolo_model,
        tracker=options.tracker,
        confidence_threshold=options.confidence_threshold,
        wedetect_repo=options.wedetect_repo,
        wedetect_ref_module=options.wedetect_ref_module,
        wedetect_ref_script=options.wedetect_ref_script,
        wedetect_ref_model_dir=options.wedetect_ref_model_dir,
        wedetect_uni_checkpoint=options.wedetect_uni_checkpoint,
        wedetect_cache_dir=options.wedetect_cache_dir,
        wedetect_attn=options.wedetect_attn,
        wedetect_dtype=options.wedetect_dtype,
        wedetect_num_proposals=options.wedetect_num_proposals,
        wedetect_score_threshold=options.wedetect_score_threshold,
        preflight=options.preflight,
    )
    return build_validation_pipeline(config, validation_options)


def _process_one_live_vision_frame(
    transport: InMemoryZmqVisionTransport,
    runtime: VisionRuntime,
) -> bool:
    frame = transport.recv_frame()
    if frame is None:
        return False
    header, payload = frame
    result = process_frame_safely(
        runtime,
        ts_req=int(header["ts_req"]),
        query=str(header.get("query", "")),
        frame_bytes=payload,
        redetect=bool(header.get("redetect", False)),
    )
    return transport.send_result(result)


def _approximate_sensor_sample(options: LiveStackOptions) -> SensorSample:
    return SensorSample(
        ts=now_us(),
        tof_mm=options.approximate_tof_mm,
        ultrasonic_mm=options.approximate_ultrasonic_mm,
        limit_switch_active=False,
    )


def _inference_source(runtime: VisionRuntime, production: bool) -> str:
    pipeline = runtime.pipeline
    if pipeline is None:
        return "none"
    fallback = "scripted" if not production else "unknown"
    return str(getattr(pipeline, "last_inference_source", fallback))


def _live_frame_label(
    web_view: str,
    edge_state: str,
    source: str,
    result: TrackingResult | None,
) -> str:
    if result is None or result.bbox is None:
        return f"WEB={web_view} EDGE={edge_state} {source.upper()} no bbox"
    track = "-" if result.track_id is None else str(result.track_id)
    return (
        f"WEB={web_view} EDGE={edge_state} "
        f"{source.upper()} conf={result.confidence:.2f} id={track}"
    )


def _render_live_frame(
    st,
    image_slot,
    metric_slot,
    table_slot,
    records: list[LiveStackFrame],
) -> None:
    latest = records[-1]
    image_slot.image(latest.annotated_jpeg, use_container_width=True)
    with metric_slot.container():
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Web", latest.web_view)
        col2.metric("Edge", latest.edge_state)
        col3.metric("Vision", latest.inference_source)
        col4.metric("Confidence", f"{latest.confidence:.2f}")
        col5.metric("Pan/Tilt", f"{latest.pan_deg:.1f}/{latest.tilt_deg:.1f}")
    table_slot.dataframe(
        _table_rows(records[-40:]),
        use_container_width=True,
        hide_index=True,
    )


def _render_records(st, records: list[LiveStackFrame]) -> None:
    latest = records[-1]
    left, right = st.columns([2, 1])
    left.image(latest.annotated_jpeg, use_container_width=True)
    right.json(_record_summary(latest), expanded=True)
    st.dataframe(_table_rows(records), use_container_width=True, hide_index=True)


def _load_streamlit_config(path_text: str) -> AppConfig:
    path_text = path_text.strip()
    if not path_text:
        return load_config(None)
    path = Path(path_text)
    return load_config(path if path.exists() else None)


def _table_rows(records: list[LiveStackFrame]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        row = _record_summary(record)
        row["detected"] = record.detected
        rows.append(row)
    return rows


def _record_summary(record: LiveStackFrame) -> dict[str, Any]:
    data = asdict(record)
    data.pop("annotated_jpeg", None)
    return data


def _bbox_tuple(bbox: BBox | None) -> tuple[float, float, float, float] | None:
    if bbox is None:
        return None
    return (
        round(bbox.x1, 2),
        round(bbox.y1, 2),
        round(bbox.x2, 2),
        round(bbox.y2, 2),
    )


if __name__ == "__main__":
    main()
