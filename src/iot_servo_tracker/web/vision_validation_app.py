"""Streamlit webcam validation for the real WeDetect + YOLO vision path."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from iot_servo_tracker.common.config import AppConfig, ServerConfig, load_config
from iot_servo_tracker.common.packets import BBox, CommandPacket, CommandType
from iot_servo_tracker.common.timebase import now_us
from iot_servo_tracker.edge.camera import OpenCvCamera
from iot_servo_tracker.server.vision import WeDetectYoloPipeline, build_wedetect_client


DEFAULT_WEDETECT_MODULE = "iot_servo_tracker.server.wedetect_ref_runtime:detect"


@dataclass(frozen=True)
class VisionValidationOptions:
    query: str = "person"
    frames: int = 12
    camera_index: int = 0
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
    preflight: bool = True


@dataclass(frozen=True)
class VisionValidationFrame:
    frame_index: int
    command: str
    inference_source: str
    bbox: tuple[float, float, float, float] | None
    confidence: float
    track_id: int | None
    yolo_lost_count: int
    yolo_suspect_count: int
    reject_reason: str
    annotated_jpeg: bytes

    @property
    def detected(self) -> bool:
        return self.bbox is not None


def run_webcam_vision_validation(
    config: AppConfig,
    options: VisionValidationOptions,
) -> list[VisionValidationFrame]:
    """Capture webcam frames and run the production vision pipeline."""

    command = CommandPacket.create(CommandType.TRACK, query=options.query)
    pipeline = build_validation_pipeline(config, options)
    camera = OpenCvCamera(
        options.camera_index,
        width=config.camera.width,
        height=config.camera.height,
    )
    records: list[VisionValidationFrame] = []
    try:
        for frame_index in range(max(1, options.frames)):
            frame_bytes = camera.read_jpeg()
            result = pipeline.process_frame(
                ts_req=now_us(),
                query=command.query,
                frame_bytes=frame_bytes,
                frame_index=frame_index,
            )
            source = getattr(pipeline, "last_inference_source", "unknown")
            label = _frame_label(source, result.confidence, result.track_id, result.bbox)
            records.append(
                VisionValidationFrame(
                    frame_index=frame_index,
                    command=command.cmd_type.value,
                    inference_source=source,
                    bbox=_bbox_tuple(result.bbox),
                    confidence=round(result.confidence, 4),
                    track_id=result.track_id,
                    yolo_lost_count=getattr(pipeline, "yolo_lost_count", 0),
                    yolo_suspect_count=getattr(pipeline, "yolo_suspect_count", 0),
                    reject_reason=getattr(pipeline, "last_yolo_reject_reason", ""),
                    annotated_jpeg=_annotate_frame_jpeg(frame_bytes, result.bbox, label),
                )
            )
    finally:
        camera.close()
    return records


def build_validation_pipeline(
    config: AppConfig,
    options: VisionValidationOptions,
) -> WeDetectYoloPipeline:
    server = build_validation_server_config(config.server, options)
    _apply_wedetect_environment(options)
    wedetect_client = build_wedetect_client(server)
    if options.preflight:
        _preflight_wedetect_client(wedetect_client)
    return WeDetectYoloPipeline(
        wedetect_client=wedetect_client,
        yolo_model=server.yolo_model,
        tracker=server.tracker,
        confidence_threshold=server.confidence_threshold,
        yolo_lost_frames=server.yolo_lost_frames,
        yolo_suspect_frames=server.yolo_suspect_frames,
        yolo_max_center_jump_px=server.yolo_max_center_jump_px,
        yolo_max_area_growth_ratio=server.yolo_max_area_growth_ratio,
        yolo_max_frame_area_ratio=server.yolo_max_frame_area_ratio,
        yolo_max_aspect_ratio_change=server.yolo_max_aspect_ratio_change,
        yolo_min_iou_on_id_change=server.yolo_min_iou_on_id_change,
        camera=config.camera,
    )


def build_validation_server_config(
    server: ServerConfig,
    options: VisionValidationOptions,
) -> ServerConfig:
    return replace(
        server,
        wedetect_cache_dir=options.wedetect_cache_dir or server.wedetect_cache_dir,
        wedetect_ref_model_dir=(
            options.wedetect_ref_model_dir or server.wedetect_ref_model_dir
        ),
        wedetect_uni_checkpoint=(
            options.wedetect_uni_checkpoint or server.wedetect_uni_checkpoint
        ),
        wedetect_ref_module=(
            options.wedetect_ref_module
            or server.wedetect_ref_module
            or DEFAULT_WEDETECT_MODULE
        ),
        wedetect_ref_script=options.wedetect_ref_script or server.wedetect_ref_script,
        wedetect_device=options.device or server.wedetect_device,
        yolo_model=options.yolo_model or server.yolo_model,
        tracker=options.tracker or server.tracker,
        confidence_threshold=(
            options.confidence_threshold
            if options.confidence_threshold is not None
            else server.confidence_threshold
        ),
    )


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError("Install optional dependency: pip install '.[web]'") from exc

    st.set_page_config(page_title="Vision Validation", layout="wide")
    st.title("Vision Validation")

    with st.sidebar.form("vision_validation_form"):
        config_path = st.text_input("Config", value="config/settings.toml")
        query = st.text_input("Target", value="person")
        camera_index = st.number_input("Camera index", min_value=0, value=0, step=1)
        frames = st.slider("Frames", min_value=2, max_value=60, value=12, step=1)
        device = st.text_input("Device", value="cuda:0")
        yolo_model = st.text_input("YOLO model", value="")
        tracker = st.text_input("Tracker", value="")
        confidence = st.slider("Confidence", 0.05, 0.90, 0.25, 0.05)
        preflight = st.checkbox("Preflight", value=True)
        with st.expander("WeDetect"):
            wedetect_repo = st.text_input("WEDETECT_REPO", value="")
            wedetect_ref_module = st.text_input(
                "Ref module",
                value=DEFAULT_WEDETECT_MODULE,
            )
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
        submitted = st.form_submit_button("Run")

    if submitted:
        try:
            config = _load_streamlit_config(config_path)
            options = VisionValidationOptions(
                query=query,
                frames=int(frames),
                camera_index=int(camera_index),
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
                preflight=preflight,
            )
            with st.spinner("Running webcam vision validation"):
                records = run_webcam_vision_validation(config, options)
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            return
        st.session_state.vision_validation_records = records
        st.session_state.vision_validation_command = CommandPacket.create(
            CommandType.TRACK,
            query=query,
        ).to_json()

    records = st.session_state.get("vision_validation_records", [])
    if not records:
        st.info("Run validation to capture webcam frames.")
        return

    first_lock = _first_detected(records)
    yolo_hits = sum(
        1 for record in records if record.inference_source == "yolo" and record.detected
    )
    wedetect_hits = sum(
        1 for record in records if record.inference_source == "wedetect" and record.detected
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Frames", len(records))
    col2.metric("First lock", "-" if first_lock is None else first_lock)
    col3.metric("WeDetect hits", wedetect_hits)
    col4.metric("YOLO hits", yolo_hits)

    st.code(st.session_state.get("vision_validation_command", ""), language="json")
    selected = st.slider("Frame", 0, len(records) - 1, len(records) - 1)
    current = records[selected]
    left, right = st.columns([2, 1])
    left.image(current.annotated_jpeg, use_container_width=True)
    right.json(_record_summary(current), expanded=True)
    st.dataframe(_table_rows(records), use_container_width=True, hide_index=True)


def _apply_wedetect_environment(options: VisionValidationOptions) -> None:
    if options.wedetect_repo:
        os.environ["WEDETECT_REPO"] = str(Path(options.wedetect_repo).expanduser())
    os.environ["WEDETECT_ATTN_IMPLEMENTATION"] = options.wedetect_attn
    os.environ["WEDETECT_DTYPE"] = options.wedetect_dtype
    os.environ["WEDETECT_NUM_PROPOSALS"] = str(options.wedetect_num_proposals)
    os.environ["WEDETECT_SCORE_THRE"] = str(options.wedetect_score_threshold)


def _preflight_wedetect_client(client: Any) -> None:
    preflight = getattr(client, "preflight", None)
    if not callable(preflight):
        raise RuntimeError("production WeDetect client does not expose preflight()")
    preflight()


def _load_streamlit_config(path_text: str) -> AppConfig:
    path_text = path_text.strip()
    if not path_text:
        return load_config(None)
    path = Path(path_text)
    return load_config(path if path.exists() else None)


def _first_detected(records: list[VisionValidationFrame]) -> int | None:
    for record in records:
        if record.detected:
            return record.frame_index
    return None


def _table_rows(records: list[VisionValidationFrame]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        row = asdict(record)
        row.pop("annotated_jpeg", None)
        row["detected"] = record.detected
        rows.append(row)
    return rows


def _record_summary(record: VisionValidationFrame) -> dict[str, Any]:
    data = asdict(record)
    data.pop("annotated_jpeg", None)
    data["detected"] = record.detected
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


def _frame_label(
    source: str,
    confidence: float,
    track_id: int | None,
    bbox: BBox | None,
) -> str:
    if bbox is None:
        return f"{source.upper()} no bbox"
    track = "-" if track_id is None else str(track_id)
    return f"{source.upper()} conf={confidence:.2f} id={track}"


def _annotate_frame_jpeg(frame_bytes: bytes, bbox: BBox | None, label: str) -> bytes:
    cv2 = _require_module("cv2")
    np = _require_module("numpy")
    array = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("failed to decode webcam JPEG frame")

    if bbox is None:
        _draw_label(cv2, frame, label, 16, 28, color=(0, 0, 255))
    else:
        height, width = frame.shape[:2]
        x1 = _clamp_int(bbox.x1, 0, width - 1)
        y1 = _clamp_int(bbox.y1, 0, height - 1)
        x2 = _clamp_int(bbox.x2, 0, width - 1)
        y2 = _clamp_int(bbox.y2, 0, height - 1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        _draw_label(cv2, frame, label, x1, max(24, y1 - 8), color=(0, 220, 0))

    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("failed to encode annotated webcam frame")
    return encoded.tobytes()


def _draw_label(cv2, frame, label: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    cv2.putText(
        frame,
        label,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def _clamp_int(value: float, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(round(value))))


def _require_module(name: str):
    try:
        return __import__(name)
    except ImportError as exc:
        raise RuntimeError(f"missing Python dependency: {name}") from exc


if __name__ == "__main__":
    main()
