"""Run local and RTX/webcam validation for the IoT servo tracker project.

Default mode stays dependency-light and validates the control/safety contracts.
Use ``--rtx-webcam`` on the RTX3060 laptop to run the real webcam + WeDetect-Ref
+ YOLO path. The real mode downloads Hugging Face artifacts when local paths are
not supplied and imports the official WeChatCV/WeDetect repository.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_OUTPUT_DIR = ROOT / "validation_outputs"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sys.path.insert(0, str(SRC))

    checks: list[tuple[str, Callable[[], None]]] = []
    if not args.skip_quick:
        checks.extend(
            [
                ("unit tests", run_unit_tests),
                ("no network WeDetect adapter", check_no_network_adapter),
                ("WeDetect-Ref config", check_wedetect_ref_config),
                ("YOLO guard smoke", check_yolo_guard_smoke),
                ("edge safety smoke", check_edge_safety_smoke),
            ]
        )
    if args.rtx_webcam:
        checks.extend(
            [
                ("RTX3060 CUDA environment", lambda: check_rtx_environment(args)),
                ("webcam capture", lambda: check_webcam_capture(args)),
                ("Hugging Face WeDetect artifacts", lambda: check_wedetect_artifacts(args)),
                ("real WeDetect-Ref on webcam", lambda: check_real_wedetect_ref(args)),
                ("real YOLO webcam tracking", lambda: check_real_yolo(args)),
                ("real WeDetect-Ref + YOLO pipeline", lambda: check_real_pipeline(args)),
            ]
        )

    failed = False
    for name, check in checks:
        print(f"[validate] {name} ...", flush=True)
        try:
            check()
        except Exception as exc:  # noqa: BLE001
            failed = True
            print(f"[validate] FAIL {name}: {exc}", flush=True)
        else:
            print(f"[validate] OK {name}", flush=True)
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate IoT servo tracker")
    parser.add_argument(
        "--rtx-webcam",
        action="store_true",
        help="Run real RTX3060 + webcam + WeDetect-Ref + YOLO validation",
    )
    parser.add_argument("--skip-quick", action="store_true", help="Skip unit/smoke checks")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--query", default="person")
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--yolo-model", default="yolo26n.pt")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--wedetect-repo", default=os.environ.get("WEDETECT_REPO", ""))
    parser.add_argument("--wedetect-ref-model-dir", default="")
    parser.add_argument("--wedetect-uni-checkpoint", default="")
    parser.add_argument("--wedetect-cache-dir", default="")
    parser.add_argument("--wedetect-ref-repo-id", default="fushh7/WeDetect-Ref-2B")
    parser.add_argument("--wedetect-uni-repo-id", default="fushh7/WeDetect")
    parser.add_argument("--wedetect-uni-filename", default="wedetect_base_uni.pth")
    parser.add_argument("--wedetect-attn", default="sdpa")
    parser.add_argument("--wedetect-dtype", default="auto")
    parser.add_argument("--wedetect-num-proposals", type=int, default=100)
    parser.add_argument("--wedetect-score-thre", type=float, default=-1.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--allow-non-3060",
        action="store_true",
        help="Allow CUDA GPUs whose name does not contain 3060",
    )
    return parser


def run_unit_tests() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        cwd=ROOT,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"unittest failed with exit code {completed.returncode}")


def check_no_network_adapter() -> None:
    forbidden = [
        "Http" + "WeDetect",
        "wedetect_" + "endpoint",
        "wedetect_" + "mode",
        "url" + "lib" + ".request",
    ]
    scanned_suffixes = {".py", ".toml", ".md"}
    offenders: list[str] = []
    for path in ROOT.rglob("*"):
        if "__pycache__" in path.parts or path.suffix not in scanned_suffixes:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in forbidden:
            if token in text:
                offenders.append(f"{path.relative_to(ROOT)} contains {token}")
    if offenders:
        raise RuntimeError("; ".join(offenders))


def check_wedetect_ref_config() -> None:
    from iot_servo_tracker.common.config import AppConfig
    from iot_servo_tracker.server.vision import (
        HuggingFaceWeDetectRefClient,
        build_wedetect_client,
    )

    config = AppConfig()
    client = build_wedetect_client(config.server)
    if not isinstance(client, HuggingFaceWeDetectRefClient):
        raise RuntimeError(f"unexpected WeDetect client: {type(client).__name__}")
    if client.ref_repo_id != "fushh7/WeDetect-Ref-2B":
        raise RuntimeError(f"unexpected ref repo: {client.ref_repo_id}")
    if client.uni_repo_id != "fushh7/WeDetect":
        raise RuntimeError(f"unexpected uni repo: {client.uni_repo_id}")


def check_yolo_guard_smoke() -> None:
    from iot_servo_tracker.common.config import AppConfig
    from iot_servo_tracker.common.packets import BBox, TrackingResult
    from iot_servo_tracker.server.vision import WeDetectYoloPipeline

    class TensorList:
        def __init__(self, data):
            self.data = data

        def cpu(self):
            return self

        def tolist(self):
            return self.data

    class TensorItems:
        def __init__(self, data):
            self.data = data

        def __getitem__(self, index):
            return TensorScalar(self.data[index])

    class TensorScalar:
        def __init__(self, value):
            self.value = value

        def cpu(self):
            return self

        def item(self):
            return self.value

    class Boxes:
        def __init__(self, candidates):
            self.xyxy = TensorList([candidate[0] for candidate in candidates])
            self.conf = TensorItems([candidate[1] for candidate in candidates])
            self.id = TensorItems([candidate[2] for candidate in candidates])

    class Result:
        def __init__(self, candidates):
            self.boxes = Boxes(candidates)

    class Yolo:
        def __init__(self, candidates):
            self.candidates = candidates

        def track(self, *args, **kwargs):
            return [Result(self.candidates)]

    class EmptyRef:
        def __init__(self):
            self.calls = 0

        def detect(self, frame_bytes, query, ts_req):
            self.calls += 1
            return TrackingResult.empty(ts_req=ts_req, query=query)

    def build(candidates):
        pipe = object.__new__(WeDetectYoloPipeline)
        pipe.yolo = Yolo(candidates)
        pipe.tracker = "bytetrack.yaml"
        pipe.confidence_threshold = 0.25
        pipe.yolo_lost_frames = 2
        pipe.yolo_suspect_frames = 1
        pipe.yolo_max_center_jump_px = 120.0
        pipe.yolo_max_area_growth_ratio = 4.0
        pipe.yolo_max_frame_area_ratio = 0.35
        pipe.yolo_max_aspect_ratio_change = 3.0
        pipe.yolo_min_iou_on_id_change = 0.10
        pipe.camera = AppConfig().camera
        pipe.wedetect_client = EmptyRef()
        pipe.locked_bbox = BBox(300, 220, 340, 260)
        pipe.locked_track_id = 7
        pipe.redetect_reference_bbox = None
        pipe.yolo_lost_count = 0
        pipe.yolo_suspect_count = 0
        pipe.last_yolo_reject_reason = ""
        pipe._decode_frame = lambda frame_bytes: object()
        return pipe

    similar = build([([520, 220, 560, 260], 0.99, 8)])
    if similar.process_frame(10, "red cup", b"jpeg").bbox is not None:
        raise RuntimeError("similar-object candidate reached servo path")

    large = build([([100, 80, 600, 460], 0.99, 7)])
    if large.process_frame(20, "red cup", b"jpeg").bbox is not None:
        raise RuntimeError("large background bbox reached servo path")


def check_edge_safety_smoke() -> None:
    from iot_servo_tracker.common.config import AppConfig, SafetyConfig
    from iot_servo_tracker.common.packets import BBox, SensorSample, TrackingResult
    from iot_servo_tracker.common.timebase import now_us
    from iot_servo_tracker.control.states import SystemState
    from iot_servo_tracker.edge.runtime import EdgeRuntime

    config = AppConfig(
        safety=SafetyConfig(
            bbox_frame_area_threshold=0.20,
            safe_hold_rescan_delay_s=10.0,
        )
    )
    runtime = EdgeRuntime(config)
    runtime.state_machine.state = SystemState.SAFE_HOLD
    runtime.safe_hold_started_us = now_us()
    runtime.last_valid_result = TrackingResult(
        packet="tracking_result",
        ts_req=1,
        ts_resp=2,
        bbox=BBox(300, 220, 340, 260),
        confidence=0.9,
        track_id=7,
        query="red cup",
    )
    runtime.handle_tracking_result(
        TrackingResult(
            packet="tracking_result",
            ts_req=now_us(),
            ts_resp=now_us(),
            bbox=BBox(100, 80, 600, 460),
            confidence=0.99,
            track_id=7,
            query="red cup",
        ),
        SensorSample(ts=now_us(), tof_mm=620.0, ultrasonic_mm=650.0),
    )
    if runtime.state == SystemState.TRACKING:
        raise RuntimeError("safe hold recovered to a large absorbed bbox")


def check_rtx_environment(args: argparse.Namespace) -> None:
    torch = require_module("torch")
    torchvision = require_module("torchvision")
    require_module("cv2")
    require_module("numpy")
    ultralytics = require_module("ultralytics")
    require_module("huggingface_hub")
    transformers = require_module("transformers")
    accelerate = require_module("accelerate")
    require_module("trl")
    require_module("tqdm")
    require_module("requests")
    require_module("packaging")
    require_module("PIL")

    require_min_version("torch", getattr(torch, "__version__", "0"), (2, 5, 1))
    require_min_version("torchvision", getattr(torchvision, "__version__", "0"), (0, 20, 1))
    require_min_version("ultralytics", getattr(ultralytics, "__version__", "0"), (8, 4, 0))
    require_min_version("transformers", getattr(transformers, "__version__", "0"), (4, 57, 1))
    require_min_version("accelerate", getattr(accelerate, "__version__", "0"), (1, 10, 0))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run this on the RTX3060 laptop")
    device_index = int(args.device.split(":", 1)[1]) if ":" in args.device else 0
    name = torch.cuda.get_device_name(device_index)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    print(
        "[validate] CUDA "
        f"{name}, free={free_bytes / 1024**3:.2f}GiB, total={total_bytes / 1024**3:.2f}GiB",
        flush=True,
    )
    if "3060" not in name and not args.allow_non_3060:
        raise RuntimeError(f"expected an RTX3060-class GPU, got {name!r}")
    if total_bytes < 5.5 * 1024**3:
        raise RuntimeError("GPU memory is below 5.5GiB; WeDetect-Ref-2B may not fit")


def check_webcam_capture(args: argparse.Namespace) -> None:
    frame_bytes = capture_webcam_jpeg(args)
    output_dir = ensure_output_dir(args)
    frame_path = output_dir / "webcam_frame.jpg"
    frame_path.write_bytes(frame_bytes)
    print(f"[validate] saved webcam frame: {frame_path}", flush=True)


def check_wedetect_artifacts(args: argparse.Namespace) -> None:
    client = build_real_wedetect_client(args)
    ref_model_dir, uni_checkpoint = client.resolve_artifacts()
    print(f"[validate] WeDetect-Ref: {ref_model_dir}", flush=True)
    print(f"[validate] WeDetect-Uni: {uni_checkpoint}", flush=True)
    if not Path(ref_model_dir).exists():
        raise RuntimeError(f"WeDetect-Ref directory does not exist: {ref_model_dir}")
    if not Path(uni_checkpoint).exists():
        raise RuntimeError(f"WeDetect-Uni checkpoint does not exist: {uni_checkpoint}")


def check_real_wedetect_ref(args: argparse.Namespace) -> None:
    frame_bytes = capture_webcam_jpeg(args)
    client = build_real_wedetect_client(args)
    result = client.detect(frame_bytes, args.query, ts_req=1)
    output_dir = ensure_output_dir(args)
    (output_dir / "wedetect_ref_result.json").write_text(
        result.to_json(),
        encoding="utf-8",
    )
    if result.bbox is None:
        raise RuntimeError(
            "WeDetect-Ref returned no bbox. Use a visible query such as --query person "
            "or lower --wedetect-score-thre."
        )
    print(
        f"[validate] WeDetect bbox={result.bbox} confidence={result.confidence:.4f}",
        flush=True,
    )


def check_real_yolo(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    frames = capture_webcam_frames(args, max(1, args.frames))
    model = YOLO(args.yolo_model)
    detections = 0
    for frame_bytes in frames:
        frame = decode_jpeg(frame_bytes)
        results = model.track(
            frame,
            persist=True,
            tracker=args.tracker,
            conf=args.confidence_threshold,
            verbose=False,
        )
        if results and getattr(results[0], "boxes", None) is not None:
            boxes = results[0].boxes
            if boxes.xyxy is not None and len(boxes.xyxy) > 0:
                detections += 1
    if detections == 0:
        raise RuntimeError(
            "YOLO produced no webcam detections. Point the camera at a COCO object, "
            "or use a custom --yolo-model trained for the target."
        )
    print(f"[validate] YOLO detections on {detections}/{len(frames)} frames", flush=True)


def check_real_pipeline(args: argparse.Namespace) -> None:
    from iot_servo_tracker.common.config import AppConfig
    from iot_servo_tracker.common.timebase import now_us
    from iot_servo_tracker.server.vision import WeDetectYoloPipeline

    frames = capture_webcam_frames(args, max(2, args.frames))
    client = build_real_wedetect_client(args)
    pipeline = WeDetectYoloPipeline(
        wedetect_client=client,
        yolo_model=args.yolo_model,
        tracker=args.tracker,
        confidence_threshold=args.confidence_threshold,
        camera=AppConfig().camera,
    )
    results = []
    for frame_bytes in frames:
        result = pipeline.process_frame(now_us(), args.query, frame_bytes=frame_bytes)
        results.append(json.loads(result.to_json()))
    output_dir = ensure_output_dir(args)
    (output_dir / "production_pipeline_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if results[0]["bbox"] is None:
        raise RuntimeError("initial WeDetect lock did not return a bbox")
    yolo_hits = sum(1 for result in results[1:] if result["bbox"] is not None)
    if yolo_hits == 0:
        raise RuntimeError(
            "WeDetect locked the first frame, but YOLO did not keep a bbox on "
            "following webcam frames. Use --query person for a first RTX3060 smoke "
            "test or provide a target-specific YOLO model."
        )
    print(f"[validate] production pipeline YOLO hits={yolo_hits}", flush=True)


def build_real_wedetect_client(args: argparse.Namespace):
    from iot_servo_tracker.common.config import AppConfig
    from iot_servo_tracker.server.vision import build_wedetect_client

    if args.wedetect_repo:
        repo = Path(args.wedetect_repo).expanduser().resolve()
        if not (repo / "infer_wedetect_ref.py").exists():
            raise RuntimeError(
                f"--wedetect-repo must point to WeChatCV/WeDetect, got {repo}"
            )
        os.environ["WEDETECT_REPO"] = str(repo)
    elif not os.environ.get("WEDETECT_REPO"):
        raise RuntimeError(
            "Set --wedetect-repo to a cloned WeChatCV/WeDetect repository. "
            "Example: git clone https://github.com/WeChatCV/WeDetect external/WeDetect"
        )
    os.environ["WEDETECT_ATTN_IMPLEMENTATION"] = args.wedetect_attn
    os.environ["WEDETECT_DTYPE"] = args.wedetect_dtype
    os.environ["WEDETECT_NUM_PROPOSALS"] = str(args.wedetect_num_proposals)
    os.environ["WEDETECT_SCORE_THRE"] = str(args.wedetect_score_thre)

    config = AppConfig()
    server = replace(
        config.server,
        wedetect_ref_repo_id=args.wedetect_ref_repo_id,
        wedetect_uni_repo_id=args.wedetect_uni_repo_id,
        wedetect_uni_filename=args.wedetect_uni_filename,
        wedetect_cache_dir=args.wedetect_cache_dir,
        wedetect_ref_model_dir=args.wedetect_ref_model_dir,
        wedetect_uni_checkpoint=args.wedetect_uni_checkpoint,
        wedetect_ref_module="iot_servo_tracker.server.wedetect_ref_runtime:detect",
        wedetect_ref_script="",
        wedetect_device=args.device,
    )
    return build_wedetect_client(server)


def capture_webcam_jpeg(args: argparse.Namespace) -> bytes:
    return capture_webcam_frames(args, 1)[0]


def capture_webcam_frames(args: argparse.Namespace, count: int) -> list[bytes]:
    cv2 = require_module("cv2")
    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"failed to open webcam index {args.camera_index}")
    frames: list[bytes] = []
    try:
        for _ in range(count):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("failed to read webcam frame")
            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                raise RuntimeError("failed to encode webcam frame")
            frames.append(encoded.tobytes())
    finally:
        capture.release()
    return frames


def decode_jpeg(frame_bytes: bytes):
    cv2 = require_module("cv2")
    np = require_module("numpy")
    array = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("failed to decode captured JPEG frame")
    return frame


def ensure_output_dir(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args.output_dir


def require_module(name: str):
    try:
        return __import__(name)
    except ImportError as exc:
        raise RuntimeError(f"missing Python dependency: {name}") from exc


def require_min_version(name: str, installed: str, minimum: tuple[int, int, int]) -> None:
    actual = parse_version_tuple(installed)
    if actual < minimum:
        minimum_text = ".".join(str(part) for part in minimum)
        raise RuntimeError(
            f"{name} {installed} is too old; install {name}>={minimum_text}"
        )


def parse_version_tuple(version_text: str) -> tuple[int, int, int]:
    normalized = version_text.split("+", 1)[0]
    parts: list[int] = []
    for raw_part in normalized.split("."):
        digits = "".join(character for character in raw_part if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
        if len(parts) == 3:
            break
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


if __name__ == "__main__":
    raise SystemExit(main())
