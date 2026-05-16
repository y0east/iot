"""Run full local validation for the IoT servo tracker project.

This script is intentionally dependency-light. It does not download Hugging Face
models or start hardware loops; it verifies the local safety contracts that should
hold before running on the Raspberry Pi/RTX setup.
"""

from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def main() -> int:
    sys.path.insert(0, str(SRC))
    checks = [
        ("unit tests", run_unit_tests),
        ("no network WeDetect adapter", check_no_network_adapter),
        ("WeDetect-Ref config", check_wedetect_ref_config),
        ("YOLO guard smoke", check_yolo_guard_smoke),
        ("edge safety smoke", check_edge_safety_smoke),
    ]
    failed = False
    for name, check in checks:
        print(f"[validate] {name} ...", flush=True)
        try:
            check()
        except Exception as exc:
            failed = True
            print(f"[validate] FAIL {name}: {exc}", flush=True)
        else:
            print(f"[validate] OK {name}", flush=True)
    return 1 if failed else 0


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
    similar_result = similar.process_frame(10, "red cup", b"jpeg")
    if similar_result.bbox is not None:
        raise RuntimeError("similar-object candidate reached servo path")

    large = build([([100, 80, 600, 460], 0.99, 7)])
    large_result = large.process_frame(20, "red cup", b"jpeg")
    if large_result.bbox is not None:
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
        SensorSample.empty(),
    )
    if runtime.state == SystemState.TRACKING:
        raise RuntimeError("safe hold recovered to a large absorbed bbox")


if __name__ == "__main__":
    raise SystemExit(main())
