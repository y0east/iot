"""Validate the live stack with the actual local webcam."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from iot_servo_tracker.common.config import load_config
from iot_servo_tracker.web.live_stack_app import (
    DEFAULT_WEDETECT_MODULE,
    LiveStackFrame,
    LiveStackOptions,
    run_live_stack_validation,
)


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path if config_path and config_path.exists() else None)
    options = LiveStackOptions(
        query=args.query,
        frames=args.frames,
        camera_index=args.camera_index,
        sleep_s=args.sleep_s,
        approximate_tof_mm=args.tof_mm,
        approximate_ultrasonic_mm=args.ultrasonic_mm,
        production=args.production,
        preflight=not args.skip_preflight,
        device=args.device,
        yolo_model=args.yolo_model,
        tracker=args.tracker,
        confidence_threshold=args.confidence,
        wedetect_repo=args.wedetect_repo,
        wedetect_ref_module=args.wedetect_ref_module,
        wedetect_ref_script=args.wedetect_ref_script,
    )

    records: list[LiveStackFrame] = []

    def on_frame(record: LiveStackFrame) -> None:
        records.append(record)
        if args.jsonl:
            print(json.dumps(_record_summary(record), ensure_ascii=False, separators=(",", ":")))

    run_live_stack_validation(config, options, on_frame=on_frame)
    if not records:
        raise RuntimeError("no webcam frames were processed")
    if args.save_last_frame:
        Path(args.save_last_frame).write_bytes(records[-1].annotated_jpeg)
    if not args.jsonl:
        _print_summary(records, args.save_last_frame)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run web/MQTT/edge/ZMQ/vision/status validation on a real webcam",
    )
    parser.add_argument("--config", default=None, help="Path to settings TOML")
    parser.add_argument("--query", default="person", help="Web target text")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--sleep-s", type=float, default=0.03)
    parser.add_argument("--tof-mm", type=float, default=620.0)
    parser.add_argument("--ultrasonic-mm", type=float, default=650.0)
    parser.add_argument("--jsonl", action="store_true", help="Print every frame as JSONL")
    parser.add_argument("--save-last-frame", default="", help="Write the last annotated JPEG")
    parser.add_argument(
        "--production",
        action="store_true",
        help="Use real WeDetect + YOLO instead of scripted vision",
    )
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--yolo-model", default="")
    parser.add_argument("--tracker", default="")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--wedetect-repo", default="")
    parser.add_argument("--wedetect-ref-module", default=DEFAULT_WEDETECT_MODULE)
    parser.add_argument("--wedetect-ref-script", default="")
    return parser


def _print_summary(records: Iterable[LiveStackFrame], image_path: str) -> None:
    records = list(records)
    latest = records[-1]
    tracking_frames = sum(1 for record in records if record.web_view == "TRACKING")
    sent_frames = sum(1 for record in records if record.frame_sent)
    processed_frames = sum(1 for record in records if record.vision_processed)
    detected_frames = sum(1 for record in records if record.detected)
    print(f"frames={len(records)} sent={sent_frames} vision={processed_frames}")
    print(f"detected={detected_frames} tracking={tracking_frames}")
    print(
        "latest="
        f"WEB={latest.web_view} EDGE={latest.edge_state} "
        f"vision={latest.inference_source} bbox={latest.bbox} "
        f"pan={latest.pan_deg:.1f} tilt={latest.tilt_deg:.1f}"
    )
    if image_path:
        print(f"saved_last_frame={image_path}")


def _record_summary(record: LiveStackFrame) -> dict:
    data = asdict(record)
    data.pop("annotated_jpeg", None)
    data["detected"] = record.detected
    return data


if __name__ == "__main__":
    main()
