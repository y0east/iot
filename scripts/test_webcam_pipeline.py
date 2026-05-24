#!/usr/bin/env python3
"""Interactive webcam test script for WeDetect lock-on and YOLO tracking.

This script allows you to test the complete vision pipeline locally using a webcam 
without needing any Raspberry Pi hardware. It runs a live OpenCV window showing the 
bounding box:
- BLUE: WeDetect Open-Vocabulary Lock-on
- GREEN: YOLO + ByteTrack high-speed tracking
- RED: Lost / Safe Hold state

Usage:
  python scripts/test_webcam_pipeline.py --query "cell phone" --wedetect-repo path/to/WeDetect
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Add src to python path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from iot_servo_tracker.common.config import load_config
from iot_servo_tracker.common.packets import BBox
from iot_servo_tracker.common.timebase import now_us
from iot_servo_tracker.server.vision import WeDetectYoloPipeline, build_wedetect_client


def main() -> int:
    parser = argparse.ArgumentParser(description="Webcam WeDetect + YOLO Pipeline Tester")
    parser.add_argument("--query", default="person", help="Target search query (e.g. 'person', 'cup')")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam device index")
    parser.add_argument("--device", default="cuda:0", help="Inference device (cuda:0 or cpu)")
    parser.add_argument("--wedetect-repo", default=os.environ.get("WEDETECT_REPO", "external/WeDetect"), 
                        help="Path to cloned WeChatCV/WeDetect repository")
    parser.add_argument("--yolo-model", default="yolo26m.pt", help="YOLO checkpoint")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="YOLO tracker configuration")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold")
    parser.add_argument("--wedetect-conf", type=float, default=0.10, help="WeDetect confidence threshold")
    parser.add_argument("--yolo-max-area-ratio", type=float, default=0.85, help="YOLO maximum frame area ratio")
    parser.add_argument("--config", default=None, help="Path to config TOML")
    args = parser.parse_args()

    # Verify dependencies
    try:
        import cv2
        import numpy as np
        import torch
    except ImportError as exc:
        print(f"[-] Missing dependencies: {exc}", file=sys.stderr)
        print("Please install opencv-python, numpy, and torch.", file=sys.stderr)
        return 1

    # Verify WeChatCV/WeDetect repo
    if args.wedetect_repo:
        repo_path = Path(args.wedetect_repo).expanduser().resolve()
        if not (repo_path / "infer_wedetect_ref.py").exists():
            print(f"[-] --wedetect-repo path does not look like WeChatCV/WeDetect: {repo_path}", file=sys.stderr)
            return 1
        os.environ["WEDETECT_REPO"] = str(repo_path)
    elif not os.environ.get("WEDETECT_REPO"):
        print("[-] Error: WeDetect repository path is not provided.", file=sys.stderr)
        print("WeDetect requires the official WeChatCV/WeDetect repository.", file=sys.stderr)
        print("Please clone it first: git clone https://github.com/WeChatCV/WeDetect external/WeDetect", file=sys.stderr)
        print("Then run this script with: --wedetect-repo external/WeDetect", file=sys.stderr)
        return 1

    print("[+] Loading configuration...")
    config = load_config(args.config)
    
    # Override server settings with CLI args
    server_config = config.server
    from dataclasses import replace
    server_config = replace(
        server_config,
        wedetect_ref_module="iot_servo_tracker.server.wedetect_ref_runtime:detect",
        wedetect_device=args.device,
        yolo_model=args.yolo_model,
        tracker=args.tracker,
        confidence_threshold=args.conf,
        wedetect_confidence_threshold=args.wedetect_conf,
        yolo_max_frame_area_ratio=args.yolo_max_area_ratio,
    )
    
    print("[+] Initializing WeDetect client (this might download weights if not present)...")
    try:
        wedetect_client = build_wedetect_client(server_config)
        # Run preflight snapshot download/verify
        preflight = getattr(wedetect_client, "preflight", None)
        if callable(preflight):
            preflight()
    except Exception as exc:
        print(f"[-] Failed to initialize WeDetect: {exc}", file=sys.stderr)
        return 1

    print("[+] Initializing WeDetectYoloPipeline...")
    pipeline = WeDetectYoloPipeline(
        wedetect_client=wedetect_client,
        yolo_model=server_config.yolo_model,
        tracker=server_config.tracker,
        confidence_threshold=server_config.confidence_threshold,
        wedetect_confidence_threshold=server_config.wedetect_confidence_threshold,
        yolo_max_frame_area_ratio=server_config.yolo_max_frame_area_ratio,
        camera=config.camera,
    )

    print(f"[+] Opening webcam device {args.camera_index}...")
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"[-] Failed to open webcam at index {args.camera_index}", file=sys.stderr)
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.height)

    window_name = "WeDetect + YOLO Webcam Pipeline Tester"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"\n[+] Starting live preview window. Target Query: '{args.query}'")
    print("    - Press 'q' to quit.")
    print("    - Press 'r' to force REDETECT (resets YOLO, runs WeDetect again).")
    
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[-] Failed to read frame from webcam", file=sys.stderr)
                break

            # Encode frame to jpeg bytes as the pipeline expects
            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            frame_bytes = encoded.tobytes()

            # Process frame through WeDetect -> YOLO pipeline
            ts_req = now_us()
            start_time = time.monotonic()
            
            # If redetect was requested or we have no locked bbox, it runs WeDetect. Otherwise, it tracks with YOLO.
            result = pipeline.process_frame(
                ts_req=ts_req,
                query=args.query,
                frame_bytes=frame_bytes,
                frame_index=frame_index
            )
            
            fps = 1.0 / max(0.001, time.monotonic() - start_time)

            # Draw visualization overlay
            draw_frame = frame.copy()
            
            # Determine color and tracking mode name
            if result.bbox is not None:
                # BBox coordinates
                x1, y1, x2, y2 = map(int, [result.bbox.x1, result.bbox.y1, result.bbox.x2, result.bbox.y2])
                
                # Check if it was YOLO or WeDetect
                if pipeline.locked_track_id is not None:
                    # YOLO tracking (Green)
                    color = (0, 255, 0)
                    mode_text = f"YOLO Track (ID: {pipeline.locked_track_id})"
                else:
                    # WeDetect detection (Blue)
                    color = (255, 0, 0)
                    mode_text = "WeDetect Lock-on"
                
                # Draw bbox
                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 3)
                
                # Draw text background and text
                label = f"{mode_text} Conf: {result.confidence:.2f}"
                cv2.putText(draw_frame, label, (x1, max(15, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                # Lost / Search mode (Red)
                color = (0, 0, 255)
                # Show yolo reject reason if any
                reason = pipeline.last_yolo_reject_reason or "Target Lost"
                cv2.putText(draw_frame, f"LOST: {reason}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw status info
            cv2.putText(draw_frame, f"Query: '{args.query}'", (20, config.camera.height - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(draw_frame, f"FPS: {fps:.1f}", (20, config.camera.height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Show frame
            cv2.imshow(window_name, draw_frame)
            frame_index += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("[*] Manual REDETECT triggered! Clearing YOLO locks...")
                pipeline.redetect(frame_bytes, args.query, now_us())

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[+] Webcam release and resources cleaned up.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
