"""Command-line entry point for the vision server."""

from __future__ import annotations

import argparse
import os
import sys

from iot_servo_tracker.common.config import load_config
from iot_servo_tracker.common.packets import TrackingResult
from iot_servo_tracker.common.timebase import now_us
from iot_servo_tracker.comms.zmq_socket import ZmqVisionTransport
from iot_servo_tracker.server.runtime import VisionRuntime
from iot_servo_tracker.server.vision import WeDetectYoloPipeline, build_wedetect_client
from iot_servo_tracker.server.mjpeg import BackgroundMjpegServer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="IoT servo tracker vision server")
    parser.add_argument("--config", default=None)
    parser.add_argument("--query", default="red cup")
    parser.add_argument("--serve", action="store_true", help="Run the ZMQ vision server loop")
    parser.add_argument("--production", action="store_true", help="Use WeDetect + YOLO pipeline")
    parser.add_argument("--wedetect-ref-module", default=None)
    parser.add_argument("--wedetect-ref-script", default=None)
    parser.add_argument("--wedetect-repo", default=None)
    parser.add_argument("--mjpeg", action="store_true", help="Start background MJPEG stream on port 8000")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.wedetect_repo:
        os.environ["WEDETECT_REPO"] = args.wedetect_repo
    if args.wedetect_ref_module or args.wedetect_ref_script:
        config = _override_wedetect_config(
            config,
            module=args.wedetect_ref_module,
            script=args.wedetect_ref_script,
        )
    pipeline = None
    if args.production:
        wedetect_client = build_wedetect_client(config.server)
        preflight_production(wedetect_client)
        pipeline = WeDetectYoloPipeline(
            wedetect_client=wedetect_client,
            yolo_model=config.server.yolo_model,
            tracker=config.server.tracker,
            confidence_threshold=config.server.confidence_threshold,
            wedetect_confidence_threshold=config.server.wedetect_confidence_threshold,
            yolo_lost_frames=config.server.yolo_lost_frames,
            yolo_suspect_frames=config.server.yolo_suspect_frames,
            yolo_max_center_jump_px=config.server.yolo_max_center_jump_px,
            yolo_max_area_growth_ratio=config.server.yolo_max_area_growth_ratio,
            yolo_max_frame_area_ratio=config.server.yolo_max_frame_area_ratio,
            yolo_max_aspect_ratio_change=config.server.yolo_max_aspect_ratio_change,
            yolo_min_iou_on_id_change=config.server.yolo_min_iou_on_id_change,
            camera=config.camera,
        )
    runtime = VisionRuntime(config, pipeline=pipeline)
    
    mjpeg_server = None
    if args.mjpeg:
        mjpeg_server = BackgroundMjpegServer(port=8000)
        mjpeg_server.start()
        print("[server] Background MJPEG stream started at http://127.0.0.1:8000/stream.mjpg", file=sys.stderr, flush=True)

    if args.serve:
        try:
            serve(config, runtime, mjpeg_server)
        finally:
            if mjpeg_server:
                mjpeg_server.stop()
        return
    result = runtime.process(ts_req=now_us(), query=args.query, frame_bytes=b"demo")
    print(result.to_json())


def serve(config, runtime: VisionRuntime, mjpeg_server: BackgroundMjpegServer | None = None) -> None:
    transport = ZmqVisionTransport(
        config.zmq.frame_bind_endpoint,
        config.zmq.result_bind_endpoint,
        frame_rcv_hwm=config.zmq.frame_rcv_hwm,
        result_snd_hwm=config.zmq.result_snd_hwm,
    )
    import threading
    import time
    
    # AI 쓰레드와 메인 쓰레드가 공유할 최신 프레임
    state_lock = threading.Lock()
    latest_frame = None

    def inference_loop():
        nonlocal latest_frame
        while True:
            with state_lock:
                frame_to_process = latest_frame
                latest_frame = None
            
            if frame_to_process is None:
                time.sleep(0.01)
                continue
                
            header, payload = frame_to_process
            result = process_frame_safely(
                runtime,
                ts_req=int(header["ts_req"]),
                query=str(header.get("query", "")),
                frame_bytes=payload,
                redetect=bool(header.get("redetect", False)),
            )
            transport.send_result(result)
            
            if mjpeg_server is not None:
                query = header.get("query", "")
                label = f"{query} conf={result.confidence:.2f}" if result.bbox else f"{query} no bbox"
                mjpeg_server.update_bbox(result.bbox, label)

    inference_thread = threading.Thread(target=inference_loop, daemon=True)
    inference_thread.start()

    try:
        while True:
            frame = transport.recv_frame(timeout_ms=1000)
            if frame is None:
                continue
            header, payload = frame
            
            # 영상은 들어오는 즉시(15FPS) 딜레이 없이 렌더링
            if mjpeg_server is not None:
                mjpeg_server.update_raw_jpeg(payload)
                
            # 무거운 AI 처리를 위해 최신 프레임 업데이트
            if header.get("query"):
                with state_lock:
                    latest_frame = frame
    except KeyboardInterrupt:
        return
    finally:
        transport.close()


def preflight_production(wedetect_client) -> None:
    preflight = getattr(wedetect_client, "preflight", None)
    if not callable(preflight):
        raise RuntimeError("production WeDetect client does not expose preflight()")
    ref_model_dir, uni_checkpoint = preflight()
    print(f"[server] WeDetect-Ref ready: {ref_model_dir}", file=sys.stderr, flush=True)
    print(f"[server] WeDetect-Uni ready: {uni_checkpoint}", file=sys.stderr, flush=True)


def process_frame_safely(
    runtime: VisionRuntime,
    ts_req: int,
    query: str,
    frame_bytes: bytes,
    redetect: bool,
) -> TrackingResult:
    try:
        return runtime.process(
            ts_req=ts_req,
            query=query,
            frame_bytes=frame_bytes,
            redetect=redetect,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[server] frame inference failed: {exc}", file=sys.stderr, flush=True)
        return TrackingResult.empty(ts_req=ts_req, query=query)


def _override_wedetect_config(config, module: str | None, script: str | None):
    from dataclasses import replace

    server = config.server
    if module is not None:
        server = replace(server, wedetect_ref_module=module)
    if script is not None:
        server = replace(server, wedetect_ref_script=script)
    return replace(config, server=server)


if __name__ == "__main__":
    main()
