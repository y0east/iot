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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="IoT servo tracker vision server")
    parser.add_argument("--config", default=None)
    parser.add_argument("--query", default="red cup")
    parser.add_argument("--serve", action="store_true", help="Run the ZMQ vision server loop")
    parser.add_argument("--production", action="store_true", help="Use WeDetect + YOLO pipeline")
    parser.add_argument("--wedetect-ref-module", default=None)
    parser.add_argument("--wedetect-ref-script", default=None)
    parser.add_argument("--wedetect-repo", default=None)
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
    if args.serve:
        serve(config, runtime)
        return
    result = runtime.process(ts_req=now_us(), query=args.query, frame_bytes=b"demo")
    print(result.to_json())


def serve(config, runtime: VisionRuntime) -> None:
    transport = ZmqVisionTransport(
        config.zmq.frame_bind_endpoint,
        config.zmq.result_bind_endpoint,
        frame_rcv_hwm=config.zmq.frame_rcv_hwm,
        result_snd_hwm=config.zmq.result_snd_hwm,
    )
    try:
        while True:
            frame = transport.recv_frame(timeout_ms=1000)
            if frame is None:
                continue
            header, payload = frame
            result = process_frame_safely(
                runtime,
                ts_req=int(header["ts_req"]),
                query=str(header.get("query", "")),
                frame_bytes=payload,
                redetect=bool(header.get("redetect", False)),
            )
            transport.send_result(result)
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
