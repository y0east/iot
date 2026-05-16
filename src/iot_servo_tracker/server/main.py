"""Command-line entry point for the vision server."""

from __future__ import annotations

import argparse

from iot_servo_tracker.common.config import load_config
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
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.wedetect_ref_module or args.wedetect_ref_script:
        config = _override_wedetect_config(
            config,
            module=args.wedetect_ref_module,
            script=args.wedetect_ref_script,
        )
    pipeline = None
    if args.production:
        pipeline = WeDetectYoloPipeline(
            wedetect_client=build_wedetect_client(config.server),
            yolo_model=config.server.yolo_model,
            tracker=config.server.tracker,
            confidence_threshold=config.server.confidence_threshold,
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
    )
    try:
        while True:
            frame = transport.recv_frame(timeout_ms=1000)
            if frame is None:
                continue
            header, payload = frame
            result = runtime.process(
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
