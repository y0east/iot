"""Command-line entry points for the edge runtime."""

from __future__ import annotations

import argparse
import json
import time

from iot_servo_tracker.common.config import load_config
from iot_servo_tracker.common.packets import CommandPacket, CommandType, SensorSample
from iot_servo_tracker.common.timebase import now_us
from iot_servo_tracker.edge.runtime import EdgeRuntime
from iot_servo_tracker.server.vision import SimulatedVisionPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IoT servo tracker edge runtime")
    parser.add_argument("--config", default=None, help="Path to settings TOML")
    parser.add_argument("--simulate", action="store_true", help="Run a local simulation")
    parser.add_argument("--query", default="red cup", help="Simulation target query")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.simulate:
        simulate(["--config", args.config] if args.config else [])
        return
    config = load_config(args.config)
    runtime = EdgeRuntime(config=config)
    status = runtime.handle_command(CommandPacket.create(CommandType.CENTER))
    print(status.to_json())


def simulate(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run local edge/server simulation")
    parser.add_argument("--config", default=None)
    parser.add_argument("--query", default="red cup")
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    edge = EdgeRuntime(config=config)
    vision = SimulatedVisionPipeline(config.camera)
    command = CommandPacket.create(CommandType.TRACK, query=args.query)
    print(edge.handle_command(command).to_json())

    for step in range(args.steps):
        ts = now_us()
        edge.capture_frame(b"simulated-frame", ts_us=ts)
        result = vision.process_frame(ts_req=ts, query=args.query, frame_index=step)
        sensor = SensorSample(ts=now_us(), tof_mm=620.0, ultrasonic_mm=650.0)
        status = edge.handle_tracking_result(result, sensor, dt_s=0.033)
        if step % 20 == 0 or step == args.steps - 1:
            print(json.dumps({"step": step, "status": json.loads(status.to_json())}))
        time.sleep(0.001)


if __name__ == "__main__":
    main()
