"""Command-line entry point for the vision server."""

from __future__ import annotations

import argparse

from iot_servo_tracker.common.config import load_config
from iot_servo_tracker.common.timebase import now_us
from iot_servo_tracker.server.runtime import VisionRuntime


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="IoT servo tracker vision server")
    parser.add_argument("--config", default=None)
    parser.add_argument("--query", default="red cup")
    args = parser.parse_args(argv)

    runtime = VisionRuntime(load_config(args.config))
    result = runtime.process(ts_req=now_us(), query=args.query, frame_bytes=b"demo")
    print(result.to_json())


if __name__ == "__main__":
    main()
