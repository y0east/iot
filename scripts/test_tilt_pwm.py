"""Move only the tilt servo through the native PWM driver."""

from __future__ import annotations

import argparse
import time

from iot_servo_tracker.common.config import load_config
from iot_servo_tracker.control.pd_controller import ServoCommand
from iot_servo_tracker.control.servo import NativeSysfsServoDriver


def main() -> None:
    parser = argparse.ArgumentParser(description="Test tilt servo on native PWM channel 1")
    parser.add_argument("--config", default="config/settings.toml")
    parser.add_argument("--pan", type=float, default=None, help="Pan angle to hold")
    parser.add_argument("--min", type=float, default=None, help="Minimum tilt angle")
    parser.add_argument("--max", type=float, default=None, help="Maximum tilt angle")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--pause", type=float, default=0.8)
    args = parser.parse_args()

    config = load_config(args.config)
    pan_hold = config.control.pan.center_deg if args.pan is None else args.pan
    tilt_min = config.control.tilt.min_deg if args.min is None else args.min
    tilt_max = config.control.tilt.max_deg if args.max is None else args.max
    tilt_center = config.control.tilt.center_deg

    driver = NativeSysfsServoDriver(config.control.pan, config.control.tilt)
    sequence = [tilt_center, tilt_min, tilt_center, tilt_max, tilt_center]

    for cycle in range(args.cycles):
        print(f"cycle {cycle + 1}/{args.cycles}")
        for tilt in sequence:
            print(f"tilt={tilt:.1f} pan_hold={pan_hold:.1f}")
            driver.apply(
                ServoCommand(
                    pan_deg=pan_hold,
                    tilt_deg=tilt,
                    pan_pwm_us=0,
                    tilt_pwm_us=0,
                    pan_omega_deg_s=0.0,
                    tilt_omega_deg_s=0.0,
                )
            )
            time.sleep(args.pause)


if __name__ == "__main__":
    main()
