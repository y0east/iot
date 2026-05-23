"""No-Raspberry-Pi simulation runner.

This module drives the real edge runtime with simulated camera frames, simulated
vision results, simulated distance sensors, and the existing simulated servo
driver. It is intentionally dependency-light so a laptop can exercise the
control loop without GPIO, PCA9685, camera, MQTT, or ZMQ setup.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

from iot_servo_tracker.common.config import AppConfig, load_config
from iot_servo_tracker.common.packets import (
    CommandPacket,
    CommandType,
    SensorSample,
    StatusAck,
    TrackingResult,
)
from iot_servo_tracker.common.timebase import now_us
from iot_servo_tracker.edge.runtime import EdgeRuntime
from iot_servo_tracker.server.vision import SimulatedVisionPipeline


@dataclass(frozen=True)
class NoPiScenario:
    """Inputs that shape a deterministic desktop simulation run."""

    query: str = "red cup"
    steps: int = 120
    dt_s: float = 1.0 / 30.0
    rtt_ms: float = 25.0
    tof_mm: float = 620.0
    ultrasonic_mm: float = 650.0
    occlusion_start_step: int | None = None
    occlusion_steps: int = 0
    sensor_dropout_start_step: int | None = None
    sensor_dropout_steps: int = 0
    limit_switch_step: int | None = None
    frame_payload: bytes = b"simulated-frame"

    def is_occluded(self, step: int) -> bool:
        return _in_window(step, self.occlusion_start_step, self.occlusion_steps)

    def sensors_unavailable(self, step: int) -> bool:
        return _in_window(step, self.sensor_dropout_start_step, self.sensor_dropout_steps)

    def sensor_sample(self, step: int) -> SensorSample:
        if self.sensors_unavailable(step):
            return SensorSample.empty()
        return SensorSample(
            ts=now_us(),
            tof_mm=self.tof_mm,
            ultrasonic_mm=self.ultrasonic_mm,
            limit_switch_active=self.limit_switch_step == step,
        )


@dataclass(frozen=True)
class NoPiStepRecord:
    step: int
    status: StatusAck
    sensor: SensorSample
    result: TrackingResult | None

    def as_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "status": json.loads(self.status.to_json()),
            "sensor": json.loads(self.sensor.to_json()),
            "result": json.loads(self.result.to_json()) if self.result else None,
        }


@dataclass(frozen=True)
class NoPiSimulationResult:
    command_status: StatusAck
    records: list[NoPiStepRecord]

    @property
    def final_status(self) -> StatusAck:
        if not self.records:
            return self.command_status
        return self.records[-1].status

    def as_dict(self) -> dict[str, object]:
        return {
            "command_status": json.loads(self.command_status.to_json()),
            "final_status": json.loads(self.final_status.to_json()),
            "steps": [record.as_dict() for record in self.records],
        }


class NoPiSimulation:
    """Run the edge control loop against fully simulated dependencies."""

    def __init__(
        self,
        config: AppConfig | None = None,
        scenario: NoPiScenario | None = None,
    ) -> None:
        self.config = config or AppConfig()
        self.scenario = scenario or NoPiScenario()
        self.edge = EdgeRuntime(config=self.config)
        self.vision = SimulatedVisionPipeline(self.config.camera)

    def run(self, realtime: bool = False) -> NoPiSimulationResult:
        command_status = self.edge.handle_command(
            CommandPacket.create(CommandType.TRACK, query=self.scenario.query)
        )
        records: list[NoPiStepRecord] = []
        for step in range(max(0, self.scenario.steps)):
            started = time.monotonic()
            records.append(self.step(step))
            if realtime:
                elapsed = time.monotonic() - started
                time.sleep(max(0.0, self.scenario.dt_s - elapsed))
        return NoPiSimulationResult(command_status=command_status, records=records)

    def step(self, step: int) -> NoPiStepRecord:
        ts_req = now_us()
        self.edge.capture_frame(self.scenario.frame_payload, ts_us=ts_req)
        query, redetect = self.edge.next_frame_request()
        result = self._vision_result(step, ts_req, query, redetect)
        sensor = self.scenario.sensor_sample(step)
        if result is None:
            status = self.edge.control_step(dt_s=self.scenario.dt_s, sensor_sample=sensor)
        else:
            status = self.edge.handle_tracking_result(
                result,
                sensor,
                dt_s=self.scenario.dt_s,
                received_ts_us=ts_req + int(self.scenario.rtt_ms * 1_000),
            )
        return NoPiStepRecord(step=step, status=status, sensor=sensor, result=result)

    def _vision_result(
        self,
        step: int,
        ts_req: int,
        query: str,
        redetect: bool,
    ) -> TrackingResult | None:
        del redetect
        if not query:
            return None
        if self.scenario.is_occluded(step):
            return TrackingResult.empty(ts_req=ts_req, query=query)
        return self.vision.process_frame(
            ts_req=ts_req,
            query=query,
            frame_bytes=self.scenario.frame_payload,
            frame_index=step,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the tracker without Raspberry Pi hardware")
    parser.add_argument("--config", default=None, help="Path to settings TOML")
    parser.add_argument("--query", default="red cup", help="Target query for TRACK command")
    parser.add_argument("--steps", type=int, default=120, help="Simulation step count")
    parser.add_argument("--dt-ms", type=float, default=33.0, help="Control step duration")
    parser.add_argument("--rtt-ms", type=float, default=25.0, help="Simulated inference RTT")
    parser.add_argument("--tof-mm", type=float, default=620.0, help="Simulated ToF distance")
    parser.add_argument(
        "--ultrasonic-mm",
        type=float,
        default=650.0,
        help="Simulated ultrasonic distance",
    )
    parser.add_argument("--occlude-start", type=int, default=None, help="First missing-vision step")
    parser.add_argument(
        "--occlude-steps",
        type=int,
        default=0,
        help="Number of missing-vision steps",
    )
    parser.add_argument(
        "--sensor-dropout-start",
        type=int,
        default=None,
        help="First step with unavailable distance sensors",
    )
    parser.add_argument(
        "--sensor-dropout-steps",
        type=int,
        default=0,
        help="Number of unavailable-sensor steps",
    )
    parser.add_argument("--limit-switch-step", type=int, default=None)
    parser.add_argument("--jsonl", action="store_true", help="Emit one JSON object per line")
    parser.add_argument("--realtime", action="store_true", help="Sleep between steps")
    parser.add_argument("--summary-every", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scenario = NoPiScenario(
        query=args.query,
        steps=args.steps,
        dt_s=max(args.dt_ms, 1.0) / 1_000.0,
        rtt_ms=max(args.rtt_ms, 0.0),
        tof_mm=args.tof_mm,
        ultrasonic_mm=args.ultrasonic_mm,
        occlusion_start_step=args.occlude_start,
        occlusion_steps=max(0, args.occlude_steps),
        sensor_dropout_start_step=args.sensor_dropout_start,
        sensor_dropout_steps=max(0, args.sensor_dropout_steps),
        limit_switch_step=args.limit_switch_step,
    )
    simulation = NoPiSimulation(config=load_config(args.config), scenario=scenario)
    result = simulation.run(realtime=args.realtime)
    if args.jsonl:
        print(
            json.dumps(
                {"event": "command", "status": json.loads(result.command_status.to_json())}
            )
        )
        for record in result.records:
            print(json.dumps(record.as_dict(), ensure_ascii=False))
    else:
        _print_summary(result, every=max(1, args.summary_every))
    return 1 if result.final_status.system_state == "ERROR" else 0


def _print_summary(result: NoPiSimulationResult, every: int) -> None:
    command = result.command_status
    print(
        "No-Pi simulation started: "
        f"ack={command.ack} state={command.system_state} message={command.message}"
    )
    for record in result.records:
        if record.step % every != 0 and record.step != len(result.records) - 1:
            continue
        status = record.status
        bbox = record.result.bbox if record.result else None
        bbox_text = "none" if bbox is None else (
            f"({bbox.x1:.0f},{bbox.y1:.0f})-({bbox.x2:.0f},{bbox.y2:.0f})"
        )
        print(
            f"step={record.step:04d} state={status.system_state:<18} "
            f"pan={status.pan_deg:7.2f} tilt={status.tilt_deg:7.2f} "
            f"conf={status.confidence:.2f} bbox={bbox_text} msg={status.message}"
        )
    final = result.final_status
    print(
        "No-Pi simulation finished: "
        f"state={final.system_state} pan={final.pan_deg:.2f} "
        f"tilt={final.tilt_deg:.2f} message={final.message}"
    )


def _in_window(step: int, start: int | None, length: int) -> bool:
    if start is None or length <= 0:
        return False
    return start <= step < start + length


if __name__ == "__main__":
    raise SystemExit(main())
