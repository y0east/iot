"""Offline edge/server simulation without Raspberry Pi hardware."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from typing import Iterable

from iot_servo_tracker.common.config import AppConfig, CameraConfig, load_config
from iot_servo_tracker.common.packets import (
    BBox,
    CommandPacket,
    CommandType,
    SensorSample,
    TrackingResult,
)
from iot_servo_tracker.common.timebase import now_us
from iot_servo_tracker.control.servo import SimulatedServoDriver
from iot_servo_tracker.edge.camera import SimulatedCamera
from iot_servo_tracker.edge.runtime import EdgeRuntime
from iot_servo_tracker.edge.sensors import SimulatedSensorReader
from iot_servo_tracker.server.vision import SimulatedVisionPipeline


SCENARIOS = ("normal", "lost", "retarget", "sensor")


@dataclass(frozen=True)
class SimulationOptions:
    query: str = "red cup"
    steps: int = 120
    scenario: str = "normal"
    switch_step: int | None = None
    switch_query: str = "blue cup"
    lost_start: int | None = None
    lost_steps: int = 0
    sensor_spike_start: int | None = None
    sensor_spike_steps: int = 0
    dt_s: float = 0.033
    sleep_s: float = 0.0
    network_delay_ms: float = 0.0
    print_every: int = 10
    jsonl: bool = False


@dataclass(frozen=True)
class SimulationEvent:
    step: int
    state: str
    query: str
    redetect: bool
    bbox: tuple[float, float, float, float] | None
    confidence: float
    pan_deg: float
    tilt_deg: float
    message: str
    command: str = ""


class ScriptedVisionPipeline:
    """Deterministic vision source that can drop frames and change targets."""

    def __init__(
        self,
        camera: CameraConfig,
        lost_start: int | None = None,
        lost_steps: int = 0,
        confidence: float = 0.91,
    ) -> None:
        self.camera = camera
        self.lost_start = lost_start
        self.lost_steps = max(0, lost_steps)
        self.confidence = confidence
        self._base = SimulatedVisionPipeline(camera, confidence=confidence)

    def process_frame(
        self,
        ts_req: int,
        query: str,
        frame_bytes: bytes = b"",
        frame_index: int = 0,
        redetect: bool = False,
    ) -> TrackingResult:
        del redetect
        if self._is_lost(frame_index):
            return TrackingResult.empty(ts_req=ts_req, query=query)

        result = self._base.process_frame(
            ts_req=ts_req,
            query=query,
            frame_bytes=frame_bytes,
            frame_index=frame_index,
        )
        offset_x, offset_y = _query_offset(query)
        bbox = result.bbox
        if bbox is not None:
            bbox = bbox.shifted(offset_x, offset_y)
        return result.__class__(
            packet=result.packet,
            ts_req=result.ts_req,
            ts_resp=result.ts_resp,
            bbox=bbox,
            confidence=result.confidence,
            track_id=_track_id(query),
            query=query,
        )

    def _is_lost(self, frame_index: int) -> bool:
        return (
            self.lost_start is not None
            and self.lost_start <= frame_index < self.lost_start + self.lost_steps
        )


class ScriptedSensorReader:
    """Sensor source that can inject an ultrasonic occlusion-like spike."""

    def __init__(
        self,
        spike_start: int | None = None,
        spike_steps: int = 0,
        normal_tof_mm: float = 620.0,
        normal_ultrasonic_mm: float = 650.0,
    ) -> None:
        self.base = SimulatedSensorReader(
            tof_mm=normal_tof_mm,
            ultrasonic_mm=normal_ultrasonic_mm,
        )
        self.spike_start = spike_start
        self.spike_steps = max(0, spike_steps)

    def read(self, step: int) -> SensorSample:
        sample = self.base.read()
        if (
            self.spike_start is not None
            and self.spike_start <= step < self.spike_start + self.spike_steps
        ):
            return SensorSample(ts=sample.ts, tof_mm=sample.tof_mm, ultrasonic_mm=430.0)
        return sample


def run_offline_simulation(
    config: AppConfig,
    options: SimulationOptions,
) -> list[SimulationEvent]:
    options = _with_scenario_defaults(options)
    config = _with_simulation_safety_defaults(config, options)
    servo = SimulatedServoDriver()
    edge = EdgeRuntime(config=config, servo=servo)
    camera = SimulatedCamera()
    vision = ScriptedVisionPipeline(
        config.camera,
        lost_start=options.lost_start,
        lost_steps=options.lost_steps,
    )
    sensors = ScriptedSensorReader(
        spike_start=options.sensor_spike_start,
        spike_steps=options.sensor_spike_steps,
    )

    events: list[SimulationEvent] = []
    command_status = edge.handle_command(
        CommandPacket.create(CommandType.TRACK, query=options.query)
    )
    active_query = options.query

    for step in range(options.steps):
        command_label = ""
        if options.switch_step is not None and step == options.switch_step:
            active_query = options.switch_query
            command_status = edge.handle_command(
                CommandPacket.create(CommandType.TRACK, query=active_query)
            )
            command_label = f"TRACK {active_query}"

        frame = camera.read_jpeg()
        ts_req = now_us()
        edge.capture_frame(frame, ts_us=ts_req)
        query, redetect = edge.next_frame_request()
        sensor = sensors.read(step)
        if query:
            result = vision.process_frame(
                ts_req=ts_req,
                query=query,
                frame_bytes=frame,
                frame_index=step,
                redetect=redetect,
            )
            status = edge.handle_tracking_result(
                result,
                sensor,
                dt_s=options.dt_s,
                received_ts_us=ts_req + int(options.network_delay_ms * 1_000),
            )
        else:
            result = None
            status = edge.control_step(dt_s=options.dt_s, sensor_sample=sensor)

        if command_label:
            message = f"{command_status.message}; {status.message}"
        else:
            message = status.message
        events.append(
            SimulationEvent(
                step=step,
                state=status.system_state,
                query=query or active_query,
                redetect=redetect,
                bbox=_bbox_tuple(result.bbox if result is not None else None),
                confidence=status.confidence,
                pan_deg=status.pan_deg,
                tilt_deg=status.tilt_deg,
                message=message,
                command=command_label,
            )
        )
        if options.sleep_s > 0:
            time.sleep(options.sleep_s)
    camera.close()
    return events


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    options = SimulationOptions(
        query=args.query,
        steps=args.steps,
        scenario=args.scenario,
        switch_step=args.switch_step,
        switch_query=args.switch_query,
        lost_start=args.lost_start,
        lost_steps=args.lost_steps,
        sensor_spike_start=args.sensor_spike_start,
        sensor_spike_steps=args.sensor_spike_steps,
        dt_s=args.dt_s,
        sleep_s=args.sleep_s,
        network_delay_ms=args.network_delay_ms,
        print_every=args.print_every,
        jsonl=args.jsonl,
    )
    events = run_offline_simulation(load_config(args.config), options)
    _print_events(events, options)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a Raspberry Pi-free local tracking simulation",
    )
    parser.add_argument("--config", default=None, help="Path to settings TOML")
    parser.add_argument("--query", default="red cup", help="Initial target query")
    parser.add_argument("--steps", type=int, default=120, help="Number of simulated frames")
    parser.add_argument("--scenario", choices=SCENARIOS, default="normal")
    parser.add_argument("--switch-step", type=int, default=None)
    parser.add_argument("--switch-query", default="blue cup")
    parser.add_argument("--lost-start", type=int, default=None)
    parser.add_argument("--lost-steps", type=int, default=0)
    parser.add_argument("--sensor-spike-start", type=int, default=None)
    parser.add_argument("--sensor-spike-steps", type=int, default=0)
    parser.add_argument("--dt-s", type=float, default=0.033)
    parser.add_argument("--sleep-s", type=float, default=0.0)
    parser.add_argument(
        "--network-delay-ms",
        type=float,
        default=0.0,
        help="Fixed simulated vision result delay in milliseconds",
    )
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--jsonl", action="store_true", help="Print every event as JSONL")
    return parser


def _with_scenario_defaults(options: SimulationOptions) -> SimulationOptions:
    if options.scenario == "normal":
        return options
    if options.scenario == "lost":
        return _replace_defaults(options, lost_start=40, lost_steps=18)
    if options.scenario == "retarget":
        return _replace_defaults(
            options,
            lost_start=35,
            lost_steps=25,
            switch_step=52,
        )
    if options.scenario == "sensor":
        return _replace_defaults(options, sensor_spike_start=45, sensor_spike_steps=15)
    raise ValueError(f"unknown simulation scenario: {options.scenario}")


def _with_simulation_safety_defaults(config: AppConfig, options: SimulationOptions) -> AppConfig:
    if options.scenario == "normal":
        return config
    required_frames = max(1, min(config.safety.consecutive_frames, 5))
    return replace(
        config,
        safety=replace(config.safety, consecutive_frames=required_frames),
    )


def _replace_defaults(options: SimulationOptions, **defaults) -> SimulationOptions:
    data = asdict(options)
    for key, value in defaults.items():
        if key.endswith("_start") or key == "switch_step":
            if data[key] is None:
                data[key] = value
        elif data[key] in {0, None}:
            data[key] = value
    return SimulationOptions(**data)


def _print_events(events: Iterable[SimulationEvent], options: SimulationOptions) -> None:
    previous_state = ""
    for event in events:
        if options.jsonl:
            print(json.dumps(asdict(event), ensure_ascii=False, separators=(",", ":")))
            continue
        state_changed = event.state != previous_state
        previous_state = event.state
        if (
            event.step == 0
            or event.command
            or event.redetect
            or state_changed
            or event.step % max(1, options.print_every) == 0
        ):
            bbox = "-" if event.bbox is None else _format_bbox(event.bbox)
            command = f" {event.command}" if event.command else ""
            print(
                f"{event.step:04d} {event.state:<18} query={event.query!r:<14} "
                f"bbox={bbox:<23} pan={event.pan_deg:7.2f} "
                f"tilt={event.tilt_deg:7.2f} redetect={str(event.redetect):<5} "
                f"{event.message}{command}"
            )


def _bbox_tuple(bbox: BBox | None) -> tuple[float, float, float, float] | None:
    if bbox is None:
        return None
    return (
        round(bbox.x1, 2),
        round(bbox.y1, 2),
        round(bbox.x2, 2),
        round(bbox.y2, 2),
    )


def _format_bbox(bbox: tuple[float, float, float, float]) -> str:
    return f"({bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f})"


def _query_offset(query: str) -> tuple[float, float]:
    seed = sum(ord(char) for char in query)
    x = ((seed % 9) - 4) * 18.0
    y = (((seed // 9) % 7) - 3) * 12.0
    return x, y


def _track_id(query: str) -> int:
    return 1 + sum((index + 1) * ord(char) for index, char in enumerate(query)) % 997


if __name__ == "__main__":
    main()
