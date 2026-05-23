"""End-to-end web, edge, and vision simulation without external services."""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Iterable

from iot_servo_tracker.common.config import AppConfig, load_config
from iot_servo_tracker.common.packets import (
    BBox,
    CommandPacket,
    CommandType,
    StatusAck,
    TrackingResult,
)
from iot_servo_tracker.common.timebase import now_us
from iot_servo_tracker.comms.mqtt import (
    CommandPublisher,
    InMemoryMqttBus,
    MqttTopics,
    StatusPublisher,
)
from iot_servo_tracker.comms.zmq_socket import MultipartFrame
from iot_servo_tracker.control.servo import SimulatedServoDriver
from iot_servo_tracker.edge.camera import OpenCvCamera, SimulatedCamera
from iot_servo_tracker.edge.runtime import EdgeRuntime
from iot_servo_tracker.server.main import process_frame_safely
from iot_servo_tracker.server.runtime import VisionRuntime
from iot_servo_tracker.sim.offline import ScriptedSensorReader, ScriptedVisionPipeline


SCENARIOS = ("normal", "lost", "retarget", "sensor")


@dataclass(frozen=True)
class FullStackSimulationOptions:
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
    webcam: bool = False
    camera_index: int = 0


@dataclass(frozen=True)
class FullStackEvent:
    step: int
    web_view: str
    web_target: str
    command: str
    edge_state: str
    query: str
    redetect: bool
    frame_source: str
    frame_sent: bool
    vision_processed: bool
    bbox: tuple[float, float, float, float] | None
    confidence: float
    pan_deg: float
    tilt_deg: float
    mqtt_commands: int
    mqtt_statuses: int
    message: str


@dataclass
class SimulatedWebClient:
    """Web-side MQTT client that records what the Streamlit page would display."""

    bus: InMemoryMqttBus
    topics: MqttTopics
    publisher: CommandPublisher = field(init=False)
    commands: list[CommandPacket] = field(default_factory=list)
    statuses: list[StatusAck] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.publisher = CommandPublisher(self.bus, self.topics)
        self.bus.subscribe(self.topics.status, self._on_status)

    @property
    def last_status(self) -> StatusAck | None:
        if not self.statuses:
            return None
        return self.statuses[-1]

    @property
    def target(self) -> str:
        for command in reversed(self.commands):
            if command.cmd_type == CommandType.TRACK:
                return command.query
        return ""

    @property
    def view_state(self) -> str:
        return _web_view_state(self.last_status)

    def start_tracking(
        self,
        query: str,
        scan_range_deg: float,
        max_speed_deg_s: float,
    ) -> CommandPacket:
        return self.publish(
            CommandPacket.create(
                CommandType.TRACK,
                query=query,
                scan_range_deg=scan_range_deg,
                max_speed_deg_s=max_speed_deg_s,
            )
        )

    def publish(self, command: CommandPacket) -> CommandPacket:
        self.commands.append(command)
        self.publisher.publish(command)
        return command

    def _on_status(self, payload: str) -> None:
        self.statuses.append(StatusAck.from_json(payload))


class SimulatedMqttEdgeBridge:
    """Edge-side MQTT bridge backed by the in-memory bus."""

    def __init__(
        self,
        bus: InMemoryMqttBus,
        topics: MqttTopics,
        edge: EdgeRuntime,
    ) -> None:
        self.edge = edge
        self.status_publisher = StatusPublisher(bus, topics)
        bus.subscribe(topics.command, self._on_command)

    def publish_status(self, status: StatusAck) -> None:
        self.status_publisher.publish(status)

    def _on_command(self, payload: str) -> None:
        try:
            command = CommandPacket.from_json(payload)
            status = self.edge.handle_command(command)
        except Exception as exc:  # noqa: BLE001
            status = StatusAck(ack=False, message=f"invalid command: {exc}")
        self.publish_status(status)


@dataclass
class InMemoryZmqBus:
    """ZMQ-like multipart queues for frame and result packets."""

    frame_parts: deque[list[bytes]] = field(default_factory=deque)
    result_parts: deque[list[bytes]] = field(default_factory=deque)


class InMemoryZmqEdgeTransport:
    """Edge-side transport with the same shape as ``ZmqEdgeTransport``."""

    def __init__(self, bus: InMemoryZmqBus, high_water_mark: int = 1) -> None:
        self.bus = bus
        self.high_water_mark = max(1, high_water_mark)

    def send_frame(
        self,
        ts_req: int,
        query: str,
        frame_bytes: bytes,
        frame_index: int,
        redetect: bool = False,
    ) -> bool:
        frame = MultipartFrame(
            header={
                "packet": "frame",
                "ts_req": ts_req,
                "query": query,
                "frame_index": frame_index,
                "redetect": redetect,
            },
            payload=frame_bytes,
        )
        self.bus.frame_parts.append(frame.encode())
        _trim_left(self.bus.frame_parts, self.high_water_mark)
        return True

    def recv_result(self) -> TrackingResult | None:
        latest: TrackingResult | None = None
        while self.bus.result_parts:
            frame = MultipartFrame.decode(self.bus.result_parts.popleft())
            latest = TrackingResult.from_json(frame.header["result"])
        return latest


class InMemoryZmqVisionTransport:
    """Vision-side transport with the same shape as ``ZmqVisionTransport``."""

    def __init__(self, bus: InMemoryZmqBus, high_water_mark: int = 1) -> None:
        self.bus = bus
        self.high_water_mark = max(1, high_water_mark)

    def recv_frame(self) -> tuple[dict, bytes] | None:
        latest: MultipartFrame | None = None
        while self.bus.frame_parts:
            latest = MultipartFrame.decode(self.bus.frame_parts.popleft())
        if latest is None:
            return None
        return latest.header, latest.payload

    def send_result(self, result: TrackingResult) -> bool:
        frame = MultipartFrame(
            header={"packet": "tracking_result", "result": result.to_json()},
            payload=b"",
        )
        self.bus.result_parts.append(frame.encode())
        _trim_left(self.bus.result_parts, self.high_water_mark)
        return True


def run_full_stack_simulation(
    config: AppConfig,
    options: FullStackSimulationOptions,
) -> list[FullStackEvent]:
    options = _with_scenario_defaults(options)
    mqtt_bus = InMemoryMqttBus()
    topics = MqttTopics.from_config(config.mqtt)
    edge = EdgeRuntime(config=config, servo=SimulatedServoDriver())
    bridge = SimulatedMqttEdgeBridge(mqtt_bus, topics, edge)
    web = SimulatedWebClient(mqtt_bus, topics)
    zmq_bus = InMemoryZmqBus()
    edge_transport = InMemoryZmqEdgeTransport(zmq_bus, config.zmq.frame_snd_hwm)
    vision_transport = InMemoryZmqVisionTransport(zmq_bus, config.zmq.frame_rcv_hwm)
    camera, frame_source = _build_camera(config, options)
    sensors = ScriptedSensorReader(
        spike_start=options.sensor_spike_start,
        spike_steps=options.sensor_spike_steps,
    )
    vision = VisionRuntime(
        config,
        pipeline=ScriptedVisionPipeline(
            config.camera,
            lost_start=options.lost_start,
            lost_steps=options.lost_steps,
        ),
    )

    events: list[FullStackEvent] = []
    frame_index = 0
    web.start_tracking(
        options.query,
        scan_range_deg=config.scan.range_deg,
        max_speed_deg_s=config.control.max_speed_deg_s,
    )

    try:
        for step in range(options.steps):
            command_label = ""
            if options.switch_step is not None and step == options.switch_step:
                web.start_tracking(
                    options.switch_query,
                    scan_range_deg=config.scan.range_deg,
                    max_speed_deg_s=config.control.max_speed_deg_s,
                )
                command_label = f"TRACK {options.switch_query}"

            frame = camera.read_jpeg()
            ts_req = now_us()
            edge.capture_frame(frame, ts_us=ts_req)
            query, redetect = edge.next_frame_request()
            frame_sent = False
            if query:
                frame_sent = edge_transport.send_frame(
                    ts_req,
                    query,
                    frame,
                    frame_index,
                    redetect=redetect,
                )
                if frame_sent:
                    frame_index += 1

            vision_processed = _process_one_vision_frame(vision_transport, vision)
            result = edge_transport.recv_result()
            sensor = sensors.read(step)
            if result is not None:
                status = edge.handle_tracking_result(
                    result,
                    sensor,
                    dt_s=options.dt_s,
                    received_ts_us=result.ts_req + int(options.network_delay_ms * 1_000),
                )
            else:
                status = edge.control_step(dt_s=options.dt_s, sensor_sample=sensor)
            bridge.publish_status(status)

            events.append(
                FullStackEvent(
                    step=step,
                    web_view=web.view_state,
                    web_target=web.target,
                    command=command_label,
                    edge_state=status.system_state,
                    query=query or web.target,
                    redetect=redetect,
                    frame_source=frame_source,
                    frame_sent=frame_sent,
                    vision_processed=vision_processed,
                    bbox=_bbox_tuple(result.bbox if result is not None else None),
                    confidence=status.confidence,
                    pan_deg=status.pan_deg,
                    tilt_deg=status.tilt_deg,
                    mqtt_commands=len(web.commands),
                    mqtt_statuses=len(web.statuses),
                    message=status.message,
                )
            )
            if options.sleep_s > 0:
                time.sleep(options.sleep_s)
    finally:
        camera.close()
    return events


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    options = FullStackSimulationOptions(
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
        webcam=args.webcam,
        camera_index=args.camera_index,
    )
    events = run_full_stack_simulation(load_config(args.config), options)
    _print_events(events, options)


def _process_one_vision_frame(
    transport: InMemoryZmqVisionTransport,
    runtime: VisionRuntime,
) -> bool:
    frame = transport.recv_frame()
    if frame is None:
        return False
    header, payload = frame
    result = process_frame_safely(
        runtime,
        ts_req=int(header["ts_req"]),
        query=str(header.get("query", "")),
        frame_bytes=payload,
        redetect=bool(header.get("redetect", False)),
    )
    return transport.send_result(result)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a web-to-vision-to-web full stack simulation",
    )
    parser.add_argument("--config", default=None, help="Path to settings TOML")
    parser.add_argument("--query", default="red cup", help="Initial web target query")
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
    parser.add_argument("--network-delay-ms", type=float, default=0.0)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--jsonl", action="store_true", help="Print every event as JSONL")
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Capture real webcam JPEG frames instead of generated frame bytes",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    return parser


def _with_scenario_defaults(
    options: FullStackSimulationOptions,
) -> FullStackSimulationOptions:
    if options.scenario == "normal":
        return options
    data = asdict(options)
    if options.scenario == "lost":
        _set_if_default(data, "lost_start", 40)
        _set_if_default(data, "lost_steps", 18)
    elif options.scenario == "retarget":
        _set_if_default(data, "lost_start", 35)
        _set_if_default(data, "lost_steps", 25)
        _set_if_default(data, "switch_step", 52)
    elif options.scenario == "sensor":
        _set_if_default(data, "sensor_spike_start", 45)
        _set_if_default(data, "sensor_spike_steps", 7)
    else:
        raise ValueError(f"unknown simulation scenario: {options.scenario}")
    return FullStackSimulationOptions(**data)


def _set_if_default(data: dict, key: str, value) -> None:
    if key.endswith("_start") or key == "switch_step":
        if data[key] is None:
            data[key] = value
    elif data[key] in {0, None}:
        data[key] = value


def _build_camera(config: AppConfig, options: FullStackSimulationOptions):
    if options.webcam:
        return (
            OpenCvCamera(
                options.camera_index,
                config.camera.width,
                config.camera.height,
            ),
            f"webcam:{options.camera_index}",
        )
    return SimulatedCamera(payload=b"simulated-webcam-frame"), "simulated-webcam"


def _web_view_state(status: StatusAck | None) -> str:
    if status is None:
        return "IDLE"
    if status.system_state in {"SCAN", "DELAY_COMPENSATION", "LIMITED_RESCAN"}:
        return "DETECTING"
    if status.system_state == "TRACKING":
        return "TRACKING"
    return status.system_state


def _print_events(
    events: Iterable[FullStackEvent],
    options: FullStackSimulationOptions,
) -> None:
    previous_web_view = ""
    for event in events:
        if options.jsonl:
            print(json.dumps(asdict(event), ensure_ascii=False, separators=(",", ":")))
            continue
        view_changed = event.web_view != previous_web_view
        previous_web_view = event.web_view
        if (
            event.step == 0
            or event.command
            or event.redetect
            or view_changed
            or event.step % max(1, options.print_every) == 0
        ):
            bbox = "-" if event.bbox is None else _format_bbox(event.bbox)
            command = f" command={event.command}" if event.command else ""
            print(
                f"{event.step:04d} WEB={event.web_view:<10} EDGE={event.edge_state:<18} "
                f"target={event.web_target!r:<14} query={event.query!r:<14} "
                f"bbox={bbox:<23} frame={str(event.frame_sent):<5} "
                f"vision={str(event.vision_processed):<5} mqtt={event.mqtt_commands}/"
                f"{event.mqtt_statuses} pan={event.pan_deg:7.2f} "
                f"tilt={event.tilt_deg:7.2f} {event.message}{command}"
            )


def _trim_left(queue: deque, max_len: int) -> None:
    while len(queue) > max_len:
        queue.popleft()


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


if __name__ == "__main__":
    main()
