"""Edge runtime orchestration for Raspberry Pi control."""

from __future__ import annotations

from dataclasses import dataclass, field

from iot_servo_tracker.common.config import AppConfig
from iot_servo_tracker.common.packets import (
    CommandPacket,
    CommandType,
    SensorSample,
    StatusAck,
    TrackingResult,
)
from iot_servo_tracker.common.timebase import now_us
from iot_servo_tracker.control.pd_controller import PDServoController, ServoCommand
from iot_servo_tracker.control.safety import DelayStats, SensorValidator
from iot_servo_tracker.control.servo import ServoDriver, SimulatedServoDriver
from iot_servo_tracker.control.state_machine import StateMachine
from iot_servo_tracker.control.states import Event, SystemState
from iot_servo_tracker.edge.ring_buffer import DetectionHistory, RingBuffer


@dataclass
class EdgeRuntime:
    config: AppConfig
    servo: ServoDriver = field(default_factory=SimulatedServoDriver)
    state_machine: StateMachine = field(default_factory=StateMachine)
    controller: PDServoController = field(init=False)
    validator: SensorValidator = field(init=False)
    delay_stats: DelayStats = field(default_factory=DelayStats)
    frame_buffer: RingBuffer[bytes] = field(default_factory=lambda: RingBuffer(maxlen=60))
    detections: DetectionHistory = field(default_factory=DetectionHistory)
    current_cmd_id: str = ""
    current_query: str = ""
    last_status: StatusAck = field(default_factory=StatusAck)
    last_command: ServoCommand | None = None

    def __post_init__(self) -> None:
        self.controller = PDServoController(self.config)
        self.validator = SensorValidator(self.config.safety)

    @property
    def state(self) -> SystemState:
        return self.state_machine.state

    def handle_command(self, command: CommandPacket) -> StatusAck:
        command.validate()
        if command.cmd_id == self.current_cmd_id and command.cmd_type == CommandType.TRACK:
            return self._status("duplicate command ignored")

        self.current_cmd_id = command.cmd_id
        if command.cmd_type == CommandType.TRACK:
            self.current_query = command.query
            self.state_machine.apply(Event.TRACK_COMMAND)
        elif command.cmd_type == CommandType.STOP:
            self.state_machine.apply(Event.STOP_COMMAND)
        elif command.cmd_type == CommandType.CENTER:
            self.state_machine.apply(Event.CENTER_COMMAND)
        elif command.cmd_type == CommandType.REDETECT:
            self.state_machine.apply(Event.REDETECT_COMMAND)
        return self._status(f"accepted {command.cmd_type}")

    def capture_frame(self, frame_bytes: bytes, ts_us: int | None = None) -> None:
        self.frame_buffer.append(ts_us or now_us(), frame_bytes)

    def handle_tracking_result(
        self,
        result: TrackingResult,
        sensor_sample: SensorSample | None = None,
        dt_s: float = 0.033,
    ) -> StatusAck:
        sensor_sample = sensor_sample or SensorSample.empty()
        rtt_ms = max((result.ts_resp - result.ts_req) / 1_000.0, 0.0)
        if self.delay_stats.is_delayed(rtt_ms) and self.state == SystemState.TRACKING:
            self.state_machine.apply(Event.COMMS_DELAY)

        if self.state == SystemState.SCAN and result.bbox and result.confidence >= 0.5:
            self.state_machine.apply(Event.DETECTION_LOCKED)
        if self.state == SystemState.DELAY_COMPENSATION:
            self.state_machine.apply(Event.SYNC_READY)

        self.detections.append(result)
        corrected_bbox = self.detections.estimate_current(result, now_us())
        validation = self.validator.evaluate(corrected_bbox, sensor_sample)

        if self.state == SystemState.TRACKING:
            if validation.safe_hold:
                self.state_machine.apply(Event.SENSOR_ANOMALY)
            elif corrected_bbox is not None:
                command = self.controller.update(corrected_bbox, dt_s)
                self.servo.apply(command)
                self.last_command = command
                self.state_machine.apply(Event.TRACK_OK)
        elif self.state == SystemState.SAFE_HOLD:
            command = self.controller.soft_stop(dt_s)
            self.servo.apply(command)
            self.last_command = command
            if corrected_bbox is not None and not validation.safe_hold:
                self.state_machine.apply(Event.RECOVERED)
        elif self.state == SystemState.CENTERING:
            self.center_step(dt_s)

        return self._status(validation.reason, rtt_ms=rtt_ms, confidence=result.confidence)

    def center_step(self, dt_s: float = 0.033) -> StatusAck:
        command = self.controller.center_step(dt_s)
        self.servo.apply(command)
        self.last_command = command
        if self.controller.is_centered():
            self.state_machine.apply(Event.CENTERED)
        return self._status("centering")

    def safe_stop_step(self, dt_s: float = 0.033) -> StatusAck:
        command = self.controller.soft_stop(dt_s)
        self.servo.apply(command)
        self.last_command = command
        return self._status("soft stop")

    def _status(
        self,
        message: str,
        rtt_ms: float = 0.0,
        confidence: float = 0.0,
    ) -> StatusAck:
        command = self.last_command
        status = StatusAck(
            cmd_id=self.current_cmd_id,
            ack=True,
            system_state=self.state.value,
            pan_deg=command.pan_deg if command else self.controller.state.pan_deg,
            tilt_deg=command.tilt_deg if command else self.controller.state.tilt_deg,
            rtt_ms=rtt_ms,
            confidence=confidence,
            message=message,
        )
        self.last_status = status
        return status
