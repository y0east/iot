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
from iot_servo_tracker.control.safety import (
    DelayStats,
    SensorValidator,
    ValidationCategory,
    dynamic_timeout_s,
)
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
    scan_left_deg: float = 0.0
    scan_right_deg: float = 0.0
    scan_target_deg: float = 0.0
    scan_passes_completed: int = 0
    scan_lock_key: str = ""
    scan_lock_count: int = 0
    safe_hold_started_us: int | None = None
    limited_rescan_center_deg: float = 0.0
    last_valid_result: TrackingResult | None = None
    last_loss_velocity_px_s: float = 0.0
    last_status: StatusAck = field(default_factory=StatusAck)
    last_command: ServoCommand | None = None

    def __post_init__(self) -> None:
        self.controller = PDServoController(self.config)
        self.validator = SensorValidator(self.config.safety, self.config.camera)
        self.delay_stats = DelayStats(
            default_threshold_ms=self.config.safety.default_ping_threshold_ms
        )

    @property
    def state(self) -> SystemState:
        return self.state_machine.state

    def handle_command(self, command: CommandPacket) -> StatusAck:
        command.validate()
        if command.cmd_id == self.current_cmd_id:
            return self._status("duplicate command ignored")

        if command.cmd_type == CommandType.TRACK:
            if self.state != SystemState.IDLE:
                return self._status(
                    f"TRACK rejected while state is {self.state.value}",
                    ack=False,
                )
            self.current_cmd_id = command.cmd_id
            self.current_query = command.query
            self._begin_scan(command.scan_range_deg)
            self.state_machine.apply(Event.TRACK_COMMAND)
        elif command.cmd_type == CommandType.STOP:
            self.current_cmd_id = command.cmd_id
            self.state_machine.apply(Event.STOP_COMMAND)
        elif command.cmd_type == CommandType.CENTER:
            self.current_cmd_id = command.cmd_id
            self.state_machine.apply(Event.CENTER_COMMAND)
        elif command.cmd_type == CommandType.REDETECT:
            self.current_cmd_id = command.cmd_id
            if self.state == SystemState.SAFE_HOLD:
                self.state_machine.apply(Event.RESCAN_REQUIRED)
                self._begin_limited_rescan()
            elif self.state != SystemState.LIMITED_RESCAN:
                return self._status(
                    f"REDETECT ignored while state is {self.state.value}",
                    ack=False,
                )
        return self._status(f"accepted {command.cmd_type.value}")

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
        entered_safe_hold = False
        if self.delay_stats.is_delayed(rtt_ms) and self.state == SystemState.TRACKING:
            self._enter_safe_hold()
            self.state_machine.apply(Event.COMMS_DELAY)
            entered_safe_hold = True

        if self.state == SystemState.SCAN:
            if self._scan_candidate_locked(result):
                self.last_valid_result = result
                self.state_machine.apply(Event.DETECTION_LOCKED)
            else:
                self.control_step(dt_s, sensor_sample)
        if self.state == SystemState.DELAY_COMPENSATION:
            self.state_machine.apply(Event.SYNC_READY)

        self.detections.append(result)
        corrected_bbox = self.detections.estimate_current(result, now_us())
        validation = self.validator.evaluate(corrected_bbox, sensor_sample)

        if validation.category == ValidationCategory.LIMIT_SWITCH:
            self.safe_stop_step(dt_s)
            self.state_machine.apply(Event.CALIBRATION_ERROR)
            return self._status(validation.reason, rtt_ms=rtt_ms, confidence=result.confidence, ack=False)

        if self.state == SystemState.TRACKING:
            if validation.safe_hold:
                self._enter_safe_hold()
                self.state_machine.apply(Event.SENSOR_ANOMALY)
            elif corrected_bbox is not None:
                command = self.controller.update(corrected_bbox, dt_s)
                self.servo.apply(command)
                self.last_command = command
                self.last_valid_result = TrackingResult(
                    packet=result.packet,
                    ts_req=result.ts_req,
                    ts_resp=result.ts_resp,
                    bbox=corrected_bbox,
                    confidence=result.confidence,
                    track_id=result.track_id,
                    query=result.query,
                )
                self.state_machine.apply(Event.TRACK_OK)
        elif self.state == SystemState.SAFE_HOLD:
            command = self.controller.soft_stop(dt_s)
            self.servo.apply(command)
            self.last_command = command
            if (
                corrected_bbox is not None
                and not validation.safe_hold
                and not entered_safe_hold
                and result.confidence >= self.config.scan.confidence_threshold
                and self._is_same_target(result)
            ):
                self.last_valid_result = result
                self.safe_hold_started_us = None
                self.state_machine.apply(Event.RECOVERED)
            else:
                self._advance_safe_hold(dt_s)
        elif self.state == SystemState.LIMITED_RESCAN:
            if (
                corrected_bbox is not None
                and result.confidence >= self.config.scan.confidence_threshold
                and self._is_same_target(result)
            ):
                self.last_valid_result = result
                self.state_machine.apply(Event.RESCAN_SUCCESS)
                self.state_machine.apply(Event.SYNC_READY)
            else:
                self.control_step(dt_s, sensor_sample)
        elif self.state == SystemState.CENTERING:
            self.center_step(dt_s)

        return self._status(validation.reason, rtt_ms=rtt_ms, confidence=result.confidence)

    def control_step(
        self,
        dt_s: float = 0.033,
        sensor_sample: SensorSample | None = None,
    ) -> StatusAck:
        sensor_sample = sensor_sample or SensorSample.empty()
        if sensor_sample.limit_switch_active:
            self.safe_stop_step(dt_s)
            self.state_machine.apply(Event.CALIBRATION_ERROR)
            return self._status("limit switch is active", ack=False)
        if self.state == SystemState.SCAN:
            self._scan_step(dt_s)
        elif self.state == SystemState.SAFE_HOLD:
            self._advance_safe_hold(dt_s)
        elif self.state == SystemState.LIMITED_RESCAN:
            self._limited_rescan_step(dt_s)
        elif self.state == SystemState.CENTERING:
            self.center_step(dt_s)
        return self._status("control step")

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

    def _begin_scan(self, scan_range_deg: float) -> None:
        limit = min(abs(scan_range_deg), self.config.scan.range_deg)
        self.scan_left_deg = max(-limit, self.config.control.pan.min_deg)
        self.scan_right_deg = min(limit, self.config.control.pan.max_deg)
        self.scan_target_deg = self.scan_left_deg
        self.scan_passes_completed = 0
        self.scan_lock_key = ""
        self.scan_lock_count = 0

    def _scan_step(self, dt_s: float) -> None:
        command = self.controller.scan_pan_step(
            self.scan_target_deg,
            self.config.scan.speed_deg_s,
            dt_s,
        )
        self.servo.apply(command)
        self.last_command = command
        if abs(command.pan_deg - self.scan_target_deg) > 0.5:
            return
        if self.scan_target_deg == self.scan_left_deg:
            self.scan_target_deg = self.scan_right_deg
        else:
            self.scan_target_deg = self.scan_left_deg
            self.scan_passes_completed += 1
        if self.scan_passes_completed >= self.config.scan.passes:
            self.state_machine.apply(Event.SCAN_FAILED)

    def _scan_candidate_locked(self, result: TrackingResult) -> bool:
        if result.bbox is None or result.confidence < self.config.scan.confidence_threshold:
            self.scan_lock_key = ""
            self.scan_lock_count = 0
            return False
        key = str(result.track_id) if result.track_id is not None else _bbox_grid_key(result)
        if key == self.scan_lock_key:
            self.scan_lock_count += 1
        else:
            self.scan_lock_key = key
            self.scan_lock_count = 1
        return self.scan_lock_count >= self.config.scan.confirmation_frames

    def _enter_safe_hold(self) -> None:
        if self.safe_hold_started_us is None:
            self.safe_hold_started_us = now_us()
            self.last_loss_velocity_px_s = self._estimate_loss_velocity()

    def _advance_safe_hold(self, dt_s: float) -> None:
        command = self.controller.soft_stop(dt_s)
        self.servo.apply(command)
        self.last_command = command
        started = self.safe_hold_started_us or now_us()
        elapsed_s = (now_us() - started) / 1_000_000.0
        timeout_s = dynamic_timeout_s(self.last_loss_velocity_px_s, self.config.safety)
        if elapsed_s >= timeout_s:
            self.state_machine.apply(Event.TIMEOUT)
            return
        if elapsed_s >= self.config.safety.safe_hold_rescan_delay_s:
            self.state_machine.apply(Event.RESCAN_REQUIRED)
            self._begin_limited_rescan()

    def _begin_limited_rescan(self) -> None:
        self.limited_rescan_center_deg = self.controller.state.pan_deg
        span = self.config.safety.limited_rescan_range_deg
        self.scan_left_deg = max(
            self.limited_rescan_center_deg - span,
            self.config.control.pan.min_deg,
        )
        self.scan_right_deg = min(
            self.limited_rescan_center_deg + span,
            self.config.control.pan.max_deg,
        )
        self.scan_target_deg = self.scan_left_deg
        self.scan_passes_completed = 0
        self.scan_lock_key = ""
        self.scan_lock_count = 0

    def _limited_rescan_step(self, dt_s: float) -> None:
        self._scan_step(dt_s)
        if self.state != SystemState.LIMITED_RESCAN:
            return
        started = self.safe_hold_started_us or now_us()
        elapsed_s = (now_us() - started) / 1_000_000.0
        timeout_s = dynamic_timeout_s(self.last_loss_velocity_px_s, self.config.safety)
        if elapsed_s >= timeout_s:
            self.state_machine.apply(Event.TIMEOUT)

    def _is_same_target(self, result: TrackingResult) -> bool:
        if self.last_valid_result is None:
            return result.bbox is not None
        if (
            result.track_id is not None
            and self.last_valid_result.track_id is not None
            and result.track_id == self.last_valid_result.track_id
        ):
            return True
        if result.bbox is None or self.last_valid_result.bbox is None:
            return False
        x0, y0 = self.last_valid_result.bbox.center
        x1, y1 = result.bbox.center
        distance = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        return distance <= self.config.safety.pixel_jump_threshold * 2.0

    def _estimate_loss_velocity(self) -> float:
        if len(self.detections.results) < 2:
            return 0.0
        prev, latest = self.detections.results[-2], self.detections.results[-1]
        if prev.bbox is None or latest.bbox is None:
            return 0.0
        dt_s = max((latest.ts_resp - prev.ts_resp) / 1_000_000.0, 1e-3)
        x0, y0 = prev.bbox.center
        x1, y1 = latest.bbox.center
        return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / dt_s

    def _status(
        self,
        message: str,
        rtt_ms: float = 0.0,
        confidence: float = 0.0,
        ack: bool = True,
    ) -> StatusAck:
        command = self.last_command
        status = StatusAck(
            cmd_id=self.current_cmd_id,
            ack=ack,
            system_state=self.state.value,
            pan_deg=command.pan_deg if command else self.controller.state.pan_deg,
            tilt_deg=command.tilt_deg if command else self.controller.state.tilt_deg,
            rtt_ms=rtt_ms,
            confidence=confidence,
            message=message,
        )
        self.last_status = status
        return status


def _bbox_grid_key(result: TrackingResult) -> str:
    if result.bbox is None:
        return ""
    x, y = result.bbox.center
    return f"{round(x / 20)}:{round(y / 20)}"
