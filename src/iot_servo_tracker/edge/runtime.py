"""Edge runtime orchestration for Raspberry Pi control."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import RLock

from iot_servo_tracker.common.config import AppConfig
from iot_servo_tracker.common.packets import (
    BBox,
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
    recovery_confirm_count: int = 0
    last_result_ts_req: int = 0
    redetect_requested: bool = False
    processed_cmd_ids: set[str] = field(default_factory=set)
    processed_cmd_id_order: deque[str] = field(default_factory=deque)
    last_status: StatusAck = field(default_factory=StatusAck)
    last_command: ServoCommand | None = None
    is_predicting: bool = False
    predicted_bbox: BBox | None = None
    locked_target_pan: float | None = None
    locked_target_tilt: float | None = None
    _lock: RLock = field(default_factory=RLock, repr=False)

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
        with self._lock:
            return self._handle_command_unlocked(command)

    def _handle_command_unlocked(self, command: CommandPacket) -> StatusAck:
        command.validate()
        if command.cmd_id in self.processed_cmd_ids:
            return self._status("duplicate command ignored", ack=False, cmd_id=command.cmd_id)
        self._remember_command(command.cmd_id)

        if command.cmd_type == CommandType.TRACK:
            if self.state == SystemState.ERROR:
                return self._status(
                    f"TRACK rejected while state is {self.state.value}",
                    ack=False,
                    cmd_id=command.cmd_id,
                )
            self.current_cmd_id = command.cmd_id
            self.current_query = command.query
            self.controller.set_max_speed_limit(command.max_speed_deg_s)
            self._reset_tracking_session()
            self._begin_scan(command.scan_range_deg)
            self.state_machine.apply(Event.TRACK_COMMAND)
        elif command.cmd_type == CommandType.STOP:
            self.current_cmd_id = command.cmd_id
            self.current_query = ""
            self.controller.set_max_speed_limit(None)
            self.state_machine.apply(Event.STOP_COMMAND)
        elif command.cmd_type == CommandType.CENTER:
            self.current_cmd_id = command.cmd_id
            self.current_query = ""
            self.controller.set_max_speed_limit(None)
            self.state_machine.apply(Event.CENTER_COMMAND)
        elif command.cmd_type == CommandType.REDETECT:
            if self.state == SystemState.SAFE_HOLD:
                self.current_cmd_id = command.cmd_id
                self.redetect_requested = True
                self.state_machine.apply(Event.RESCAN_REQUIRED)
                self._begin_limited_rescan()
            elif self.state == SystemState.LIMITED_RESCAN:
                self.current_cmd_id = command.cmd_id
                self.redetect_requested = True
            else:
                return self._status(
                    f"REDETECT ignored while state is {self.state.value}",
                    ack=False,
                    cmd_id=command.cmd_id,
                )
        return self._status(f"accepted {command.cmd_type.value}")

    def capture_frame(self, frame_bytes: bytes, ts_us: int | None = None) -> None:
        with self._lock:
            self.frame_buffer.append(ts_us or now_us(), frame_bytes)

    def consume_redetect_request(self) -> bool:
        with self._lock:
            requested = self.redetect_requested
            self.redetect_requested = False
            return requested

    def next_frame_request(self) -> tuple[str, bool]:
        with self._lock:
            vision_states = {
                SystemState.SCAN,
                SystemState.DELAY_COMPENSATION,
                SystemState.TRACKING,
                SystemState.SAFE_HOLD,
                SystemState.LIMITED_RESCAN,
            }
            if not self.current_query or self.state not in vision_states:
                self.redetect_requested = False
                return "", False
            requested = self.redetect_requested
            self.redetect_requested = False
            return self.current_query, requested

    def handle_tracking_result(
        self,
        result: TrackingResult,
        sensor_sample: SensorSample | None = None,
        dt_s: float = 0.033,
        received_ts_us: int | None = None,
    ) -> StatusAck:
        with self._lock:
            return self._handle_tracking_result_unlocked(
                result,
                sensor_sample,
                dt_s,
                received_ts_us,
            )

    def _handle_tracking_result_unlocked(
        self,
        result: TrackingResult,
        sensor_sample: SensorSample | None = None,
        dt_s: float = 0.033,
        received_ts_us: int | None = None,
    ) -> StatusAck:
        self.is_predicting = False
        self.predicted_bbox = None
        sensor_sample = sensor_sample or SensorSample.empty()
        received_ts_us = received_ts_us or now_us()
        rtt_ms = max((received_ts_us - result.ts_req) / 1_000.0, 0.0)
        if self.current_query and result.query != self.current_query:
            return self._status(
                "tracking result query does not match current target",
                rtt_ms=rtt_ms,
                confidence=result.confidence,
                ack=False,
            )
        if self.last_result_ts_req and result.ts_req <= self.last_result_ts_req:
            return self._status(
                "stale tracking result ignored",
                rtt_ms=rtt_ms,
                confidence=result.confidence,
                ack=False,
            )
        self.last_result_ts_req = result.ts_req
        entered_safe_hold = False
        if self.delay_stats.is_delayed(rtt_ms) and self.state == SystemState.TRACKING:
            self._enter_safe_hold()
            self.state_machine.apply(Event.COMMS_DELAY)
            entered_safe_hold = True

        if self.state == SystemState.SCAN:
            if self._scan_candidate_locked(result):
                self.last_valid_result = result
                self.controller.reset_history()
                
                # Update absolute target IMMEDIATELY upon locking.
                # If we lose the bbox in the very next frame, we must have a valid absolute coordinate to fallback on,
                # otherwise we will violently swing towards a stale target from the previous tracking session.
                if result.bbox is not None:
                    from iot_servo_tracker.control.geometry import pixel_error_to_angle_deg
                    yaw_err, pitch_err = pixel_error_to_angle_deg(result.bbox, self.config.camera)
                    self.locked_target_pan = self.controller.state.pan_deg + yaw_err
                    self.locked_target_tilt = self.controller.state.tilt_deg + pitch_err
                    
                self.state_machine.apply(Event.DETECTION_LOCKED)
            else:
                self._control_step_unlocked(dt_s, sensor_sample)
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
                
                from iot_servo_tracker.control.geometry import pixel_error_to_angle_deg
                yaw_err, pitch_err = pixel_error_to_angle_deg(corrected_bbox, self.config.camera)
                self.locked_target_pan = self.controller.state.pan_deg + yaw_err
                self.locked_target_tilt = self.controller.state.tilt_deg + pitch_err

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
            elif validation.category == ValidationCategory.MISSING:
                if self.locked_target_pan is not None and self.locked_target_tilt is not None:
                    yaw_error = self.locked_target_pan - self.controller.state.pan_deg
                    pitch_error = self.locked_target_tilt - self.controller.state.tilt_deg
                    command = self.controller.update_from_angle(yaw_error, pitch_error, dt_s)
                    self.servo.apply(command)
                    self.last_command = command
                    self.is_predicting = True
                    self.predicted_bbox = None
                self.state_machine.apply(Event.TRACK_OK)
        elif self.state == SystemState.SAFE_HOLD:
            command = self.controller.soft_stop(dt_s)
            self.servo.apply(command)
            self.last_command = command
            if (
                corrected_bbox is not None
                and not validation.safe_hold
                and not entered_safe_hold
                and result.confidence >= self.config.server.confidence_threshold
                and self._is_same_target(result)
            ):
                self.recovery_confirm_count += 1
                if self.recovery_confirm_count >= self.config.safety.recovery_confirm_frames:
                    self.last_valid_result = result
                    self.safe_hold_started_us = None
                    self.recovery_confirm_count = 0
                    self.controller.reset_history()
                    self.state_machine.apply(Event.RECOVERED)
            else:
                self.recovery_confirm_count = 0
                self._advance_safe_hold(dt_s)
        elif self.state == SystemState.LIMITED_RESCAN:
            if (
                corrected_bbox is not None
                and result.confidence >= self.config.server.confidence_threshold
                and self._is_same_target(result)
            ):
                self.last_valid_result = result
                self.controller.reset_history()
                self.state_machine.apply(Event.RESCAN_SUCCESS)
                self.state_machine.apply(Event.SYNC_READY)
            else:
                self._control_step_unlocked(dt_s, sensor_sample)
        elif self.state == SystemState.CENTERING:
            self._center_step_unlocked(dt_s)

        return self._status(validation.reason, rtt_ms=rtt_ms, confidence=result.confidence)

    def control_step(
        self,
        dt_s: float = 0.033,
        sensor_sample: SensorSample | None = None,
    ) -> StatusAck:
        with self._lock:
            return self._control_step_unlocked(dt_s, sensor_sample)

    def _control_step_unlocked(
        self,
        dt_s: float = 0.033,
        sensor_sample: SensorSample | None = None,
    ) -> StatusAck:
        sensor_sample = sensor_sample or SensorSample.empty()
        if sensor_sample.limit_switch_active:
            self._safe_stop_step_unlocked(dt_s)
            self.state_machine.apply(Event.CALIBRATION_ERROR)
            return self._status("limit switch is active", ack=False)
        if self.state == SystemState.SCAN:
            self._scan_step(dt_s)
        elif self.state == SystemState.SAFE_HOLD:
            self._advance_safe_hold(dt_s)
        elif self.state == SystemState.LIMITED_RESCAN:
            self._limited_rescan_step(dt_s)
        elif self.state == SystemState.CENTERING:
            self._center_step_unlocked(dt_s)
        return self._status("control step")

    def center_step(self, dt_s: float = 0.033) -> StatusAck:
        with self._lock:
            return self._center_step_unlocked(dt_s)

    def _center_step_unlocked(self, dt_s: float = 0.033) -> StatusAck:
        command = self.controller.center_step(dt_s)
        self.servo.apply(command)
        self.last_command = command
        if self.controller.is_centered():
            self.state_machine.apply(Event.CENTERED)
            self.current_query = ""
        return self._status("centering")

    def safe_stop_step(self, dt_s: float = 0.033) -> StatusAck:
        with self._lock:
            return self._safe_stop_step_unlocked(dt_s)

    def _safe_stop_step_unlocked(self, dt_s: float = 0.033) -> StatusAck:
        command = self.controller.soft_stop(dt_s)
        self.servo.apply(command)
        self.last_command = command
        return self._status("soft stop")

    def _remember_command(self, cmd_id: str) -> None:
        if cmd_id in self.processed_cmd_ids:
            return
        while len(self.processed_cmd_id_order) >= 256:
            expired = self.processed_cmd_id_order.popleft()
            self.processed_cmd_ids.discard(expired)
        self.processed_cmd_id_order.append(cmd_id)
        self.processed_cmd_ids.add(cmd_id)

    def _reset_tracking_session(self) -> None:
        self.validator = SensorValidator(self.config.safety, self.config.camera)
        self.detections = DetectionHistory()
        self.last_valid_result = None
        self.last_loss_velocity_px_s = 0.0
        self.last_result_ts_req = 0
        self.safe_hold_started_us = None
        self.limited_rescan_center_deg = 0.0
        self.redetect_requested = False
        self.recovery_confirm_count = 0
        self.is_predicting = False
        self.predicted_bbox = None

    def _begin_scan(self, scan_range_deg: float) -> None:
        limit = min(abs(scan_range_deg), self.config.scan.range_deg)
        self.scan_left_deg = max(-limit, self.config.control.pan.min_deg)
        self.scan_right_deg = min(limit, self.config.control.pan.max_deg)
        self.scan_target_deg = self.scan_left_deg
        self.scan_passes_completed = 0
        self.scan_lock_key = ""
        self.scan_lock_count = 0
        self.redetect_requested = True

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
        if not self._is_initial_scan_candidate(result):
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
            self.redetect_requested = True
            self.recovery_confirm_count = 0

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
            self.redetect_requested = True
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
        if result.bbox is None or self.last_valid_result.bbox is None:
            return False
        if not self._bbox_continuity_ok(result.bbox, self.last_valid_result.bbox):
            return False
        if (
            result.track_id is not None
            and self.last_valid_result.track_id is not None
            and result.track_id == self.last_valid_result.track_id
        ):
            return True
        x0, y0 = self.last_valid_result.bbox.center
        x1, y1 = result.bbox.center
        distance = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        return distance <= self.config.safety.pixel_jump_threshold * 2.0

    def _bbox_continuity_ok(self, bbox: BBox, previous: BBox) -> bool:
        frame_area = self.config.camera.width * self.config.camera.height
        if frame_area > 0 and bbox.area / frame_area > self.config.safety.bbox_frame_area_threshold:
            return False
        if previous.area > 0 and bbox.area / previous.area > self.config.safety.bbox_area_growth_threshold:
            return False
        if (
            _aspect_ratio_change(bbox, previous)
            > self.config.safety.bbox_aspect_ratio_change_threshold
        ):
            return False
        return True

    def _is_initial_scan_candidate(self, result: TrackingResult) -> bool:
        if result.bbox is None or result.confidence < self.config.scan.confidence_threshold:
            return False
        center_x, center_y = result.bbox.center
        dx = (center_x - self.config.camera.width / 2.0) / self.config.camera.width
        dy = (center_y - self.config.camera.height / 2.0) / self.config.camera.height
        center_distance = (dx**2 + dy**2) ** 0.5
        if center_distance > self.config.scan.max_center_distance_ratio:
            return False
        frame_area = self.config.camera.width * self.config.camera.height
        if frame_area <= 0:
            return False
        return result.bbox.area / frame_area >= self.config.scan.min_box_area_ratio

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
        cmd_id: str | None = None,
    ) -> StatusAck:
        command = self.last_command
        status = StatusAck(
            cmd_id=cmd_id or self.current_cmd_id,
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


def _aspect_ratio_change(current: BBox, previous: BBox) -> float:
    current_width = max(current.x2 - current.x1, 1e-6)
    current_height = max(current.y2 - current.y1, 1e-6)
    previous_width = max(previous.x2 - previous.x1, 1e-6)
    previous_height = max(previous.y2 - previous.y1, 1e-6)
    current_ratio = current_width / current_height
    previous_ratio = previous_width / previous_height
    return max(current_ratio / previous_ratio, previous_ratio / current_ratio)
