"""Finite-state machine for the edge control process."""

from __future__ import annotations

from dataclasses import dataclass

from iot_servo_tracker.control.states import Event, SystemState


@dataclass(frozen=True)
class Transition:
    previous: SystemState
    event: Event
    current: SystemState
    reason: str


class StateMachine:
    def __init__(self, initial: SystemState = SystemState.IDLE) -> None:
        self.state = initial
        self.last_reason = "initialized"

    def apply(self, event: Event) -> Transition:
        previous = self.state
        next_state, reason = self._next(previous, event)
        self.state = next_state
        self.last_reason = reason
        return Transition(previous, event, next_state, reason)

    @staticmethod
    def _next(state: SystemState, event: Event) -> tuple[SystemState, str]:
        if event == Event.STOP_COMMAND:
            return SystemState.IDLE, "operator requested stop"
        if event == Event.CENTER_COMMAND:
            return SystemState.CENTERING, "operator requested center"
        if event == Event.CALIBRATION_ERROR:
            return SystemState.ERROR, "calibration or limit switch error"
        if event == Event.TRACK_COMMAND and state != SystemState.ERROR:
            return SystemState.SCAN, "tracking command accepted"

        table: dict[tuple[SystemState, Event], tuple[SystemState, str]] = {
            (SystemState.SCAN, Event.DETECTION_LOCKED): (
                SystemState.DELAY_COMPENSATION,
                "initial target locked",
            ),
            (SystemState.SCAN, Event.SCAN_FAILED): (
                SystemState.CENTERING,
                "scan passes exhausted",
            ),
            (SystemState.DELAY_COMPENSATION, Event.SYNC_READY): (
                SystemState.TRACKING,
                "buffer and result timestamps synchronized",
            ),
            (SystemState.TRACKING, Event.TRACK_OK): (
                SystemState.TRACKING,
                "tracking remains valid",
            ),
            (SystemState.TRACKING, Event.SENSOR_ANOMALY): (
                SystemState.SAFE_HOLD,
                "sensor validation rejected vision result",
            ),
            (SystemState.TRACKING, Event.COMMS_DELAY): (
                SystemState.SAFE_HOLD,
                "communication delay exceeded threshold",
            ),
            (SystemState.SAFE_HOLD, Event.RECOVERED): (
                SystemState.TRACKING,
                "same target recovered and validated",
            ),
            (SystemState.SAFE_HOLD, Event.RESCAN_REQUIRED): (
                SystemState.LIMITED_RESCAN,
                "safe hold persisted; start limited rescan",
            ),
            (SystemState.LIMITED_RESCAN, Event.RESCAN_SUCCESS): (
                SystemState.DELAY_COMPENSATION,
                "limited rescan found candidate",
            ),
            (SystemState.SAFE_HOLD, Event.TIMEOUT): (
                SystemState.CENTERING,
                "safe hold timeout",
            ),
            (SystemState.LIMITED_RESCAN, Event.TIMEOUT): (
                SystemState.CENTERING,
                "limited rescan timeout",
            ),
            (SystemState.CENTERING, Event.CENTERED): (
                SystemState.IDLE,
                "neutral angle reached",
            ),
            (SystemState.ERROR, Event.CENTER_COMMAND): (
                SystemState.CENTERING,
                "manual recovery requested",
            ),
        }
        return table.get((state, event), (state, f"ignored event {event} in {state}"))
