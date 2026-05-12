import unittest

from iot_servo_tracker.control.state_machine import StateMachine
from iot_servo_tracker.control.states import Event, SystemState


class StateMachineTests(unittest.TestCase):
    def test_tracking_path(self) -> None:
        machine = StateMachine()
        self.assertEqual(machine.apply(Event.TRACK_COMMAND).current, SystemState.SCAN)
        self.assertEqual(
            machine.apply(Event.DETECTION_LOCKED).current,
            SystemState.DELAY_COMPENSATION,
        )
        self.assertEqual(machine.apply(Event.SYNC_READY).current, SystemState.TRACKING)

    def test_stop_overrides_any_state(self) -> None:
        machine = StateMachine(SystemState.TRACKING)
        self.assertEqual(machine.apply(Event.STOP_COMMAND).current, SystemState.CENTERING)


if __name__ == "__main__":
    unittest.main()
