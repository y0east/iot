import unittest

from iot_servo_tracker.common.config import AppConfig
from iot_servo_tracker.sim.full_stack import (
    FullStackSimulationOptions,
    run_full_stack_simulation,
)


class FullStackSimulationTests(unittest.TestCase):
    def test_web_command_reaches_edge_and_status_returns_to_web(self) -> None:
        events = run_full_stack_simulation(
            AppConfig(),
            FullStackSimulationOptions(steps=15, print_every=999),
        )

        self.assertEqual(events[0].web_view, "DETECTING")
        self.assertTrue(any(event.frame_sent for event in events))
        self.assertTrue(any(event.vision_processed for event in events))
        self.assertTrue(any(event.web_view == "TRACKING" for event in events))
        self.assertEqual(events[-1].mqtt_commands, 1)
        self.assertGreater(events[-1].mqtt_statuses, 1)

    def test_full_stack_lost_flow_reaches_web_safe_hold(self) -> None:
        events = run_full_stack_simulation(
            AppConfig(),
            FullStackSimulationOptions(scenario="lost", steps=55),
        )

        self.assertTrue(any(event.web_view == "SAFE_HOLD" for event in events))
        self.assertTrue(
            any(
                event.edge_state == "SAFE_HOLD"
                and "vision result is missing" in event.message
                for event in events
            )
        )

    def test_full_stack_retarget_flow_reaches_tracking_again(self) -> None:
        events = run_full_stack_simulation(
            AppConfig(),
            FullStackSimulationOptions(scenario="retarget", steps=70),
        )

        switch = next(event for event in events if event.command == "TRACK blue cup")
        after_switch = [event for event in events if event.step >= switch.step]

        self.assertEqual(switch.web_view, "DETECTING")
        self.assertEqual(switch.web_target, "blue cup")
        self.assertTrue(all(event.web_target == "blue cup" for event in after_switch))
        self.assertTrue(any(event.web_view == "TRACKING" for event in after_switch))
        self.assertEqual(events[-1].mqtt_commands, 2)


if __name__ == "__main__":
    unittest.main()
