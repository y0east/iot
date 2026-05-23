import unittest

from iot_servo_tracker.common.config import AppConfig
from iot_servo_tracker.sim.offline import SimulationOptions, run_offline_simulation


class OfflineSimulationTests(unittest.TestCase):
    def test_normal_scenario_locks_and_tracks(self) -> None:
        events = run_offline_simulation(
            AppConfig(),
            SimulationOptions(steps=12, print_every=999),
        )

        self.assertEqual(events[0].state, "SCAN")
        self.assertIn("TRACKING", [event.state for event in events])
        self.assertTrue(any(event.bbox is not None for event in events))

    def test_lost_scenario_enters_safe_hold_after_missing_bboxes(self) -> None:
        events = run_offline_simulation(
            AppConfig(),
            SimulationOptions(scenario="lost", steps=55),
        )

        safe_hold_events = [event for event in events if event.state == "SAFE_HOLD"]
        self.assertTrue(safe_hold_events)
        self.assertTrue(any(event.bbox is None for event in safe_hold_events))
        self.assertIn("vision result is missing", safe_hold_events[0].message)

    def test_retarget_scenario_restarts_scan_for_new_query(self) -> None:
        events = run_offline_simulation(
            AppConfig(),
            SimulationOptions(scenario="retarget", steps=70, switch_query="blue cup"),
        )

        switch = next(event for event in events if event.command == "TRACK blue cup")
        after_switch = [event for event in events if event.step >= switch.step]

        self.assertEqual(switch.state, "SCAN")
        self.assertTrue(all(event.query == "blue cup" for event in after_switch))
        self.assertTrue(any(event.state == "TRACKING" for event in after_switch))

    def test_sensor_scenario_can_trigger_safe_hold_without_hardware(self) -> None:
        events = run_offline_simulation(
            AppConfig(),
            SimulationOptions(scenario="sensor", steps=60),
        )

        self.assertTrue(
            any(
                event.state == "SAFE_HOLD"
                and "ultrasonic distance dropped abruptly" in event.message
                for event in events
            )
        )


if __name__ == "__main__":
    unittest.main()
