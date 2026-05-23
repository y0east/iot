import unittest

from iot_servo_tracker.common.config import AppConfig, SafetyConfig
from iot_servo_tracker.control.states import SystemState
from iot_servo_tracker.simulation.nopi import NoPiScenario, NoPiSimulation, build_parser


class NoPiSimulationTests(unittest.TestCase):
    def test_tracks_with_only_simulated_dependencies(self) -> None:
        simulation = NoPiSimulation(
            scenario=NoPiScenario(query="red cup", steps=8, rtt_ms=20.0)
        )

        result = simulation.run()

        self.assertTrue(result.command_status.ack)
        self.assertEqual(result.final_status.system_state, SystemState.TRACKING.value)
        self.assertGreater(simulation.edge.servo.applied_count, 0)
        self.assertEqual(len(result.records), 8)

    def test_occlusion_enters_safe_hold_without_hardware(self) -> None:
        config = AppConfig(safety=SafetyConfig(consecutive_frames=2))
        simulation = NoPiSimulation(
            config=config,
            scenario=NoPiScenario(
                query="red cup",
                steps=8,
                occlusion_start_step=4,
                occlusion_steps=3,
            ),
        )

        result = simulation.run()

        states = [record.status.system_state for record in result.records]
        self.assertIn(SystemState.SAFE_HOLD.value, states)

    def test_parser_exposes_no_pi_scenario_knobs(self) -> None:
        args = build_parser().parse_args(
            [
                "--query",
                "person",
                "--steps",
                "5",
                "--occlude-start",
                "2",
                "--occlude-steps",
                "3",
                "--jsonl",
            ]
        )

        self.assertEqual(args.query, "person")
        self.assertEqual(args.steps, 5)
        self.assertEqual(args.occlude_start, 2)
        self.assertEqual(args.occlude_steps, 3)
        self.assertTrue(args.jsonl)


if __name__ == "__main__":
    unittest.main()
