import time
import unittest

from iot_servo_tracker.common.config import AppConfig, SafetyConfig, ScanConfig
from iot_servo_tracker.common.packets import BBox, CommandPacket, CommandType, SensorSample, TrackingResult
from iot_servo_tracker.common.timebase import now_us
from iot_servo_tracker.control.states import SystemState
from iot_servo_tracker.edge.runtime import EdgeRuntime


class EdgeRuntimeTests(unittest.TestCase):
    def test_track_command_is_rejected_while_not_idle(self) -> None:
        runtime = EdgeRuntime(AppConfig())
        first = runtime.handle_command(CommandPacket.create(CommandType.TRACK, query="red cup"))
        second = runtime.handle_command(CommandPacket.create(CommandType.TRACK, query="blue cup"))
        self.assertTrue(first.ack)
        self.assertFalse(second.ack)
        self.assertEqual(runtime.current_query, "red cup")

    def test_scan_requires_consecutive_initial_detections(self) -> None:
        config = AppConfig(scan=ScanConfig(confirmation_frames=3))
        runtime = EdgeRuntime(config)
        runtime.handle_command(CommandPacket.create(CommandType.TRACK, query="red cup"))
        for index in range(2):
            result = TrackingResult(
                packet="tracking_result",
                ts_req=now_us(),
                ts_resp=now_us(),
                bbox=BBox(300, 220, 340, 260),
                confidence=0.9,
                track_id=7,
                query="red cup",
            )
            runtime.handle_tracking_result(result, SensorSample.empty())
            self.assertEqual(runtime.state, SystemState.SCAN)
        result = TrackingResult(
            packet="tracking_result",
            ts_req=now_us(),
            ts_resp=now_us(),
            bbox=BBox(300, 220, 340, 260),
            confidence=0.9,
            track_id=7,
            query="red cup",
        )
        runtime.handle_tracking_result(result, SensorSample.empty())
        self.assertEqual(runtime.state, SystemState.TRACKING)

    def test_safe_hold_times_out_to_centering(self) -> None:
        config = AppConfig(
            safety=SafetyConfig(
                timeout_min_s=0.01,
                timeout_max_s=0.01,
                safe_hold_rescan_delay_s=10.0,
            )
        )
        runtime = EdgeRuntime(config)
        runtime.state_machine.state = SystemState.SAFE_HOLD
        runtime.safe_hold_started_us = now_us()
        time.sleep(0.02)
        runtime.control_step()
        self.assertEqual(runtime.state, SystemState.CENTERING)


if __name__ == "__main__":
    unittest.main()
