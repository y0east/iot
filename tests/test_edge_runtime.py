import time
import unittest

from iot_servo_tracker.common.config import AppConfig, SafetyConfig, ScanConfig
from iot_servo_tracker.common.packets import BBox, CommandPacket, CommandType, SensorSample, TrackingResult
from iot_servo_tracker.common.timebase import now_us
from iot_servo_tracker.control.states import SystemState
from iot_servo_tracker.edge.runtime import EdgeRuntime


def sensor_sample() -> SensorSample:
    return SensorSample(ts=now_us(), tof_mm=620.0, ultrasonic_mm=650.0)


class EdgeRuntimeTests(unittest.TestCase):
    def test_track_command_is_rejected_while_not_idle(self) -> None:
        runtime = EdgeRuntime(AppConfig())
        first = runtime.handle_command(CommandPacket.create(CommandType.TRACK, query="red cup"))
        second = runtime.handle_command(CommandPacket.create(CommandType.TRACK, query="blue cup"))
        self.assertTrue(first.ack)
        self.assertFalse(second.ack)
        self.assertEqual(runtime.current_query, "red cup")

    def test_replayed_command_id_is_ignored_after_current_command_changes(self) -> None:
        runtime = EdgeRuntime(AppConfig())
        track = CommandPacket.create(CommandType.TRACK, query="red cup")
        runtime.handle_command(track)
        runtime.handle_command(CommandPacket.create(CommandType.CENTER))
        runtime.center_step()
        replay = runtime.handle_command(track)
        self.assertFalse(replay.ack)
        self.assertEqual(runtime.state, SystemState.IDLE)

    def test_track_command_speed_limit_is_applied(self) -> None:
        runtime = EdgeRuntime(AppConfig())
        runtime.handle_command(
            CommandPacket.create(CommandType.TRACK, query="red cup", max_speed_deg_s=5)
        )
        self.assertEqual(runtime.controller.max_speed_deg_s, 5)

    def test_scan_requires_consecutive_initial_detections(self) -> None:
        config = AppConfig(scan=ScanConfig(confirmation_frames=3))
        runtime = EdgeRuntime(config)
        runtime.handle_command(CommandPacket.create(CommandType.TRACK, query="red cup"))
        for _ in range(2):
            result = TrackingResult(
                packet="tracking_result",
                ts_req=now_us(),
                ts_resp=now_us(),
                bbox=BBox(300, 220, 340, 260),
                confidence=0.9,
                track_id=7,
                query="red cup",
            )
            runtime.handle_tracking_result(result, sensor_sample())
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
        runtime.handle_tracking_result(result, sensor_sample())
        self.assertEqual(runtime.state, SystemState.TRACKING)

    def test_scan_rejects_far_or_tiny_initial_candidate(self) -> None:
        config = AppConfig(scan=ScanConfig(confirmation_frames=1))
        runtime = EdgeRuntime(config)
        runtime.handle_command(CommandPacket.create(CommandType.TRACK, query="red cup"))
        result = TrackingResult(
            packet="tracking_result",
            ts_req=now_us(),
            ts_resp=now_us(),
            bbox=BBox(620, 450, 630, 460),
            confidence=0.99,
            track_id=7,
            query="red cup",
        )
        runtime.handle_tracking_result(result, sensor_sample())
        self.assertEqual(runtime.state, SystemState.SCAN)

    def test_stale_tracking_result_is_ignored(self) -> None:
        runtime = EdgeRuntime(AppConfig())
        runtime.state_machine.state = SystemState.TRACKING
        ts = now_us()
        fresh = TrackingResult(
            packet="tracking_result",
            ts_req=ts,
            ts_resp=0,
            bbox=BBox(300, 220, 340, 260),
            confidence=0.9,
            track_id=7,
            query="red cup",
        )
        stale = TrackingResult(
            packet="tracking_result",
            ts_req=ts - 1,
            ts_resp=0,
            bbox=BBox(100, 100, 150, 150),
            confidence=0.9,
            track_id=7,
            query="red cup",
        )
        runtime.handle_tracking_result(fresh, sensor_sample())
        applied_count = runtime.servo.applied_count
        status = runtime.handle_tracking_result(stale, sensor_sample())
        self.assertFalse(status.ack)
        self.assertEqual(status.message, "stale tracking result ignored")
        self.assertEqual(runtime.servo.applied_count, applied_count)

    def test_rtt_uses_edge_receive_time_not_server_response_clock(self) -> None:
        runtime = EdgeRuntime(AppConfig())
        runtime.state_machine.state = SystemState.TRACKING
        ts_req = now_us()
        result = TrackingResult(
            packet="tracking_result",
            ts_req=ts_req,
            ts_resp=42,
            bbox=BBox(300, 220, 340, 260),
            confidence=0.9,
            track_id=7,
            query="red cup",
        )

        status = runtime.handle_tracking_result(
            result,
            sensor_sample(),
            received_ts_us=ts_req + 50_000,
        )

        self.assertAlmostEqual(status.rtt_ms, 50.0)
        self.assertEqual(runtime.state, SystemState.TRACKING)

    def test_consecutive_sensor_unavailable_enters_safe_hold(self) -> None:
        config = AppConfig(safety=SafetyConfig(consecutive_frames=2))
        runtime = EdgeRuntime(config)
        runtime.state_machine.state = SystemState.TRACKING

        for _ in range(2):
            runtime.handle_tracking_result(
                TrackingResult(
                    packet="tracking_result",
                    ts_req=now_us(),
                    ts_resp=0,
                    bbox=BBox(300, 220, 340, 260),
                    confidence=0.9,
                    track_id=7,
                    query="red cup",
                ),
                SensorSample.empty(),
            )

        self.assertEqual(runtime.state, SystemState.SAFE_HOLD)

    def test_redetect_command_sets_one_shot_request(self) -> None:
        runtime = EdgeRuntime(AppConfig())
        runtime.state_machine.state = SystemState.SAFE_HOLD
        runtime.safe_hold_started_us = now_us()
        status = runtime.handle_command(CommandPacket.create(CommandType.REDETECT))
        self.assertTrue(status.ack)
        self.assertTrue(runtime.consume_redetect_request())
        self.assertFalse(runtime.consume_redetect_request())

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

    def test_safe_hold_does_not_recover_to_large_bbox_even_with_same_track_id(self) -> None:
        config = AppConfig(
            safety=SafetyConfig(
                bbox_frame_area_threshold=0.20,
                safe_hold_rescan_delay_s=10.0,
            )
        )
        runtime = EdgeRuntime(config)
        runtime.state_machine.state = SystemState.SAFE_HOLD
        runtime.safe_hold_started_us = now_us()
        runtime.last_valid_result = TrackingResult(
            packet="tracking_result",
            ts_req=1,
            ts_resp=2,
            bbox=BBox(300, 220, 340, 260),
            confidence=0.9,
            track_id=7,
            query="red cup",
        )

        status = runtime.handle_tracking_result(
            TrackingResult(
                packet="tracking_result",
                ts_req=now_us(),
                ts_resp=now_us(),
                bbox=BBox(100, 80, 600, 460),
                confidence=0.99,
                track_id=7,
                query="red cup",
            ),
            sensor_sample(),
        )

        self.assertEqual(runtime.state, SystemState.SAFE_HOLD)
        self.assertNotEqual(status.system_state, SystemState.TRACKING.value)


if __name__ == "__main__":
    unittest.main()
