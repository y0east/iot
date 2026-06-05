import time
import unittest

from iot_servo_tracker.common.config import AppConfig, ControlConfig, SafetyConfig, ScanConfig
from iot_servo_tracker.common.packets import BBox, CommandPacket, CommandType, SensorSample, TrackingResult
from iot_servo_tracker.common.timebase import now_us
from iot_servo_tracker.control.states import Event, SystemState
from iot_servo_tracker.edge.runtime import EdgeRuntime
from iot_servo_tracker.edge.status_light import BLUE, GREEN, RED, SimulatedStatusLight


def sensor_sample() -> SensorSample:
    return SensorSample(ts=now_us(), tof_mm=620.0, ultrasonic_mm=650.0)


class EdgeRuntimeTests(unittest.TestCase):
    def test_track_command_restarts_while_not_idle(self) -> None:
        runtime = EdgeRuntime(AppConfig())
        first = runtime.handle_command(CommandPacket.create(CommandType.TRACK, query="red cup"))
        runtime.last_valid_result = TrackingResult(
            packet="tracking_result",
            ts_req=1,
            ts_resp=2,
            bbox=BBox(300, 220, 340, 260),
            confidence=0.9,
            track_id=7,
            query="red cup",
        )
        runtime.validator.evaluate(BBox(300, 220, 340, 260), sensor_sample())
        runtime.detections.append(runtime.last_valid_result)
        runtime.safe_hold_started_us = now_us()

        second = runtime.handle_command(CommandPacket.create(CommandType.TRACK, query="blue cup"))

        self.assertTrue(first.ack)
        self.assertTrue(second.ack)
        self.assertEqual(runtime.current_query, "blue cup")
        self.assertEqual(runtime.state, SystemState.SCAN)
        self.assertIsNone(runtime.last_valid_result)
        self.assertIsNone(runtime.validator.prev_bbox)
        self.assertIsNone(runtime.safe_hold_started_us)
        self.assertEqual(len(runtime.detections.results), 0)
        query, redetect = runtime.next_frame_request()
        self.assertEqual(query, "blue cup")
        self.assertTrue(redetect)

    def test_track_command_is_rejected_in_error_state(self) -> None:
        runtime = EdgeRuntime(AppConfig())
        runtime.state_machine.state = SystemState.ERROR

        status = runtime.handle_command(CommandPacket.create(CommandType.TRACK, query="blue cup"))

        self.assertFalse(status.ack)
        self.assertEqual(runtime.state, SystemState.ERROR)

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

    def test_stop_command_holds_without_centering(self) -> None:
        runtime = EdgeRuntime(AppConfig())
        runtime.state_machine.state = SystemState.TRACKING
        runtime.controller.state.pan_deg = 20.0
        runtime.controller.state.tilt_deg = 10.0

        status = runtime.handle_command(CommandPacket.create(CommandType.STOP))

        self.assertTrue(status.ack)
        self.assertEqual(runtime.state, SystemState.IDLE)
        self.assertEqual(runtime.current_query, "")
        self.assertNotEqual(runtime.last_command.pan_deg, runtime.config.control.pan.center_deg)

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

    def test_scan_accepts_small_but_valid_center_candidate(self) -> None:
        config = AppConfig(scan=ScanConfig(confirmation_frames=1))
        runtime = EdgeRuntime(config)
        runtime.handle_command(CommandPacket.create(CommandType.TRACK, query="red cup"))

        runtime.handle_tracking_result(
            TrackingResult(
                packet="tracking_result",
                ts_req=now_us(),
                ts_resp=now_us(),
                bbox=BBox(310, 230, 330, 250),
                confidence=0.9,
                track_id=7,
                query="red cup",
            ),
            sensor_sample(),
        )

        self.assertEqual(runtime.state, SystemState.TRACKING)

    def test_pan_error_sign_can_be_reversed_for_mirrored_servo_mount(self) -> None:
        config = AppConfig(control=ControlConfig(pan_error_sign=-1.0))
        runtime = EdgeRuntime(config)
        runtime.state_machine.state = SystemState.TRACKING

        runtime.handle_tracking_result(
            TrackingResult(
                packet="tracking_result",
                ts_req=now_us(),
                ts_resp=now_us(),
                bbox=BBox(500, 220, 560, 280),
                confidence=0.9,
                track_id=7,
                query="red cup",
            ),
            sensor_sample(),
        )

        self.assertLess(runtime.last_command.pan_omega_deg_s, 0.0)

    def test_tracking_does_not_chase_suddenly_tiny_bbox(self) -> None:
        config = AppConfig(safety=SafetyConfig(bbox_area_shrink_threshold=0.25))
        runtime = EdgeRuntime(config)
        runtime.state_machine.state = SystemState.TRACKING
        runtime.handle_tracking_result(
            TrackingResult(
                packet="tracking_result",
                ts_req=now_us(),
                ts_resp=now_us(),
                bbox=BBox(260, 180, 380, 300),
                confidence=0.9,
                track_id=7,
                query="red cup",
            ),
            sensor_sample(),
        )
        before = runtime.last_command

        status = runtime.handle_tracking_result(
            TrackingResult(
                packet="tracking_result",
                ts_req=now_us(),
                ts_resp=now_us(),
                bbox=BBox(500, 220, 515, 235),
                confidence=0.9,
                track_id=7,
                query="red cup",
            ),
            sensor_sample(),
        )

        self.assertEqual(status.message, "vision bbox shrank too much between frames")
        self.assertLessEqual(abs(runtime.last_command.pan_omega_deg_s), abs(before.pan_omega_deg_s))

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

    def test_previous_query_result_is_ignored_after_restart(self) -> None:
        runtime = EdgeRuntime(AppConfig())
        runtime.handle_command(CommandPacket.create(CommandType.TRACK, query="red cup"))
        runtime.handle_command(CommandPacket.create(CommandType.TRACK, query="blue cup"))

        status = runtime.handle_tracking_result(
            TrackingResult(
                packet="tracking_result",
                ts_req=now_us(),
                ts_resp=now_us(),
                bbox=BBox(300, 220, 340, 260),
                confidence=0.9,
                track_id=7,
                query="red cup",
            ),
            sensor_sample(),
        )

        self.assertFalse(status.ack)
        self.assertEqual(status.message, "tracking result query does not match current target")
        self.assertEqual(runtime.state, SystemState.SCAN)

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

    def test_status_light_follows_runtime_state_and_infrared_override(self) -> None:
        light = SimulatedStatusLight()
        runtime = EdgeRuntime(AppConfig(), status_light=light)
        runtime.state_machine.state = SystemState.TRACKING
        runtime.control_step(sensor_sample=sensor_sample())
        self.assertEqual(light.last_state, SystemState.TRACKING)
        self.assertEqual(light.last_color, GREEN)

        runtime.control_step(
            sensor_sample=SensorSample(ts=now_us(), infrared_active=True),
        )

        self.assertEqual(runtime.state, SystemState.SAFE_HOLD)
        self.assertEqual(light.last_state, SystemState.SAFE_HOLD)
        self.assertEqual(light.last_color, BLUE)

        runtime.control_step(sensor_sample=sensor_sample())

        self.assertEqual(light.last_color, RED)

    def test_tracking_loss_sets_status_light_red(self) -> None:
        light = SimulatedStatusLight()
        runtime = EdgeRuntime(AppConfig(), status_light=light)
        runtime.state_machine.state = SystemState.TRACKING
        runtime.current_query = "red cup"

        status = runtime.handle_tracking_result(
            TrackingResult.empty(now_us(), query="red cup"),
            sensor_sample(),
        )

        self.assertEqual(status.message, "vision result is missing")
        self.assertEqual(light.last_color, RED)

    def test_ultrasonic_stable_jump_is_not_used_for_servo_control(self) -> None:
        light = SimulatedStatusLight()
        config = AppConfig(
            safety=SafetyConfig(
                consecutive_frames=2,
                pixel_jump_threshold=40.0,
                ultrasonic_stable_delta_threshold_mm=20.0,
            )
        )
        runtime = EdgeRuntime(config, status_light=light)
        runtime.state_machine.state = SystemState.TRACKING
        runtime.current_query = "red cup"
        base_ts = now_us()
        jump_ts = base_ts + 300_000
        baseline = TrackingResult(
            packet="tracking_result",
            ts_req=base_ts,
            ts_resp=base_ts,
            bbox=BBox(300, 220, 340, 260),
            confidence=0.9,
            track_id=7,
            query="red cup",
        )
        runtime.handle_tracking_result(
            baseline,
            SensorSample(ts=now_us(), ultrasonic_mm=650.0),
            received_ts_us=base_ts + 10_000,
        )
        previous_valid = runtime.last_valid_result

        status = runtime.handle_tracking_result(
            TrackingResult(
                packet="tracking_result",
                ts_req=jump_ts,
                ts_resp=jump_ts,
                bbox=BBox(20, 220, 60, 260),
                confidence=0.9,
                track_id=7,
                query="red cup",
            ),
            SensorSample(ts=now_us(), ultrasonic_mm=655.0),
            received_ts_us=jump_ts + 10_000,
        )

        self.assertEqual(
            status.message,
            "vision center jumped but ultrasonic distance stayed stable",
        )
        self.assertEqual(runtime.state, SystemState.TRACKING)
        self.assertEqual(runtime.last_valid_result, previous_valid)
        self.assertEqual(light.last_color, RED)

    def test_redetect_command_sets_one_shot_request(self) -> None:
        runtime = EdgeRuntime(AppConfig())
        runtime.state_machine.state = SystemState.SAFE_HOLD
        runtime.safe_hold_started_us = now_us()
        status = runtime.handle_command(CommandPacket.create(CommandType.REDETECT))
        self.assertTrue(status.ack)
        self.assertTrue(runtime.consume_redetect_request())
        self.assertFalse(runtime.consume_redetect_request())

    def test_centering_state_does_not_keep_requesting_vision_frames(self) -> None:
        runtime = EdgeRuntime(AppConfig())
        runtime.handle_command(CommandPacket.create(CommandType.TRACK, query="smartphone"))
        runtime.state_machine.apply(Event.SCAN_FAILED)

        query, redetect = runtime.next_frame_request()

        self.assertEqual(runtime.state, SystemState.CENTERING)
        self.assertEqual(query, "")
        self.assertFalse(redetect)

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
