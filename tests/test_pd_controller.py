import unittest

from iot_servo_tracker.common.config import AppConfig
from iot_servo_tracker.common.packets import BBox
from iot_servo_tracker.control.pd_controller import PDServoController


class PDControllerTests(unittest.TestCase):
    def test_center_detection_stays_near_center(self) -> None:
        config = AppConfig()
        controller = PDServoController(config)
        bbox = BBox(300, 220, 340, 260)
        command = controller.update(bbox, dt_s=0.033)
        self.assertAlmostEqual(command.pan_deg, 0.0, delta=0.2)
        self.assertAlmostEqual(command.tilt_deg, 0.0, delta=0.2)

    def test_right_side_detection_commands_pan_motion(self) -> None:
        config = AppConfig()
        controller = PDServoController(config)
        bbox = BBox(500, 220, 560, 280)
        command = controller.update(bbox, dt_s=0.033)
        self.assertGreater(command.pan_omega_deg_s, 0.0)
        self.assertLessEqual(abs(command.pan_omega_deg_s), config.control.max_speed_deg_s)

    def test_soft_stop_limits_velocity_change(self) -> None:
        config = AppConfig()
        controller = PDServoController(config)
        controller.update(BBox(500, 220, 560, 280), dt_s=0.033)
        before = controller.state.pan_omega_deg_s
        command = controller.soft_stop(dt_s=0.033)
        self.assertLess(abs(command.pan_omega_deg_s), abs(before))


if __name__ == "__main__":
    unittest.main()
