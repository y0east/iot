import time
from iot_servo_tracker.control.servo import NativeSysfsServoDriver
from iot_servo_tracker.config.settings import load_config
from iot_servo_tracker.common.packets import ServoCommand

def main():
    print("====================================")
    print("두 모터 모두 스무스하게 움직이는지 테스트 시작...")
    config = load_config("config/settings.toml")
    servo = NativeSysfsServoDriver(config.control.pan, config.control.tilt)

    # 0도 -> 45도 -> -45도 -> 0도 순서로 아주 부드럽게 이동합니다
    for angle in list(range(0, 45, 1)) + list(range(45, -45, -1)) + list(range(-45, 0, 1)):
        servo.apply(ServoCommand(pan_deg=angle, tilt_deg=angle))
        time.sleep(0.015)
        
    print("테스트 완료! 지터 없이 부드러운가요?")
    print("====================================")

if __name__ == "__main__":
    main()
