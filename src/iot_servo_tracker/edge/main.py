"""Command-line entry points for the edge runtime."""

from __future__ import annotations

import argparse
import time

from iot_servo_tracker.common.config import load_config
from iot_servo_tracker.common.packets import CommandPacket, CommandType, SensorSample
from iot_servo_tracker.common.timebase import now_us
from iot_servo_tracker.comms.mqtt import MqttEdgeBridge
from iot_servo_tracker.comms.zmq_socket import ZmqEdgeTransport
from iot_servo_tracker.control.servo import Pca9685ServoDriver, SimulatedServoDriver, DirectGpioServoDriver, NativeSysfsServoDriver
from iot_servo_tracker.edge.camera import OpenCvCamera, SimulatedCamera, RpiCamVidCamera
from iot_servo_tracker.edge.runtime import EdgeRuntime
from iot_servo_tracker.edge.sensors import RaspberryPiSensorReader, SimulatedSensorReader


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IoT servo tracker edge runtime")
    parser.add_argument("--config", default=None, help="Path to settings TOML")
    parser.add_argument("--simulate", action="store_true", help="Run a local simulation")
    parser.add_argument("--run", action="store_true", help="Run the MQTT/ZMQ edge loop")
    parser.add_argument("--query", default="red cup", help="Simulation target query")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--rpicam", action="store_true", help="Use rpicam-vid for native Raspberry Pi Camera")
    parser.add_argument("--simulated-camera", action="store_true")
    parser.add_argument("--hardware-servo", action="store_true")
    parser.add_argument("--direct-servo", action="store_true", help="Connect servos directly to RPi GPIO without PCA9685")
    parser.add_argument("--native-pwm", action="store_true", help="Use zero-jitter sysfs hardware PWM")
    parser.add_argument("--hardware-sensors", action="store_true")
    parser.add_argument("--voice", action="store_true", help="Enable voice command NLU via microphone")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.simulate:
        simulate_args = ["--query", args.query]
        if args.config:
            simulate_args.extend(["--config", args.config])
        simulate(simulate_args)
        return
    if args.run:
        run_edge(args)
        return
    config = load_config(args.config)
    if args.hardware_servo:
        servo = Pca9685ServoDriver(config.control.pan, config.control.tilt)
    elif args.native_pwm:
        servo = NativeSysfsServoDriver(config.control.pan, config.control.tilt)
    elif args.direct_servo:
        servo = DirectGpioServoDriver(config.control.pan, config.control.tilt)
    else:
        servo = SimulatedServoDriver()
    runtime = EdgeRuntime(config=config, servo=servo)
    status = runtime.handle_command(CommandPacket.create(CommandType.CENTER))
    print(status.to_json())


def run_edge(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.hardware_servo:
        servo = Pca9685ServoDriver(config.control.pan, config.control.tilt)
    elif args.native_pwm:
        servo = NativeSysfsServoDriver(config.control.pan, config.control.tilt)
    elif args.direct_servo:
        servo = DirectGpioServoDriver(config.control.pan, config.control.tilt)
    else:
        servo = SimulatedServoDriver()
    runtime = EdgeRuntime(config=config, servo=servo)
    transport = ZmqEdgeTransport(
        config.zmq.frame_connect_endpoint,
        config.zmq.result_connect_endpoint,
        frame_snd_hwm=config.zmq.frame_snd_hwm,
        result_rcv_hwm=config.zmq.result_rcv_hwm,
    )
    if args.simulated_camera:
        camera = SimulatedCamera()
    elif args.rpicam:
        camera = RpiCamVidCamera(config.camera.width, config.camera.height)
    else:
        camera = OpenCvCamera(args.camera_index, config.camera.width, config.camera.height)
    sensors = RaspberryPiSensorReader() if args.hardware_sensors else SimulatedSensorReader()
    try:
        bridge = MqttEdgeBridge(config.mqtt, runtime.handle_command)
        bridge.start()
    except Exception as e:
        print(f"[Warning] Failed to connect to MQTT ({e}). Running without MQTT status reporting.")
        bridge = None

    voice_commander = None
    if args.voice:
        try:
            from iot_servo_tracker.edge.audio import VoiceCommander
            voice_commander = VoiceCommander(runtime.handle_command)
            voice_commander.start()
        except ImportError as e:
            print(f"[Warning] Voice command dependencies missing: {e}")

    frame_index = 0
    last_status = runtime.last_status
    last_loop_s = time.monotonic()
    last_control_s = last_loop_s
    last_frame_send_s = 0.0
    last_sent_frame_sequence = None
    min_frame_interval_s = 1.0 / 15.0
    last_status_publish_s = 0.0
    last_status_event_key = None
    status_publish_interval_s = 0.10
    try:
        while True:
            loop_s = time.monotonic()
            dt_s = _bounded_dt_s(loop_s - last_loop_s)
            last_loop_s = loop_s
            ts_req = now_us()
            frame = camera.read_jpeg()
            query, redetect = runtime.next_frame_request()

            # 카메라 프레임레이트(15FPS)에 맞춰 초당 15번만 전송하도록 제한 (TCP Bufferbloat 렉 방지)
            if query and (loop_s - last_frame_send_s >= 0.066):
                if transport.send_frame(
                    ts_req,
                    query,
                    frame,
                    frame_index,
                    redetect=redetect,
                ):
                    frame_index += 1
                    last_frame_send_s = loop_s
            result = transport.recv_result(timeout_ms=1)
            sensor = _read_sensor_sample(sensors)
            control_dt_s = _bounded_dt_s(loop_s - last_control_s)
            if result is not None:
                last_status = runtime.handle_tracking_result(
                    result,
                    sensor,
                    dt_s=control_dt_s,
                    received_ts_us=now_us(),
                )
                last_control_s = loop_s
            else:
                control_states = {"SCAN", "SAFE_HOLD", "LIMITED_RESCAN", "CENTERING"}
                if runtime.state.value in control_states:
                    last_status = runtime.control_step(
                        dt_s=control_dt_s,
                        sensor_sample=sensor,
                    )
                    last_control_s = loop_s
                else:
                    last_status = runtime.control_step(dt_s=dt_s, sensor_sample=sensor)
            status_event_key = (
                last_status.system_state,
                last_status.ack,
                last_status.message,
            )
            if (
                bridge is not None and (
                    loop_s - last_status_publish_s >= status_publish_interval_s
                    or status_event_key != last_status_event_key
                )
            ):
                bridge.publish_status(last_status)
                last_status_publish_s = loop_s
                last_status_event_key = status_event_key
            time.sleep(0.01)
    except KeyboardInterrupt:
        if bridge is not None:
            bridge.publish_status(runtime.handle_command(CommandPacket.create(CommandType.CENTER)))
    finally:
        if bridge is not None:
            bridge.stop()
        if voice_commander is not None:
            voice_commander.stop()
        transport.close()
        camera.close()


def _read_sensor_sample(sensors) -> SensorSample:
    try:
        return sensors.read()
    except Exception:
        return SensorSample.empty()


def _bounded_dt_s(dt_s: float) -> float:
    return max(0.001, min(dt_s, 0.25))


def simulate(argv: list[str] | None = None) -> None:
    from iot_servo_tracker.sim.offline import main as simulate_offline

    simulate_offline(argv)


if __name__ == "__main__":
    main()
