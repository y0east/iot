"""Command-line entry points for the edge runtime."""

from __future__ import annotations

import argparse
import json
import time

from iot_servo_tracker.common.config import load_config
from iot_servo_tracker.common.packets import CommandPacket, CommandType, SensorSample
from iot_servo_tracker.common.timebase import now_us
from iot_servo_tracker.comms.mqtt import MqttEdgeBridge
from iot_servo_tracker.comms.zmq_socket import ZmqEdgeTransport
from iot_servo_tracker.edge.camera import OpenCvCamera, SimulatedCamera
from iot_servo_tracker.edge.runtime import EdgeRuntime
from iot_servo_tracker.edge.sensors import RaspberryPiSensorReader, SimulatedSensorReader
from iot_servo_tracker.control.servo import Pca9685ServoDriver, SimulatedServoDriver
from iot_servo_tracker.server.vision import SimulatedVisionPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IoT servo tracker edge runtime")
    parser.add_argument("--config", default=None, help="Path to settings TOML")
    parser.add_argument("--simulate", action="store_true", help="Run a local simulation")
    parser.add_argument("--run", action="store_true", help="Run the MQTT/ZMQ edge loop")
    parser.add_argument("--query", default="red cup", help="Simulation target query")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--simulated-camera", action="store_true")
    parser.add_argument("--hardware-servo", action="store_true")
    parser.add_argument("--hardware-sensors", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.simulate:
        simulate(["--config", args.config] if args.config else [])
        return
    if args.run:
        run_edge(args)
        return
    config = load_config(args.config)
    servo = (
        Pca9685ServoDriver(config.control.pan, config.control.tilt)
        if args.hardware_servo
        else SimulatedServoDriver()
    )
    runtime = EdgeRuntime(config=config, servo=servo)
    status = runtime.handle_command(CommandPacket.create(CommandType.CENTER))
    print(status.to_json())


def run_edge(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    runtime = EdgeRuntime(config=config)
    transport = ZmqEdgeTransport(
        config.zmq.frame_connect_endpoint,
        config.zmq.result_connect_endpoint,
    )
    camera = (
        SimulatedCamera()
        if args.simulated_camera
        else OpenCvCamera(args.camera_index, config.camera.width, config.camera.height)
    )
    sensors = RaspberryPiSensorReader() if args.hardware_sensors else SimulatedSensorReader()
    bridge = MqttEdgeBridge(config.mqtt, runtime.handle_command)
    bridge.start()
    frame_index = 0
    last_status = runtime.last_status
    try:
        while True:
            ts_req = now_us()
            frame = camera.read_jpeg()
            if runtime.current_query:
                transport.send_frame(ts_req, runtime.current_query, frame, frame_index)
                frame_index += 1
            result = transport.recv_result(timeout_ms=1)
            sensor = sensors.read()
            if result is not None:
                last_status = runtime.handle_tracking_result(result, sensor)
            else:
                last_status = runtime.control_step(sensor_sample=sensor)
            bridge.publish_status(last_status)
            time.sleep(0.01)
    except KeyboardInterrupt:
        bridge.publish_status(runtime.handle_command(CommandPacket.create(CommandType.CENTER)))
    finally:
        bridge.stop()
        transport.close()
        camera.close()


def simulate(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run local edge/server simulation")
    parser.add_argument("--config", default=None)
    parser.add_argument("--query", default="red cup")
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    edge = EdgeRuntime(config=config)
    vision = SimulatedVisionPipeline(config.camera)
    command = CommandPacket.create(CommandType.TRACK, query=args.query)
    print(edge.handle_command(command).to_json())

    for step in range(args.steps):
        ts = now_us()
        edge.capture_frame(b"simulated-frame", ts_us=ts)
        result = vision.process_frame(ts_req=ts, query=args.query, frame_index=step)
        sensor = SensorSample(ts=now_us(), tof_mm=620.0, ultrasonic_mm=650.0)
        status = edge.handle_tracking_result(result, sensor, dt_s=0.033)
        if step % 20 == 0 or step == args.steps - 1:
            print(json.dumps({"step": step, "status": json.loads(status.to_json())}))
        time.sleep(0.001)


if __name__ == "__main__":
    main()
