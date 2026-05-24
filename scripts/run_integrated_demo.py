#!/usr/bin/env python3
"""Run a full integrated system test with real webcam, WeDetect, and YOLO.

This script launches the three main components of the system:
1. Streamlit Web app
2. RTX Server (WeDetect + YOLO) via ZMQ
3. Simulated Edge loop (Webcam -> ZMQ -> Server, status -> MQTT)

It also hosts a lightweight HTTP MJPEG stream at http://127.0.0.1:8000/stream.mjpg
to feed the annotated camera frame back into the Streamlit interface.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import threading
import tempfile
import socket
import uuid
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# Add src to Python Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from iot_servo_tracker.common.config import load_config
from iot_servo_tracker.common.packets import SensorSample
from iot_servo_tracker.common.timebase import now_us
from iot_servo_tracker.comms.mqtt import MqttEdgeBridge
from iot_servo_tracker.comms.zmq_socket import ZmqEdgeTransport
from iot_servo_tracker.edge.camera import OpenCvCamera
from iot_servo_tracker.edge.sensors import SimulatedSensorReader
from iot_servo_tracker.edge.runtime import EdgeRuntime
from iot_servo_tracker.control.servo import SimulatedServoDriver

# Global reference to hold the latest annotated JPEG frame
LATEST_ANNOTATED_FRAME: bytes | None = None
FRAME_LOCK = threading.Lock()


class MJPEGStreamHandler(BaseHTTPRequestHandler):
    """Serve MJPEG stream to Streamlit Web App."""
    def do_GET(self):
        global LATEST_ANNOTATED_FRAME
        if self.path != "/stream.mjpg":
            self.send_error(404, "File Not Found")
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with FRAME_LOCK:
                    frame = LATEST_ANNOTATED_FRAME
                if frame is None:
                    time.sleep(0.03)
                    continue
                header = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame)).encode() + b"\r\n"
                    b"\r\n"
                )
                self.wfile.write(header)
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                time.sleep(0.04)  # ~25 FPS limit
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            pass

    def log_message(self, format, *args):
        """Suppress per-request log spam."""
        pass


def resolve_mqtt_host(args) -> str:
    """Resolve the MQTT broker host. Probes localhost if not specified."""
    mqtt_host = args.mqtt_host
    if mqtt_host:
        return mqtt_host

    # Check if local broker is active on 1883
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.5)
        s.connect(("127.0.0.1", 1883))
        s.close()
        print("[+] Detected local MQTT broker on 127.0.0.1:1883")
        return "127.0.0.1"
    except Exception:
        raise RuntimeError(
            "No local MQTT broker detected on port 1883. "
            "Please start mosquitto or specify a broker using --mqtt-host."
        )


def build_demo_config(
    config_path: Path,
    temp_dir: str,
    mqtt_host: str,
    mqtt_topic_prefix: str | None,
    device: str,
    relax_safety: bool,
) -> Path:
    """Generate a modified config file in the temp directory, leaving the original intact."""
    if not config_path.exists():
        raise FileNotFoundError(f"Base configuration file not found at {config_path}")

    content = config_path.read_text(encoding="utf-8")
    new_lines = []
    in_web = False
    in_mqtt = False
    in_server = False
    in_safety = False

    for line in content.splitlines():
        stripped = line.strip()
        if stripped == "[web]":
            in_web = True
            in_mqtt = False
            in_server = False
            in_safety = False
        elif stripped == "[mqtt]":
            in_web = False
            in_mqtt = True
            in_server = False
            in_safety = False
        elif stripped == "[server]":
            in_web = False
            in_mqtt = False
            in_server = True
            in_safety = False
        elif stripped == "[safety]":
            in_web = False
            in_mqtt = False
            in_server = False
            in_safety = True
        elif stripped.startswith("[") and stripped.endswith("]"):
            in_web = False
            in_mqtt = False
            in_server = False
            in_safety = False

        if in_web and stripped.startswith("processed_stream_url"):
            new_lines.append('processed_stream_url = "http://127.0.0.1:8000/stream.mjpg"')
        elif in_mqtt and stripped.startswith("host"):
            new_lines.append(f'host = "{mqtt_host}"')
        elif in_mqtt and stripped.startswith("command_topic") and mqtt_topic_prefix:
            new_lines.append(f'command_topic = "{mqtt_topic_prefix}/command"')
        elif in_mqtt and stripped.startswith("status_topic") and mqtt_topic_prefix:
            new_lines.append(f'status_topic = "{mqtt_topic_prefix}/status"')
        elif in_server and stripped.startswith("wedetect_device"):
            new_lines.append(f'wedetect_device = "{device}"')
        elif in_safety and stripped.startswith("pixel_jump_threshold") and relax_safety:
            new_lines.append('pixel_jump_threshold = 999.0')
        else:
            new_lines.append(line)

    temp_config_path = Path(temp_dir) / "settings.integrated.toml"
    temp_config_path.write_text("\n".join(new_lines), encoding="utf-8")
    return temp_config_path


def start_server(config_path: Path, env: dict) -> subprocess.Popen:
    """Launch the RTX Server process."""
    print("[+] Launching RTX Server process via ZMQ...")
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "iot_servo_tracker.server.main",
            "--config",
            str(config_path),
            "--serve",
            "--production",
        ],
        env=env,
    )


def start_streamlit(env: dict) -> subprocess.Popen:
    """Launch the Streamlit Web App process."""
    print("[+] Launching Streamlit Web App...")
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "src/iot_servo_tracker/web/app.py",
        ],
        env=env,
    )


def terminate_process(proc: subprocess.Popen | None, name: str, timeout: float = 5.0) -> None:
    """Cleanly terminate or force kill a subprocess."""
    if proc is None:
        return
    if proc.poll() is not None:
        return

    print(f"[*] Terminating {name}...")
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[!] {name} did not exit within {timeout}s; killing...")
        proc.kill()
        proc.wait(timeout=timeout)


def draw_overlay(
    img,
    state,
    current_query,
    last_bbox,
    last_confidence,
    last_track_id,
    is_predicting=False,
):
    """Draw tracking overlay on OpenCV image frame."""
    import cv2
    h, w = img.shape[:2]
    if current_query:
        # Semi-transparent dark banner for text readability
        img = img.copy()  # Ensure we have a clean copy to prevent trailing artifacts
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

        # Row 1: Query name (white)
        cv2.putText(img, f"Query: {current_query}", (30, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Row 2: State with color coding
        state_colors = {
            "TRACKING": (0, 255, 0),       # green
            "SCAN": (255, 200, 0),          # cyan-ish
            "SAFE_HOLD": (0, 165, 255),     # orange
            "LIMITED_RESCAN": (0, 100, 255), # deep orange
            "DELAY_COMPENSATION": (255, 255, 0),  # cyan
        }
        state_color = state_colors.get(state.value, (200, 200, 200))
        
        display_state_text = state.value
        if state.value == "SCAN":
            display_state_text = "SCAN (Panning...)"
        elif state.value == "DELAY_COMPENSATION":
            display_state_text = "TRACKING (Syncing...)"
            
        cv2.putText(img, f"State: {display_state_text}", (30, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)

        # Row 3: Additional info
        if state.value in ("SAFE_HOLD", "LIMITED_RESCAN"):
            cv2.putText(img, "Holding - waiting for stable re-detection...", (30, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

        # Draw BBox: use last valid bbox while TRACKING or SAFE_HOLD
        show_bbox = False
        if state.value in ("TRACKING", "DELAY_COMPENSATION", "SAFE_HOLD"):
            show_bbox = last_bbox is not None

        if show_bbox and last_bbox is not None:
            x1, y1, x2, y2 = map(int, [last_bbox.x1, last_bbox.y1,
                                        last_bbox.x2, last_bbox.y2])
            if is_predicting:
                color = (255, 0, 255)  # purple
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                cv2.putText(img, "PREDICTING...", (x1, max(15, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            elif state.value == "SAFE_HOLD":
                # Dimmed box during SAFE_HOLD
                color = (0, 165, 255)  # orange
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"HOLD Conf: {last_confidence:.2f}",
                            (x1, max(15, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                # Active tracking box
                is_yolo = last_track_id is not None
                color = (0, 255, 0) if is_yolo else (255, 100, 0)
                mode_text = f"YOLO ({last_track_id})" if is_yolo else "WeDetect"
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cv2.putText(img, f"{mode_text} Conf: {last_confidence:.2f}",
                            (x1, max(15, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    else:
        # IDLE state
        img = img.copy()
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
        cv2.putText(img, "IDLE - Enter target on Streamlit Web UI", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img


def edge_loop_thread(args, config, stop_event: threading.Event):
    global LATEST_ANNOTATED_FRAME
    import cv2
    import numpy as np

    print("[+] Edge thread: Starting camera & MQTT bridge...")
    servo = SimulatedServoDriver()
    runtime = EdgeRuntime(config=config, servo=servo)
    
    # Initialize ZMQ transport
    transport = ZmqEdgeTransport(
        config.zmq.frame_connect_endpoint,
        config.zmq.result_connect_endpoint,
        frame_snd_hwm=config.zmq.frame_snd_hwm,
        result_rcv_hwm=config.zmq.result_rcv_hwm,
    )
    
    # Initialize camera and sensors
    if getattr(args, "camera_source", None):
        try:
            device_index = int(args.camera_source)
        except ValueError:
            print(f"[-] Invalid camera source: {args.camera_source}, using default index")
            device_index = args.camera_index
        camera = OpenCvCamera(device_index, config.camera.width, config.camera.height)
    else:
        camera = OpenCvCamera(args.camera_index, config.camera.width, config.camera.height)
    sensors = SimulatedSensorReader()

    # Initialize MQTT Bridge to receive Streamlit commands
    bridge = MqttEdgeBridge(config.mqtt, runtime.handle_command)
    bridge.start()

    frame_index = 0
    last_status = runtime.last_status
    last_loop_s = time.monotonic()
    last_display_bbox = None
    last_display_confidence = 0.0
    last_display_track_id = None

    print("[+] Edge loop is active and processing frames.")
    try:
        while not stop_event.is_set():
            loop_s = time.monotonic()
            dt_s = max(0.001, min(loop_s - last_loop_s, 0.25))
            last_loop_s = loop_s
            ts_req = now_us()

            # Read frame from webcam
            try:
                frame_bytes = camera.read_jpeg()
            except Exception as e:
                print(f"[-] Camera read failed: {e}")
                time.sleep(0.1)
                continue

            query, redetect = runtime.next_frame_request()
            
            # Send frame to RTX Server if query is set (Start Tracking from Streamlit)
            if query:
                sent = transport.send_frame(
                    ts_req,
                    query,
                    frame_bytes,
                    frame_index,
                    redetect=redetect,
                )
                if sent:
                    frame_index += 1

            # Recv tracking result from ZMQ server
            result = transport.recv_result(timeout_ms=1)
            sensor = sensors.read()

            if result is not None:
                last_status = runtime.handle_tracking_result(
                    result,
                    sensor,
                    dt_s=dt_s,
                    received_ts_us=now_us(),
                )
            else:
                last_status = runtime.control_step(dt_s=dt_s, sensor_sample=sensor)

            # Publish status back to Streamlit
            bridge.publish_status(last_status)

            # Draw overlay on the frame for MJPEG stream
            array = np.frombuffer(frame_bytes, dtype=np.uint8)
            img = cv2.imdecode(array, cv2.IMREAD_COLOR)
            if img is None:
                print("[-] Failed to decode JPEG frame")
                time.sleep(0.02)
                continue

            # Update last known valid bbox for visual persistence using latency-compensated box
            if runtime.last_valid_result is not None and runtime.last_valid_result.bbox is not None:
                last_display_bbox = runtime.last_valid_result.bbox
                last_display_confidence = runtime.last_valid_result.confidence
                last_display_track_id = runtime.last_valid_result.track_id

            state = runtime.state
            is_predicting = getattr(runtime, "is_predicting", False)
            
            # If we are currently predicting, override the display box
            display_bbox = last_display_bbox
            if is_predicting and getattr(runtime, "predicted_bbox", None) is not None:
                display_bbox = runtime.predicted_bbox

            img = draw_overlay(
                img,
                state,
                runtime.current_query,
                display_bbox,
                last_display_confidence,
                last_display_track_id,
                is_predicting=is_predicting,
            )

            # Encode annotated frame back to JPEG
            ok, encoded_img = cv2.imencode(".jpg", img)
            if not ok:
                print("[-] Failed to encode annotated frame")
                time.sleep(0.02)
                continue

            with FRAME_LOCK:
                LATEST_ANNOTATED_FRAME = encoded_img.tobytes()

            time.sleep(0.02)
    finally:
        bridge.stop()
        transport.close()
        camera.close()
        print("[+] Edge thread: Cleaned up resources.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Integrated Webcam/WeDetect/YOLO/Web UI Test Loop")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam device index")
    parser.add_argument("--camera-source", default=None, help="Path to video/image file or camera index (int)")
    parser.add_argument("--device", default="cuda:0", help="CUDA GPU or cpu device")
    parser.add_argument("--wedetect-repo", default="external/WeDetect", help="Path to WeChatCV/WeDetect")
    parser.add_argument("--mqtt-host", default="", help="MQTT Broker host (probes local, or pass custom)")
    parser.add_argument(
        "--mqtt-topic-prefix",
        default=None,
        help="MQTT topic prefix (defaults to settings.toml or random suffix if using public broker)",
    )
    parser.add_argument(
        "--relax-safety-for-demo",
        action="store_true",
        help="Relax pixel jump threshold for local demo only (sets pixel_jump_threshold to 999.0)",
    )
    args = parser.parse_args()

    if args.relax_safety_for_demo:
        print("\n" + "!" * 60)
        print("[!] WARNING: --relax-safety-for-demo option is active.")
        print("[!] This relaxes pixel_jump_threshold to 999.0 to bypass bbox jump safety.")
        print("[!] DO NOT USE THIS SETTING WITH REAL SERVO HARDWARE.")
        print("!" * 60 + "\n")

    # Resolve MQTT Host (throws error if not found and no override given)
    try:
        mqtt_host = resolve_mqtt_host(args)
    except RuntimeError as e:
        print(f"[-] Error: {e}", file=sys.stderr)
        return 1

    # Determine topic prefix
    topic_prefix = args.mqtt_topic_prefix
    if not topic_prefix and mqtt_host != "127.0.0.1" and mqtt_host != "localhost":
        # Generate unique prefix for external broker to avoid collision
        topic_prefix = f"iot_servo_tracker/{uuid.uuid4().hex[:8]}"
        print(f"[+] External MQTT broker. Using unique topic prefix: {topic_prefix}")

    config_path = ROOT / "config" / "settings.toml"
    # If the user has IOT_CONFIG env var set already, use it as baseline
    config_env = os.getenv("IOT_CONFIG")
    if config_env:
        config_path = Path(config_env)

    # Set up WeDetect repository path
    repo_path = Path(args.wedetect_repo).expanduser().resolve()
    if not (repo_path / "infer_wedetect_ref.py").exists():
        print(f"[-] Error: WeDetect repository path not found: {repo_path}", file=sys.stderr)
        return 1

    # Create temporary directory to store our modified settings.toml
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            temp_config_path = build_demo_config(
                config_path=config_path,
                temp_dir=temp_dir,
                mqtt_host=mqtt_host,
                mqtt_topic_prefix=topic_prefix,
                device=args.device,
                relax_safety=args.relax_safety_for_demo,
            )
            print(f"[+] Created temporary runtime configuration at: {temp_config_path}")
        except Exception as e:
            print(f"[-] Failed to generate temporary configuration: {e}", file=sys.stderr)
            return 1

        # Load final configuration for edge runtime thread
        config = load_config(temp_config_path)

        # Set environment variables for child processes
        os.environ["IOT_CONFIG"] = str(temp_config_path)
        os.environ["WEDETECT_REPO"] = str(repo_path)

        server_env = os.environ.copy()
        server_env["PYTHONPATH"] = str(SRC)

        # 3. Start RTX Server Process
        server_process = start_server(temp_config_path, server_env)

        # Startup health check for server
        time.sleep(2.0)
        if server_process.poll() is not None:
            print(f"[-] Error: RTX server process failed to start (exit code: {server_process.returncode})", file=sys.stderr)
            terminate_process(server_process, "RTX Server")
            return 1

        # 4. Start HTTP MJPEG Streamer Thread
        mjpeg_server = ThreadingHTTPServer(("127.0.0.1", 8000), MJPEGStreamHandler)
        mjpeg_thread = threading.Thread(target=mjpeg_server.serve_forever, daemon=True)
        mjpeg_thread.start()

        # 5. Start Edge loop thread
        stop_event = threading.Event()
        edge_thread = threading.Thread(
            target=edge_loop_thread,
            args=(args, config, stop_event),
            daemon=True,
        )
        edge_thread.start()

        # 6. Start Streamlit Web UI Process
        web_env = os.environ.copy()
        web_env["PYTHONPATH"] = str(SRC)
        streamlit_process = start_streamlit(web_env)

        # Startup health check for Streamlit
        time.sleep(1.0)
        if streamlit_process.poll() is not None:
            print(f"[-] Error: Streamlit Web UI process failed to start (exit code: {streamlit_process.returncode})", file=sys.stderr)
            stop_event.set()
            terminate_process(streamlit_process, "Streamlit Web App")
            terminate_process(server_process, "RTX Server")
            mjpeg_server.shutdown()
            mjpeg_server.server_close()
            return 1

        print("\n" + "="*60)
        print("ALL SERVICES INITIALIZED!")
        print("- RTX Server running (ZMQ)")
        print("- Edge loop running (Webcam -> ZMQ)")
        print("- MJPEG Stream server running at http://127.0.0.1:8000/stream.mjpg")
        print("- Streamlit Web UI starting (usually at http://localhost:8501)")
        print("Press Ctrl+C to stop all services and exit.")
        print("="*60 + "\n")

        try:
            while True:
                # Monitor running processes
                if server_process.poll() is not None:
                    print(f"[-] Alert: RTX Server process died early (exit code: {server_process.returncode})")
                    break
                if streamlit_process.poll() is not None:
                    print(f"[-] Alert: Streamlit process died early (exit code: {streamlit_process.returncode})")
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[*] Stopping all services...")
        finally:
            # Clean up
            stop_event.set()
            
            terminate_process(streamlit_process, "Streamlit Web App", timeout=5.0)
            terminate_process(server_process, "RTX Server", timeout=5.0)

            print("[*] Shutting down MJPEG streaming server...")
            mjpeg_server.shutdown()
            mjpeg_server.server_close()

            if edge_thread.is_alive():
                print("[*] Waiting for Edge thread to exit...")
                edge_thread.join(timeout=5.0)

            print("[+] Cleanup complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
