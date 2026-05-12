"""Streamlit web control UI."""

from __future__ import annotations

import json
from pathlib import Path

from iot_servo_tracker.common.config import load_config
from iot_servo_tracker.common.packets import CommandPacket, CommandType, PacketError
from iot_servo_tracker.comms.mqtt import build_paho_client


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError("Install optional dependency: pip install '.[web]'") from exc

    st.set_page_config(page_title="IoT Servo Tracker", layout="wide")
    config_path = Path("config/settings.toml")
    config = load_config(config_path if config_path.exists() else None)

    st.title("IoT Servo Tracker")
    st.caption("Web command layer. GPIO/PWM control stays on the Raspberry Pi edge process.")

    if "last_command" not in st.session_state:
        st.session_state.last_command = None
    if "last_status" not in st.session_state:
        st.session_state.last_status = None

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Live Video")
        st.info("Connect the processed server stream here.")

    with right:
        st.subheader("Command")
        with st.form("track_form"):
            query = st.text_input("Target", placeholder="red cup")
            scan_range = st.slider("Scan range (deg)", 10, 90, int(config.scan.range_deg))
            max_speed = st.slider("Max speed (deg/s)", 5, 60, int(config.control.max_speed_deg_s))
            submitted = st.form_submit_button("Start tracking")
        if submitted:
            _publish_or_show(st, config, CommandType.TRACK, query, scan_range, max_speed)

        c1, c2, c3 = st.columns(3)
        if c1.button("Stop"):
            _publish_or_show(st, config, CommandType.STOP)
        if c2.button("Redetect"):
            _publish_or_show(st, config, CommandType.REDETECT)
        if c3.button("Center"):
            _publish_or_show(st, config, CommandType.CENTER)

        st.subheader("Last command")
        st.json(st.session_state.last_command or {})
        st.subheader("Last status")
        st.json(st.session_state.last_status or {})


def _publish_or_show(st, config, cmd_type: CommandType, query: str = "", scan=45, speed=20) -> None:
    try:
        command = CommandPacket.create(cmd_type, query=query, scan_range_deg=scan, max_speed_deg_s=speed)
    except PacketError as exc:
        st.error(str(exc))
        return

    payload = command.to_json()
    st.session_state.last_command = json.loads(payload)
    try:
        client = build_paho_client(config.mqtt)
        client.publish(config.mqtt.command_topic, payload, qos=1)
        client.disconnect()
        st.success(f"Published {command.cmd_type} command")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"MQTT publish skipped: {exc}")


if __name__ == "__main__":
    main()
