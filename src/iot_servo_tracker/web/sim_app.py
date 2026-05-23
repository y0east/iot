"""Streamlit dashboard for the in-process full-stack simulation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from iot_servo_tracker.common.config import load_config
from iot_servo_tracker.sim.full_stack import (
    SCENARIOS,
    FullStackSimulationOptions,
    run_full_stack_simulation,
)


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError("Install optional dependency: pip install '.[web]'") from exc

    st.set_page_config(page_title="IoT Servo Tracker Simulation", layout="wide")
    config_path = Path("config/settings.toml")
    config = load_config(config_path if config_path.exists() else None)

    st.title("IoT Servo Tracker Simulation")
    st.caption("Web, MQTT, edge, ZMQ, vision, and status return path")

    with st.sidebar.form("simulation_form"):
        query = st.text_input("Target", value="red cup")
        scenario = st.selectbox("Scenario", SCENARIOS, index=0)
        steps = st.slider("Frames", min_value=10, max_value=160, value=80, step=5)
        print_every = st.slider("Table stride", min_value=1, max_value=20, value=5)
        webcam = st.checkbox("Use webcam frames")
        production = st.checkbox("Use real WeDetect + YOLO")
        skip_preflight = st.checkbox("Skip production preflight")
        wedetect_repo = st.text_input("WEDETECT_REPO", value="")
        yolo_model = st.text_input("YOLO model", value="")
        tracker = st.text_input("Tracker", value="")
        camera_index = st.number_input("Camera index", min_value=0, value=0, step=1)
        submitted = st.form_submit_button("Run")

    if not submitted and "full_stack_events" not in st.session_state:
        submitted = True

    if submitted:
        try:
            events = run_full_stack_simulation(
                config,
                FullStackSimulationOptions(
                    query=query,
                    scenario=scenario,
                    steps=steps,
                    print_every=print_every,
                    webcam=webcam,
                    camera_index=int(camera_index),
                    production=production,
                    preflight_production=not skip_preflight,
                    wedetect_repo=wedetect_repo or None,
                    yolo_model=yolo_model or None,
                    tracker=tracker or None,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            return
        st.session_state.full_stack_events = events

    events = st.session_state.get("full_stack_events", [])
    if not events:
        st.info("No simulation events yet.")
        return

    latest = events[-1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Web", latest.web_view)
    col2.metric("Edge", latest.edge_state)
    col3.metric("Target", latest.web_target)
    col4.metric("Vision", latest.vision_mode)

    rows = [asdict(event) for event in events]
    st.subheader("Timeline")
    st.dataframe(rows[:: max(1, print_every)], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
