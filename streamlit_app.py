"""
================================================================================
     CLEAR-DRIVE AI : NEXT-GEN PREDICTIVE V2X COCKPIT (STREAMLIT CLOUD)
================================================================================
Streamlit Community Cloud Deployment Entrypoint
Combines YOLOv8 vehicle kinematics, atmospheric dehazing, Retinex night vision,
AR lane laser guidance, pothole detection, and real-time ADAS telemetry.
================================================================================
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import cv2
import time
import numpy as np
import streamlit as st
from engine import OmniVisionEngine

# Page configuration with Cyber Automotive styling
st.set_page_config(
    page_title="ClearDrive AI | Automotive ADAS Cockpit",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cyberpunk HUD CSS
st.markdown("""
<style>
    .main {
        background-color: #050811;
        color: #e0f2fe;
    }
    .stApp {
        background: linear-gradient(180deg, #050811 0%, #0a0f1d 100%);
    }
    .hud-title {
        font-family: 'Courier New', monospace;
        color: #00f3ff;
        font-size: 2.0rem;
        font-weight: 900;
        letter-spacing: 2px;
        text-shadow: 0 0 15px rgba(0, 243, 255, 0.6);
        margin-bottom: 2px;
    }
    .hud-sub {
        color: #64748b;
        font-size: 0.85rem;
        letter-spacing: 1px;
        margin-bottom: 20px;
    }
    .metric-card {
        background: rgba(10, 20, 40, 0.7);
        border: 1px solid rgba(0, 243, 255, 0.3);
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.1);
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 900;
        font-family: 'Courier New', monospace;
    }
    .val-good { color: #00ff66; text-shadow: 0 0 10px rgba(0, 255, 102, 0.5); }
    .val-warn { color: #ffaa00; text-shadow: 0 0 10px rgba(255, 170, 0, 0.5); }
    .val-danger { color: #ff3344; text-shadow: 0 0 10px rgba(255, 51, 68, 0.5); }
    .val-cyan { color: #00f3ff; text-shadow: 0 0 10px rgba(0, 243, 255, 0.5); }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_engine():
    """Initializes and caches the OmniVision perception engine."""
    return OmniVisionEngine(yolo_model='yolov8n.pt')


def main():
    st.markdown('<div class="hud-title">🚗 CLEAR-DRIVE AI // OMNI-VISION COCKPIT</div>', unsafe_allow_html=True)
    st.markdown('<div class="hud-sub">PREDICTIVE V2X PERCEPTION ENGINE // REAL-TIME ACCIDENT PREVENTION SUITE</div>', unsafe_allow_html=True)

    engine = load_engine()

    # Video Sources
    video_options = {
        "🚗 Demo 8: Highway Traffic & Metric Kinematics": "traffic_dashcam.mp4",
        "🌫️ Demo 1: Dense Himalayan Fog": "foggy_dashcam.mp4",
        "🌧️ Demo 2: Heavy Monsoon Rain": "indian_rain.mp4",
        "🌙 Demo 3: Dark Night Highway": "night_dashcam.mp4",
        "☀️ Demo 4: High-Beam Glare & Snow": "glare_dashcam.mp4",
        "⚠️ Demo 6: Urban Road Potholes": "pothole_dashcam.mp4"
    }

    # Sidebar: Perception Matrix & Tactical Controls
    st.sidebar.markdown("### 🎛️ PERCEPTION CONTROLS")
    selected_label = st.sidebar.selectbox("Select Road Benchmark Stream:", list(video_options.keys()), index=0)
    video_path = video_options[selected_label]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👁️ MULTI-FEATURE MATRIX")

    col_s1, col_s2 = st.sidebar.columns(2)
    with col_s1:
        feat_lanes = st.checkbox("🛣️ AR Lanes", value=True)
        feat_radar = st.checkbox("🚗 YOLO Radar", value=True)
        feat_potholes = st.checkbox("⚠️ Potholes", value=True)
        feat_fog = st.checkbox("🌫️ Dehaze", value=False)
    with col_s2:
        feat_night = st.checkbox("🌙 Night Vision", value=False)
        feat_glare = st.checkbox("😎 Anti-Glare", value=False)
        feat_lidar = st.checkbox("🌐 Cyber-LIDAR", value=False)
        feat_thermal = st.checkbox("🔥 Thermal", value=False)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎚️ DISPLAY MODES")
    split_view = st.sidebar.checkbox("Side-by-Side (Raw vs AI)", value=False)
    raw_only = st.sidebar.checkbox("Raw Sensor Bypass", value=False)
    run_stream = st.sidebar.toggle("▶️ Live Perception Stream", value=True)

    features = {
        'fog': feat_fog,
        'night': feat_night,
        'lidar': feat_lidar,
        'thermal': feat_thermal,
        'lanes': feat_lanes,
        'glare': feat_glare,
        'potholes': feat_potholes,
        'radar': feat_radar,
        'raw': raw_only
    }

    # Top KPI Metrics Strip
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        kpi_speed = st.empty()
    with m2:
        kpi_closest = st.empty()
    with m3:
        kpi_status = st.empty()
    with m4:
        kpi_road = st.empty()

    # Viewport for live perception feed
    stream_placeholder = st.empty()
    diag_placeholder = st.empty()

    if not os.path.exists(video_path):
        st.error(f"Video file '{video_path}' not found. Please verify repository assets.")
        return

    cap = cv2.VideoCapture(video_path)

    prev_time = time.time()
    frame_idx = 0

    while run_stream and cap.isOpened():
        ret, raw_frame = cap.read()
        if not ret or raw_frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_idx += 1
        # Process every frame with size 640x360 for high FPS
        frame = cv2.resize(raw_frame, (640, 360), interpolation=cv2.INTER_LINEAR)

        start_t = time.time()
        dashboard_frame, emergency_brake, min_ttc, tele = engine.process(
            frame,
            mode='auto',
            active_features=features,
            split_view=split_view,
            active_video_name=video_path,
            live_speed=None
        )
        latency_ms = (time.time() - start_t) * 1000.0

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-5)
        prev_time = now

        # Convert BGR to RGB for Streamlit rendering
        frame_rgb = cv2.cvtColor(dashboard_frame, cv2.COLOR_BGR2RGB)
        stream_placeholder.image(frame_rgb, use_container_width=True)

        # Update KPI Cards every 3 frames to conserve compute
        if frame_idx % 3 == 0:
            spd = tele.get("speed_kmh", 74)
            limit = tele.get("speed_limit", 80)
            overspeed = tele.get("overspeed", False)
            spd_class = "val-danger" if overspeed else "val-good"

            kpi_speed.markdown(f"""
            <div class="metric-card">
                <div style="color: #64748b; font-size: 0.75rem;">RADAR SPEED</div>
                <div class="metric-val {spd_class}">{spd} <span style="font-size: 0.9rem;">km/h</span></div>
                <div style="color: #94a3b8; font-size: 0.70rem;">LIMIT: {limit} km/h</div>
            </div>
            """, unsafe_allow_html=True)

            closest_val = tele.get("closest_vehicle", "--")
            dist_class = "val-danger" if (isinstance(closest_val, (int, float)) and closest_val < 8) else "val-cyan"
            kpi_closest.markdown(f"""
            <div class="metric-card">
                <div style="color: #64748b; font-size: 0.75rem;">CLOSEST TARGET</div>
                <div class="metric-val {dist_class}">{closest_val}</div>
                <div style="color: #94a3b8; font-size: 0.70rem;">VRU / VEHICLE TRACKER</div>
            </div>
            """, unsafe_allow_html=True)

            brake = tele.get("brake_alert", False)
            brake_str = "⚠️ BRAKE ARMED" if brake else "OPTIMAL [SAFE]"
            brake_class = "val-danger" if brake else "val-good"
            kpi_status.markdown(f"""
            <div class="metric-card">
                <div style="color: #64748b; font-size: 0.75rem;">ADAS COLLISION AVOIDANCE</div>
                <div class="metric-val {brake_class}">{brake_str}</div>
                <div style="color: #94a3b8; font-size: 0.70rem;">TTC: {tele.get('ttc', 0.0)}s</div>
            </div>
            """, unsafe_allow_html=True)

            kpi_road.markdown(f"""
            <div class="metric-card">
                <div style="color: #64748b; font-size: 0.75rem;">ENVIRONMENT PROFILE</div>
                <div class="metric-val val-cyan" style="font-size: 1.05rem; padding-top: 5px;">{tele.get('climate_profile', 'OPTIMAL')}</div>
                <div style="color: #94a3b8; font-size: 0.70rem;">TEXTURE VAR: {tele.get('texture_var', 0.0)}</div>
            </div>
            """, unsafe_allow_html=True)

            enh_list = " | ".join(tele.get("enhancements", ["CALIBRATING"]))
            diag_placeholder.markdown(f"""
            <div style="background: rgba(5, 10, 20, 0.8); border: 1px solid rgba(0, 243, 255, 0.2); border-radius: 6px; padding: 8px 14px; font-family: monospace; font-size: 0.75rem; color: #38bdf8;">
                ⚡ <b>ACTIVE PIPELINE:</b> {enh_list} &nbsp;|&nbsp; <b>INFERENCE:</b> {latency_ms:.1f}ms &nbsp;|&nbsp; <b>FPS:</b> {fps:.1f}
            </div>
            """, unsafe_allow_html=True)

        time.sleep(0.02)

    cap.release()


if __name__ == "__main__":
    main()
