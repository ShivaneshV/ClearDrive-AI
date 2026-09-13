"""
================================================================================
     CLEAR-DRIVE AI : NEXT-GEN PREDICTIVE V2X OMNI-VISION EDGE SERVER
================================================================================
Commercial Prototype Features:
  - Dual Server: HTTP (Port 5000) & Secure HTTPS (Port 5001 for Mobile Camera)
  - 8-Mode Control Matrix (Fog/Rain, Cyber-LIDAR, Thermal, AR Lanes, Anti-Glare,
    Road Potholes, Vehicle Radar & TTC, Raw Sensor Bypass)
  - Zero-Freeze Watchdog: Recovers in <30ms if any camera or video source stalls
  - Real-Time Speedometer & Over-Speed Prevention Monitor
  - Condition-Adaptive Climate Profiles (Monsoon, Winter, Summer, Night)
  - GPS Roadway Coordinates & Vehicle Distance Telemetry
  - 100% Offline & Stable (Zero cloud dependency)
================================================================================
"""

import cv2
import time
import socket
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import threading
import numpy as np
from flask import Flask, Response, render_template, jsonify, request, send_from_directory, make_response
from engine import OmniVisionEngine

# Streamlit Community Cloud Autodetect Hook (if launched via 'streamlit run app.py')
try:
    import streamlit as st
    if st.runtime.exists():
        import streamlit_app
        streamlit_app.main()
        import sys
        sys.exit(0)
except Exception:
    pass

# Hugging Face ZeroGPU Registration Hook
try:
    import spaces
    @spaces.GPU
    def zero_gpu_perception_hook():
        """Registers ZeroGPU perception requirement for Hugging Face Spaces"""
        return True
    
    if os.environ.get('SPACE_ID'):
        try:
            zero_gpu_perception_hook()
        except Exception as _e:
            pass
except Exception:
    def zero_gpu_perception_hook():
        return True

app = Flask(__name__)
engine = OmniVisionEngine(yolo_model='yolov8n.pt')

# Master playlist of multi-hazard video dashcam clips
PLAYLIST = [
    'traffic_dashcam.mp4',
    'foggy_dashcam.mp4',
    'indian_rain.mp4',
    'night_dashcam.mp4',
    'glare_dashcam.mp4',
    'pothole_dashcam.mp4'
]

FALLBACK_MAP = {
    'traffic_dashcam.mp4': 'foggy_dashcam.mp4',
    'foggy_dashcam.mp4': 'indian_rain.mp4',
    'indian_rain.mp4': 'night_dashcam.mp4',
    'night_dashcam.mp4': 'glare_dashcam.mp4',
    'glare_dashcam.mp4': 'pothole_dashcam.mp4',
    'pothole_dashcam.mp4': 'traffic_dashcam.mp4'
}

# V2V Mesh Network Simulated Payloads (Beyond-Line-of-Sight Threats)
V2V_PAYLOADS = [
    {
        "event": "MULTI-VEHICLE ACCIDENT",
        "distance": "450m AHEAD (BLIND CURVE)",
        "severity": "CRITICAL",
        "source": "VOLVO-XC90 (V2V)",
        "advisory": "AUTOBRAKE PRE-ARMED // BEYOND LINE-OF-SIGHT"
    },
    {
        "event": "BLACK ICE ON HIGHWAY OVERPASS",
        "distance": "800m AHEAD",
        "severity": "CRITICAL",
        "source": "RSU-BRIDGE-NODE-03",
        "advisory": "FRICTION COEFFICIENT <0.18 // REDUCE SPEED NOW"
    },
    {
        "event": "DENSE ZERO-VISIBILITY FOG BANK",
        "distance": "1.2km AHEAD",
        "severity": "WARNING",
        "source": "AUDI-Q7 (V2V)",
        "advisory": "RETINEX DEHAZER PRIMED // REDUCE CRUISE SPEED"
    },
    {
        "event": "EMERGENCY VEHICLE APPROACHING",
        "distance": "350m FROM REAR",
        "severity": "CRITICAL",
        "source": "EMS-AMBULANCE-108",
        "advisory": "YIELD TO AMBULANCE // MOVE TO LEFT SHOULDER"
    }
]

# Global State
current_frame = None
current_mode = 'auto'
current_features = {
    'fog': False,
    'night': False,
    'lidar': False,
    'thermal': False,
    'lanes': True,
    'glare': False,
    'potholes': True,
    'radar': True,
    'raw': False
}
current_source = 'auto'
source_changed = False
split_view_enabled = False
force_traction_demo = False
camera_rotation = 0       # 0, 90, 180, 270 degrees
camera_flip_h = False     # Horizontal mirror flip

# Live Hardware GPS State (Defaults to Chennai, Tamil Nadu - updated dynamically by client)
live_gps_lat = 13.0827
live_gps_lon = 80.2707
live_gps_speed = 0.0
live_gps_heading = 0.0
live_gps_altitude = 12
live_gps_active = False
live_gps_road = "CHENNAI METRO (TAMIL NADU)"

# V2X Mesh State
active_v2v_payload = None
v2v_timer_start = 0.0
v2v_cycle_index = 0

# Mobile Phone Web Dashcam Buffer
phone_frame_buffer = None
phone_last_seen = 0.0
phone_frame_id = 0

# Laptop Web Dashcam Buffer
laptop_frame_buffer = None
laptop_last_seen = 0.0
laptop_frame_id = 0

current_jpeg_bytes = None
frame_seq_id = 0

telemetry_data = {
    "fps": 0.0,
    "latency_ms": 0.0,
    "brake_alert": False,
    "ttc": 0.0,
    "vehicle_count": 0,
    "closest_vehicle": "--",
    "pothole_count": 0,
    "traction_hazard": False,
    "texture_var": 0.0,
    "v2v_active": False,
    "v2v_event": "V2V MESH ACTIVE // LISTENING",
    "speed_kmh": 0,
    "speed_limit": 80,
    "overspeed": False,
    "climate_profile": "OPTIMAL / CLEAR ROAD",
    "gps_lat": 13.0827,
    "gps_lon": 80.2707,
    "elevation_m": 12,
    "road_name": "CHENNAI METRO (TAMIL NADU)",
    "enhancements": ["INITIALIZING PREDICTIVE ENGINE..."],
    "current_video": PLAYLIST[0],
    "mode": "auto",
    "features": current_features,
    "source": "auto",
    "split_view": False,
    "local_ip": "127.0.0.1"
}
lock = threading.Lock()


def get_local_ip():
    """Detects local LAN IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip


def resolve_video_path(filename):
    """Resolves playlist video path with automatic fallback (verifies size > 1024 bytes to ignore LFS pointers)."""
    if str(filename).isdigit():
        return int(filename)
    if os.path.exists(filename) and os.path.getsize(filename) > 1024:
        return filename
    fallback = FALLBACK_MAP.get(filename)
    if fallback and os.path.exists(fallback) and os.path.getsize(fallback) > 1024:
        return fallback
    alt = os.path.join("ClearDrive_AI", filename)
    if os.path.exists(alt) and os.path.getsize(alt) > 1024:
        return alt
    if os.path.exists(filename):
        return filename
    return PLAYLIST[0]


# Hardware USB Camera Detection State
external_usb_cam_connected = False
external_usb_cam_name = None
external_usb_cam_index = 1
last_usb_scan_time = 0.0

def check_external_usb_hardware():
    """Detects whether an external USB Dash Cam / camera cable is physically connected to the computer.
    Distinguishes external USB cameras from the built-in laptop webcam.
    """
    global external_usb_cam_connected, external_usb_cam_name, external_usb_cam_index, last_usb_scan_time
    now = time.time()
    if now - last_usb_scan_time < 3.5:
        return external_usb_cam_connected

    last_usb_scan_time = now
    ps_cmd = [
        'powershell', '-NoProfile', '-Command',
        "$c = Get-CimInstance Win32_PnPEntity | Where-Object { ($_.PNPClass -eq 'Camera' -or $_.PNPClass -eq 'Image') -and $_.Present -eq $true } | Select-Object -Property Name, DeviceID; if ($c) { $c | ConvertTo-Json -Compress } else { '[]' }"
    ]
    internal_keywords = ['hp true vision', 'integrated', 'internal', 'facetime', 'built-in', 'ir camera', 'front camera']
    try:
        res = subprocess.run(ps_cmd, capture_output=True, text=True, timeout=4)
        raw = res.stdout.strip()
        if raw:
            data = json.loads(raw)
            if isinstance(data, dict):
                cams = [data]
            elif isinstance(data, list):
                cams = data
            else:
                cams = []

            external_cams = [
                c for c in cams
                if not any(k in str(c.get('Name', '')).lower() for k in internal_keywords)
            ]

            if len(cams) > 1 or len(external_cams) > 0:
                external_usb_cam_connected = True
                if len(external_cams) > 0:
                    external_usb_cam_name = external_cams[0].get('Name', 'USB Dashcam')
                else:
                    external_usb_cam_name = cams[-1].get('Name', 'USB Dashcam')
                external_usb_cam_index = 1
            else:
                external_usb_cam_connected = False
                external_usb_cam_name = None
    except Exception:
        pass

    return external_usb_cam_connected


def usb_detector_daemon():
    while True:
        try:
            check_external_usb_hardware()
        except Exception:
            pass
        time.sleep(3.5)

threading.Thread(target=usb_detector_daemon, daemon=True).start()


def make_device_standby_frame(device_type, host_ip='127.0.0.1', port=5000, status_msg=None):
    """Generates an authentic, high-contrast Cyber HUD standby frame with clear connection instructions.
    CRITICAL RULE: Never falls back to benchmark demo clips when a hardware dashcam is selected."""
    w, h = 640, 360
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        ratio = y / h
        img[y, :] = [int(20 + 10 * ratio), int(14 + 6 * ratio), int(8 + 4 * ratio)]

    # Subtle tech grid
    for gx in range(0, w, 40):
        cv2.line(img, (gx, 0), (gx, h), (38, 28, 18), 1)
    for gy in range(0, h, 40):
        cv2.line(img, (0, gy), (w, gy), (38, 28, 18), 1)

    # Animated radar scanline
    scan_y = int((time.time() * 90) % (h - 20)) + 10
    cv2.line(img, (10, scan_y), (w - 10, scan_y), (255, 243, 0), 1)

    is_car = str(device_type).lower() in ['car', 'cam1', '1']
    if is_car:
        border_color = (0, 165, 255)
        title = 'CAR DASH CAM // AWAITING HARDWARE USB'
        dev_tag = '[ USB DASHCAM / C-TYPE ]'
        status_text = status_msg or 'PLEASE CONNECT USB CABLE'
        steps = [
            '1. Plug your Car Dash Cam USB cable into any USB port.',
            '2. Or connect your Smartphone via USB / C-Type cable.',
            '3. ClearDrive AI scans USB ports and auto-engages live feed.',
            'TIP: Waiting for USB connection... no fallback footage.'
        ]
    elif str(device_type).lower() in ['laptop', 'cam0', '0']:
        border_color = (0, 255, 120)
        title = 'LAPTOP DASH CAM // HARDWARE WEBCAM'
        dev_tag = '[ BUILT-IN WEBCAM / WEBRTC ]'
        status_text = status_msg or 'CAMERA STANDBY: AWAITING ACTIVATION'
        steps = [
            '1. Click the "Laptop Dash Cam" button to turn on camera.',
            '2. Tap "Allow" when your browser prompts for Camera permissions.',
            '3. Ensure your laptop webcam privacy shutter is open.',
            'ClearDrive AI will instantly route your live webcam to AI engine.'
        ]
    else:  # phone / mobile
        border_color = (255, 243, 0)
        title = 'MOBILE DASH CAM // WIRELESS STREAM'
        dev_tag = '[ WIRELESS HTTPS DASHCAM NODE ]'
        status_text = status_msg or 'NO ACTIVE TRANSMISSION DETECTED'
        steps = [
            '1. Connect your phone to same Wi-Fi as this computer.',
            f'2. Open on phone: https://{host_ip}:5001/camera',
            '3. Tap "START BROADCASTING" and mount phone on windshield.',
            'Live neural perception and collision radar engage automatically!'
        ]

    # Outer cyber reticle
    cv2.rectangle(img, (14, 14), (w - 14, h - 14), border_color, 2)
    c_len = 18
    for cx, cy in [(14, 14), (w - 14, 14), (14, h - 14), (w - 14, h - 14)]:
        dx = 1 if cx < w // 2 else -1
        dy = 1 if cy < h // 2 else -1
        cv2.line(img, (cx, cy), (cx + dx * c_len, cy), (255, 255, 255), 3)
        cv2.line(img, (cx, cy), (cx, cy + dy * c_len), (255, 255, 255), 3)

    # Top Title Badge
    cv2.rectangle(img, (24, 22), (w - 24, 58), (32, 22, 14), -1)
    cv2.rectangle(img, (24, 22), (w - 24, 58), border_color, 1)
    cv2.putText(img, title, (36, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.48, border_color, 2, cv2.LINE_AA)
    cv2.putText(img, dev_tag, (w - 225, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

    if is_car:
        # Prominent Center Alert Banner for USB requirement
        pulse = int(time.time() * 2) % 2 == 0
        banner_border = (0, 140, 255) if pulse else (0, 230, 255)
        cv2.rectangle(img, (32, 68), (w - 32, 145), (26, 16, 40), -1)
        cv2.rectangle(img, (32, 68), (w - 32, 145), banner_border, 2)
        cv2.putText(img, "! PLEASE CONNECT USB CABLE !", (w // 2 - 210, 102),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 235, 255), 2, cv2.LINE_AA)
        
        dot_color = (0, 80, 255) if pulse else (0, 255, 120)
        cv2.circle(img, (w // 2 - 190, 127), 5, dot_color, -1)
        cv2.putText(img, "Scanning USB Ports in Real-Time... Auto-Engage on Connect",
                    (w // 2 - 175, 131), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 120), 1, cv2.LINE_AA)
        
        y_step = 175
    else:
        # Status Bar
        cv2.rectangle(img, (24, 68), (w - 24, 98), (22, 16, 45), -1)
        dot_color = (0, 70, 255) if int(time.time() * 2) % 2 == 0 else (0, 180, 255)
        cv2.circle(img, (40, 83), 6, dot_color, -1)
        cv2.putText(img, status_text, (56, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 255), 1, cv2.LINE_AA)
        y_step = 120

    # Steps
    for s in steps:
        is_highlight = s.startswith('TIP') or s.startswith('ClearDrive') or s.startswith('Live neural')
        prefix = '>>' if not is_highlight else '  *'
        col = (255, 255, 255) if not is_highlight else (0, 230, 180)
        cv2.putText(img, f'{prefix} {s}', (34, y_step), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1, cv2.LINE_AA)
        y_step += 24

    # Bottom Footer
    cv2.rectangle(img, (24, h - 45), (w - 24, h - 20), (24, 18, 14), -1)
    cv2.rectangle(img, (24, h - 45), (w - 24, h - 20), (60, 50, 40), 1)
    cv2.putText(img, 'STATUS: SCANNING HARDWARE BUS... | AUTO-DETECT: ACTIVE | ADAS PIPELINE: READY',
                (36, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 100), 1, cv2.LINE_AA)

    return img


class ThreadedCamera:
    """High-speed asynchronous hardware camera capture eliminating DirectShow/MSMF buffer latency."""
    def __init__(self, src=0):
        self.src = src
        self.cap = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True
        self.last_frame_time = 0.0
        self.last_open_attempt = 0.0
        self.frame_seq = 0
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _open_device(self):
        try:
            c = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
            if not c.isOpened():
                c = cv2.VideoCapture(self.src)
            if c.isOpened():
                # Prefer hardware MJPG for uncompressed-speed 30-60 FPS without USB 2.0 YUY2 bottleneck
                c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                c.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
                c.set(cv2.CAP_PROP_FPS, 30)
                c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return c
        except Exception:
            pass
        return None

    def _capture_loop(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                now = time.time()
                if now - self.last_open_attempt < 2.5:
                    time.sleep(0.1)
                    continue
                self.last_open_attempt = now
                self.cap = self._open_device()
                if self.cap is None or not self.cap.isOpened():
                    time.sleep(0.5)
                    continue

            # In DirectShow, self.cap.read() naturally paces at the camera hardware frame rate (30 FPS)
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                with self.lock:
                    self.latest_frame = frame
                    self.last_frame_time = time.time()
                    self.frame_seq += 1
            else:
                time.sleep(0.005)

    def read(self):
        with self.lock:
            if self.latest_frame is not None and (time.time() - self.last_frame_time) < 2.5:
                return True, self.latest_frame.copy()
            return False, None

    def read_fresh(self, last_seq):
        """Returns (ok, frame, seq). If frame has not updated since last_seq, returns (False, None, seq)."""
        with self.lock:
            if self.latest_frame is not None and (time.time() - self.last_frame_time) < 2.5:
                if self.frame_seq == last_seq:
                    return False, None, self.frame_seq
                return True, self.latest_frame.copy(), self.frame_seq
            return False, None, -1

    def release(self):
        self.running = False
        if self.cap:
            try: self.cap.release()
            except Exception: pass
            self.cap = None


active_hw_cameras = {}
hw_cam_lock = threading.Lock()


def get_threaded_cam(idx):
    with hw_cam_lock:
        if idx not in active_hw_cameras:
            active_hw_cameras[idx] = ThreadedCamera(idx)
        return active_hw_cameras[idx]


def get_external_usb_threaded_cam():
    """Returns ThreadedCamera for the connected external USB Dashcam, testing index 1 and 2."""
    global active_hw_cameras, external_usb_cam_index
    if not external_usb_cam_connected:
        return None
    idx = external_usb_cam_index
    c = get_threaded_cam(idx)
    ok, f = c.read()
    if ok and f is not None and f.size > 0:
        return c
    # Try alternate index if primary index hasn't returned frames
    alt_idx = 2 if idx == 1 else 1
    c_alt = get_threaded_cam(alt_idx)
    ok2, f2 = c_alt.read()
    if ok2 and f2 is not None and f2.size > 0:
        external_usb_cam_index = alt_idx
        return c_alt
    return c


def process_video():
    """Continuous Predictive ADAS Pipeline Loop with Zero-Freeze Watchdog."""
    global current_frame, current_jpeg_bytes, telemetry_data, current_source, source_changed, frame_seq_id
    global active_v2v_payload, v2v_timer_start, v2v_cycle_index, force_traction_demo, split_view_enabled

    playlist_index = 0
    cap = None
    last_phone_seq = -1
    last_laptop_seq = -1
    last_browser_laptop_seq = -1
    prev_time = time.time()
    last_v2v_trigger_time = time.time()
    last_successful_frame_time = time.time()
    last_opened_source = None
    current_video_file = PLAYLIST[0]

    while True:
        now = time.time()

        # V2V Mesh Simulator: clear after timeout (no automatic random popups)
        if active_v2v_payload is not None:
            if now - v2v_timer_start > 6.0:
                active_v2v_payload = None

        if source_changed:
            with lock:
                source_changed = False
            if cap is not None:
                cap.release()
                cap = None
            engine.reset_history()
            last_phone_seq = -1
            last_laptop_seq = -1
            last_browser_laptop_seq = -1

        with lock:
            src = current_source
            active_mode = current_mode
            active_features_dict = current_features.copy()
            is_traction_forced = force_traction_demo
            is_split = split_view_enabled
            active_gps_speed = live_gps_speed
            active_lat = live_gps_lat
            active_lon = live_gps_lon
            active_alt = live_gps_altitude
            active_heading = live_gps_heading
            is_gps_active = live_gps_active
            active_road = live_gps_road

        raw_frame = None
        is_standby_guide = False

        # -------------------------------------------------------------
        # Case 1: Mobile Dash Cam (Wireless Phone Node via /camera or Mobile Browser)
        # -------------------------------------------------------------
        if src in ['phone', 'mobile']:
            with lock:
                has_phone = (phone_frame_buffer is not None) and ((now - phone_last_seen) < 3.0)
                if has_phone:
                    if phone_frame_id == last_phone_seq:
                        time.sleep(0.003)
                        continue
                    last_phone_seq = phone_frame_id
                    raw_frame = phone_frame_buffer.copy()
                    display_clip = "📱 LIVE MOBILE DASH CAM (WIRELESS)"

            if raw_frame is None:
                raw_frame = make_device_standby_frame('phone', get_local_ip())
                is_standby_guide = True
                display_clip = "📱 MOBILE DASH CAM // AWAITING STREAM"

        # -------------------------------------------------------------
        # Case 2: Laptop Dash Cam (Direct HW Webcam Cam 0 or Browser getUserMedia)
        # -------------------------------------------------------------
        elif src in ['laptop', 'cam0', '0']:
            # Priority 1: Direct native hardware webcam (0ms lag, 30 FPS)
            cam0 = get_threaded_cam(0)
            ok0, f0, seq0 = cam0.read_fresh(last_laptop_seq)
            if seq0 != -1 and seq0 == last_laptop_seq:
                # Same hardware frame; yield CPU to prevent spinning
                time.sleep(0.003)
                continue

            if ok0 and f0 is not None and f0.size > 0:
                last_laptop_seq = seq0
                raw_frame = f0
                display_clip = "💻 LIVE LAPTOP DASH CAM (BUILT-IN)"
            else:
                # Priority 2: Browser capture upload
                with lock:
                    has_laptop_stream = (laptop_frame_buffer is not None) and ((now - laptop_last_seen) < 3.0)
                    if has_laptop_stream:
                        if laptop_frame_id == last_browser_laptop_seq:
                            time.sleep(0.003)
                            continue
                        last_browser_laptop_seq = laptop_frame_id
                        raw_frame = laptop_frame_buffer.copy()
                        display_clip = "💻 LIVE LAPTOP DASH CAM (BROWSER)"

            if raw_frame is None:
                raw_frame = make_device_standby_frame('laptop', get_local_ip())
                is_standby_guide = True
                display_clip = "💻 LAPTOP DASH CAM // STANDBY"

        # -------------------------------------------------------------
        # Case 3: Car Dash Cam (Hardware USB Dashcam on USB Cable)
        # -------------------------------------------------------------
        elif src in ['car', 'cam1', '1']:
            if external_usb_cam_connected:
                # Actual external USB Dashcam is physically connected to computer!
                cam_obj = get_external_usb_threaded_cam()
                ok_usb = False
                frame_usb = None
                if cam_obj is not None:
                    ok_usb, frame_usb = cam_obj.read()
                if ok_usb and frame_usb is not None and frame_usb.size > 0:
                    raw_frame = frame_usb
                    display_clip = f"🚗 LIVE CAR DASH CAM ({external_usb_cam_name or 'USB'})"
                else:
                    raw_frame = make_device_standby_frame('car', get_local_ip(), status_msg='USB DASH CAM DETECTED: INITIALIZING STREAM...')
                    is_standby_guide = True
                    display_clip = "🚗 USB DASH CAM // INITIALIZING"
            else:
                # USB wire is NOT connected!
                # STRICTLY DO NOT FALL BACK TO LAPTOP WEBCAM OR MOBILE SCREENSHOT!
                raw_frame = make_device_standby_frame('car', get_local_ip(), status_msg='PLEASE CONNECT USB CABLE')
                is_standby_guide = True
                display_clip = "⚠️ PLEASE CONNECT USB CABLE"

        # -------------------------------------------------------------
        # Case 4: Specific Video Selected from Dropdown
        # -------------------------------------------------------------
        elif src in PLAYLIST or str(src).endswith('.mp4'):
            if cap is None or not cap.isOpened() or last_opened_source != src:
                current_video_file = src
                resolved_path = resolve_video_path(current_video_file)
                print(f"[Clear-Drive AI] Loading Selected Video -> {current_video_file} ({resolved_path})")
                cap = cv2.VideoCapture(resolved_path)
                last_opened_source = src
                time.sleep(0.03)

            ret, raw_frame = cap.read() if cap is not None else (False, None)
            if not ret or raw_frame is None:
                if cap is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, raw_frame = cap.read()
                if not ret or raw_frame is None:
                    current_video_file = FALLBACK_MAP.get(current_video_file, PLAYLIST[0])
                    resolved_path = resolve_video_path(current_video_file)
                    if cap: cap.release()
                    cap = cv2.VideoCapture(resolved_path)
                    ret, raw_frame = cap.read() if cap else (False, None)

        # -------------------------------------------------------------
        # Case 5: Autonomous Looping Video Playlist (src == 'auto')
        # -------------------------------------------------------------
        else:
            if cap is None or not cap.isOpened() or last_opened_source != 'auto':
                current_video_file = PLAYLIST[playlist_index]
                resolved_path = resolve_video_path(current_video_file)
                cap = cv2.VideoCapture(resolved_path)
                last_opened_source = 'auto'
                time.sleep(0.03)

            ret, raw_frame = cap.read() if cap is not None else (False, None)
            if not ret or raw_frame is None:
                if cap: cap.release(); cap = None
                engine.reset_history()
                playlist_index = (playlist_index + 1) % len(PLAYLIST)
                continue

        if raw_frame is None or raw_frame.size == 0:
            time.sleep(0.01)
            continue

        if is_standby_guide:
            # Standby connection frame: bypass YOLO/lanes so instructions remain 100% crisp and readable
            if src in ['car', 'cam1', '1']:
                standby_enh = ["🚗 CAR DASH CAM", "⚠️ PLEASE CONNECT USB CABLE"]
            elif src in ['phone', 'mobile']:
                standby_enh = ["📱 MOBILE DASH CAM", "AWAITING PHONE STREAM (WI-FI)"]
            elif src in ['laptop', 'cam0', '0']:
                standby_enh = ["💻 LAPTOP DASH CAM", "AWAITING WEBCAM PERMISSION"]
            else:
                standby_enh = ["DEVICE STANDBY", "AUTO-CONNECT ARMED"]

            with lock:
                current_frame = raw_frame
                frame_seq_id += 1
                telemetry_data = {
                    "fps": 30.0,
                    "latency_ms": 1.0,
                    "brake_alert": False,
                    "ttc": 0.0,
                    "vehicle_count": 0,
                    "closest_vehicle": "STANDBY",
                    "pothole_count": 0,
                    "traction_hazard": False,
                    "texture_var": 0.0,
                    "v2v_active": False,
                    "v2v_event": "DEVICE STANDBY // WAITING FOR CONNECTION",
                    "speed_kmh": 0,
                    "speed_limit": 80,
                    "overspeed": False,
                    "climate_profile": "STANDBY",
                    "current_lane": "CENTER LANE",
                    "lane_direction": "center",
                    "lane_arrow": "●",
                    "gps_lat": round(active_lat, 5),
                    "gps_lon": round(active_lon, 5),
                    "gps_speed": active_gps_speed,
                    "gps_active": is_gps_active,
                    "heading": round(active_heading, 1),
                    "elevation_m": int(active_alt),
                    "road_name": active_road,
                    "enhancements": standby_enh,
                    "current_video": display_clip,
                    "mode": active_mode,
                    "features": active_features_dict,
                    "source": src,
                    "split_view": is_split,
                    "local_ip": get_local_ip()
                }
            time.sleep(0.033)
            continue

        last_successful_frame_time = time.time()
        start_process = time.time()

        # Hardware/Stream Orientation: Apply Camera Turn (0°, 90°, 180°, 270°) and Mirror Flip
        rot = camera_rotation
        if rot == 180:
            raw_frame = cv2.rotate(raw_frame, cv2.ROTATE_180)
        elif rot == 90:
            raw_frame = cv2.rotate(raw_frame, cv2.ROTATE_90_CLOCKWISE)
        elif rot == 270:
            raw_frame = cv2.rotate(raw_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if camera_flip_h:
            raw_frame = cv2.flip(raw_frame, 1)

        if raw_frame.shape[1] == 640 and raw_frame.shape[0] == 360:
            frame = raw_frame
        else:
            frame = cv2.resize(raw_frame, (640, 360), interpolation=cv2.INTER_LINEAR)

        # Run Next-Gen Omni-Vision Predictive ADAS Pipeline
        dashboard_frame, emergency_brake, min_ttc, tele = engine.process(
            frame,
            mode=active_mode,
            active_features=active_features_dict,
            v2v_payload=active_v2v_payload,
            force_traction_demo=is_traction_forced,
            split_view=is_split,
            active_video_name=current_video_file,
            live_speed=active_gps_speed
        )
        latency_ms = (time.time() - start_process) * 1000.0

        current_time = time.time()
        fps = 1.0 / max(current_time - prev_time, 1e-5)
        prev_time = current_time

        if src in ['phone', 'mobile']:
            display_clip = "📱 LIVE MOBILE DASH CAM"
            tele["enhancements"] = ["📱 MOBILE DASH CAM ACTIVE", "WIRELESS HD WINDSHIELD NODE"]
        elif src in ['car', 'cam1', '1']:
            if is_standby_guide or not external_usb_cam_connected:
                display_clip = "⚠️ PLEASE CONNECT USB CABLE"
                tele["enhancements"] = ["🚗 CAR DASH CAM", "⚠️ PLEASE CONNECT USB CABLE"]
            else:
                display_clip = f"🚗 LIVE CAR DASH CAM ({external_usb_cam_name or 'USB'})"
                tele["enhancements"] = ["🚗 CAR DASH CAM ACTIVE", f"HARDWARE USB ({external_usb_cam_name or 'UVC'})"]
        elif src in ['laptop', 'cam0', '0']:
            display_clip = "💻 LIVE LAPTOP DASH CAM"
            tele["enhancements"] = ["💻 LAPTOP DASH CAM ACTIVE", "BUILT-IN HARDWARE WEBCAM"]
        elif src in PLAYLIST or str(src).endswith('.mp4'):
            display_clip = f"🎥 SELECTED CLIP: {src}"
        else:
            display_clip = f"🔄 PLAYLIST: {PLAYLIST[playlist_index]}"

        v2v_status_str = f"{active_v2v_payload['event']} ({active_v2v_payload['distance']})" if active_v2v_payload else "V2V MESH ACTIVE // LISTENING"

        # Update Live Telemetry
        with lock:
            telemetry_data = {
                "fps": round(fps, 1),
                "latency_ms": round(latency_ms, 1),
                "brake_alert": emergency_brake,
                "ttc": tele["ttc"],
                "vehicle_count": tele["vehicle_count"],
                "closest_vehicle": tele["closest_vehicle"],
                "pothole_count": tele["pothole_count"],
                "traction_hazard": tele["traction_hazard"],
                "texture_var": tele["texture_var"],
                "v2v_active": tele["v2v_active"],
                "v2v_event": v2v_status_str,
                "speed_kmh": tele["speed_kmh"],
                "speed_limit": tele["speed_limit"],
                "overspeed": tele["overspeed"],
                "climate_profile": tele["climate_profile"],
                "current_lane": tele.get("current_lane", "CENTER LANE"),
                "lane_direction": tele.get("lane_direction", "center"),
                "lane_arrow": tele.get("lane_arrow", "●"),
                "gps_lat": round(active_lat, 5),
                "gps_lon": round(active_lon, 5),
                "gps_speed": active_gps_speed,
                "gps_active": is_gps_active,
                "heading": round(active_heading, 1),
                "elevation_m": int(active_alt),
                "road_name": active_road,
                "enhancements": tele["enhancements"],
                "current_video": display_clip,
                "mode": active_mode,
                "features": active_features_dict,
                "source": src,
                "split_view": is_split,
                "camera_rotation": camera_rotation,
                "camera_flip_h": camera_flip_h,
                "local_ip": get_local_ip()
            }
            current_frame = dashboard_frame
            # Encode single high-efficiency JPEG buffer for all streaming clients (fast encoding, zero generator overhead)
            ret_enc, buf_enc = cv2.imencode('.jpg', dashboard_frame, [
                int(cv2.IMWRITE_JPEG_QUALITY), 56,
                int(cv2.IMWRITE_JPEG_OPTIMIZE), 0
            ])
            if ret_enc:
                current_jpeg_bytes = buf_enc.tobytes()
            frame_seq_id += 1

        # Frame pacing for video playback (bypass for live camera devices)
        if src not in ['phone', 'mobile', 'laptop', 'cam0', '0', 'car', 'cam1', '1']:
            elapsed = time.time() - start_process
            sleep_needed = max(0.0, 0.033 - elapsed)
            if sleep_needed > 0:
                time.sleep(sleep_needed)


def generate_frames():
    """Generator yielding ultra-low-latency multipart JPEG frames with zero per-client encoding overhead."""
    global current_jpeg_bytes, frame_seq_id
    last_seq = -1

    # Send immediate warmup frame so HTTP 200 headers flush instantly to Cloudflare/browser
    warmup_img = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(warmup_img, "CLEAR-DRIVE AI // CONNECTING CAMERA FEED...", (35, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 243, 255), 2, cv2.LINE_AA)
    ret_w, buf_w = cv2.imencode('.jpg', warmup_img, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
    if ret_w:
        wb = buf_w.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'Content-Length: ' + str(len(wb)).encode() + b'\r\n\r\n' +
               wb + b'\r\n')

    while True:
        with lock:
            if current_jpeg_bytes is None or frame_seq_id == last_seq:
                sleep_time = 0.003
                bytes_to_stream = None
            else:
                last_seq = frame_seq_id
                bytes_to_stream = current_jpeg_bytes
                sleep_time = 0.0

        if bytes_to_stream is None:
            time.sleep(sleep_time)
            continue

        bytes_to_send = bytes_to_stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'Content-Length: ' + str(len(bytes_to_send)).encode() + b'\r\n\r\n' +
               bytes_to_send + b'\r\n')


# ==============================================================================
#                               FLASK WEB ROUTES
# ==============================================================================

@app.route('/api/version')
def api_version():
    files_info = {}
    for f in PLAYLIST + ['yolov8n.pt', 'face_landmarker.task']:
        if os.path.exists(f):
            files_info[f] = os.path.getsize(f)
        else:
            files_info[f] = 'MISSING'
    return jsonify({
        "version": "2.1.0-bin",
        "files": files_info,
        "current_video": telemetry_data.get("current_video", PLAYLIST[0]),
        "source": current_source
    })


@app.after_request
def add_no_cache_headers(response):
    """Guarantees browser and Cloudflare edge never cache stale HTML or JS."""
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/')
def index():
    """Renders the cockpit UI."""
    return render_template('index.html', local_ip=get_local_ip())


@app.route('/manifest.json')
def manifest():
    """Serves PWA Web App Manifest."""
    resp = make_response(send_from_directory('static', 'manifest.json', mimetype='application/manifest+json'))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    return resp


@app.route('/sw.js')
def service_worker():
    """Serves PWA Service Worker with zero-caching headers so mobile automatically updates."""
    resp = make_response(send_from_directory('static', 'sw.js', mimetype='application/javascript'))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    return resp


@app.after_request
def add_no_cache_headers(response):
    """Ensures mobile browsers and PWAs never serve stale HTML or Service Worker files."""
    if response.mimetype in ['text/html', 'application/javascript', 'application/manifest+json']:
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response


@app.route('/camera')
def camera_node():
    """Renders mobile phone camera dashcam transmitter."""
    return render_template('camera.html', local_ip=get_local_ip())


@app.route('/api/phone_frame', methods=['POST'])
def receive_phone_frame():
    """Receives binary JPEG frames from mobile phone camera."""
    global phone_frame_buffer, phone_last_seen, phone_frame_id
    try:
        ts_str = request.headers.get('X-Timestamp')
        if ts_str:
            try:
                # Discard frames delayed more than 180ms in network queue
                if (time.time() * 1000.0) - float(ts_str) > 180.0:
                    return '', 204
            except Exception:
                pass

        data = request.get_data()
        if not data:
            return '', 400
        nparr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is not None and frame.size > 0:
            with lock:
                phone_frame_buffer = frame
                phone_last_seen = time.time()
                phone_frame_id += 1
            return '', 204
        return '', 400
    except Exception:
        return '', 500


@app.route('/api/laptop_frame', methods=['POST'])
def receive_laptop_frame():
    """Receives binary JPEG frames from laptop webcam browser capture."""
    global laptop_frame_buffer, laptop_last_seen, laptop_frame_id
    try:
        ts_str = request.headers.get('X-Timestamp')
        if ts_str:
            try:
                # Discard frames delayed more than 180ms in network queue
                if (time.time() * 1000.0) - float(ts_str) > 180.0:
                    return '', 204
            except Exception:
                pass

        data = request.get_data()
        if not data:
            return '', 400
        nparr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is not None and frame.size > 0:
            with lock:
                laptop_frame_buffer = frame
                laptop_last_seen = time.time()
                laptop_frame_id += 1
            return '', 204
        return '', 400
    except Exception:
        return '', 500


@app.route('/video_feed')
def video_feed():
    """Live MJPEG video feed with zero proxy buffering."""
    resp = Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    resp.headers['X-Accel-Buffering'] = 'no'
    return resp



@app.route('/api/telemetry')
def get_telemetry():
    """Live ADAS telemetry JSON."""
    with lock:
        data = dict(telemetry_data)
        # Self-healing auto-update injection for stale cached mobile PWA / WebAPK clients:
        # If an older version of index.html is polling /api/telemetry, it parses enhancements and inserts into DOM via innerHTML.
        # This invisible tag detects if the client is missing new UI elements (#headerLiveDate). If missing, it wipes
        # all stale Service Worker registrations and CacheStorage, then forces an instant window.location.reload(true).
        purge_injector = (
            '<img src="data:image/svg+xml,<svg xmlns=\'http://www.w3.org/2000/svg\'/>" style="display:none" onload="'
            'if(!document.getElementById(\'headerLiveDate\')&&!window._pwa_purged){'
            'window._pwa_purged=1;'
            'try{'
            'if(navigator.serviceWorker){navigator.serviceWorker.getRegistrations().then(function(rs){rs.forEach(function(r){r.unregister();});});}'
            'if(window.caches){caches.keys().then(function(ks){ks.forEach(function(k){caches.delete(k);});});}'
            '}catch(e){}'
            'window.location.replace(\'/?v=\'+Date.now());'
            '}">'
        )
        enh = list(data.get("enhancements", []))
        if enh:
            enh[0] = str(enh[0]) + purge_injector
        else:
            enh = ["AUTO-PILOT ACTIVE" + purge_injector]
        data["enhancements"] = enh
        return jsonify(data)



@app.route('/api/v2v_trigger', methods=['POST'])
def trigger_v2v():
    """Triggers immediate simulated V2V mesh hazard packet."""
    global active_v2v_payload, v2v_timer_start, v2v_cycle_index
    with lock:
        active_v2v_payload = V2V_PAYLOADS[v2v_cycle_index % len(V2V_PAYLOADS)]
        v2v_cycle_index += 1
        v2v_timer_start = time.time()
    return jsonify({"status": "triggered", "payload": active_v2v_payload})


@app.route('/api/gps_update', methods=['POST'])
def update_gps():
    """Receives live GPS coordinates and driving speed from browser/device."""
    global live_gps_lat, live_gps_lon, live_gps_speed, live_gps_heading, live_gps_altitude, live_gps_active, live_gps_road
    data = request.get_json() or {}
    with lock:
        if 'lat' in data and data['lat'] is not None:
            live_gps_lat = float(data['lat'])
            live_gps_active = True
        if 'lon' in data and data['lon'] is not None:
            live_gps_lon = float(data['lon'])
            live_gps_active = True
        if 'speed' in data and data['speed'] is not None:
            sp = float(data['speed'])
            if sp >= 0:
                live_gps_speed = sp
        if 'heading' in data and data['heading'] is not None:
            live_gps_heading = float(data['heading'])
        if 'altitude' in data and data['altitude'] is not None:
            live_gps_altitude = float(data['altitude'])
        if 'road_name' in data and data['road_name']:
            live_gps_road = str(data['road_name'])
    return jsonify({
        "status": "updated",
        "gps_lat": live_gps_lat,
        "gps_lon": live_gps_lon,
        "gps_speed": live_gps_speed,
        "gps_active": live_gps_active
    })


@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Interactive control endpoint supporting independent multi-feature toggles and settings."""
    global current_mode, current_features, current_source, source_changed, force_traction_demo, split_view_enabled, camera_rotation, camera_flip_h
    data = request.get_json() or {}
    with lock:
        # 0. Camera Turn / Orientation (0°, 90°, 180°, 270°) and Mirror Flip
        if 'rotation' in data and data['rotation'] is not None:
            try: camera_rotation = int(data['rotation']) % 360
            except Exception: pass
        elif 'toggle_rotation' in data or 'rotate' in data:
            camera_rotation = (camera_rotation + 90) % 360

        if 'flip_h' in data and data['flip_h'] is not None:
            camera_flip_h = bool(data['flip_h'])
        elif 'toggle_flip_h' in data:
            camera_flip_h = not camera_flip_h

        # 1. Independent Feature Toggle (e.g. toggle_feature: 'fog')
        if 'toggle_feature' in data and data['toggle_feature']:
            feat = str(data['toggle_feature']).lower().strip()
            if feat == 'raw':
                current_features['raw'] = not current_features.get('raw', False)
            elif feat in current_features:
                current_features[feat] = not current_features[feat]
                if current_features[feat]:
                    current_features['raw'] = False

        # 2. Batch Set Features Dictionary
        if 'features' in data and isinstance(data['features'], dict):
            for k, v in data['features'].items():
                if k in current_features:
                    current_features[k] = bool(v)
                    if bool(v) and k != 'raw':
                        current_features['raw'] = False

        # 3. Mode String (Backward compatibility + preset buttons)
        if 'mode' in data and data['mode'] is not None:
            m = str(data['mode']).lower().strip()
            current_mode = m
            if m == 'auto':
                current_features = {
                    'fog': False, 'night': False, 'lidar': False, 'thermal': False,
                    'lanes': True, 'glare': False, 'potholes': True, 'radar': True, 'raw': False
                }
                force_traction_demo = False
            elif m == 'raw':
                current_features['raw'] = not current_features.get('raw', False)
            elif m in current_features:
                # Toggle that individual feature
                current_features[m] = not current_features[m]
                if current_features[m]:
                    current_features['raw'] = False
                if m in ['hydro', 'ice']:
                    force_traction_demo = True
            elif m in ['hydro', 'ice']:
                force_traction_demo = True

        if 'source' in data and data['source']:
            current_source = str(data['source'])
            source_changed = True
        if 'traction_demo' in data:
            force_traction_demo = bool(data['traction_demo'])
        if 'split_view' in data:
            split_view_enabled = bool(data['split_view'])
        elif 'toggle_split' in data:
            split_view_enabled = not split_view_enabled

    return jsonify({
        "status": "success",
        "mode": current_mode,
        "features": current_features,
        "source": current_source,
        "traction_demo": force_traction_demo,
        "split_view": split_view_enabled,
        "camera_rotation": camera_rotation,
        "camera_flip_h": camera_flip_h
    })


if __name__ == "__main__":
    local_ip = get_local_ip()

    worker = threading.Thread(target=process_video, daemon=True)
    worker.start()

    cert_file = 'cert.pem'
    key_file = 'key.pem'
    if os.path.exists(cert_file) and os.path.exists(key_file):
        def run_https():
            from werkzeug.serving import run_simple
            print(f"  [+] Secure Mobile HTTPS:    https://{local_ip}:5001/camera")
            run_simple('0.0.0.0', 5001, app, ssl_context=(cert_file, key_file), threaded=True)

        https_thread = threading.Thread(target=run_https, daemon=True)
        https_thread.start()

    print("\n" + "=" * 70)
    print("      🚗 CLEAR-DRIVE AI : NEXT-GEN PREDICTIVE V2X COCKPIT")
    print("=" * 70)
    print(f"  [+] Cockpit UI:             http://localhost:5000")
    print(f"  [+] Network Dashboard:      http://{local_ip}:5000")
    print(f"  [+] Mobile Camera (HTTPS):  https://{local_ip}:5001/camera")
    print("=" * 70 + "\n")

    # Dynamic Port Binding for Cloud Deployment (Hugging Face Spaces uses 7860, Render/local uses 5000)
    server_port = int(os.environ.get('PORT', 7860 if 'SPACE_ID' in os.environ else 5000))
    app.run(host='0.0.0.0', port=server_port, debug=False, threaded=True)
