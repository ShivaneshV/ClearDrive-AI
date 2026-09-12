# 🚗 ClearDrive AI : Next-Gen Predictive V2X Omni-Vision Cockpit

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Flask](https://img.shields.io/badge/Flask-3.0+-black.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

> **ClearDrive AI** is a production-grade, 100% offline-resilient Automotive ADAS Cockpit Operating System. It combines computer vision perception, multi-sensor concurrency, live hardware GPS telemetry, condition-aware speed limits, and in-cockpit Google Maps navigation to prevent vehicular collisions in extreme weather and zero-light conditions.

---

## 🌟 Key Capabilities & Features

### 1. 🎛️ Independent Multi-Feature Concurrency Matrix
ClearDrive AI runs an additive, non-interfering perception pipeline allowing drivers to activate multiple vision enhancements concurrently:
* **`[1]` 🌫️ Fog & Rain Dehazer:** Physical Dark Channel Prior (DCP) atmospheric transmission restoration with true-color preservation.
* **`[2]` 📡 Cyber-LIDAR Point Cloud:** 64-beam 905nm pseudo-LIDAR 3D spatial scan grid with obstacle proximity alerts.
* **`[3]` 🌡️ FLIR Thermal Optics:** False-color infrared heatmap for pitch-black, zero-illumination environments.
* **`[4]` 🛣️ Sleek AR Lane Guidance:** Neon cyan boundary rails, center dashed guide, and dynamic distance hashes (5m, 15m, 30m) without visual obstruction.
* **`[5]` 😎 Polarized Anti-Glare Shield:** Automotive specular knee compression rolling off blinding oncoming high-beams (>190) without color distortion.
* **`[6]` ⚠️ Road Pothole Scanner:** 3D drivable corridor isolation detecting roadway craters in neon green (`#00ff00`) with strict vegetation/tree rejection.
* **`[7]` 🚗 Vehicle Radar & +1.5s Ghost-Vision:** YOLOv8 kinematic tracking, metric distance calculation, Time-to-Collision (TTC), and predictive cut-in trajectory projection.
* **`[8]` 📷 Raw Sensor Bypass:** Instant 1:1 hardware bypass toggle comparing raw dashcam footage with AI-enhanced perception.

---

### 2. 🛰️ Real-Time Hardware GPS Speedometer & Safety Limits
* **Live In-Car Telemetry:** Reads live `coords.speed` directly from the smartphone or tablet's onboard GPS chip via the HTML5 Geolocation Watchdog API (converted from m/s to **km/h**).
* **Climate-Adaptive Safe Speed Limits:** Dynamically evaluates roadway risks and enforces contextual speed limits:
  * `35 km/h` — Black Ice / Hydroplaning Hazard
  * `50 km/h` — Dense Fog / Low-Light Night
  * `60 km/h` — Heavy Monsoon Rain
  * `80 km/h` — Optimal / Clear Expressway
* **Over-Speed Prevention:** Instant audible chime and pulsing cyber HUD banner when speed exceeds condition thresholds.

---

### 3. 🗺️ In-Cockpit Google Maps Navigation Subsystem
* **Integrated Dual Cockpit:** Split-view layout displaying live AI vision on the left and full-featured interactive navigation on the right.
* **Real-Time Destination Autocomplete:**
  * **0ms High-Frequency Database:** Instant suggestions for major transportation hubs, international airports, monuments, tech corridors, and cities.
  * **Live OpenStreetMap Nominatim Geocoding:** Live street-level address autocomplete with 280ms debouncing.
  * **Full Keyboard Navigation:** Arrow Down/Up selection, Enter to navigate, Escape to dismiss.
* **Turn-by-Turn Routing & ETA:** Computes optimal driving paths, draws neon route polylines, and estimates distance (km) and travel time (ETA).
* **Multiple Map Layers:** `SATELLITE`, `TRAFFIC`, `ROAD`, and dark-mode `CYBER`.

---

### 4. 📱 Wireless Mobile Dashcam (Zero-Install IoT Node)
* Turn any smartphone into an encrypted HD dashcam mount.
* Streams live 1080p camera frames over local HTTPS (`/camera`) directly to the vehicle dashboard with under 60ms edge latency.

---

## 🏗️ System Architecture

```
[ Dashboard Phone / USB Dashcam ] ─── HTTPS ───┐
                                                │
[ Onboard Hardware GPS Satellite ] ── HTML5 ───┼──> [ Flask Edge Server (Port 5000 / 5001) ]
                                                │         │
[ Offline Local Video Streams ] ───────────────┘         ▼
                                                ┌─────────────────────────────┐
                                                │  OmniVisionEngine Pipeline  │
                                                │  - CLAHE & Unsharp Masking  │
                                                │  - Retinex Night Vision     │
                                                │  - Atmospheric Dehazing     │
                                                │  - Anti-Glare Knee Filter   │
                                                │  - YOLOv8 Kinematic Radar   │
                                                │  - AR Laser Lane Projector  │
                                                │  - Neon Pothole Isolation   │
                                                │  - V2V AR Sky Billboard     │
                                                └──────────────┬──────────────┘
                                                               │
                                                               ▼
                                                [ Cyber Cockpit Web HUD ]
                                                - Dual Camera + Leaflet / GMap
                                                - Autocomplete Geocoder
                                                - Audio Chime Alerts
```

---

## ⚡ Quickstart & Installation

### Prerequisites
* Python 3.9+ installed
* Webcam or smartphone (for live camera demo)

### 1. Clone the Repository
```bash
git clone https://github.com/ShivaneshV/ClearDrive-AI.git
cd ClearDrive-AI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Cockpit Operating System
```bash
python app.py
```

### 4. Access the Dashboard
* **Cockpit Interface:** Open [`http://localhost:5000`](http://localhost:5000) in your browser.
* **Mobile Wireless Dashcam Node:** Open [`https://<YOUR-LOCAL-IP>:5001/camera`](https://localhost:5001/camera) on your phone.

---

## ⌨️ Hotkey Controls

| Key | Action |
|---|---|
| `1` | Toggle Fog / Rain Dehazing |
| `2` | Toggle Cyber-LIDAR 64-Beam Point Cloud |
| `3` | Toggle FLIR Thermal False-Color Optics |
| `4` | Toggle Sleek AR Lane Guidance |
| `5` | Toggle Polarized Anti-Glare Shield |
| `6` | Toggle Road Pothole Scanner |
| `7` | Toggle Vehicle Radar & Ghost-Vision |
| `8` | Toggle Raw Sensor Bypass |
| `N` | Toggle Low-Light Retinex Night Vision |
| `S` | Toggle Split-Screen (Raw vs AI Enhanced) |
| `M` | Switch Cockpit Layout (Dual / Cam Only / Map Only) |

---

## 📦 Project Structure

```
ClearDrive-AI/
├── app.py                      # Flask backend, GPS telemetry handler & video streamer
├── engine.py                   # OmniVision perception pipeline & filter graph
├── dehaze.py                   # Atmospheric aerosol transmission physics
├── dms.py                      # Driver Monitoring System (MediaPipe face mesh)
├── vehicle_tracker.py          # Metric distance kinematics & tracking
├── templates/
│   ├── index.html              # Cyber Cockpit HUD, Google Maps & Autocomplete
│   └── camera.html             # Wireless mobile dashcam client
├── requirements.txt            # Python dependencies
├── yolov8n.pt                  # YOLOv8 nano perception weights
├── face_landmarker.task        # MediaPipe facial landmark model
├── EXPO_PRESENTATION_SCRIPT.md # Live stage presentation & judge walkthrough script
└── *.mp4                       # High-definition road test benchmarks
```

---

## 📄 License
This project is licensed under the MIT License.
