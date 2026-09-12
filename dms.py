"""
================================================================================
            CLEAR-DRIVE AI : BIOMETRIC DRIVER MONITORING SYSTEM (DMS)
================================================================================
Domain: Computer Vision & Biometric Safety Engineering
Functionality:
  1. Real-time Inward Driver Tracking via Google MediaPipe 3D Neural Mesh (468 points)
  2. Microscopic Eyelid Distance Measurement:
       - Left Eye: dist(159, 145)
       - Right Eye: dist(386, 374)
  3. Temporal Drowsiness Filter (eyes closed > 15 frames ~ 0.5s triggers ALARM)
  4. Cyberpunk PiP (Picture-in-Picture) Inset with Neural Mesh Overlay & Reticles
================================================================================
"""

import cv2
import math
import os
import numpy as np
import mediapipe as mp

class DriverMonitor:
    """
    Biometric Driver Monitoring System (DMS).
    Monitors driver alertness in real-time using microscopic eyelid tracking.
    """

    def __init__(self, model_asset_path="face_landmarker.task"):
        self.sleep_frames = 0
        self.is_sleeping = False
        self.backend = "none"
        self.detector = None
        self.mp_face_mesh = None

        # Try Approach 1: Legacy mp.solutions.face_mesh (Python 3.10/3.11)
        if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
            try:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.mp_drawing = mp.solutions.drawing_utils
                self.mp_drawing_styles = mp.solutions.drawing_styles
                self.backend = "solutions"
            except Exception as e:
                print(f"[DMS] Legacy mp.solutions failed: {e}")

        # Try Approach 2: Modern MediaPipe Tasks API (Python 3.12/3.13/3.14+)
        if self.backend == "none":
            try:
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision

                # Search model path in current directory or ClearDrive_AI
                resolved_path = model_asset_path
                if not os.path.exists(resolved_path):
                    alt_path = os.path.join(os.path.dirname(__file__), model_asset_path)
                    if os.path.exists(alt_path):
                        resolved_path = alt_path

                if os.path.exists(resolved_path):
                    base_options = python.BaseOptions(model_asset_path=resolved_path)
                    options = vision.FaceLandmarkerOptions(
                        base_options=base_options,
                        running_mode=vision.RunningMode.IMAGE,
                        num_faces=1,
                        min_face_detection_confidence=0.45,
                        min_face_presence_confidence=0.45,
                        min_tracking_confidence=0.45
                    )
                    self.detector = vision.FaceLandmarker.create_from_options(options)
                    self.backend = "tasks"
                    print(f"[DMS] Initialized MediaPipe FaceLandmarker Tasks API (Model: {resolved_path})")
            except Exception as e:
                print(f"[DMS] MediaPipe Tasks API initialization failed: {e}")

        # Fallback Approach 3: OpenCV Haar Cascade Eyes / Face
        if self.backend == "none":
            print("[DMS] Operating in Haar Cascade / Optical Fallback Mode")
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
            self.backend = "haar"

        # Key landmark topological connections for Cyberpunk 3D Neural Mesh
        self.LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61]
        self.LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]
        self.RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466, 263]
        self.LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
        self.RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
        self.NOSE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2]
        self.FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
        self.SAMPLE_MESH_POINTS = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 11, 12, 13, 14, 15, 16, 17, 18, 200, 199, 175, 152]

    def _draw_neural_mesh(self, frame, landmarks, w, h, is_drowsy):
        """Renders glowing futuristic neural mesh lines and nodes onto PiP frame."""
        node_color = (0, 0, 255) if is_drowsy else (0, 255, 200)       # Red or Neon Aqua
        line_color = (0, 50, 200) if is_drowsy else (80, 180, 40)       # Dark Red or Cyber Green
        eye_color = (0, 0, 255) if is_drowsy else (0, 243, 255)        # Bright Red or Cyan

        # Helper to draw connected path
        def draw_path(indices, color, thick=1, is_closed=False):
            pts = []
            for idx in indices:
                if idx < len(landmarks):
                    lm = landmarks[idx]
                    pts.append((int(lm.x * w), int(lm.y * h)))
            if len(pts) > 1:
                pts_arr = np.array(pts, dtype=np.int32)
                cv2.polylines(frame, [pts_arr], is_closed, color, thick, cv2.LINE_AA)

        # Draw Face Contours & Feature Meshes
        draw_path(self.FACE_OVAL, line_color, 1, True)
        draw_path(self.LEFT_EYEBROW, line_color, 1, False)
        draw_path(self.RIGHT_EYEBROW, line_color, 1, False)
        draw_path(self.NOSE, line_color, 1, False)
        draw_path(self.LIPS, line_color, 1, True)
        draw_path(self.LEFT_EYE, eye_color, 1, True)
        draw_path(self.RIGHT_EYE, eye_color, 1, True)

        # Draw Eyelid Vertical Measurement Indicators
        if 159 < len(landmarks) and 145 < len(landmarks):
            p1 = (int(landmarks[159].x * w), int(landmarks[159].y * h))
            p2 = (int(landmarks[145].x * w), int(landmarks[145].y * h))
            cv2.line(frame, p1, p2, (0, 255, 255) if not is_drowsy else (0, 0, 255), 2, cv2.LINE_AA)
        if 386 < len(landmarks) and 374 < len(landmarks):
            p1 = (int(landmarks[386].x * w), int(landmarks[386].y * h))
            p2 = (int(landmarks[374].x * w), int(landmarks[374].y * h))
            cv2.line(frame, p1, p2, (0, 255, 255) if not is_drowsy else (0, 0, 255), 2, cv2.LINE_AA)

        # Draw Neural Node Dots
        for idx in self.SAMPLE_MESH_POINTS:
            if idx < len(landmarks):
                px = int(landmarks[idx].x * w)
                py = int(landmarks[idx].y * h)
                cv2.circle(frame, (px, py), 1, node_color, -1)

    def process(self, frame):
        """
        Processes driver camera frame and returns (pip_frame, is_sleeping).
        """
        # Resize to PiP (Picture-in-Picture) size for the dashboard corner
        h, w = 195, 260
        if frame is None or frame.size == 0:
            standby = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.rectangle(standby, (4, 4), (w - 4, h - 4), (0, 100, 180), 1)
            cv2.putText(standby, "[ DMS : STANDBY ]", (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(standby, "CAMERA 0 CONNECTED", (35, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 160, 180), 1, cv2.LINE_AA)
            return standby, False

        pip_frame = cv2.resize(frame, (w, h))
        rgb_frame = cv2.cvtColor(pip_frame, cv2.COLOR_BGR2RGB)
        
        is_sleeping = False
        face_detected = False
        left_eye_dist = 0.03
        right_eye_dist = 0.03

        # Backend 1: Legacy mp.solutions
        if self.backend == "solutions" and self.face_mesh is not None:
            results = self.face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                face_detected = True
                face_landmarks = results.multi_face_landmarks[0]

                def dist_sol(p1, p2):
                    return math.hypot(face_landmarks.landmark[p1].x - face_landmarks.landmark[p2].x,
                                      face_landmarks.landmark[p1].y - face_landmarks.landmark[p2].y)

                left_eye_dist = dist_sol(159, 145)
                right_eye_dist = dist_sol(386, 374)

                is_currently_closed = (left_eye_dist < 0.015 and right_eye_dist < 0.015)
                if is_currently_closed:
                    self.sleep_frames += 1
                else:
                    self.sleep_frames = max(0, self.sleep_frames - 2)

                is_sleeping = (self.sleep_frames > 15)
                self.mp_drawing.draw_landmarks(
                    image=pip_frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

        # Backend 2: Modern MediaPipe Tasks API
        elif self.backend == "tasks" and self.detector is not None:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.detector.detect(mp_image)
            if results.face_landmarks and len(results.face_landmarks) > 0:
                face_detected = True
                landmarks = results.face_landmarks[0]

                def dist_task(p1, p2):
                    return math.hypot(landmarks[p1].x - landmarks[p2].x,
                                      landmarks[p1].y - landmarks[p2].y)

                left_eye_dist = dist_task(159, 145)
                right_eye_dist = dist_task(386, 374)

                is_currently_closed = (left_eye_dist < 0.015 and right_eye_dist < 0.015)
                if is_currently_closed:
                    self.sleep_frames += 1
                else:
                    self.sleep_frames = max(0, self.sleep_frames - 2)

                is_sleeping = (self.sleep_frames > 15)
                self._draw_neural_mesh(pip_frame, landmarks, w, h, is_sleeping)

        # Backend 3: Haar Cascade Fallback
        elif self.backend == "haar":
            gray = cv2.cvtColor(pip_frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.2, 4)
            if len(faces) > 0:
                face_detected = True
                fx, fy, fw, fh = faces[0]
                face_roi = gray[fy:fy + fh, fx:fx + fw]
                eyes = self.eye_cascade.detectMultiScale(face_roi, 1.15, 3)
                if len(eyes) == 0:
                    self.sleep_frames += 1
                else:
                    self.sleep_frames = max(0, self.sleep_frames - 2)
                is_sleeping = (self.sleep_frames > 15)
                cv2.rectangle(pip_frame, (fx, fy), (fx + fw, fy + fh), (0, 243, 255), 1)

        # Render HUD Status Overlay on PiP Frame
        avg_gap = (left_eye_dist + right_eye_dist) / 2.0
        if face_detected:
            if is_sleeping:
                # Flashing Blood-Red Warning Border
                cv2.rectangle(pip_frame, (0, 0), (w, h), (0, 0, 255), 6)
                cv2.rectangle(pip_frame, (6, 6), (w - 6, 42), (0, 0, 180), -1)
                cv2.putText(pip_frame, "DROWSY!", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.88, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(pip_frame, f"EYES SHUT ({self.sleep_frames}f)", (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                # Cyber Aqua / Cyan Awake Border
                cv2.rectangle(pip_frame, (0, 0), (w, h), (0, 255, 255), 2)
                cv2.rectangle(pip_frame, (6, 6), (145, 32), (10, 15, 25), -1)
                cv2.putText(pip_frame, "AWAKE", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2, cv2.LINE_AA)
                cv2.putText(pip_frame, f"GAP: {avg_gap:.3f}", (w - 95, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            # Standby / No Driver
            cv2.rectangle(pip_frame, (0, 0), (w, h), (80, 90, 100), 2)
            cv2.rectangle(pip_frame, (6, 6), (150, 30), (10, 15, 25), -1)
            cv2.putText(pip_frame, "NO DRIVER", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (140, 150, 160), 2, cv2.LINE_AA)

        self.is_sleeping = is_sleeping
        return pip_frame, is_sleeping
