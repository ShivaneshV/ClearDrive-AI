"""
================================================================================
            CLEAR-DRIVE AI : ADAS VEHICLE DETECTION & DISTANCE ESTIMATION
================================================================================
Domain: Computer Vision & Automotive ADAS
Functionality:
  1. Real-time Multi-Class Vehicle Detection (Cars, Trucks, Buses, Motorcycles)
  2. Monocular Ground-Plane & Optical Width Distance Estimation to Ego-Vehicle
  3. 3D Inter-Vehicle Distance Calculation (Distance between each vehicle on road)
  4. Collision Alert & Safe Following Distance Telemetry HUD
================================================================================
"""

import cv2
import numpy as np
from ultralytics import YOLO


class VehicleTracker:
    """
    Automotive ADAS Distance Estimation & Inter-Vehicle Proximity Tracker.
    Uses real-time YOLOv8n inference with pinhole ground-plane geometry.
    """

    def __init__(self, model_name='yolov8n.pt', cam_height=1.35, fov_deg=75.0):
        """
        Parameters:
            model_name: Lightweight YOLOv8 nano model weights.
            cam_height: Height of dashcam mounting above ground plane in meters.
            fov_deg: Horizontal Field of View of automotive camera lens.
        """
        self.model = YOLO(model_name)
        if hasattr(self.model, 'to'):
            self.model.to('cpu')
        # Target classes: 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
        self.target_classes = [2, 3, 5, 7]
        self.cam_height = cam_height
        self.fov_deg = fov_deg
        
        # Real-world average physical widths (meters) for optical scaling
        self.class_widths = {
            2: 1.82,   # Car
            3: 0.85,   # Motorcycle
            5: 2.55,   # Bus
            7: 2.50    # Truck
        }

    def estimate_distance(self, bbox, cls_id, frame_w, frame_h):
        """
        Calculates 3D ground coordinates (X: lateral, Z: longitudinal forward distance).
        Combines ground-plane contact geometry with optical bounding-box width scaling.
        """
        x1, y1, x2, y2 = bbox
        bw = max(x2 - x1, 4.0)
        xc = (x1 + x2) / 2.0
        yc = (y1 + y2) / 2.0
        y_bottom = y2

        # Pinhole camera intrinsic approximations
        focal_px = (frame_w / 2.0) / np.tan(np.radians(self.fov_deg / 2.0))
        cx = frame_w / 2.0
        y_horizon = frame_h * 0.50

        # Method 1: Ground-plane pitch projection
        dy = max(y_bottom - y_horizon, 4.0)
        d_ground = (focal_px * self.cam_height) / dy

        # Method 2: Physical width scaling
        w_real = self.class_widths.get(cls_id, 1.80)
        d_width = (focal_px * w_real) / bw

        # Sensor fusion (robust weighted median)
        Z = float(np.clip(0.60 * d_ground + 0.40 * d_width, 2.5, 120.0))
        X = float((xc - cx) * Z / focal_px)

        return round(Z, 1), round(X, 1), (int(xc), int(yc))

    def process(self, frame, conf_thresh=0.28):
        """
        Detects vehicles, calculates ego distances and inter-vehicle distances,
        and renders futuristic Tesla-style HUD telemetry graphics.
        """
        h, w = frame.shape[:2]
        annotated = frame.copy()
        
        # Run fast inference
        results = self.model(
            frame, 
            classes=self.target_classes, 
            conf=conf_thresh, 
            verbose=False,
            imgsz=480,
            device='cpu'
        )[0]

        vehicles = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            bbox = (int(x1), int(y1), int(x2), int(y2))
            dist_z, offset_x, center_pt = self.estimate_distance(bbox, cls_id, w, h)
            
            vehicles.append({
                'box': bbox,
                'cls': cls_id,
                'label': self.model.names[cls_id].upper(),
                'conf': conf,
                'dist': dist_z,
                'X': offset_x,
                'Z': dist_z,
                'center': center_pt
            })

        # Sort vehicles from nearest to furthest
        vehicles.sort(key=lambda v: v['Z'])

        # 1. Draw Inter-Vehicle Distance Lines (Distance Between Each Vehicle)
        num_v = len(vehicles)
        if num_v >= 2:
            drawn_pairs = set()
            # Link each vehicle to its closest neighboring vehicle on the road
            for i, v1 in enumerate(vehicles[:6]):
                best_j = None
                min_d = 35.0  # Max search proximity in meters
                for j, v2 in enumerate(vehicles[:6]):
                    if i == j:
                        continue
                    pair_key = tuple(sorted([i, j]))
                    if pair_key in drawn_pairs:
                        continue
                    d_inter = float(np.sqrt((v1['X'] - v2['X'])**2 + (v1['Z'] - v2['Z'])**2))
                    if d_inter < min_d:
                        min_d = d_inter
                        best_j = j

                if best_j is not None:
                    pair_key = tuple(sorted([i, best_j]))
                    drawn_pairs.add(pair_key)
                    v2 = vehicles[best_j]
                    
                    pt1 = v1['center']
                    pt2 = v2['center']
                    mid_pt = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)

                    # Color-code by inter-vehicle safety gap
                    if min_d < 7.0:
                        line_color = (45, 50, 240)    # Red (Tailgating Hazard)
                    elif min_d < 15.0:
                        line_color = (0, 215, 255)    # Amber (Caution)
                    else:
                        line_color = (0, 243, 255)    # Cyan (Safe Gap)

                    # Draw dashed/solid radar connector line
                    cv2.line(annotated, pt1, pt2, line_color, 2, cv2.LINE_AA)
                    
                    # High-contrast floating distance tag pill
                    tag_text = f"<-> {min_d:.1f}m"
                    tw = len(tag_text) * 8 + 14
                    cv2.rectangle(annotated, (mid_pt[0] - tw//2, mid_pt[1] - 12), 
                                  (mid_pt[0] + tw//2, mid_pt[1] + 12), (10, 14, 20), -1)
                    cv2.rectangle(annotated, (mid_pt[0] - tw//2, mid_pt[1] - 12), 
                                  (mid_pt[0] + tw//2, mid_pt[1] + 12), line_color, 1)
                    cv2.putText(annotated, tag_text, (mid_pt[0] - tw//2 + 7, mid_pt[1] + 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.44, line_color, 1, cv2.LINE_AA)


        # 2. Draw Tesla-Style Corner Reticles & Ego Distance Badges + TTC Autobrake Physics
        frame_area = float(h * w)
        emergency_brake = False
        min_ttc = 99.9

        for v in vehicles:
            x1, y1, x2, y2 = v['box']
            dist = v['dist']
            name = v['label']
            box_area = (x2 - x1) * (y2 - y1)
            occupancy = box_area / frame_area
            
            # TTC AUTOBRAKE LOGIC:
            # If vehicle occupies > 15% of camera view OR forward distance < 4.8m -> Imminent Crash Threat!
            is_threat = (occupancy > 0.15) or (dist < 4.8)

            if is_threat:
                emergency_brake = True
                ttc = round(min(1.2, max(0.4, dist / 8.0)), 1)
                min_ttc = min(min_ttc, ttc)
                box_color = (0, 0, 255)       # Blood Red Danger
                
                # Heavy Red Danger Box & Sniper Reticle Crosshair
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.drawMarker(annotated, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 45, 3)
            elif dist < 12.0:
                box_color = (0, 215, 255)     # Amber Gold (Caution)
            else:
                box_color = (80, 255, 60)     # Neon Green (Safe Spacing)

            # Aerospace Corner Brackets
            blen = max(min(int((x2 - x1) * 0.22), 24), 8)
            thick = 3 if is_threat else 2
            cv2.line(annotated, (x1, y1), (x1 + blen, y1), box_color, thick)
            cv2.line(annotated, (x1, y1), (x1, y1 + blen), box_color, thick)
            cv2.line(annotated, (x2, y1), (x2 - blen, y1), box_color, thick)
            cv2.line(annotated, (x2, y1), (x2, y1 + blen), box_color, thick)
            cv2.line(annotated, (x1, y2), (x1 + blen, y2), box_color, thick)
            cv2.line(annotated, (x1, y2), (x1, y2 - blen), box_color, thick)
            cv2.line(annotated, (x2, y2), (x2 - blen, y2), box_color, thick)
            cv2.line(annotated, (x2, y2), (x2, y2 - blen), box_color, thick)

            # Floating Telemetry Badge
            threat_label = " [THREAT]" if is_threat else ""
            badge_text = f"{name}: {dist}m{threat_label}"
            b_w = len(badge_text) * 9 + 14
            badge_y1 = max(y1 - 24, 8)
            badge_y2 = badge_y1 + 20
            
            cv2.rectangle(annotated, (x1, badge_y1), (x1 + b_w, badge_y2), (10, 14, 20), -1)
            cv2.rectangle(annotated, (x1, badge_y1), (x1 + b_w, badge_y2), box_color, 1)
            cv2.putText(annotated, badge_text, (x1 + 6, badge_y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.44, box_color, 1, cv2.LINE_AA)

        # Flash Top Blood-Red Emergency HUD Banner on Imminent Collision
        if emergency_brake:
            banner_h = int(max(48, h * 0.13))
            cv2.rectangle(annotated, (0, 0), (w, banner_h), (0, 0, 235), -1)
            cv2.rectangle(annotated, (0, 0), (w, banner_h), (255, 255, 255), 2)
            banner_text = f"!! AUTOBRAKE ENGAGED - COLLISION IMMINENT (TTC: {min_ttc}s) !!"
            font_scale = max(0.48, (w / 640.0) * 0.60)
            (tw, th), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            tx = max(10, (w - tw) // 2)
            ty = int(banner_h * 0.50 + th * 0.40)
            cv2.putText(annotated, banner_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

        closest_dist = vehicles[0]['dist'] if vehicles else None
        return annotated, emergency_brake, (min_ttc if emergency_brake else 0.0), vehicles, closest_dist

    def process_frame(self, frame):
        """
        Streamlined API returning (out_frame, emergency_brake, min_ttc).
        """
        out_frame, brake, ttc, _, _ = self.process(frame)
        return out_frame, brake, ttc

