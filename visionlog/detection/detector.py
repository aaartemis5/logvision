"""
detection/detector.py
YOLO-pose + ByteTrack wrapper.
Returns a list of Detection objects per frame — one per tracked person.
"""
import math
import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from ultralytics import YOLO
from visionlog import config


@dataclass
class Detection:
    local_id:       int             # ByteTrack ID (per-camera, resets on restart)
    global_id:      Optional[str]   # Re-ID global (Phase 2 — None for now)
    camera_id:      str
    frame_number:   int
    timestamp:      float           # unix time
    bbox:           tuple           # (x1, y1, x2, y2)
    confidence:     float
    zone:           str
    position_pct:   dict            # {x: 0.0-1.0, y: 0.0-1.0}
    keypoints:      list            # raw [(x,y)] * 17
    kp_conf:        list            # confidence per keypoint
    pose:           str             # STANDING/SITTING/CROUCHING/LYING/RUNNING/UNKNOWN
    previous_pose:  str
    arms_raised:    bool
    lean:           str
    speed_px:       float           # pixel movement vs last frame
    movement_state: str             # stationary / walking / running


class Detector:
    """
    Runs YOLO-pose inference + ByteTrack on each frame.
    Maintains per-person state: pose history, speed, dwell time.
    """

    def __init__(self, camera_id: str, frame_w: int, frame_h: int,
                 zone_fn=None):
        self.camera_id  = camera_id
        self.frame_w    = frame_w
        self.frame_h    = frame_h
        self.zone_fn    = zone_fn       # callable: bbox -> zone name

        self.model = YOLO(config.YOLO_MODEL)
        print(f"[Detector] {camera_id}: loaded {config.YOLO_MODEL}")

        # Per-track state memory
        self._prev_center:  dict[int, tuple] = {}   # id → last (cx,cy)
        self._prev_pose:    dict[int, str]   = {}   # id → last pose label
        self._speed_hist:   dict[int, list]  = {}   # id → rolling speed list

    # ── Keypoint indices ─────────────────────────────────────────────
    _NOSE, _L_SH, _R_SH = 0, 5, 6
    _L_HIP, _R_HIP       = 11, 12
    _L_KNEE, _R_KNEE      = 13, 14
    _L_ANK, _R_ANK        = 15, 16
    _L_WR, _R_WR          = 9, 10

    def _kp(self, xy, conf, idx, min_c=0.3):
        return xy[idx] if conf[idx] >= min_c else None

    def _mid(self, a, b):
        if a is None or b is None: return None
        return ((a[0]+b[0])/2, (a[1]+b[1])/2)

    def _dist(self, a, b):
        if a is None or b is None: return None
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def _classify_pose(self, xy, conf, bbox_h):
        ls = self._kp(xy, conf, self._L_SH)
        rs = self._kp(xy, conf, self._R_SH)
        lh = self._kp(xy, conf, self._L_HIP)
        rh = self._kp(xy, conf, self._R_HIP)
        lk = self._kp(xy, conf, self._L_KNEE)
        rk = self._kp(xy, conf, self._R_KNEE)
        la = self._kp(xy, conf, self._L_ANK)
        ra = self._kp(xy, conf, self._R_ANK)
        lw = self._kp(xy, conf, self._L_WR)
        rw = self._kp(xy, conf, self._R_WR)

        sh_mid  = self._mid(ls, rs)
        hip_mid = self._mid(lh, rh)
        kn_mid  = self._mid(lk, rk)
        an_mid  = self._mid(la, ra)

        pose = "UNKNOWN"
        if sh_mid and hip_mid:
            torso = self._dist(sh_mid, hip_mid) or 1
            if an_mid:
                vspan = abs(sh_mid[1] - an_mid[1])
                if vspan < bbox_h * 0.35:
                    pose = "LYING DOWN"
                elif kn_mid and abs(hip_mid[1] - kn_mid[1]) < bbox_h * 0.15:
                    pose = "SITTING"
                elif kn_mid and self._dist(hip_mid, kn_mid) and self._dist(hip_mid, kn_mid) < torso * 0.6:
                    pose = "CROUCHING"
                else:
                    pose = "STANDING"
            elif kn_mid:
                pose = "SITTING" if self._dist(hip_mid, kn_mid) and self._dist(hip_mid, kn_mid) < torso * 0.6 else "STANDING"
            else:
                pose = "STANDING"

        # Arms raised?
        arms_raised = False
        if sh_mid:
            for wr in [lw, rw]:
                if wr and wr[1] < sh_mid[1] - 20:
                    arms_raised = True

        # Lean
        lean = "centre"
        if ls and rs:
            slope = rs[1] - ls[1]
            if slope > 15:   lean = "leaning LEFT"
            elif slope < -15: lean = "leaning RIGHT"

        return pose, arms_raised, lean

    def process(self, frame, frame_number: int) -> list[Detection]:
        """Run inference on one frame, return list of Detection objects."""
        ts = time.time()
        results = self.model.track(
            frame,
            classes=config.YOLO_CLASSES,
            conf=config.YOLO_CONF,
            tracker=config.TRACKER,
            persist=True,
            verbose=False
        )
        r = results[0]
        detections = []

        if r.boxes is None or r.keypoints is None:
            return detections

        for box, kps in zip(r.boxes, r.keypoints):
            if box.id is None:
                continue

            tid   = int(box.id.item())
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            conf  = box.conf.item()
            bbox_h = max(y2 - y1, 1)

            kp_xy   = kps.xy[0].tolist()
            kp_conf = kps.conf[0].tolist() if kps.conf is not None else [1.0]*17

            # Pose
            pose, arms, lean = self._classify_pose(kp_xy, kp_conf, bbox_h)
            prev_pose = self._prev_pose.get(tid, pose)
            self._prev_pose[tid] = pose

            # Speed
            cx, cy = (x1+x2)//2, (y1+y2)//2
            prev   = self._prev_center.get(tid)
            spd    = math.hypot(cx - prev[0], cy - prev[1]) if prev else 0.0
            self._prev_center[tid] = (cx, cy)

            hist = self._speed_hist.setdefault(tid, [])
            hist.append(spd)
            if len(hist) > 30: hist.pop(0)
            avg_spd = sum(hist) / len(hist) if hist else 1
            if (spd > avg_spd * config.SPEED_SPIKE_X
                    and spd > config.MIN_RUNNING_PX
                    and pose == "STANDING"):
                pose = "RUNNING"
            move_state = "stationary" if spd < 3 else ("running" if spd > 15 else "walking")

            # Zone
            zone = self.zone_fn((x1,y1,x2,y2)) if self.zone_fn else "UNKNOWN"

            detections.append(Detection(
                local_id      = tid,
                global_id     = None,
                camera_id     = self.camera_id,
                frame_number  = frame_number,
                timestamp     = ts,
                bbox          = (x1, y1, x2, y2),
                confidence    = conf,
                zone          = zone,
                position_pct  = {"x": round(cx/self.frame_w, 3),
                                  "y": round(cy/self.frame_h, 3)},
                keypoints     = kp_xy,
                kp_conf       = kp_conf,
                pose          = pose,
                previous_pose = prev_pose,
                arms_raised   = arms,
                lean          = lean,
                speed_px      = round(spd, 2),
                movement_state= move_state,
            ))

        return detections
