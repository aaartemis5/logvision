"""
enrichment/head_pose_tracker.py

Estimates head direction (yaw) from YOLO-pose facial keypoints and detects
"surveillance-checking" behaviour — the rapid looking-around pattern seen
before shoplifting, assault, or other pre-crime behaviour.

Method: uses ONLY the 5 facial YOLO keypoints (no extra model required)
  KP 0: nose
  KP 1: left eye      KP 2: right eye
  KP 3: left ear      KP 4: right ear

Head yaw logic:
  - Both ears visible, symmetric      → FRONT (facing camera)
  - Left ear hidden, right ear visible → LOOKING LEFT  (from person's POV)
  - Right ear hidden, left ear visible → LOOKING RIGHT
  - Only nose visible, no ears         → AWAY (back to camera)

Scan detection:
  A "scan" = direction changes from LEFT→RIGHT or RIGHT→LEFT within the
  same 10-second window. ≥3 scans in 10s → suspicious surveillance loop.
"""
import math
import time
from collections import deque


# ── Head direction constants ──────────────────────────────────────────
DIR_FRONT = "front"
DIR_LEFT  = "looking_left"
DIR_RIGHT = "looking_right"
DIR_AWAY  = "away"
DIR_UNK   = "unknown"

CONF_MIN  = 0.3    # minimum keypoint confidence to count as visible

# KP indices (YOLO COCO order)
_NOSE, _L_EYE, _R_EYE, _L_EAR, _R_EAR = 0, 1, 2, 3, 4


def _visible(kp_conf: list, idx: int) -> bool:
    return len(kp_conf) > idx and kp_conf[idx] >= CONF_MIN


def estimate_yaw(kp_xy: list, kp_conf: list) -> str:
    """
    Returns one of: DIR_FRONT, DIR_LEFT, DIR_RIGHT, DIR_AWAY, DIR_UNK
    Uses the visibility pattern of ears, eyes, and nose.
    """
    has_l_ear  = _visible(kp_conf, _L_EAR)
    has_r_ear  = _visible(kp_conf, _R_EAR)
    has_l_eye  = _visible(kp_conf, _L_EYE)
    has_r_eye  = _visible(kp_conf, _R_EYE)
    has_nose   = _visible(kp_conf, _NOSE)

    if not has_nose and not has_l_eye and not has_r_eye:
        return DIR_UNK

    # Both ears hidden → away from camera
    if not has_l_ear and not has_r_ear and not has_l_eye and not has_r_eye:
        return DIR_AWAY

    # Both ears visible → roughly front-facing
    if has_l_ear and has_r_ear:
        # Refine using eye symmetry vs nose
        if has_l_eye and has_r_eye and has_nose:
            nose_x  = kp_xy[_NOSE][0]
            l_eye_x = kp_xy[_L_EYE][0]
            r_eye_x = kp_xy[_R_EYE][0]
            eye_mid = (l_eye_x + r_eye_x) / 2
            offset  = nose_x - eye_mid
            # If nose is offset significantly from eye midpoint → slight turn
            if offset < -8:   return DIR_LEFT
            if offset > 8:    return DIR_RIGHT
        return DIR_FRONT

    # Only left ear → person turned right (their left ear shows when facing right)
    if has_l_ear and not has_r_ear:
        return DIR_RIGHT

    # Only right ear → person turned left
    if has_r_ear and not has_l_ear:
        return DIR_LEFT

    # Fallback: use eye visibility
    if has_l_eye and not has_r_eye:
        return DIR_RIGHT
    if has_r_eye and not has_l_eye:
        return DIR_LEFT

    return DIR_FRONT


class HeadPoseTracker:
    """
    Per-person head direction history tracker.
    Detects "surveillance checking" — rapid left/right scan pattern.

    Usage:
        tracker = HeadPoseTracker()
        result  = tracker.update(det.local_id, det.keypoints, det.kp_conf)
    """

    def __init__(self,
                 window_s: float = 10.0,  # sliding window in seconds
                 scan_threshold: int = 3): # scans in window → suspicious
        self._window_s      = window_s
        self._scan_threshold = scan_threshold
        # per-person: deque of (timestamp, direction)
        self._history:  dict[int, deque] = {}
        self._last_dir: dict[int, str]   = {}

    def update(self, local_id: int, kp_xy: list, kp_conf: list) -> dict:
        """
        Returns:
          {
            direction:         str,    # current estimated head direction
            scans_in_window:   int,    # direction reversals in last N seconds
            scan_rate_per_min: float,  # extrapolated scan rate
            surveillance_flag: bool,   # True if scan_threshold exceeded
          }
        """
        now = time.time()
        direction = estimate_yaw(kp_xy, kp_conf)

        # Get or create history for this person
        hist = self._history.setdefault(local_id, deque())

        # Append current direction entry
        hist.append((now, direction))

        # Prune entries older than the window
        while hist and (now - hist[0][0]) > self._window_s:
            hist.popleft()

        # Count direction reversals (scans) in the window
        scans = 0
        last_horiz = None
        for _, d in hist:
            if d in (DIR_LEFT, DIR_RIGHT):
                if last_horiz is not None and d != last_horiz:
                    scans += 1
                last_horiz = d

        # Rate per minute (extrapolate from window)
        elapsed = min(now - hist[0][0], self._window_s) if hist else 1.0
        rate_per_min = round((scans / max(elapsed, 0.1)) * 60, 1)

        self._last_dir[local_id] = direction

        return {
            "direction":         direction,
            "scans_in_window":   scans,
            "scan_rate_per_min": rate_per_min,
            "surveillance_flag": scans >= self._scan_threshold,
        }

    def evict(self, active_ids: set[int]):
        """Remove history for persons no longer in frame."""
        for tid in list(self._history.keys()):
            if tid not in active_ids:
                del self._history[tid]
                self._last_dir.pop(tid, None)
