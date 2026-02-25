"""
anomaly/anomaly_detector.py
Rule-based anomaly detection.
Each rule adds confidence weight. Final score determines is_anomaly + severity.
"""
import time
import datetime
from visionlog import config
from visionlog.detection.detector import Detection


SEVERITY_MAP = [
    (0.90, "CRITICAL"),
    (0.70, "HIGH"),
    (0.40, "MEDIUM"),
    (0.10, "LOW"),
    (0.00, "NORMAL"),
]


class AnomalyDetector:
    """
    Stateful rule engine. Tracks dwell time and history per person per camera.
    Call evaluate(detection) each frame to get anomaly verdict.
    """

    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self._first_seen:   dict[int, float] = {}
        self._zone_entry:   dict[int, dict]  = {}   # {id: {zone: entry_time}}
        self._anomalous:    set[int]          = set()   # ids flagged this session

    def evaluate(self, det: Detection,
                 hand_result: dict | None = None,
                 head_result: dict | None = None) -> dict:
        """
        Returns anomaly dict:
        {is_anomaly, confidence, severity, reasons, rule}
        """
        tid = det.local_id
        now = det.timestamp

        # Track first seen
        if tid not in self._first_seen:
            self._first_seen[tid] = now

        # Track zone entry time
        zone_times = self._zone_entry.setdefault(tid, {})
        if det.zone not in zone_times:
            zone_times[det.zone] = now
        zone_dwell = now - zone_times[det.zone]

        reasons    = []
        confidence = 0.0

        # ── Rule: LYING DOWN ─────────────────────────────────────────
        if det.pose == "LYING DOWN":
            reasons.append("person_lying_down")
            confidence += 0.55

        # ── Rule: ARMS RAISED ─────────────────────────────────────────
        if det.arms_raised:
            reasons.append("arms_raised")
            confidence += 0.35

        # ── Rule: RUNNING ─────────────────────────────────────────────
        if det.pose == "RUNNING":
            reasons.append("running_detected")
            confidence += 0.25

        # ── Rule: CROUCHING NEAR SHELF ────────────────────────────────
        if det.pose == "CROUCHING" and det.zone == "SHELF_AREA":
            if zone_dwell > config.CROUCH_NEAR_SHELF_S:
                reasons.append("crouching_near_shelf")
                confidence += 0.40

        # ── Rule: DWELL TIME ─────────────────────────────────────────
        # Skip zones where long dwell is perfectly normal (checkout queue etc.)
        if det.zone not in config.DWELL_EXEMPT_ZONES:
            # DISPLAY_AREA gets a more lenient threshold — browsing displays is normal
            med_threshold  = (config.DWELL_MEDIUM_S * 1.5
                              if det.zone == "DISPLAY_AREA"
                              else config.DWELL_MEDIUM_S)
            high_threshold = (config.DWELL_HIGH_S * 1.5
                              if det.zone == "DISPLAY_AREA"
                              else config.DWELL_HIGH_S)
            if zone_dwell > high_threshold:
                reasons.append(f"dwell_exceeded_{int(zone_dwell)}s")
                confidence += 0.30
            elif zone_dwell > med_threshold:
                reasons.append(f"dwell_exceeded_{int(zone_dwell)}s")
                confidence += 0.15

        # ── Rule: LATE NIGHT ─────────────────────────────────────────
        hour = datetime.datetime.fromtimestamp(now).hour
        nh_start, nh_end = config.NIGHT_HOURS
        is_night = hour >= nh_start or hour < nh_end
        if is_night and reasons:   # night hours amplify existing rules
            confidence += 0.15
            reasons.append("late_night_activity")

        # ── Rule: POSE STATE CHANGED ──────────────────────────────────
        if det.pose != det.previous_pose and det.pose in ("CROUCHING", "LYING DOWN"):
            reasons.append(f"pose_changed_to_{det.pose.lower().replace(' ','_')}")
            confidence += 0.10

        # ── Rule: SURVEILLANCE CHECKING (head pose scanning) ────────────
        if head_result and head_result.get("surveillance_flag"):
            scans = head_result.get("scans_in_window", 0)
            rate  = head_result.get("scan_rate_per_min", 0)
            reasons.append(f"surveillance_checking_{scans}scans_{int(rate)}rpm")
            confidence += 0.40   # strong signal when combined with dwell

        # ── Rule: CONCEALING / BAG-STUFFING (MediaPipe Hands) ────────────
        if hand_result and hand_result.get("bag_stuff_suspected"):
            reasons.append("concealing_gesture")
            confidence += 0.50   # highest-weight rule — very specific signal

        # ── Rule: REACHING IN SHELF ZONE ─────────────────────────────────
        if hand_result and hand_result.get("gesture") == "reaching" \
                and det.zone == "SHELF_AREA":
            reasons.append("reaching_in_shelf")
            confidence += 0.15

        # Clip to [0, 1]
        confidence = min(round(confidence, 3), 1.0)

        # Determine severity
        severity = "NORMAL"
        for thresh, label in SEVERITY_MAP:
            if confidence >= thresh:
                severity = label
                break

        is_anomaly = confidence >= 0.10
        primary_rule = reasons[0].upper() if reasons else "NONE"

        if is_anomaly:
            self._anomalous = getattr(self, '_anomalous', set())
            self._anomalous.add(tid)
        else:
            getattr(self, '_anomalous', set()).discard(tid)

        return {
            "is_anomaly":   is_anomaly,
            "confidence":   confidence,
            "severity":     severity,
            "reasons":      reasons,
            "rule":         primary_rule,
            "zone_dwell_s": round(zone_dwell, 1),
        }

    def is_currently_anomalous(self, local_id: int) -> bool:
        """True if the person was flagged anomalous in the most recent evaluate()."""
        return local_id in getattr(self, '_anomalous', set())

    def cleanup_stale(self, active_ids: set[int]):
        """Remove tracking data for persons no longer in frame."""
        for tid in list(self._first_seen.keys()):
            if tid not in active_ids:
                del self._first_seen[tid]
                self._zone_entry.pop(tid, None)
                getattr(self, '_anomalous', set()).discard(tid)
