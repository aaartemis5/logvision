"""
events/event_builder.py
Assembles the final JSON event from all module outputs.
This is the single source of truth for the event schema.
"""
import uuid
import datetime
from visionlog.detection.detector import Detection


def build_event(
    det:           Detection,
    anomaly:       dict,
    session_id:    str,
    camera_meta:   dict,
    person_attrs:  dict | None,
    scene:         dict | None,
    first_seen_ts: float,
    hand_result:   dict | None = None,   # from MediaPipe HandTracker
    head_result:   dict | None = None,   # from HeadPoseTracker
    risk_data:     dict | None = None,   # from RiskScorer / BehaviourWindow
) -> dict:
    """
    Builds the full event JSON dict. This is what gets embedded + sent to Pinecone.
    """
    now_dt  = datetime.datetime.fromtimestamp(det.timestamp)
    hour    = now_dt.hour
    nh_s, nh_e = 22, 6
    is_night    = hour >= nh_s or hour < nh_e

    # ── Person attributes (from DeepFace/colour enricher) ────────────
    pa = person_attrs or {}
    hr = hand_result  or {}
    hd = head_result  or {}
    rd = risk_data    or {}
    dwell_s = round(det.timestamp - first_seen_ts, 1)

    # ── Scene (from Florence-2) ───────────────────────────────────────
    sc = scene or {}

    # ── Build text_summary (this is the embedding target) ────────────
    anom_tag = f"⚠️ {anomaly['severity']}: {', '.join(anomaly['reasons'])}" \
               if anomaly["is_anomaly"] else ""
    vlm_line = f"VLM: '{sc.get('vlm_caption', '')}'" \
                    if sc.get("vlm_caption") else ""
    gesture_tag = (
        f" | Gesture: {hr.get('gesture','idle').upper()}"
        + (" 🚨 BAG_STUFFING" if hr.get('bag_stuff_suspected') else "")
    ) if hr.get("gesture") and hr["gesture"] != "idle" else ""

    head_tag = (
        f" | Head: {hd.get('direction','unknown').upper()}"
        + (f" 👁 SURVEILLANCE_CHECKING ({hd.get('scans_in_window',0)} scans)"
           if hd.get('surveillance_flag') else "")
    ) if hd.get("direction") and hd["direction"] not in ("unknown", "away") else ""

    # Risk tag — cumulative score, patterns, zone alerts
    risk_score    = rd.get("score", 0)
    risk_severity = rd.get("final_severity")
    patterns      = rd.get("patterns", [])
    zone_alert    = rd.get("zone_alert")

    risk_tag = ""
    if risk_score and risk_score >= 25:
        risk_tag = f" | Risk: {risk_score:.0f}"
        if risk_severity:
            risk_tag += f" [{risk_severity}]"
    if patterns:
        risk_tag += f" | Patterns: {', '.join(patterns)}"
    if zone_alert:
        risk_tag += (f" | ⚠️ ZONE_TRANSITION "
                     f"{zone_alert['from']}→{zone_alert['to']} ({zone_alert['dwell_s']}s)")

    text_summary = (
        f"{now_dt.strftime('%H:%M:%S')} | {camera_meta['camera_id']} ({camera_meta['name']}) | "
        f"Person #{det.local_id} ({pa.get('gender','?')}, ~{pa.get('age','?')}yo, "
        f"{pa.get('clothing_desc','?')}) | "
        f"Pose: {det.previous_pose} \u2192 {det.pose} | Zone: {det.zone} | "
        f"Dwell: {dwell_s}s | Move: {det.movement_state}{gesture_tag}{head_tag}"
        f"{risk_tag}. {vlm_line} {anom_tag}"
    ).strip()

    return {
        "event_id":   str(uuid.uuid4()),
        "session_id": session_id,
        "frame_number": det.frame_number,

        "time": {
            "timestamp":       now_dt.isoformat() + "Z",
            "unix":            round(det.timestamp, 3),
            "time_of_day":     _time_of_day(hour),
            "day_of_week":     now_dt.strftime("%A"),
            "is_business_hours": 9 <= hour < 18,
            "is_night":        is_night,
        },

        "camera": camera_meta,

        "person": {
            "local_id":             det.local_id,
            "global_id":            det.global_id,
            "first_seen_ts":        first_seen_ts,
            "dwell_seconds":        dwell_s,
            "age_estimate":         pa.get("age"),
            "gender":               pa.get("gender"),
            "emotion":              pa.get("emotion"),
            "clothing_desc":       pa.get("clothing_desc"),
            "hand_gesture":         hr.get("gesture"),
            "bag_stuff_suspected":  hr.get("bag_stuff_suspected", False),
            "head_direction":       hd.get("direction"),
            "head_scan_rate_rpm":   hd.get("scan_rate_per_min"),
            "surveillance_flag":    hd.get("surveillance_flag", False),
            "risk_score":           round(rd.get("score", 0), 1),
            "risk_severity":        rd.get("severity"),
            "behaviour_patterns":   rd.get("patterns", []),
            "interacting_with_ids": [],
        },

        "detection": {
            "confidence":   round(det.confidence, 3),
            "bbox":         {"x1": det.bbox[0], "y1": det.bbox[1],
                             "x2": det.bbox[2], "y2": det.bbox[3]},
            "zone":         det.zone,
            "position_pct": det.position_pct,
        },

        "pose": {
            "label":          det.pose,
            "previous_label": det.previous_pose,
            "arms_raised":    det.arms_raised,
            "lean":           det.lean,
            "speed_px_per_frame": det.speed_px,
            "movement_state": det.movement_state,
        },

        "scene": {
            "vlm_caption":           sc.get("vlm_caption"),
            "person_count_in_frame": sc.get("person_count", 1),
            "ocr_text":              sc.get("ocr_text"),
        },

        "event_type": _infer_event_type(det),

        "anomaly": anomaly,

        "video_clip_saved": None,   # set by main.py if clip is saved

        "text_summary": text_summary,

        "_meta": {
            "vlm_ran":           scene is not None,
            "insightface_ran":   person_attrs is not None,
            "hands_ran":         hand_result is not None,
            "head_pose_ran":     head_result is not None,
            "risk_scorer_ran":   risk_data is not None,
            "schema_version":    "1.1",
        }
    }


def _time_of_day(hour: int) -> str:
    if 5  <= hour < 12: return "morning"
    if 12 <= hour < 17: return "afternoon"
    if 17 <= hour < 21: return "evening"
    return "night"


def _infer_event_type(det: Detection) -> str:
    if det.pose != det.previous_pose:
        return "POSE_CHANGE"
    if det.movement_state == "running":
        return "RUNNING"
    return "UPDATE"
