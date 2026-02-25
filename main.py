"""
main.py — VisionLog AI entry point.
Usage:
    python main.py                          # uses config.CAMERAS defaults
    python main.py --camera CAM_01          # run a specific camera only
    python main.py --source myvideo.mp4     # override video path for CAM_01
    python main.py --no-pinecone            # log locally only (no Pinecone needed)

Output:
    - Terminal: real-time event stream
    - visionlog_events.log: human-readable log
    - visionlog_events.jsonl: full JSON events (if Pinecone not configured)
    - clips/: 30s anomaly clips (if SAVE_ANOMALY_CLIPS = True in config)
"""
import argparse
import re
import sys
import time
import uuid
import cv2

from visionlog import config
from visionlog.sources.video_source import VideoSource, SourceConfig
from visionlog.detection.detector   import Detector
from visionlog.zones.zone_manager   import ZoneManager
from visionlog.enrichment.attribute_enricher import AttributeEnricher
from visionlog.enrichment.reid_gallery        import ReIdGallery
from visionlog.enrichment.hand_tracker        import HandTracker
from visionlog.enrichment.head_pose_tracker  import HeadPoseTracker
from visionlog.enrichment.scene_describer    import SceneDescriber
from visionlog.anomaly.anomaly_detector      import AnomalyDetector
from visionlog.anomaly.behaviour_window      import BehaviourWindow
from visionlog.anomaly.risk_scorer           import RiskScorer, AlertEscalator
from visionlog.events.event_builder          import build_event
from visionlog.events.event_store            import store


def _norm_reason(r: str) -> str:
    """Strip trailing '_<N>s' so dwell_exceeded_113s ≡ dwell_exceeded."""
    return re.sub(r'_\d+s?$', '', r)


def run_camera(cam_id: str, cam_cfg: dict, use_pinecone: bool):
    """Run the full pipeline for a single camera."""
    print(f"\n{'='*55}")
    print(f"  VisionLog AI — {cam_cfg['name']} ({cam_id})")
    print(f"  Source : {cam_cfg['path']}")
    print(f"  Pinecone: {'enabled' if use_pinecone else 'disabled (logging locally)'}")
    print(f"{'='*55}\n")

    # ── Initialise all modules ───────────────────────────────────────
    src_cfg = SourceConfig(
        camera_id = cam_id,
        name      = cam_cfg["name"],
        path      = cam_cfg["path"],
        location  = cam_cfg.get("location", ""),
        lat       = cam_cfg.get("lat"),
        lon       = cam_cfg.get("lon"),
        floor     = cam_cfg.get("floor", 0),
        direction = cam_cfg.get("direction", ""),
    )
    source    = VideoSource(src_cfg)
    zones     = ZoneManager(cam_id, source.width, source.height)
    detector  = Detector(cam_id, source.width, source.height,
                         zone_fn=zones.get_zone)
    enricher  = AttributeEnricher()
    gallery   = ReIdGallery()          # Re-ID: persists global IDs across ByteTrack resets
    hand      = HandTracker()
    head      = HeadPoseTracker(window_s=10.0, scan_threshold=3)
    describer = SceneDescriber()
    anomaly   = AnomalyDetector(cam_id)
    behaviour = BehaviourWindow(window_s=30.0)
    scorer    = RiskScorer()
    escalator = AlertEscalator(window_s=60.0)

    cam_meta       = source.camera_meta()
    session_id     = f"RUN_{time.strftime('%Y%m%d_%H%M%S')}"
    first_seen_map = {}   # local_id → first seen unix time

    # ── Event-trigger: only log on real changes ──────────────────────
    last_state:   dict[int, dict]          = {}  # local_id → {pose, zone}
    smooth_bboxes: dict[int, tuple]         = {}  # local_id → EMA-smoothed display bbox
    _SMOOTH_ALPHA = 0.35  # 0=frozen, 1=raw jitter — 0.35 gives a nice glide
    last_detections:  list  = []          # persists across skipped frames
    last_proc_scale:  tuple = (1.0, 1.0)  # (scale_x, scale_y)
    last_anomalous:   set   = set()

    def _draw_boxes(canvas, dets, sx, sy, s_bboxes, alpha, anom_ids):
        """EMA-smooth then draw bboxes + contrast labels. Modifies canvas in-place."""
        for det in dets:
            x1, y1, x2, y2 = det.bbox
            x1, y1, x2, y2 = x1*sx, y1*sy, x2*sx, y2*sy
            tid = det.local_id
            if tid in s_bboxes:
                bx1, by1, bx2, by2 = s_bboxes[tid]
                x1 = alpha*x1 + (1-alpha)*bx1
                y1 = alpha*y1 + (1-alpha)*by1
                x2 = alpha*x2 + (1-alpha)*bx2
                y2 = alpha*y2 + (1-alpha)*by2
            s_bboxes[tid] = (x1, y1, x2, y2)
            ix1,iy1,ix2,iy2 = int(x1),int(y1),int(x2),int(y2)
            color = (0,60,255) if tid in anom_ids else (0,220,100)
            cv2.rectangle(canvas, (ix1,iy1), (ix2,iy2), color, 2)
            # Label: show global_id badge [Gn] when available
            gid_badge = f" [{det.global_id}]" if det.global_id else ""
            label = f"#{tid}{gid_badge} {det.pose}"
            font, fs, th_px = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            (tw, tah), bl = cv2.getTextSize(label, font, fs, th_px)
            lx = ix1
            ly = max(iy1 - 4, tah + 4)
            # Dark background rect for readability
            cv2.rectangle(canvas, (lx, ly-tah-bl), (lx+tw+4, ly+bl), (0,0,0), -1)
            cv2.putText(canvas, label, (lx+2, ly), font, fs, color, th_px)

    # Reason key is normalised (strip trailing numbers) so that
    # dwell_exceeded_113s and dwell_exceeded_114s count as the same type.
    cooldown_map: dict[int, dict[str, float]] = {}

    print(f"[Main] Session {session_id} started. Press Q in window to stop.\n")

    for frame, frame_meta in source:
        fn = frame_meta["frame_number"]

        # ── Frame skip ──────────────────────────────────────────────────
        if config.FRAME_SKIP > 1 and fn % config.FRAME_SKIP != 0:
            # Redraw last detections on current raw frame (smooth display)
            canvas = frame.copy()
            _draw_boxes(canvas, last_detections, *last_proc_scale,
                        smooth_bboxes, _SMOOTH_ALPHA, last_anomalous)
            cv2.putText(canvas, f"{cam_id} | {cam_cfg['name']} | Q=quit",
                        (10, canvas.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
            cv2.imshow(f"VisionLog — {cam_id}", canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # ── Optional resize (faster inference on all downstream models) ────
        if config.PROCESS_WIDTH and frame.shape[1] > config.PROCESS_WIDTH:
            scale  = config.PROCESS_WIDTH / frame.shape[1]
            proc_h = int(frame.shape[0] * scale)
            proc_frame = cv2.resize(frame, (config.PROCESS_WIDTH, proc_h))
        else:
            proc_frame = frame

        # ── Kick off async Florence (non-blocking) ──────────────────────
        if describer.should_run(cam_id):
            describer.request(proc_frame, cam_id)   # fires thread, returns immediately

        # ── Run detector on (possibly resized) frame ─────────────────────
        detections = detector.process(proc_frame, fn)
        active_ids = {d.local_id for d in detections}

        # Get latest cached Florence result (set by background thread)
        scene = describer.last_result(cam_id)

        for det in detections:
            tid = det.local_id

            # Track first-seen time
            if tid not in first_seen_map:
                first_seen_map[tid] = det.timestamp

            # ── Attribute enrichment (once per new person) ────────────
            attrs = enricher.enrich(det, proc_frame)  # proc_frame: bbox coords match

            # ── Re-ID: match appearance → stable global_id ─────────────
            det.global_id = gallery.match(
                local_id  = tid,
                embedding = attrs.get('clip_embedding'),
                zone      = det.zone,
                clothing  = attrs.get('clothing_desc', 'unknown'),
            )

            # ── Hand gesture tracking (every frame) ───────────────────
            hand_result = hand.track(det.local_id, det.bbox, proc_frame)

            # ── Head pose + surveillance-checking detection ────────────
            head_result = head.update(det.local_id, det.keypoints, det.kp_conf)

            # ── Anomaly check (passes hand + head context) ───────────
            anom_result = anomaly.evaluate(det, hand_result, head_result)

            # ── Decide whether to log this frame ─────────────────────
            prev = last_state.get(tid, {})

            # Fix: only treat pose change as meaningful between KNOWN poses
            # (ignore UNKNOWN↔STANDING oscillation from keypoint dropout)
            KNOWN_POSES = {'STANDING', 'CROUCHING', 'SITTING', 'LYING', 'RUNNING'}
            pose_changed = (
                prev.get("pose") in KNOWN_POSES and
                det.pose in KNOWN_POSES and
                prev.get("pose") != det.pose
            )
            zone_changed = prev.get("zone") != det.zone
            pose_zone_changed = pose_changed or zone_changed

            # ── Feed signals into BehaviourWindow ─────────────────────
            behaviour_signals: list[str] = []

            gesture = hand_result.get("gesture", "")
            if gesture == "GESTURE_REACHING":
                behaviour.add(tid, "reaching")
                behaviour_signals.append("reaching_in_shelf")
            elif gesture == "GESTURE_CONCEALING":
                behaviour.add(tid, "concealing")
                behaviour_signals.append("concealing_gesture")

            head_dir = head_result.get("direction", "").lower()
            if head_dir in {"looking_left", "looking_right", "away"}:
                behaviour.add(tid, head_dir)
            if head_result.get("surveillance_flag"):
                behaviour_signals.append("surveillance_checking")

            for reason in anom_result.get("reasons", []):
                norm = _norm_reason(reason)
                behaviour.add(tid, norm)
                behaviour_signals.append(norm)

            # ── Pattern checks ──────────────────────────────────────────
            detected_patterns = behaviour.get_all_patterns(tid)
            for p in detected_patterns:
                behaviour_signals.append(p["pattern"])

            # ── Zone transition check ─────────────────────────────────
            # scorer tracks per-zone entry time internally — no dwell_s needed
            zone_alert = scorer.check_zone_transition(tid, det.zone)
            if zone_alert:
                behaviour_signals.append("SUSPICIOUS_ZONE_TRANSITION")

            # ── Risk scoring (READ-ONLY before gate) ───────────────────────
            # We only READ the current decayed score here to inform severity.
            # add_signals() is called AFTER the log-gate so the score only
            # accumulates once per logged event (~every 5s), not per frame.
            risk_score    = scorer.get_score(tid)      # decayed current score
            risk_severity = scorer.get_severity(tid)   # derived from score

            # ── Alert escalation ──────────────────────────────────────
            base_severity = anom_result.get("severity") or "NONE"
            norm_reasons  = [_norm_reason(r) for r in anom_result.get("reasons", [])]
            escalated_severity = escalator.process(tid, norm_reasons, base_severity)

            # Final severity = best (highest) of escalated vs risk-score severity
            _sev_order = {None: 0, "NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
            final_severity = max(
                [escalated_severity, risk_severity],
                key=lambda s: _sev_order.get(s, 0)
            )

            # Assemble risk payload for event_builder
            risk_data = {
                "score":     risk_score,
                "severity":  risk_severity,
                "patterns":  [p["pattern"] for p in detected_patterns],
                "zone_alert": zone_alert,
                "final_severity": final_severity,
            }

            # ── Cooldown / log-gate ──────────────────────────────────────
            norm_reason_set = frozenset(norm_reasons)
            if hand_result.get("bag_stuff_suspected"):
                norm_reason_set |= {"concealing_gesture"}
            if head_result.get("surveillance_flag"):
                norm_reason_set |= {"surveillance_checking"}
            if detected_patterns:
                norm_reason_set |= {p["pattern"] for p in detected_patterns}

            now_ts    = det.timestamp
            person_cd = cooldown_map.setdefault(tid, {})

            def _should_log() -> bool:
                if pose_zone_changed:
                    return True           # always log real transitions
                if fn % config.HEARTBEAT_FRAMES == 0:
                    return True           # periodic heartbeat
                # Always log high-risk patterns immediately
                if final_severity in ("MEDIUM", "HIGH"):
                    cd = config.ANOMALY_COOLDOWN_S
                    for nr in norm_reason_set:
                        if now_ts - person_cd.get(nr, 0) >= cd:
                            return True
                    return False
                if not norm_reason_set:
                    return False
                cd = (
                    config.GESTURE_COOLDOWN_S
                    if norm_reason_set & {"concealing_gesture", "reaching_in_shelf",
                                          "surveillance_checking"}
                    else config.ANOMALY_COOLDOWN_S
                )
                for nr in norm_reason_set:
                    if now_ts - person_cd.get(nr, 0) >= cd:
                        return True
                return False

            if not _should_log():
                continue

            for nr in norm_reason_set:
                person_cd[nr] = now_ts

            last_state[tid] = {"pose": det.pose, "zone": det.zone}

            # ── NOW add signals (after gate) ─────────────────────────────────
            # Score accumulates only on logged events, preventing per-frame inflation
            risk_score = scorer.add_signals(tid, behaviour_signals)
            risk_data["score"] = risk_score  # update payload with actual new score

            # ── Build and store event ─────────────────────────────────
            event = build_event(
                det           = det,
                anomaly       = anom_result,
                session_id    = session_id,
                camera_meta   = cam_meta,
                person_attrs  = attrs,
                scene         = scene,
                first_seen_ts = first_seen_map[tid],
                hand_result   = hand_result,
                head_result   = head_result,
                risk_data     = risk_data,
            )
            store(event, use_pinecone=use_pinecone)

        # ── Draw annotations on the ORIGINAL (full-res) frame ────────────
        scale_x = frame.shape[1] / proc_frame.shape[1] if proc_frame is not frame else 1.0
        scale_y = frame.shape[0] / proc_frame.shape[0] if proc_frame is not frame else 1.0
        last_proc_scale = (scale_x, scale_y)
        last_detections = detections
        last_anomalous  = {d.local_id for d in detections
                           if anomaly.is_currently_anomalous(d.local_id)}
        annotated = frame.copy()
        _draw_boxes(annotated, detections, scale_x, scale_y,
                    smooth_bboxes, _SMOOTH_ALPHA, last_anomalous)
        cv2.putText(annotated, f"{cam_id} | {cam_cfg['name']} | Q=quit",
                    (10, annotated.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
        cv2.imshow(f"VisionLog — {cam_id}", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # ── Cleanup stale tracks ──────────────────────────────────────
        anomaly.cleanup_stale(active_ids)
        enricher.evict(active_ids)
        hand.reset_stale(active_ids)
        head.evict(active_ids)
        behaviour.evict(active_ids)
        scorer.evict(active_ids)
        escalator.evict(active_ids)
        gallery.evict_stale_locals(active_ids)   # Re-ID: prune local→global map
        # Evict smoothed bboxes for tracks no longer active
        for tid in list(smooth_bboxes.keys()):
            if tid not in active_ids:
                del smooth_bboxes[tid]

    cv2.destroyAllWindows()
    print(f"\n[Main] Session {session_id} ended.")
    print(f"  Events logged to: {config.LOG_FILE}")
    print(f"  Full JSON log  : visionlog_events.jsonl")


def main():
    parser = argparse.ArgumentParser(description="VisionLog AI")
    parser.add_argument("--camera",     default=None, help="Camera ID from config")
    parser.add_argument("--source",     default=None, help="Override video path")
    parser.add_argument("--no-pinecone", action="store_true", help="Disable Pinecone, log locally")
    args = parser.parse_args()

    use_pinecone = not args.no_pinecone
    cameras = config.CAMERAS

    if args.camera:
        if args.camera not in cameras:
            print(f"[Error] Camera '{args.camera}' not found in config.CAMERAS")
            sys.exit(1)
        cameras = {args.camera: cameras[args.camera]}

    if args.source and cameras:
        first_key = next(iter(cameras))
        cameras[first_key] = {**cameras[first_key], "path": args.source}

    for cam_id, cam_cfg in cameras.items():
        run_camera(cam_id, cam_cfg, use_pinecone)


if __name__ == "__main__":
    main()
