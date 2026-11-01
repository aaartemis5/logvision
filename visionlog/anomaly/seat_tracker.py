"""
anomaly/seat_tracker.py — Tracks occupant seating states per camera.

Detects:
  - SIT_DOWN (with abrupt sit flag, descent speed, neighbouring seat occupancy, seat switch)
  - STAND_UP (with total seated duration)
Does NOT modify Detection objects.
"""
from collections import defaultdict
from visionlog import config
from visionlog.detection.detector import Detection


class SeatTracker:
    """
    Stateful seat tracking for a single camera session.
    Keeps track of who is sitting where, how long they stay, and transitions.
    """

    # Shared occupancy across all occupants in this camera session: seat_id -> person_id
    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self._seat_occupants: dict[str, int] = {}  # seat_id -> local_id

        # Per-person state:
        # local_id -> dict of tracking variables
        self._state: dict[int, dict] = defaultdict(lambda: {
            "seated": False,
            "seat_id": None,
            "sit_start_ts": None,
            "candidate_sit_since": None,
            "candidate_sit_zone": None,
            "candidate_stand_since": None,
            "last_seat": None,
            "last_stand_ts": None,
            "samples": [],  # list of (ts, hip_y, torso_len)
        })

    def _extract_hip_and_torso(self, det: Detection):
        """Extract average hip Y and torso length from COCO keypoints (conf >= 0.3)."""
        kps = det.keypoints
        conf = det.kp_conf
        if not kps or not conf:
            return None, None

        # Shoulders: 5 (left), 6 (right); Hips: 11 (left), 12 (right)
        ls_ok = conf[5] >= 0.3
        rs_ok = conf[6] >= 0.3
        lh_ok = conf[11] >= 0.3
        rh_ok = conf[12] >= 0.3

        if not (lh_ok and rh_ok):
            return None, None

        hip_y = (kps[11][1] + kps[12][1]) / 2.0

        torso_len = None
        if ls_ok and rs_ok:
            sh_y = (kps[5][1] + kps[6][1]) / 2.0
            torso_len = abs(hip_y - sh_y)
        if not torso_len or torso_len < 10.0:
            bbox = det.bbox
            torso_len = (bbox[3] - bbox[1]) * 0.45

        return hip_y, torso_len

    def update(self, det: Detection) -> list[dict]:
        """
        Process detection for this frame. Returns list of seat events (usually empty).
        Event dicts:
          - SIT_DOWN: {type, seat_id, abrupt, descent_speed, neighbour_occupied_by, seat_switch}
          - STAND_UP: {type, seat_id, seated_duration_s}
        """
        tid = det.local_id
        ts = det.timestamp
        pose = det.pose
        zone = det.zone
        st = self._state[tid]

        events = []
        valid_seats = config.SEAT_ZONES.get(self.camera_id, [])

        # Update keypoint samples for abrupt sit calculation (keep last ~2 seconds)
        hip_y, torso_len = self._extract_hip_and_torso(det)
        if hip_y is not None and torso_len is not None:
            st["samples"].append((ts, hip_y, torso_len))
        st["samples"] = [s for s in st["samples"] if (ts - s[0]) <= 2.0]

        is_sitting_pose = (pose in ("SITTING", "LYING DOWN"))
        in_seat_zone = (zone in valid_seats)

        # ── State Machine ─────────────────────────────────────────────
        if not st["seated"]:
            # Check for candidate SIT_DOWN
            if is_sitting_pose and in_seat_zone:
                if st["candidate_sit_since"] is None or st["candidate_sit_zone"] != zone:
                    st["candidate_sit_since"] = ts
                    st["candidate_sit_zone"] = zone
                elif (ts - st["candidate_sit_since"]) >= config.SIT_CONFIRM_S:
                    # SIT_DOWN confirmed!
                    st["seated"] = True
                    st["seat_id"] = zone
                    st["sit_start_ts"] = st["candidate_sit_since"]
                    st["candidate_sit_since"] = None
                    st["candidate_stand_since"] = None

                    # 1. Abrupt sit calculation: max downward hip speed (Δhip_y / Δt / torso_len)
                    descent_speed = 0.0
                    recent_samples = [s for s in st["samples"] if (ts - s[0]) <= 1.2]
                    if len(recent_samples) >= 2:
                        speeds = []
                        for i in range(len(recent_samples) - 1):
                            t0, y0, tlen = recent_samples[i]
                            t1, y1, _ = recent_samples[i + 1]
                            dt = t1 - t0
                            if dt > 0.01:
                                dy = y1 - y0  # downward is positive y
                                if dy > 0:
                                    speeds.append((dy / dt) / max(tlen, 10.0))
                        if speeds:
                            descent_speed = max(speeds)
                    abrupt = (descent_speed >= config.ABRUPT_SIT_TORSO_PER_S)

                    # 2. Neighbour seat check
                    neighbours = []
                    for pair in config.SEAT_NEIGHBOURS.get(self.camera_id, []):
                        if zone == pair[0]:
                            neighbours.append(pair[1])
                        elif zone == pair[1]:
                            neighbours.append(pair[0])

                    neighbour_occupied_by = [
                        self._seat_occupants[n] for n in neighbours
                        if n in self._seat_occupants and self._seat_occupants[n] != tid
                    ]

                    # 3. Seat switch check
                    seat_switch = False
                    if st["last_seat"] in neighbours and st["last_stand_ts"] is not None:
                        if (ts - st["last_stand_ts"]) <= config.SEAT_SWITCH_WINDOW_S:
                            seat_switch = True

                    # Update seat occupancy
                    self._seat_occupants[zone] = tid

                    events.append({
                        "type": "SIT_DOWN",
                        "seat_id": zone,
                        "abrupt": bool(abrupt),
                        "descent_speed": round(float(descent_speed), 2),
                        "neighbour_occupied_by": neighbour_occupied_by,
                        "seat_switch": seat_switch,
                    })
            else:
                st["candidate_sit_since"] = None
                st["candidate_sit_zone"] = None

        else:
            # Person is currently seated; check for STAND_UP
            # If not sitting or left the seat
            if (not is_sitting_pose) or (zone != st["seat_id"]):
                if st["candidate_stand_since"] is None:
                    st["candidate_stand_since"] = ts
                elif (ts - st["candidate_stand_since"]) >= config.STAND_CONFIRM_S:
                    # STAND_UP confirmed!
                    seat_id = st["seat_id"]
                    duration_s = round(max(0.0, ts - (st["sit_start_ts"] or ts)), 1)

                    if self._seat_occupants.get(seat_id) == tid:
                        del self._seat_occupants[seat_id]

                    st["last_seat"] = seat_id
                    st["last_stand_ts"] = ts
                    st["seated"] = False
                    st["seat_id"] = None
                    st["sit_start_ts"] = None
                    st["candidate_stand_since"] = None
                    st["candidate_sit_since"] = None

                    events.append({
                        "type": "STAND_UP",
                        "seat_id": seat_id,
                        "seated_duration_s": duration_s,
                    })
            else:
                st["candidate_stand_since"] = None

        return events

    def evict(self, active_ids: set[int]):
        """Clean up state for tracks that left the frame."""
        for tid in list(self._state.keys()):
            if tid not in active_ids:
                seat_id = self._state[tid].get("seat_id")
                if seat_id and self._seat_occupants.get(seat_id) == tid:
                    del self._seat_occupants[seat_id]
                del self._state[tid]
