"""
anomaly/risk_scorer.py

Two classes that sit on top of per-frame anomaly detection:

  RiskScorer     — cumulative weighted risk score per person, with exponential
                   time decay so long-ago events fade naturally.

  AlertEscalator — promotes repeated same-severity alerts:
                   LOW×5 in 60s → MEDIUM, MEDIUM×3 in 60s → HIGH.

  Both expose an evict() method so stale tracks don't accumulate memory.
"""
import time
import math
from visionlog import config


# ── Signal weights ────────────────────────────────────────────────────────────
SIGNAL_WEIGHTS: dict[str, int] = {
    'sit_down':                      5,
    'stand_up':                      5,
    'abrupt_sit':                   25,
    'seat_switch':                  15,
    'neighbour_seat_occupied':      10,
    'restless_head_movement':       20,
    'dwell_exceeded':                4,
    'reaching_in_shelf':            15,
    'reaching':                     12,
    'concealing_gesture':           30,
    'concealing':                   30,
    'surveillance_checking':        20,
    'running_detected':             10,
    'crouching_in_shelf':           15,
    'EVASION_PATTERN':              28,
    'SHOPLIFTING_SEQUENCE':         45,
    'CONCEAL_AFTER_DWELL':          40,
    'SURVEILLANCE_WHILE_REACHING':  20,
    'SUSPICIOUS_ZONE_TRANSITION':   20,
    'at_night':                     10,
}

SUSPICIOUS_TRANSITIONS: set[tuple] = {
    ('ENTRANCE',     'SHELF_AREA'),   # loiter at entrance → shelving
    ('SHELF_AREA',   'ENTRANCE'),     # leave shelves back to exit
    ('SHELF_AREA',   'CHECKOUT'),     # shelves straight to checkout
    ('DISPLAY_AREA', 'SHELF_AREA'),   # display browsing → shelves
    ('BACK_AISLE',   'SHELF_AREA'),   # back aisle approach to shelves
}

# Minimum dwell (seconds) in previous zone to consider a transition suspicious
TRANSITION_DWELL_THRESHOLD_S: float = 25.0

# Score decay: half-life in seconds (score halves every N seconds of inactivity)
DECAY_HALF_LIFE_S: float = 120.0


class RiskScorer:
    """
    Maintains a cumulative, time-decayed risk score per tracked person.

    Usage:
        scorer = RiskScorer()
        scorer.add_signals(person_id, ['reaching_in_shelf', 'dwell_exceeded'])
        score    = scorer.get_score(person_id)
        severity = scorer.get_severity(person_id)
    """

    def __init__(self):
        # person_id → (score, last_update_time)
        self._scores:       dict[int, tuple[float, float]] = {}
        # person_id → (last_zone, time_entered, dwell_at_leave)
        self._zone_state:   dict[int, tuple[str, float, float]] = {}

    # ── Score API ─────────────────────────────────────────────────────

    def add_signals(self, person_id: int, signal_keys: list[str]) -> float:
        """Add weighted points for the given signals. Returns updated score."""
        self._apply_decay(person_id)
        score, _ = self._scores.get(person_id, (0.0, time.time()))
        for key in signal_keys:
            score += SIGNAL_WEIGHTS.get(key, 0)
        now = time.time()
        self._scores[person_id] = (score, now)
        return score

    def get_score(self, person_id: int) -> float:
        """Return current (decayed) score."""
        self._apply_decay(person_id)
        return round(self._scores.get(person_id, (0.0, 0))[0], 1)

    def get_severity(self, person_id: int) -> str | None:
        score = self.get_score(person_id)
        if score >= config.RISK_HIGH:   return 'HIGH'
        if score >= config.RISK_MEDIUM: return 'MEDIUM'
        if score >= config.RISK_LOW:    return 'LOW'
        return None

    # ── Zone transition ───────────────────────────────────────────────

    def check_zone_transition(self, person_id: int,
                               current_zone: str) -> dict | None:
        """
        Call every frame with the person's current zone.
        Returns a zone-transition alert dict if a suspicious move is detected
        (caller is responsible for scoring). Otherwise returns None.
        Does NOT call add_signals internally — scoring happens after the log gate.
        """
        now        = time.time()
        prev_state = self._zone_state.get(person_id)
        if prev_state is None:
            self._zone_state[person_id] = (current_zone, now)  # (zone, entry_time)
            return None

        prev_zone, entry_ts = prev_state

        if current_zone != prev_zone:
            # How long were they in the previous zone?
            time_in_prev_zone = now - entry_ts
            # Record new zone entry
            self._zone_state[person_id] = (current_zone, now)

            if (prev_zone, current_zone) in SUSPICIOUS_TRANSITIONS:
                if time_in_prev_zone >= TRANSITION_DWELL_THRESHOLD_S:
                    return {
                        'alert':    'SUSPICIOUS_ZONE_TRANSITION',
                        'severity': 'MEDIUM',
                        'from':     prev_zone,
                        'to':       current_zone,
                        'dwell_s':  round(time_in_prev_zone, 1),
                    }
        return None

    # ── Internal ─────────────────────────────────────────────────────

    def _apply_decay(self, person_id: int) -> None:
        if person_id not in self._scores:
            return
        score, ts = self._scores[person_id]
        elapsed = time.time() - ts
        decayed = score * (0.5 ** (elapsed / DECAY_HALF_LIFE_S))
        self._scores[person_id] = (decayed, time.time())

    def evict(self, active_ids: set[int]) -> None:
        for pid in list(self._scores.keys()):
            if pid not in active_ids:
                del self._scores[pid]
        for pid in list(self._zone_state.keys()):
            if pid not in active_ids:
                del self._zone_state[pid]


class AlertEscalator:
    """
    Promotes repeated same-type alerts within a sliding time window.
      LOW   × 5 in window → MEDIUM
      MEDIUM × 3 in window → HIGH

    This ensures a shoplifter who keeps triggering LOW alerts eventually
    gets escalated to actionable severity.
    """

    # (base_severity → (count_to_escalate, promoted_severity))
    RULES: dict[str, tuple[int, str]] = {
        'LOW':    (5, 'MEDIUM'),
        'MEDIUM': (3, 'HIGH'),
    }

    def __init__(self, window_s: float = 60.0):
        self._window_s = window_s
        # (person_id, reason_base) → [timestamps]
        self._history: dict[tuple, list[float]] = {}

    def process(self, person_id: int,
                reasons: list[str],
                severity: str) -> str:
        """
        Record this alert and return the (possibly escalated) severity.
        reasons: list of normalised reason keys for this alert.
        """
        if not reasons or not severity or severity == 'NONE':
            return severity

        now     = time.time()
        cutoff  = now - self._window_s

        # Use the first reason as the discriminator key
        key = (person_id, reasons[0])
        hist = self._history.setdefault(key, [])
        hist.append(now)
        # Prune
        self._history[key] = [t for t in hist if t >= cutoff]
        count = len(self._history[key])

        threshold, promoted = self.RULES.get(severity, (999, severity))
        if count >= threshold:
            return promoted
        return severity

    def evict(self, active_ids: set[int]) -> None:
        for key in list(self._history.keys()):
            if key[0] not in active_ids:
                del self._history[key]
