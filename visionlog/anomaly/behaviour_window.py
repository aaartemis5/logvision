"""
anomaly/behaviour_window.py

Sliding per-person behavioural signal buffer.
Stores the last N seconds of signals and checks for known shoplifting patterns.

Patterns:
  EVASION_PATTERN          — looking away + reaching simultaneously
  SHOPLIFTING_SEQUENCE     — dwell + reaching + concealing in same window
  SURVEILLANCE_SCAN        — head scanning rapidly AND reaching
"""
import time
import re
from collections import defaultdict, deque


# Signals that count as "looking away from staff / camera"
LOOKING_AWAY_TYPES = {'looking_left', 'looking_right', 'away'}


class BehaviourWindow:
    """
    Per-person ring buffer of behavioural signal events over a sliding window.

    Usage:
        bw = BehaviourWindow(window_s=30)
        bw.add(person_id, 'reaching')
        bw.add(person_id, 'looking_left')
        result = bw.check_evasion(person_id)
    """

    def __init__(self, window_s: float = 30.0):
        self._window_s = window_s
        # person_id → deque of (timestamp, signal_type)
        self._events: dict[int, deque] = defaultdict(deque)

    # ── Public ────────────────────────────────────────────────────────

    def add(self, person_id: int, signal_type: str) -> None:
        """Append a signal event for this person and prune old ones."""
        now = time.time()
        q = self._events[person_id]
        q.append((now, signal_type))
        cutoff = now - self._window_s
        while q and q[0][0] < cutoff:
            q.popleft()

    def count(self, person_id: int, signal_type: str) -> int:
        return sum(1 for _, s in self._events[person_id] if s == signal_type)

    def count_any(self, person_id: int, signal_types: set) -> int:
        return sum(1 for _, s in self._events[person_id] if s in signal_types)

    def present(self, person_id: int) -> set:
        """Return set of unique signal types seen in the window."""
        return {s for _, s in self._events[person_id]}

    # ── Pattern checks ────────────────────────────────────────────────

    def check_evasion(self,
                      person_id: int,
                      head_threshold: int = 4,
                      reach_threshold: int = 2) -> dict | None:
        """
        Looking away repeatedly WHILE reaching → evasion / concealment intent.
        Most reliable single pattern for shoplifting.
        """
        lookings  = self.count_any(person_id, LOOKING_AWAY_TYPES)
        reachings = self.count(person_id, 'reaching')

        if lookings >= head_threshold and reachings >= reach_threshold:
            return {
                'pattern':      'EVASION_PATTERN',
                'severity':     'MEDIUM',
                'looking_away': lookings,
                'reaching':     reachings,
            }
        return None

    def check_shoplifting_sequence(self, person_id: int) -> dict | None:
        """
        Dwell + reaching + concealing all within the window
        = full shoplifting sequence detected.
        """
        present = self.present(person_id)
        has_dwell    = 'dwell_exceeded' in present
        has_reach    = 'reaching' in present
        has_conceal  = 'concealing' in present

        if has_dwell and has_reach and has_conceal:
            return {
                'pattern':  'SHOPLIFTING_SEQUENCE',
                'severity': 'HIGH',
            }
        # Partial: dwell + conceal (no reaching needed)
        if has_dwell and has_conceal:
            return {
                'pattern':  'CONCEAL_AFTER_DWELL',
                'severity': 'HIGH',
            }
        return None

    def check_surveillance_scan(self, person_id: int,
                                 scan_threshold: int = 3) -> dict | None:
        """Rapid left-right-left scanning AND reaching = surveillance check."""
        lookings = self.count_any(person_id, {'looking_left', 'looking_right'})
        reachings = self.count(person_id, 'reaching')
        if lookings >= scan_threshold and reachings >= 1:
            return {
                'pattern':  'SURVEILLANCE_WHILE_REACHING',
                'severity': 'MEDIUM',
                'scans':    lookings,
            }
        return None

    def get_all_patterns(self, person_id: int) -> list[dict]:
        """Run all checks and return list of detected patterns."""
        patterns = []
        for check in (self.check_evasion,
                      self.check_shoplifting_sequence,
                      self.check_surveillance_scan):
            result = check(person_id)
            if result:
                patterns.append(result)
        return patterns

    def evict(self, active_ids: set[int]) -> None:
        for pid in list(self._events.keys()):
            if pid not in active_ids:
                del self._events[pid]
