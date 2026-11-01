# REST Repurpose — Change Log

This document records all modifications made to repurpose the `logvision` codebase into **REST** (seating behaviour analysis for EcoLattice).

---

## 1. Files Touched

| File | Changes Made |
|---|---|
| `visionlog/config.py` | Updated `CAMERAS` metadata and default video path; defined `ZONES` (`SEAT_A`, `SEAT_B`, `AISLE`); set `DWELL_EXEMPT_ZONES = {"AISLE", "GENERAL_AREA"}`; added seating study parameters (`RETAIL_RULES_ENABLED`, `SEAT_ZONES`, `SEAT_NEIGHBOURS`, `SIT_CONFIRM_S`, `STAND_CONFIRM_S`, `ABRUPT_SIT_TORSO_PER_S`, `SEAT_SWITCH_WINDOW_S`). |
| `visionlog/enrichment/scene_describer.py` | Updated Qwen2.5-VL prompt from retail security analyst to seating behaviour study analyst; updated fallback disabled message to "Scene description disabled.". |
| `visionlog/anomaly/seat_tracker.py` | **NEW FILE**. Created `SeatTracker` state machine to manage occupant sit-down/stand-up transitions, calculate descent speed / abrupt sit, track neighbour seat occupancy, and detect seat switching. |
| `visionlog/anomaly/anomaly_detector.py` | Added `seat_events` parameter to `evaluate()`; wrapped retail rules (`arms_raised`, `running`, `crouching_near_shelf`, `late_night`, `concealing`, `reaching`) in `if config.RETAIL_RULES_ENABLED:`; renamed head rule reason to `restless_head_movement`; added additive scoring rules for `sit_down`, `stand_up`, `abrupt_sit`, `seat_switch`, and `neighbour_seat_occupied`. |
| `visionlog/anomaly/risk_scorer.py` | Added signal weights for seating events: `sit_down: 5`, `stand_up: 5`, `abrupt_sit: 25`, `seat_switch: 15`, `neighbour_seat_occupied: 10`, `restless_head_movement: 20`. |
| `visionlog/events/event_builder.py` | Added `seat_events` parameter to `build_event()`; updated `_infer_event_type()` to return `SIT_DOWN` or `STAND_UP` when present; populated structured `"seat"` dictionary; enriched `text_summary` with natural language seating phrases; renamed tag `SURVEILLANCE_CHECKING` to `RESTLESS_HEAD_MOVEMENT`. |
| `visionlog/events/event_store.py` | Added `"seat_id"` to Pinecone `flat_meta`; moved local JSONL writing (`visionlog_events.jsonl`) outside the `else:` block so events are consistently logged locally regardless of Pinecone status. |
| `main.py` | Renamed CLI description and terminal banners to REST; instantiated and hooked `SeatTracker`; skipped MediaPipe hand tracking when `RETAIL_RULES_ENABLED = False`; ensured seat events bypass cooldown in `_should_log()`; added skeleton overlay (COCO 17 keypoints) and seat badge to `_draw_boxes()`. |
| `query.py` | Updated `SYSTEM_PROMPT` to seating study assistant; replaced retail example queries with seating queries; updated banner and agent name to REST. |
| `app.py` | Updated FastAPI dashboard title to REST. |
| `web/index.html` | Updated page title, navigation brand, welcome message, and input placeholder to REST seating study context. |
| `tools/summarize_events.py` | **NEW FILE**. Aggregates seat metrics (sit/stand counts, abrupt sits, seated durations) and occupant shifts/restlessness from `visionlog_events.jsonl`. |
| `tests/test_seat_tracker.py` | **NEW FILE**. Unit test suite validating sit confirmation, stand-up duration timing, abrupt sit detection, and seat switching. |

---

## 2. Configuration Constants Added

```python
RETAIL_RULES_ENABLED   = False      # Disables shoplifting/concealing rules
SEAT_ZONES             = {"CAM_01": ["SEAT_A", "SEAT_B"], "CAM_02": []}
SEAT_NEIGHBOURS        = {"CAM_01": [("SEAT_A", "SEAT_B")], "CAM_02": []}
SIT_CONFIRM_S          = 0.6        # Required continuous sitting frames (s) to confirm sit
STAND_CONFIRM_S        = 1.0        # Required continuous out-of-seat frames (s) to confirm stand
ABRUPT_SIT_TORSO_PER_S = 1.0        # Downward hip velocity threshold (torso-lengths/s)
SEAT_SWITCH_WINDOW_S   = 30.0       # Max interval (s) between stand and neighbour sit for seat switch
```

---

## 3. Items Skipped or Noticed

* **Step 3 (Zone reference point refactoring):** Skipped as advised in the specification prompt. Modifying the `zone_fn` callable signature across `Detector` and `ZoneManager` was unnecessary because bounding box center alignment is well within the seating polygon boundaries.
* **Pinecone Index Names:** Left as `logvision` and `visionlog-video` per strict instructions not to alter cloud infrastructure naming.
* **Package Directory:** Kept as `visionlog/` and all internal package imports unchanged.
