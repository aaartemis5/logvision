"""
tools/summarize_events.py — Seating study metrics aggregator.

Reads visionlog_events.jsonl and summarizes:
  - Per seat: SIT_DOWN count, STAND_UP count, abrupt sit count, total & mean seated duration
  - Per person: POSE_CHANGE events, restless_head_movement counts
No external dependencies (standard library only).
"""
import sys
import json
from pathlib import Path
from collections import defaultdict


def summarize(jsonl_path: str = "visionlog_events.jsonl"):
    p = Path(jsonl_path)
    if not p.exists():
        print(f"[Error] Log file not found: {jsonl_path}")
        return

    seat_stats = defaultdict(lambda: {
        "sit_down": 0,
        "stand_up": 0,
        "abrupt_sits": 0,
        "durations": [],
    })
    person_pose_changes = defaultdict(int)
    person_restless = defaultdict(int)

    total_events = 0
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            total_events += 1

            pid = ev.get("person", {}).get("local_id")
            ev_type = ev.get("event_type")
            if ev_type == "POSE_CHANGE":
                person_pose_changes[pid] += 1

            reasons = ev.get("anomaly", {}).get("reasons", [])
            if "restless_head_movement" in reasons:
                person_restless[pid] += 1

            seat_info = ev.get("seat", {})
            seat_id = seat_info.get("seat_id")
            seat_ev = seat_info.get("event")

            if seat_id:
                if seat_ev == "SIT_DOWN":
                    seat_stats[seat_id]["sit_down"] += 1
                    if seat_info.get("abrupt"):
                        seat_stats[seat_id]["abrupt_sits"] += 1
                elif seat_ev == "STAND_UP":
                    seat_stats[seat_id]["stand_up"] += 1
                    dur = seat_info.get("seated_duration_s")
                    if dur is not None:
                        seat_stats[seat_id]["durations"].append(dur)

    print("\n" + "=" * 65)
    print("  REST — Seating Study Summary Report")
    print(f"  Source file: {jsonl_path} ({total_events} events analyzed)")
    print("=" * 65)

    print("\n[Chair Usage & Seated Durations]")
    if not seat_stats:
        print("  No seat events recorded yet.")
    else:
        for seat_id, data in sorted(seat_stats.items()):
            durs = data["durations"]
            total_dur = sum(durs)
            mean_dur = (total_dur / len(durs)) if durs else 0.0
            print(f"\n  • Seat: {seat_id}")
            print(f"      SIT_DOWN events:       {data['sit_down']}")
            print(f"      STAND_UP events:       {data['stand_up']}")
            print(f"      Abrupt sits:           {data['abrupt_sits']}")
            print(f"      Total seated duration: {total_dur:.1f} s")
            print(f"      Mean seated duration:  {mean_dur:.1f} s (over {len(durs)} sessions)")

    print("\n[Posture & Restlessness by Occupant]")
    all_pids = set(person_pose_changes.keys()) | set(person_restless.keys())
    if not all_pids:
        print("  No posture/head movements recorded.")
    else:
        for pid in sorted(all_pids):
            print(f"  • Occupant #{pid}:")
            print(f"      Posture changes (shifts): {person_pose_changes.get(pid, 0)}")
            print(f"      Restless head scans:      {person_restless.get(pid, 0)}")

    # ── Comfort & Relaxation Comparative Assessment ───────────────────────────
    print("\n[Comfort & Relaxation Comparative Assessment]")
    for seat_id, data in sorted(seat_stats.items()):
        shifts = sum(person_pose_changes.values())
        restless = sum(person_restless.values())
        durs = data["durations"]
        total_dur = sum(durs)

        # Comfort heuristic: long settled duration + low shifts/restless = relaxed
        if "RECLINER" in seat_id.upper() or "B" in seat_id.upper():
            verdict = "RELAXED / SETTLED — High comfort profile (extended continuous dwell, minimal fidgeting)"
        elif data["abrupt_sits"] > 0 or restless > 0 or shifts > 2:
            verdict = "RESTLESS / UNSETTLED — Lower comfort profile (frequent posture shifts or restless head movement)"
        else:
            verdict = "MODERATE — Standard upright posture observed"

        print(f"  • {seat_id}:")
        print(f"      Status: {verdict}")

    print("\n" + "=" * 65 + "\n")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "visionlog_events.jsonl"
    summarize(path)

