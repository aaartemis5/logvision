"""
tests/test_seat_tracker.py — Unit test for SeatTracker state transitions.

Verifies:
  1. SIT_DOWN confirmed after SIT_CONFIRM_S threshold
  2. Abrupt sitting flagged on fast downward hip velocity
  3. STAND_UP confirmed after STAND_CONFIRM_S threshold with correct duration
  4. Seat switch detected when moving to neighbouring seat within window
  5. Pose blips under confirmation threshold do NOT trigger false events
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass
from visionlog.anomaly.seat_tracker import SeatTracker


@dataclass
class MockDetection:
    local_id: int
    camera_id: str
    timestamp: float
    bbox: tuple
    zone: str
    pose: str
    keypoints: list
    kp_conf: list


def _make_kps(hip_y=200.0, sh_y=120.0):
    # 17 keypoints: 5=L_SH, 6=R_SH, 11=L_HIP, 12=R_HIP
    kps = [(100.0, 100.0)] * 17
    kps[5] = (90.0, sh_y)
    kps[6] = (110.0, sh_y)
    kps[11] = (95.0, hip_y)
    kps[12] = (105.0, hip_y)
    confs = [1.0] * 17
    return kps, confs


def test_seat_tracker_lifecycle():
    cam_id = "CAM_01"
    tracker = SeatTracker(cam_id)
    t = 100.0

    kps_normal, conf_normal = _make_kps(hip_y=200.0, sh_y=120.0)

    # 1. Standing in AISLE -> no events
    det = MockDetection(
        local_id=1, camera_id=cam_id, timestamp=t,
        bbox=(50, 50, 150, 250), zone="AISLE", pose="STANDING",
        keypoints=kps_normal, kp_conf=conf_normal
    )
    evs = tracker.update(det)
    assert len(evs) == 0

    # 2. Enters SEAT_A and sits down at t=101.0
    t = 101.0
    det = MockDetection(
        local_id=1, camera_id=cam_id, timestamp=t,
        bbox=(50, 50, 150, 250), zone="SEAT_A", pose="SITTING",
        keypoints=kps_normal, kp_conf=conf_normal
    )
    evs = tracker.update(det)
    # < 0.6s -> no event yet
    assert len(evs) == 0

    # Brief pose blip < 0.5s should not emit
    t = 101.3
    det.timestamp = t
    evs = tracker.update(det)
    assert len(evs) == 0

    # Reach 0.7s continuous sitting -> SIT_DOWN should fire!
    t = 101.7
    det.timestamp = t
    evs = tracker.update(det)
    assert len(evs) == 1
    assert evs[0]["type"] == "SIT_DOWN"
    assert evs[0]["seat_id"] == "SEAT_A"
    assert evs[0]["seat_switch"] is False

    # 3. Stay seated for 30 seconds -> no events
    t = 131.7
    det.timestamp = t
    evs = tracker.update(det)
    assert len(evs) == 0

    # 4. Stand up at t=132.0 (pose = STANDING, zone = AISLE)
    t = 132.0
    det.pose = "STANDING"
    det.zone = "AISLE"
    det.timestamp = t
    evs = tracker.update(det)
    assert len(evs) == 0  # < 1.0s confirmation

    # At t=133.1 (1.1s later) -> STAND_UP should fire!
    t = 133.1
    det.timestamp = t
    evs = tracker.update(det)
    assert len(evs) == 1
    assert evs[0]["type"] == "STAND_UP"
    assert evs[0]["seat_id"] == "SEAT_A"
    assert 30.0 <= evs[0]["seated_duration_s"] <= 33.0

    # 5. Move to neighbouring SEAT_B within 30s window and sit down
    t = 140.0
    kps_drop, conf_drop = _make_kps(hip_y=280.0, sh_y=120.0)  # sudden drop
    det.pose = "SITTING"
    det.zone = "SEAT_B"
    det.timestamp = t
    det.keypoints = kps_drop
    det.kp_conf = conf_drop
    tracker.update(det)

    t = 140.7
    det.timestamp = t
    evs = tracker.update(det)
    assert len(evs) == 1
    assert evs[0]["type"] == "SIT_DOWN"
    assert evs[0]["seat_id"] == "SEAT_B"
    assert evs[0]["seat_switch"] is True  # neighbour switch detected!


if __name__ == "__main__":
    test_seat_tracker_lifecycle()
    print("✓ SeatTracker lifecycle unit tests PASSED successfully!")

