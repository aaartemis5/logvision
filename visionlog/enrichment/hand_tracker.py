"""
enrichment/hand_tracker.py

Detects hand landmarks per tracked person using MediaPipe 0.10 Tasks API.
Classifies gestures relevant to shoplifting: reaching, concealing, bag-stuffing.

Gesture model:
  - REACHING:   hand extended away from body center (toward shelf)
  - CONCEALING: wrist crossing inward over torso repeatedly (bag-stuffing)
  - IDLE:       default / nothing suspicious

Runs per FRAME on the person bbox crop — very fast on CPU (~5ms).
"""
import time
import cv2
import numpy as np

# ── Import mediapipe Tasks API (0.10+) ───────────────────────────────
_landmarker    = None
_MP_AVAILABLE  = False
_LOAD_ATTEMPTED = False

def _get_landmarker():
    """Lazy-load the HandLandmarker (downloads model on first call)."""
    global _landmarker, _MP_AVAILABLE, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return _landmarker
    _LOAD_ATTEMPTED = True
    try:
        import mediapipe as mp
        import urllib.request, os, tempfile

        BaseOptions        = mp.tasks.BaseOptions
        HandLandmarker     = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode  = mp.tasks.vision.RunningMode

        # Download the model if not cached
        model_path = os.path.join(os.path.expanduser("~"), ".mediapipe", "hand_landmarker.task")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        if not os.path.exists(model_path):
            print("[HandTracker] Downloading MediaPipe hand model (~8MB)...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("[HandTracker] Download complete.")

        options = HandLandmarkerOptions(
            base_options       = BaseOptions(model_asset_path=model_path),
            running_mode       = VisionRunningMode.VIDEO,
            num_hands          = 2,
            min_hand_detection_confidence = 0.5,
            min_hand_presence_confidence  = 0.4,
            min_tracking_confidence       = 0.4,
        )
        _landmarker   = HandLandmarker.create_from_options(options)
        _MP_AVAILABLE = True
        print("[HandTracker] MediaPipe HandLandmarker ready.")
    except Exception as e:
        print(f"[HandTracker] MediaPipe not available: {e} — hand tracking disabled.")
        _MP_AVAILABLE = False
    return _landmarker


# ── Gesture constants ─────────────────────────────────────────────────
GESTURE_IDLE       = "idle"
GESTURE_REACHING   = "reaching"
GESTURE_CONCEALING = "concealing"

_t0 = time.monotonic()   # epoch anchor for video timestamps


def _classify_gesture(hand_landmarks, crop_w: int, crop_h: int) -> str:
    """
    hand_landmarks: list of NormalizedLandmark from MediaPipe Tasks result.
    Returns: GESTURE_IDLE | GESTURE_REACHING | GESTURE_CONCEALING
    """
    lm   = hand_landmarks
    wrist   = lm[0]    # wrist
    midtip  = lm[12]   # middle finger tip

    wx, wy = wrist.x  * crop_w, wrist.y  * crop_h
    mx, my = midtip.x * crop_w, midtip.y * crop_h

    # REACHING: fingertip near the horizontal edge of the person crop
    if midtip.x < 0.15 or midtip.x > 0.85:
        return GESTURE_REACHING

    # CONCEALING: wrist moving across the body centreline inward
    # delta = (midtip.x - wrist.x) in frame coords; negative = folding inward
    cross_px = mx - wx
    if cross_px < -18:
        return GESTURE_CONCEALING

    return GESTURE_IDLE


class HandTracker:
    """
    Per-frame MediaPipe hand gesture tracker.
    Call .track(local_id, bbox, frame) every frame.
    call .reset_stale(active_ids) each frame to evict people who left.
    """

    def __init__(self):
        self._consecutive_conceals: dict[int, int] = {}
        self._frame_count: int = 0   # monotonic frame counter for timestamps

    def track(self, local_id: int, bbox: tuple, frame: np.ndarray) -> dict:
        """
        Returns:
          {gesture, concealing_frames, bag_stuff_suspected}
        """
        lm = _get_landmarker()
        if lm is None or not _MP_AVAILABLE:
            return _empty()

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        PAD = 30
        cx1 = max(0, x1 - PAD); cy1 = max(0, y1 - PAD)
        cx2 = min(w, x2 + PAD); cy2 = min(h, y2 + PAD)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return _empty()

        crop_h, crop_w = crop.shape[:2]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        self._frame_count += 1
        timestamp_ms = int(self._frame_count * (1000 / 30))  # assume ~30fps

        try:
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = lm.detect_for_video(mp_image, timestamp_ms)
        except Exception:
            return _empty()

        gesture = GESTURE_IDLE
        if result.hand_landmarks:
            gesture = _classify_gesture(result.hand_landmarks[0], crop_w, crop_h)

        if gesture == GESTURE_CONCEALING:
            self._consecutive_conceals[local_id] = \
                self._consecutive_conceals.get(local_id, 0) + 1
        else:
            self._consecutive_conceals[local_id] = 0

        consec = self._consecutive_conceals.get(local_id, 0)
        return {
            "gesture":             gesture,
            "concealing_frames":   consec,
            "bag_stuff_suspected": consec >= 8,   # ~0.25s at 30fps
        }

    def reset_stale(self, active_ids: set[int]):
        """Remove state for persons no longer tracked."""
        for tid in list(self._consecutive_conceals.keys()):
            if tid not in active_ids:
                del self._consecutive_conceals[tid]

    def reset(self, local_id: int):
        self._consecutive_conceals.pop(local_id, None)


def _empty() -> dict:
    return {"gesture": GESTURE_IDLE, "concealing_frames": 0, "bag_stuff_suspected": False}
