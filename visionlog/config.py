"""
config.py — Central configuration for REST
Edit this file to configure cameras, zones, and thresholds.
Secrets (API keys) are loaded from a local .env file — never commit .env to git.
"""
import os
try:
    from dotenv import load_dotenv
    load_dotenv()   # loads .env from the project root if present
except ImportError:
    pass  # python-dotenv optional; keys can also be set as real env vars


# ── Camera Registry ──────────────────────────────────────────────────
# Add each camera here. Path can be: file path, 0 (webcam), or an RTSP URL.
CAMERAS = {
    "CAM_01": {
        "name":     "Seating Area A",
        "path":     r"C:/Users/Aditya/.gemini/antigravity/brain/1a3f029a-c725-4934-8bc8-3845ccc3d77d/.user_uploaded/media_1793548913656_6bfb0597.mp4",
        "location": "Seating Study Room",
        "lat":      None,
        "lon":      None,
        "floor":    1,
        "direction": "north",
    }
}

# ── Zone Definitions per Camera ──────────────────────────────────────
# Each zone is a polygon: list of (x%, y%) points as fraction of frame size.
# Example: (0.0, 0.0) = top-left corner, (1.0, 1.0) = bottom-right corner.
ZONES = {
    "CAM_01": {
        "SEAT_A": [(0.20, 0.35), (0.45, 0.35), (0.45, 0.88), (0.20, 0.88)],
        "SEAT_B": [(0.50, 0.25), (0.85, 0.25), (0.85, 0.90), (0.50, 0.90)],
        "AISLE":  [(0.00, 0.00), (1.00, 0.00), (1.00, 1.00), (0.00, 1.00)],
    },
    "CAM_02": {
        "GENERAL_AREA": [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    }
}

# ── Detection ────────────────────────────────────────────────────────
YOLO_MODEL        = "yolo26l-pose.pt"     # YOLO26 Large: NMS-free, faster inference, better accuracy
TRACKER           = "botsort_reid.yaml"  # BoT-SORT + appearance ReID enabled
YOLO_CONF         = 0.4                 # detection confidence threshold
YOLO_CLASSES      = [0]                 # 0 = person only. Remove to detect all 80 classes.
FRAME_SKIP        = 3                   # process 1-in-N frames (3 = fast inference @ 24-30fps)
PROCESS_WIDTH     = 640                 # resize to 640 for 2x faster inference
                                        # (960 = good balance; 640 = fastest; 0 = no resize)

# ── Enrichment ───────────────────────────────────────────────────────
DEEPFACE_ACTIONS  = ["age", "gender", "emotion"]
VLM_INTERVAL      = 6      # run VLM every N seconds (non-blocking background thread)
VLM_MODEL         = "Qwen/Qwen2.5-VL-3B-Instruct"  # State-of-the-Art VLM (perfect for 16GB Colab GPUs)

# ── Anomaly Thresholds ───────────────────────────────────────────────
DWELL_MEDIUM_S    = 60     # seconds before MEDIUM dwell alert
DWELL_HIGH_S      = 120    # seconds before HIGH dwell alert
SPEED_SPIKE_X     = 6.0    # multiplier over average speed to flag as RUNNING (raised from 3 to cut false positives)
MIN_RUNNING_PX    = 20     # minimum absolute pixel speed to flag as RUNNING
CROUCH_NEAR_SHELF_S = 10   # seconds crouching in shelf zone → suspicious
NIGHT_HOURS       = (22, 6) # 10pm–6am = elevated anomaly weight

# ── Zone-Specific Dwell Rules ─────────────────────────────────────────
# Zones where long dwell is completely normal — skip dwell_exceeded rule.
DWELL_EXEMPT_ZONES = {"AISLE", "GENERAL_AREA"}  # queuing / browsing far shelves

# ── Risk Scoring (Intelligence Layer) ────────────────────────────────
RISK_LOW          = 25     # score threshold → LOW severity
RISK_MEDIUM       = 50     # score threshold → MEDIUM severity
RISK_HIGH         = 80     # score threshold → HIGH severity

# ── Event Rate-Limiting ───────────────────────────────────────────────
# Same anomaly reason type won't re-log for this many seconds (per person).
# Pose/zone state changes and *new* reason types always bypass the cooldown.
ANOMALY_COOLDOWN_S  = 5.0   # seconds between repeat anomaly events (was 10)
GESTURE_COOLDOWN_S  = 5.0   # cooldown for gesture events (hand/head)
HEARTBEAT_FRAMES    = 150   # heartbeat every N frames (~5s @ 30fps, was 300)

# ── Pinecone ─────────────────────────────────────────────────────────
# Keys are loaded from .env (git-ignored) or real environment variables.
# Copy .env.example → .env and fill in your own keys.
PINECONE_API_KEY  = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX    = os.getenv("PINECONE_INDEX",   "logvision")
PINECONE_VIDEO_INDEX = os.getenv("PINECONE_VIDEO_INDEX", "visionlog-video")
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"        # free, offline, 384-dim
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY",  "")
TWELVE_LABS_API_KEY = os.getenv("TWELVE_LABS_API_KEY", "")

# ── Video Indexing (Multimodal RAG) ──────────────────────────────────
ENABLE_VIDEO_INDEX = True                      # Auto-clip and index to Twelve Labs
VIDEO_CHUNK_S     = 6                          # Duration of each indexed video clip

# ── Output ───────────────────────────────────────────────────────────
LOG_FILE          = "visionlog_events.log"     # human-readable text log
SAVE_ANOMALY_CLIPS = True                      # save 30s clip on anomaly
CLIP_BUFFER_S     = 15                         # seconds before+after anomaly
CLIP_OUTPUT_DIR   = "clips/"
SCHEMA_VERSION    = "1.0"


# ── Seating Study (REST) ─────────────────────────────────────────────
RETAIL_RULES_ENABLED   = False      # old shoplifting rules stay in code but are switched off
SEAT_ZONES             = {"CAM_01": ["SEAT_A", "SEAT_B"], "CAM_02": []}
SEAT_NEIGHBOURS        = {"CAM_01": [("SEAT_A", "SEAT_B")], "CAM_02": []}
SIT_CONFIRM_S          = 0.6    # must be SITTING in a seat zone this long to count as a sit-down
STAND_CONFIRM_S        = 1.0    # must be out of SITTING / out of seat this long to count as stand-up
ABRUPT_SIT_TORSO_PER_S = 1.0    # downward hip speed (torso-lengths/s) threshold for abrupt sit
SEAT_SWITCH_WINDOW_S   = 30.0   # sit in a neighbouring seat within this time = seat switch
