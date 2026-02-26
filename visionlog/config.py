"""
config.py — Central configuration for VisionLog AI
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
        "name":     "Store Front",
        "path":     r"C:\Users\Aditya\OneDrive\Pictures\Screenshots\Recording 2026-02-25 162753.mp4",
        "location": "Aisle 1, main entrance",
        "lat":      None,
        "lon":      None,
        "floor":    1,
        "direction": "north",
    },
    "CAM_02": {
        "name":     "Checkout Counter",
        "path":     r"C:\Users\Aditya\OneDrive\Pictures\Screenshots\Recording 2026-02-25 161441.mp4",
        "location": "Checkout area",
        "lat":      None,
        "lon":      None,
        "floor":    1,
        "direction": "south",
    }
}

# ── Zone Definitions per Camera ──────────────────────────────────────
# Each zone is a polygon: list of (x%, y%) points as fraction of frame size.
# Example: (0.0, 0.0) = top-left corner, (1.0, 1.0) = bottom-right corner.
ZONES = {
    "CAM_01": {
        # ── Derived from CCTV image analysis ─────────────────────────
        # Camera is high-angle, wide shot. Store reads left→right:
        # checkout | promo stands | shelving wall | entrance door (bottom-right)

        # Far left strip — POS terminal, staff, customer queue
        "CHECKOUT":     [(0.00, 0.00), (0.18, 0.00),
                         (0.18, 0.95), (0.00, 0.95)],

        # Mid-floor — large red display units and wire produce baskets
        "DISPLAY_AREA": [(0.18, 0.12), (0.52, 0.12),
                         (0.52, 0.88), (0.18, 0.88)],

        # Right side — main product shelves (black shelving units)
        "SHELF_AREA":   [(0.52, 0.15), (0.95, 0.15),
                         (0.95, 0.88), (0.52, 0.88)],

        # Bottom right — glass entrance door and adjacent wall
        "ENTRANCE":     [(0.55, 0.88), (1.00, 0.88),
                         (1.00, 1.00), (0.35, 1.00)],

        # Top strip — back-of-store aisle (far end of shelving)
        "BACK_AISLE":   [(0.18, 0.00), (0.52, 0.00),
                         (0.52, 0.12), (0.18, 0.12)],
    },
    "CAM_02": {
        "GENERAL_AREA": [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    }
}

# ── Detection ────────────────────────────────────────────────────────
YOLO_MODEL        = "yolo11l-pose.pt"     # Large model: better pose accuracy, slower CPU inference
TRACKER           = "botsort_reid.yaml"  # BoT-SORT + appearance ReID enabled
YOLO_CONF         = 0.4                 # detection confidence threshold
YOLO_CLASSES      = [0]                 # 0 = person only. Remove to detect all 80 classes.
FRAME_SKIP        = 2                   # process 1-in-N frames (2 = 15fps effective @ 30fps)
PROCESS_WIDTH     = 960                 # resize frames to this width before all inference
                                        # (960 = good balance; 640 = fastest; 0 = no resize)

# ── Enrichment ───────────────────────────────────────────────────────
DEEPFACE_ACTIONS  = ["age", "gender", "emotion"]
FLORENCE_INTERVAL = 15     # run Florence-2 every N seconds (non-blocking background thread)
FLORENCE_MODEL    = "microsoft/Florence-2-base"

# ── Anomaly Thresholds ───────────────────────────────────────────────
DWELL_MEDIUM_S    = 60     # seconds before MEDIUM dwell alert
DWELL_HIGH_S      = 120    # seconds before HIGH dwell alert
SPEED_SPIKE_X     = 6.0    # multiplier over average speed to flag as RUNNING (raised from 3 to cut false positives)
MIN_RUNNING_PX    = 20     # minimum absolute pixel speed to flag as RUNNING
CROUCH_NEAR_SHELF_S = 10   # seconds crouching in shelf zone → suspicious
NIGHT_HOURS       = (22, 6) # 10pm–6am = elevated anomaly weight

# ── Zone-Specific Dwell Rules ─────────────────────────────────────────
# Zones where long dwell is completely normal — skip dwell_exceeded rule.
DWELL_EXEMPT_ZONES = {"CHECKOUT", "BACK_AISLE"}  # queuing / browsing far shelves

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
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"        # free, offline, 384-dim
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY",  "")

# ── Output ───────────────────────────────────────────────────────────
LOG_FILE          = "visionlog_events.log"     # human-readable text log
SAVE_ANOMALY_CLIPS = True                      # save 30s clip on anomaly
CLIP_BUFFER_S     = 15                         # seconds before+after anomaly
CLIP_OUTPUT_DIR   = "clips/"
SCHEMA_VERSION    = "1.0"
