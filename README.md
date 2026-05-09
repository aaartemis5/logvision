# VisionLog AI — Real-Time CCTV Intelligence Platform

> **Edge-processing AI that detects, tracks, and profiles individuals from live CCTV feeds in real-time, then enables natural language querying of events and video footage via a multimodal RAG system.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![YOLO26](https://img.shields.io/badge/YOLO26-NMS--Free-red)](https://github.com/ultralytics/ultralytics)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-2024-blue?logo=intel)](https://docs.openvino.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## What Is This?

VisionLog AI is a **hybrid computer vision + Generative AI surveillance system** that runs entirely on consumer hardware (tested on GTX 1650 Ti). It combines:

- **Real-time person tracking** at 30 FPS using YOLO26-Pose + BoT-SORT
- **Occlusion-proof re-identification** — a person keeps their ID even after walking behind a shelf
- **Automatic anomaly detection** — dwell time, crouching near shelves, head-scanning behaviour
- **Multimodal RAG chatbot** — ask questions in plain English like *"Was anyone crouching near the shelves at night?"* and get a Gemini-powered intelligence report + searchable video clip evidence

The live analytics stream runs in a browser-based dashboard alongside the AI inquiry panel.

---

## Demo Video

https://drive.google.com/file/d/1S2TRzT2ZCbwqU5qCs21ysNRVv9f9U0ZO/view?usp=sharing


---

## Architecture

```
Live Camera / Video File
        │
        ▼
  YOLO26-Pose ──── Detection + 17 Skeletal Keypoints (~8ms/frame, GPU)
        │
        ▼
  BoT-SORT ──────── Temporal Tracking + Kalman Filter (~5ms, CPU)
        │
        ▼
  OSNet Re-ID ───── Occlusion-proof Identity Persistence (~15ms/crop)
        │
        ▼
  SigLIP ────────── Clothing & Attribute Extraction (async background)
        │
  ┌─────┴──────────────────────────┐
  │       AnomalyDetector          │ ← Always running, microseconds
  │  Dwell · Pose · Head · Score   │
  └─────┬──────────────────────────┘
        │ (fires on Risk Score > threshold)
        ▼
  Qwen2.5-VL ────── Scene Description (async background thread, ~6s)
        │
        ▼
  Event Log → Pinecone Vector DB (Text Index)
  Video Clips → Twelve Labs + Pinecone (Video Index)
        │
        ▼
  FastAPI Web Server (/api/chat · /api/camera/stream · /media)
        │
  ┌─────┴──────────────┐
  │                    │
  ▼                    ▼
Chat (RAG)         Live Stream (MJPEG)
  │
  ▼
Gemini 2.5 Flash ── Intelligence Report Generation
```

---

## Model Choices & Why

| Layer | Model | Why |
|---|---|---|
| Detection | **YOLO26-Pose** | NMS-Free → no CPU bottleneck. Pose keypoints enable gesture/crouch detection for free. |
| Tracking | **BoT-SORT** | Kalman filter keeps tracks alive through short occlusions. |
| Re-ID | **OSNet** (via BoxMOT) | Trained on Market-1501 CCTV dataset. Tells apart two people in similar clothing — CLIP cannot. |
| Attributes | **SigLIP SO400M** | Superior successor to CLIP. Better fine-grained colour/texture accuracy. Runs async so it never blocks the main loop. |
| Scene VLM | **Qwen2.5-VL-3B** | #1 open-source VLM at its size class. Only invoked when anomaly fires — not every frame. |
| Text Embedding | **all-MiniLM-L6-v2** | Fast, fully offline, 384-dim semantic embedding. Runs on CPU. |
| Text Vector DB | **Pinecone** | Managed, low-latency vector search. Self-hosted (ChromaDB/LanceDB) in Phase 5. |
| Video RAG | **Twelve Labs Marengo 3.0** | Temporal video embedding — understands motion, not just static frames. |
| LLM | **Gemini 2.5 Flash** | Generous free tier (1000 RPD). Replaced with local Gemma 2 in Phase 5. |
| Optimisation | **OpenVINO 2024** | YOLO26 exported to OpenVINO IR format for NPU/iGPU inference on Intel AI PCs. |

---

## Features

- ✅ Real-time 30 FPS person detection and tracking
- ✅ Configurable zone polygons per camera (SHELF_AREA, CHECKOUT, ENTRANCE, etc.)
- ✅ Dwell time alerts, crouch detection, head-scanning detection
- ✅ Risk scoring with configurable HIGH / MEDIUM / LOW thresholds
- ✅ Structured JSON event log (`visionlog_events.jsonl`) + human-readable text log
- ✅ Multimodal RAG: query both text events AND video footage in plain English
- ✅ FastAPI web dashboard with live MJPEG stream + chat interface
- ✅ Video evidence modal — see the exact clip from the database with seekable playback
- ✅ `--no-pinecone` flag for offline demo mode (logs locally only)
- ✅ `--disable-vlm` flag for low-VRAM hardware (skips Qwen inference)
- ✅ OpenVINO model export for edge deployment

---

## Project Structure

```
logvision/
├── main.py                     # Core pipeline — run this for live processing
├── app.py                      # FastAPI web server + streaming endpoint
├── query.py                    # Dual-mode RAG query engine (text + video)
├── video_rag.py                # Standalone video indexing + search script
├── visionlog/
│   ├── config.py               # All cameras, zones, thresholds — edit this
│   ├── detection/              # YOLO wrapper + frame annotation
│   ├── enrichment/
│   │   ├── attribute_enricher.py   # SigLIP clothing/colour attributes
│   │   ├── scene_describer.py      # Qwen2.5-VL async scene captioning
│   │   ├── head_pose_tracker.py    # Head rotation → surveillance checking detection
│   │   └── hand_tracker.py         # MediaPipe hand keypoint tracking
│   ├── anomaly/
│   │   ├── anomaly_detector.py     # Rule-based anomaly triggers
│   │   ├── behaviour_window.py     # Rolling time-window behaviour memory
│   │   └── risk_scorer.py          # Weighted composite risk score
│   ├── events/
│   │   ├── event_store.py          # Embeds + upserts to Pinecone
│   │   └── video_archiver.py       # Background video clip saver + Twelve Labs uploader
│   ├── zones/                  # Polygon zone manager
│   └── sources/                # Video file + RTSP stream source wrapper
├── web/
│   ├── index.html              # Dashboard UI
│   ├── style.css               # Futuristic dark-mode UI
│   └── app.js                  # Chat + MJPEG + Video Modal logic
├── botsort_reid.yaml           # BoT-SORT tracker config
├── requirements.txt
├── .env.example                # Copy to .env and fill in your API keys
└── yolo26l-pose.pt             # YOLO26 Large weights (download separately)
```

---

## Quick Start

### 1. Clone and install dependencies

```bash
git clone https://github.com/aaartemis5/logvision.git
cd logvision
pip install -r requirements.txt
```

### 2. Set up API keys

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```env
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX=logvision
GEMINI_API_KEY=your_gemini_api_key_here
TWELVE_LABS_API_KEY=your_twelve_labs_api_key_here
PINECONE_VIDEO_INDEX=visionlog-video
```

| Key | Where to get it | Free tier? |
|---|---|---|
| `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com/app/apikey) | ✅ Yes (1000 RPD) |
| `PINECONE_API_KEY` | [app.pinecone.io](https://app.pinecone.io) | ✅ Yes (1 free index) |
| `TWELVE_LABS_API_KEY` | [playground.twelvelabs.io](https://playground.twelvelabs.io/dashboard/api-keys) | ✅ Yes (600 min free) |

### 3. Configure your camera

Edit `visionlog/config.py` and point the camera path to your video file or RTSP stream:

```python
CAMERAS = {
    "CAM_01": {
        "name": "Store Front",
        "path": r"C:\path\to\your\video.mp4",  # or 0 for webcam, or rtsp://...
        ...
    }
}
```

### 4. Download the YOLO26 model weights

Download `yolo26l-pose.pt` from the [Ultralytics releases](https://github.com/ultralytics/ultralytics) and place it in the project root. The model is auto-downloaded on first run if you have `ultralytics` installed.

---

## Running the Pipeline

### Option A — Web Dashboard (recommended)

Launches the full pipeline with live browser-based MJPEG stream + AI chat interface:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open [http://localhost:8000](http://localhost:8000)

### Option B — Command Line (headless)

Runs the full pipeline in the terminal without the UI:

```bash
python main.py
```

**Useful flags:**

```bash
python main.py --no-pinecone     # Demo mode: logs events locally, no cloud writes
python main.py --headless        # No OpenCV window (for servers / Colab)
python main.py --disable-vlm    # Skip Qwen VLM (for low-VRAM GPUs like GTX 1650 Ti)
python main.py --source path/to/video.mp4   # Override video path from CLI
```

---

## Querying the RAG System

### Via the Web Dashboard
Type any question into the Inquiry Center chat box. Toggle **Dual RAG Search** on/off to include video evidence.

### Via Command Line

```bash
# Text-only query (fastest)
python query.py "Who was acting suspiciously near the shelves?"

# Video-only search (returns clip timestamps)
python query.py --mode video "Show me someone crouching"

# Both text report + video evidence
python query.py --mode both "Was there anyone near the entrance after 10pm?"

# Filter by zone or severity
python query.py --zone SHELF_AREA --severity HIGH "Summarise all high risk activity"
```

---

## Indexing Video for RAG

Before video queries work, you need to index your video file with Twelve Labs:

```bash
# Index a video file
python video_rag.py index --video "path/to/your/video.mp4"

# Test a search query after indexing
python video_rag.py query "someone crouching near shelves"
```

---

## Hardware Requirements

| Mode | Minimum | Recommended |
|---|---|---|
| Detection only (`--disable-vlm`) | Intel i5 + 8GB RAM | Any NVIDIA GPU |
| Full pipeline (with Qwen VLM) | NVIDIA GTX 1650 Ti + 8GB VRAM | RTX 3060+ |
| OpenVINO optimised | Intel Core Ultra (NPU) | Intel Arc GPU |

> **Note:** The Qwen2.5-VL model requires ~6GB VRAM. Use `--disable-vlm` if you are on a low-end GPU. All anomaly detection, tracking, and re-ID still run at full speed without it.

---

## Roadmap

- [x] Phase 1 — Standalone Video RAG (Twelve Labs + Pinecone)
- [x] Phase 2 — Live video clip auto-indexing during processing
- [x] Phase 3 — Dual-mode RAG query system (text + video)
- [x] Phase 4 — Web dashboard with live stream + chat UI
- [ ] Phase 5 — 100% local mode (LanceDB + Gemma 2 + local video embeddings)
- [ ] Phase 6 — Cross-camera Re-ID (Multi-Target Multi-Camera / MTMC)
- [ ] Phase 7 — OpenVINO NPU deployment for Intel AI PCs

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — detection backbone
- [BoxMOT](https://github.com/mikel-brostrom/boxmot) — tracking + Re-ID framework
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) — scene understanding VLM
- [Twelve Labs](https://twelvelabs.io) — multimodal video embedding API
- [Google Gemini](https://aistudio.google.com) — intelligence report generation
- [Pinecone](https://pinecone.io) — vector database
