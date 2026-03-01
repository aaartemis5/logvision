import asyncio
import base64
import json
import os
import cv2
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path

from visionlog import config
from main import run_camera_generator
from query import ask

app = FastAPI(title="VisionLog AI Dashboard")

# Mount frontend web assets
app.mount("/web", StaticFiles(directory="web"), name="web")

# ── API Models ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    mode: str = "both"  # "text", "video", or "both"

# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    """Serve the main UI directly at root."""
    return StreamingResponse(open("web/index.html", "rb"), media_type="text/html")


def video_stream_generator(camera_id: str, use_pinecone: bool = False):
    """
    Consumes the raw np.ndarray frames from main.py's generator,
    encodes them as JPEG buffer, and yields them as a Multipart MJPEG stream.
    """
    # Fetch cam config directly from dict (matching how CLI operates)
    if camera_id not in config.CAMERAS:
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n'
        return
        
    cam_cfg = config.CAMERAS[camera_id]
    
    # Run the VisionLog pipeline generator (disable heavy VLM for web streams)
    for frame in run_camera_generator(camera_id, cam_cfg, use_pinecone=use_pinecone, disable_vlm=True):
        # Frame is the annotated np.ndarray canvas
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        
        # Yield in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.get("/api/camera/stream/{camera_id}")
async def stream_camera(camera_id: str):
    """
    Endpoint that browsers connect to via <img src="/api/camera/stream/CAM_01">
    It runs VisionLog live!
    """
    # We turn off Pinecone during the demo UI stream so it doesn't pollute the index
    # (since the user will be streaming a pre-recorded file for the MVP)
    return StreamingResponse(
        video_stream_generator(camera_id, use_pinecone=False),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Endpoint for the AI Inquiry chat bubble.
    Calls query.py dual-mode logic and returns the raw result.
    """
    try:
        api_key = config.GEMINI_API_KEY
        if not api_key:
            return JSONResponse({"error": "No Gemini API key supplied in config."}, status_code=500)
            
        print(f"[API] Query received: {req.query} (mode: {req.mode})")
        
        # Run dual-query (blocks async loop but fast enough for demo)
        response_text = ask(
            query=req.query,
            api_key=api_key,
            mode=req.mode
        )
        
        # Parse output nicely for the frontend.
        # Format returns sections separated by headers.
        report = ""
        video_clips = []
        
        parts = response_text.split("================= 📝 INTELLIGENCE REPORT ============")
        if len(parts) == 2:
            report = parts[1].strip()
            video_section = parts[0]
        else:
            video_section = response_text
            
        for line in video_section.split("\n"):
            line = line.strip()
            if line.startswith("[") and "Time:" in line:
                # [1] Time: 0:23 | Camera: Recording.mp4 | Clip: 6s | Match: 0.10
                parts = line.split("|")
                time_str = parts[0].split("Time:")[1].strip()
                cam_str = parts[1].split("Camera:")[1].strip()
                match_val = float(parts[3].split("Match:")[1].strip())
                video_clips.append({
                    "time": time_str,
                    "camera": cam_str,
                    "score": match_val,
                })
                
        return {
            "report": report,
            "video_evidence": video_clips,
            "raw_output": response_text
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/media/{camera_filename}")
async def serve_video(camera_filename: str):
    """
    Serves video files for playback in the modal.
    Maps camera filename to actual video path from config.
    """
    try:
        # Find the camera config that matches this filename
        video_path = None
        for cam_id, cam_cfg in config.CAMERAS.items():
            cam_path = cam_cfg.get("path", "")
            # Extract just the filename from the full path
            if Path(cam_path).name == camera_filename:
                video_path = cam_path
                break
        
        if not video_path or not os.path.exists(video_path):
            return JSONResponse({"error": f"Video file not found: {camera_filename}"}, status_code=404)
        
        return FileResponse(
            video_path,
            media_type="video/mp4",
            headers={
                "Accept-Ranges": "bytes",
                "Content-Disposition": f'inline; filename="{camera_filename}"'
            }
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    # Ensure web directory exists
    os.makedirs("web", exist_ok=True)
    print("🚀 Starting VisionLog UI on http://localhost:8000")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
