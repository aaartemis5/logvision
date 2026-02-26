import asyncio
import io
import cv2
import json
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from visionlog import config
from query import ask

app = FastAPI(title="VisionLog AI Dashboard")

# Ensure web directory exists
os.makedirs("web", exist_ok=True)

# Mount static files for HTML/CSS/JS
app.mount("/static", StaticFiles(directory="web"), name="static")

class ChatRequest(BaseModel):
    query: str
    camera_id: str | None = None # "ALL" or specific CAM_ID

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("web/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/cameras")
async def get_cameras():
    """Return configured cameras for the UI dropdown."""
    cams = [{"id": cid, "name": cfg["name"]} for cid, cfg in config.CAMERAS.items()]
    return JSONResponse({"cameras": cams})

@app.post("/api/chat")
async def chat_api(req: ChatRequest):
    """RAG query endpoint."""
    cam_filter = None if req.camera_id in ("ALL", None, "") else req.camera_id
    
    # Run the blocking query in a thread pool to not freeze FastAPI
    try:
        reply = await asyncio.to_thread(ask, req.query, None, zone=None, severity=None, camera=cam_filter)
        return JSONResponse({"reply": reply})
    except Exception as e:
        return JSONResponse({"reply": f"Error: {str(e)}", "error": True}, status_code=500)

# ── Video Streaming Logic ───────────────────────────────────────────────

def generate_video_feed(camera_id: str):
    """
    Generator that yields JPEG frames from main.py's processing loop.
    We need to import main's logic dynamically to avoid starting it on import.
    """
    from main import run_camera_generator
    
    # Find camera config
    cam_cfg = config.CAMERAS.get(camera_id)
    if not cam_cfg:
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + b"" + b"\r\n"
        return

    # Run the generator
    # use_pinecone=False prevents writing duplicate records to Pinecone every time you view the file
    for frame_bgr in run_camera_generator(camera_id, cam_cfg, use_pinecone=False):
        ret, buffer = cv2.imencode('.jpg', frame_bgr)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/api/video/{camera_id}")
def video_feed(camera_id: str):
    """MJPEG streaming endpoint for a specific camera."""
    return StreamingResponse(
        generate_video_feed(camera_id), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    import uvicorn
    # Make sure to run the main code, not in the app scope
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
