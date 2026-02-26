"""
sources/video_source.py
Uniform interface for any video source: file, webcam, or RTSP stream.
"""
import re
import time
import cv2
from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class SourceConfig:
    camera_id:  str
    name:       str
    path:       str | int   # file path, 0 (webcam), or rtsp://... URL
    location:   str = ""
    lat:        Optional[float] = None
    lon:        Optional[float] = None
    floor:      int = 0
    direction:  str = ""


def _parse_recording_start(path) -> float:
    """
    Try to extract the recording timestamp from the filename.
    Handles patterns like 'Recording 2026-02-25 162753.mp4'
    Falls back to file mtime, then current time.
    """
    import os
    try:
        # Match 'YYYY-MM-DD HHMMSS' anywhere in the path
        m = re.search(r'(\d{4}-\d{2}-\d{2})\s+(\d{6})', str(path))
        if m:
            dt = datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H%M%S")
            return dt.timestamp()
    except Exception:
        pass
    try:
        return os.path.getmtime(str(path))
    except Exception:
        pass
    return time.time()


class VideoSource:
    """Wraps cv2.VideoCapture behind a clean interface."""

    def __init__(self, cfg: SourceConfig):
        self.cfg = cfg
        self.cap = cv2.VideoCapture(cfg.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source '{cfg.path}' for camera {cfg.camera_id}")

        self.fps       = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width     = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height    = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_num = 0

        # Anchor timestamp: recording start time from filename (or fallback)
        self.recording_start = _parse_recording_start(cfg.path)

        print(f"[VideoSource] {cfg.camera_id} ({cfg.name}) opened: "
              f"{self.width}x{self.height} @ {self.fps:.1f}fps | source: {cfg.path}")

    def read(self):
        """Returns (success, frame, frame_number, frame_ts).
        frame_ts is the unix timestamp of this frame based on the video position,
        NOT wall-clock processing time.
        """
        ret, frame = self.cap.read()
        if ret:
            self.frame_num += 1
            # Use video position in ms to compute accurate event timestamp
            video_pos_s = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frame_ts    = self.recording_start + video_pos_s
        else:
            frame_ts = time.time()
        return ret, frame, self.frame_num, frame_ts

    def release(self):
        self.cap.release()

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame, fn, frame_ts = self.read()
        if not ret:
            self.release()
            raise StopIteration
        return frame, {"frame_number": fn, "camera_id": self.cfg.camera_id, "frame_ts": frame_ts}

    def camera_meta(self) -> dict:
        return {
            "camera_id": self.cfg.camera_id,
            "name":      self.cfg.name,
            "location":  self.cfg.location,
            "lat":       self.cfg.lat,
            "lon":       self.cfg.lon,
            "floor":     self.cfg.floor,
            "direction": self.cfg.direction,
        }
