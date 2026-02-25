"""
sources/video_source.py
Uniform interface for any video source: file, webcam, or RTSP stream.
Adding a new camera = one new entry in config.CAMERAS. Nothing else changes.
"""
import cv2
from dataclasses import dataclass
from typing import Optional


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


class VideoSource:
    """
    Wraps cv2.VideoCapture behind a clean interface.
    Usage:
        src = VideoSource(cfg)
        for frame, meta in src:
            process(frame, meta)
    """

    def __init__(self, cfg: SourceConfig):
        self.cfg = cfg
        self.cap = cv2.VideoCapture(cfg.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source '{cfg.path}' for camera {cfg.camera_id}")

        self.fps       = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width     = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height    = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_num = 0

        print(f"[VideoSource] {cfg.camera_id} ({cfg.name}) opened: "
              f"{self.width}x{self.height} @ {self.fps:.1f}fps | source: {cfg.path}")

    def read(self):
        """Returns (success, frame, frame_number)."""
        ret, frame = self.cap.read()
        if ret:
            self.frame_num += 1
        return ret, frame, self.frame_num

    def release(self):
        self.cap.release()

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame, fn = self.read()
        if not ret:
            self.release()
            raise StopIteration
        return frame, {"frame_number": fn, "camera_id": self.cfg.camera_id}

    def camera_meta(self) -> dict:
        """Returns camera metadata dict for embedding in events."""
        return {
            "camera_id": self.cfg.camera_id,
            "name":      self.cfg.name,
            "location":  self.cfg.location,
            "lat":       self.cfg.lat,
            "lon":       self.cfg.lon,
            "floor":     self.cfg.floor,
            "direction": self.cfg.direction,
        }
