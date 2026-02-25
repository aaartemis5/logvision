"""
enrichment/scene_describer.py
Runs Florence-2 in a BACKGROUND THREAD — never blocks the main pipeline.

describe() always returns immediately:
  - First call: returns None (model loading or first inference pending)
  - Subsequent calls: returns the last completed caption while next one computes
"""
import threading
import time
import cv2
import numpy as np
import torch
from PIL import Image
from visionlog import config


class SceneDescriber:
    """
    Non-blocking Florence-2 wrapper.
    The main loop calls `request(frame, cam_id)` every frame;
    it triggers inference in the background if enough time has passed.
    `last_result(cam_id)` returns the most recently completed caption (or None).
    """

    def __init__(self):
        self._model       = None
        self._processor   = None
        self._lock        = threading.Lock()       # guards model load
        self._last_req:   dict[str, float] = {}    # cam_id → last request time
        self._last_result: dict[str, dict] = {}    # cam_id → last caption dict
        self._busy:       dict[str, bool]  = {}    # cam_id → inference running?

    # ── Public API ────────────────────────────────────────────────────

    def should_run(self, camera_id: str) -> bool:
        """True if enough time has passed since last request for this camera."""
        last = self._last_req.get(camera_id, 0)
        return (time.time() - last) >= config.FLORENCE_INTERVAL

    def request(self, frame: np.ndarray, camera_id: str) -> None:
        """
        Kick off a background Florence inference if not already running.
        Returns immediately — call last_result() to get the output.
        """
        if self._busy.get(camera_id, False):
            return   # still computing previous request
        self._last_req[camera_id] = time.time()
        self._busy[camera_id] = True
        # Take a copy so the thread has its own frame reference
        frame_copy = frame.copy()
        t = threading.Thread(
            target=self._run,
            args=(frame_copy, camera_id),
            daemon=True
        )
        t.start()

    def last_result(self, camera_id: str) -> dict | None:
        """Return the most recently completed caption, or None if never run."""
        return self._last_result.get(camera_id)

    # ── Backwards-compat shim (old call-site used describe + should_run) ──
    def describe(self, frame: np.ndarray, camera_id: str) -> dict | None:
        """Legacy shim: request inference and return whatever we have so far."""
        self.request(frame, camera_id)
        return self.last_result(camera_id)

    # ── Internal ─────────────────────────────────────────────────────

    def _load(self):
        """Lazy-load model (thread-safe, only loads once)."""
        with self._lock:
            if self._model is None:
                print("[SceneDescriber] Loading Florence-2... (one-time, may take a minute)")
                from transformers import AutoProcessor, AutoModelForCausalLM
                self._model = AutoModelForCausalLM.from_pretrained(
                    config.FLORENCE_MODEL, trust_remote_code=True,
                    torch_dtype=torch.float32
                )
                self._processor = AutoProcessor.from_pretrained(
                    config.FLORENCE_MODEL, trust_remote_code=True
                )
                print("[SceneDescriber] Florence-2 ready.")

    def _run(self, frame: np.ndarray, camera_id: str):
        """Background worker: run inference and store result."""
        try:
            self._load()
            # Downscale to 512px wide before sending to Florence — much faster
            h, w = frame.shape[:2]
            scale = min(1.0, 512 / w)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img   = Image.fromarray(frame_rgb)

            task   = "<DETAILED_CAPTION>"
            inputs = self._processor(text=task, images=pil_img, return_tensors="pt")
            with torch.no_grad():
                ids = self._model.generate(
                    input_ids    = inputs["input_ids"],
                    pixel_values = inputs["pixel_values"],
                    max_new_tokens = 80,
                    num_beams    = 2,
                )
            raw    = self._processor.batch_decode(ids, skip_special_tokens=False)[0]
            parsed = self._processor.post_process_generation(
                raw, task=task, image_size=(pil_img.width, pil_img.height)
            )
            caption = str(parsed.get(task, "")).strip()

            self._last_result[camera_id] = {
                "florence_caption": caption,
                "person_count":     None,
                "ocr_text":         None,
            }
        except Exception as e:
            print(f"[SceneDescriber] Error in background thread: {e}")
        finally:
            self._busy[camera_id] = False
