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

    def __init__(self, disabled: bool = False):
        self._disabled    = disabled
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
        return (time.time() - last) >= config.VLM_INTERVAL

    def request(self, frame: np.ndarray, camera_id: str) -> None:
        """
        Kick off a background Florence inference if not already running.
        Returns immediately — call last_result() to get the output.
        """
        if self._disabled:
            self._last_result[camera_id] = {
                "vlm_caption": "VLM Disabled for Web Demo Mode.",
                "person_count": None,
                "ocr_text": None,
            }
            return
            
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
                print(f"[SceneDescriber] Loading {config.VLM_MODEL}... (may take a minute)")
                try:
                    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
                    self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        config.VLM_MODEL, torch_dtype="auto", device_map="auto"
                    )
                    self._processor = AutoProcessor.from_pretrained(
                        config.VLM_MODEL
                    )
                    print("[SceneDescriber] Qwen2.5-VL ready.")
                except ImportError as e:
                    print(f"[SceneDescriber] Missing dependencies for Qwen! Ensure qwen-vl-utils is installed. Error: {e}")

    def _run(self, frame: np.ndarray, camera_id: str):
        """Background worker: run inference and store result."""
        try:
            self._load()
            if self._model is None:
                return
            
            from qwen_vl_utils import process_vision_info
            
            # Convert OpenCV BGR to RGB PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img   = Image.fromarray(frame_rgb)

            task   = (
                "You are a retail security analyst reviewing a live CCTV frame. "
                "Ignore normal shopping behaviour. ONLY respond if you see something suspicious or security-relevant. "
                "If you detect suspicious behaviour, immediately flag it and provide: "
                "(1) a description of the person — their clothing colour, garment type, approximate build; "
                "(2) their exact location in the store (e.g. near shelves, checkout, entrance); "
                "(3) precisely what they are doing that is suspicious (e.g. concealing items, loitering, unusual movements). "
                "If there is nothing suspicious, respond with exactly: 'No suspicious activity detected.'"
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": task},
                    ],
                }
            ]
            
            # Preparation for inference
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self._model.device)
            
            # Inference
            with torch.no_grad():
                generated_ids = self._model.generate(**inputs, max_new_tokens=128)
                
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            caption = self._processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

            self._last_result[camera_id] = {
                "vlm_caption": caption,
                "person_count":     None,
                "ocr_text":         None,
            }
        except Exception as e:
            print(f"[SceneDescriber] Error in background thread: {e}")
        finally:
            self._busy[camera_id] = False
