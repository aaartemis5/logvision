"""
enrichment/attribute_enricher.py — InsightFace (age/gender) + SigLIP (clothing + Re-ID).
SigLIP replaces both CLIP and OSNet for unified appearance understanding.
"""
import cv2
import numpy as np
from visionlog.detection.detector import Detection
from visionlog.enrichment.siglip_describer import get_siglip_describer


_siglip_describer = get_siglip_describer()

_insight_app = None

def _get_insight():
    global _insight_app
    if _insight_app is None:
        from insightface.app import FaceAnalysis
        _insight_app = FaceAnalysis(
            name="buffalo_sc",
            allowed_modules=["detection", "genderage"],
            providers=["CPUExecutionProvider"]
        )
        _insight_app.prepare(ctx_id=-1, det_size=(320, 320))
        print("[AttributeEnricher] InsightFace ready.")
    return _insight_app


class AttributeEnricher:
    """Enriches detections with age, gender, and clothing description. Cached per person."""

    def __init__(self):
        self._cache: dict[int, dict] = {}

    def enrich(self, det: Detection, frame: np.ndarray) -> dict:
        tid = det.local_id
        if tid in self._cache:
            return self._cache[tid]
        attrs = self._run(det, frame)
        self._cache[tid] = attrs
        return attrs

    def _run(self, det: Detection, frame: np.ndarray) -> dict:
        x1, y1, x2, y2 = det.bbox
        h, w = frame.shape[:2]
        PAD  = 50

        # InsightFace — padded crop for better face detection
        px1 = max(0, x1-PAD); py1 = max(0, y1-PAD)
        px2 = min(w, x2+PAD); py2 = min(h, y2+PAD)
        person_crop = frame[py1:py2, px1:px2]

        age, gender = None, None
        if person_crop.size > 0:
            try:
                cr_h, cr_w = person_crop.shape[:2]
                if min(cr_h, cr_w) < 160:
                    s           = 160 / min(cr_h, cr_w)
                    person_crop = cv2.resize(person_crop, (int(cr_w*s), int(cr_h*s)),
                                             interpolation=cv2.INTER_LINEAR)
                faces = _get_insight().get(person_crop)
                if faces:
                    age    = int(faces[0].age)
                    gender = "female" if faces[0].gender == 0 else "male"
            except Exception:
                pass

        # SigLIP — unified clothing description + Re-ID embedding
        # Use upper 70% of bbox for better clothing visibility
        ub = min(h, y1 + int((y2-y1)*0.70))
        siglip_crop = frame[max(0,y1):ub, max(0,x1):min(w,x2)]
        clothing, reid_embedding = _siglip_describer.describe(siglip_crop)

        return {
            "age":            age,
            "gender":         gender,
            "emotion":        None,
            "clothing_desc":  clothing,
            "reid_embedding": reid_embedding,   # 768-dim from SigLIP (replaces OSNet)
            "clip_embedding": None,              # Deprecated - kept for compatibility
        }

    def evict(self, active_ids: set[int]):
        for tid in list(self._cache.keys()):
            if tid not in active_ids:
                del self._cache[tid]
