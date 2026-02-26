"""
enrichment/attribute_enricher.py — InsightFace (age/gender) + CLIP (clothing).
Both run once per new person and are cached by local_id.
"""
import cv2
import numpy as np
from visionlog.detection.detector import Detection


class ClipDescriber:
    """
    CLIP ViT-B/32 zero-shot clothing classifier.
    Text embeddings are pre-computed at load time; inference is one image forward pass.
    Returns (clothing_label, unit-norm 512-dim embedding).
    """

    COLOUR_LABELS = [
        "black", "white", "grey",   "navy",
        "blue",  "red",   "green",  "yellow",
        "orange","purple","brown",  "pink",
    ]
    COLOUR_PROMPTS = [
        f"a photo of a person wearing {c} colored clothes" for c in COLOUR_LABELS
    ]

    GARMENT_LABELS = ["hoodie", "t-shirt", "jacket", "shirt", "sweater", "uniform"]
    GARMENT_PROMPTS = [
        "a photo of a person wearing a hoodie or sweatshirt",
        "a photo of a person wearing a t-shirt",
        "a photo of a person wearing a jacket or coat",
        "a photo of a person wearing a collared shirt",
        "a photo of a person wearing a sweater or jumper",
        "a photo of a person wearing a work uniform",
    ]

    def __init__(self):
        self._processor      = None
        self._model          = None
        self._colour_embeds  = None
        self._garment_embeds = None

    def _load(self):
        import torch
        from transformers import CLIPProcessor, CLIPModel
        print("[CLIP] Loading openai/clip-vit-base-patch32 …", flush=True)
        self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self._model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self._model.eval()
        def _extract_512(feat, proj_layer):
            if not isinstance(feat, torch.Tensor):
                feat = getattr(feat, 'text_embeds', getattr(feat, 'image_embeds', getattr(feat, 'pooler_output', feat[0])))
            if isinstance(feat, torch.Tensor) and feat.dim() > 2:
                feat = feat[:, 0, :] if feat.shape[1] > 1 else feat.squeeze(1)
            if isinstance(feat, torch.Tensor) and feat.shape[-1] != 512 and proj_layer is not None:
                if isinstance(proj_layer, torch.nn.Linear):
                    feat = proj_layer(feat)
                elif hasattr(proj_layer, 'weight'):
                    # Sometimes projection is just a parameter matrix [768, 512] or [512, 768]
                    w = proj_layer.weight if proj_layer.weight.shape[0] == 768 else proj_layer.weight.T
                    feat = feat @ w
            return feat

        with torch.no_grad():
            c_in   = self._processor(text=self.COLOUR_PROMPTS,  return_tensors="pt", padding=True)
            c_feat = _extract_512(self._model.get_text_features(**c_in), getattr(self._model, 'text_projection', None))
            self._colour_embeds = c_feat / c_feat.norm(dim=-1, keepdim=True)

            g_in   = self._processor(text=self.GARMENT_PROMPTS, return_tensors="pt", padding=True)
            g_feat = _extract_512(self._model.get_text_features(**g_in), getattr(self._model, 'text_projection', None))
            self._garment_embeds = g_feat / g_feat.norm(dim=-1, keepdim=True)
        print("[CLIP] Ready.", flush=True)

    def describe(self, person_bgr: np.ndarray) -> tuple[str, np.ndarray | None]:
        """Returns (clothing_label, unit-norm embedding) or ('unknown', None) on failure."""
        if person_bgr is None or person_bgr.size == 0:
            return "unknown", None

        h, w = person_bgr.shape[:2]
        if min(h, w) < 112:
            scale      = 112 / min(h, w)
            person_bgr = cv2.resize(person_bgr, (int(w*scale), int(h*scale)),
                                    interpolation=cv2.INTER_LINEAR)
        try:
            import torch
            from PIL import Image
            if self._model is None:
                self._load()

            pil_img  = Image.fromarray(cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB))
            with torch.no_grad():
                img_in   = self._processor(images=pil_img, return_tensors="pt")
                img_feat = self._model.get_image_features(**img_in)
                
                # Robust extraction
                if not isinstance(img_feat, torch.Tensor):
                    img_feat = getattr(img_feat, 'image_embeds', getattr(img_feat, 'pooler_output', img_feat[0]))
                if isinstance(img_feat, torch.Tensor) and img_feat.dim() > 2:
                    img_feat = img_feat[:, 0, :] if img_feat.shape[1] > 1 else img_feat.squeeze(1)
                if isinstance(img_feat, torch.Tensor) and img_feat.shape[-1] != 512 and hasattr(self._model, 'visual_projection'):
                    proj = self._model.visual_projection
                    if isinstance(proj, torch.nn.Linear):
                        img_feat = proj(img_feat)
                    elif hasattr(proj, 'weight'):
                        w = proj.weight if proj.weight.shape[0] == 768 else proj.weight.T
                        img_feat = img_feat @ w
                        
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

                colour  = self.COLOUR_LABELS[(img_feat @ self._colour_embeds.T).squeeze(0).argmax().item()]
                garment = self.GARMENT_LABELS[(img_feat @ self._garment_embeds.T).squeeze(0).argmax().item()]

            return f"{colour} {garment}", img_feat.squeeze(0).cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"[CLIP] {e}")
            return "unknown", None


_clip_describer = ClipDescriber()


class OsNetDescriber:
    """
    OSNet-x0.25 person Re-ID feature extractor (via torchreid).
    Returns a 512-dim unit-norm embedding for identity matching.
    Uses full person bbox at 256×128 — the standard input for person Re-ID models.
    """

    def __init__(self):
        self._model     = None
        self._transform = None

    def _load(self):
        import torchreid
        import torchvision.transforms as T
        print("[OSNet] Loading osnet_x0_25...", flush=True)
        self._model = torchreid.models.build_model(
            name='osnet_x0_25', num_classes=1000, pretrained=True
        )
        self._model.eval()
        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("[OSNet] Ready.", flush=True)

    def embed(self, person_bgr: np.ndarray) -> np.ndarray | None:
        """Returns 512-dim unit-norm Re-ID embedding or None on failure."""
        if person_bgr is None or person_bgr.size == 0:
            return None
        try:
            import torch
            if self._model is None:
                self._load()
            rgb = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)
            x   = self._transform(rgb).unsqueeze(0)   # (1, 3, 256, 128)
            with torch.no_grad():
                feats = self._model(x)                 # (1, 512) in eval mode
            emb  = feats.squeeze(0).cpu().numpy().astype(np.float32)
            norm = np.linalg.norm(emb)
            return (emb / norm) if norm > 0 else emb
        except Exception as e:
            print(f"[OSNet] {e}")
            return None


_osnet_describer = OsNetDescriber()

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

        # CLIP — upper 65% of bbox for clothing label
        ub        = min(h, y1 + int((y2-y1)*0.65))
        clip_crop = frame[max(0,y1):ub, max(0,x1):min(w,x2)]
        clothing, clip_embedding = _clip_describer.describe(clip_crop)

        # OSNet — full person bbox for identity Re-ID embedding
        reid_crop      = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        reid_embedding = _osnet_describer.embed(reid_crop)

        return {
            "age":            age,
            "gender":         gender,
            "emotion":        None,
            "clothing_desc":  clothing,
            "clip_embedding": clip_embedding,   # kept for legacy / debug
            "reid_embedding": reid_embedding,   # OSNet vector — use this in ReID gallery
        }

    def evict(self, active_ids: set[int]):
        for tid in list(self._cache.keys()):
            if tid not in active_ids:
                del self._cache[tid]
