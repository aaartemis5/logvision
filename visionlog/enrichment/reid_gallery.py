"""
enrichment/reid_gallery.py — Gallery-based person Re-ID using CLIP embeddings.

Each new ByteTrack local_id is compared to stored appearance vectors.
If cosine similarity ≥ REID_THRESHOLD the existing global_id is restored,
otherwise a new one (G1, G2, …) is created.
"""
import time
import numpy as np

REID_THRESHOLD = 0.90
EXPIRE_S       = 600    # forget a person after 10 min of inactivity
MAX_GALLERY    = 100


class ReIdGallery:
    """In-memory appearance gallery. One instance per camera session."""

    def __init__(self):
        self._gallery: dict[str, dict] = {}          # global_id → entry
        self._local_to_global: dict[int, str] = {}   # local_id → global_id
        self._counter = 0

    def match(self,
              local_id:  int,
              embedding: np.ndarray | None,
              zone:      str = "UNKNOWN",
              clothing:  str = "unknown") -> str | None:
        """
        Returns the global_id for this local_id.
        Fast-path if already assigned; otherwise attempts gallery match.
        Returns None if embedding not yet available.
        """
        if local_id in self._local_to_global:
            gid = self._local_to_global[local_id]
            if gid in self._gallery:
                self._gallery[gid]['last_seen'] = time.time()
                self._gallery[gid]['last_zone']  = zone
            return gid

        if embedding is None:
            return None

        self._evict_expired()

        emb = embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        best_gid, best_sim = None, 0.0
        for gid, entry in self._gallery.items():
            sim = float(np.dot(emb, entry['embedding']))
            if sim > best_sim:
                best_sim, best_gid = sim, gid

        if best_sim >= REID_THRESHOLD and best_gid:
            gid = best_gid
            print(f"[ReID] #{local_id} → {gid} (sim={best_sim:.3f})", flush=True)
        else:
            self._counter += 1
            gid = f"G{self._counter}"

        self._gallery[gid] = {
            'embedding': emb,
            'last_seen': time.time(),
            'last_zone': zone,
            'clothing':  clothing,
        }
        self._local_to_global[local_id] = gid
        return gid

    def evict_stale_locals(self, active_local_ids: set[int]):
        """Remove local→global mappings for tracks that have left the frame."""
        for lid in list(self._local_to_global.keys()):
            if lid not in active_local_ids:
                del self._local_to_global[lid]

    def get_global_id(self, local_id: int) -> str | None:
        return self._local_to_global.get(local_id)

    def _evict_expired(self):
        now = time.time()
        expired = [g for g, e in self._gallery.items() if now - e['last_seen'] > EXPIRE_S]
        for g in expired:
            del self._gallery[g]
        if len(self._gallery) > MAX_GALLERY:
            oldest = sorted(self._gallery, key=lambda g: self._gallery[g]['last_seen'])
            for g in oldest[:len(self._gallery) - MAX_GALLERY]:
                del self._gallery[g]
