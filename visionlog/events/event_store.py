"""
events/event_store.py
Embeds text_summary and upserts event to Pinecone.
Also writes a human-readable local log file as backup.
"""
import json
import time
import datetime
from pathlib import Path
from visionlog import config

# Lazy imports so app starts fast even without pinecone configured
_pc_index  = None
_embedder  = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        print(f"[EventStore] Embedding model loaded: {config.EMBEDDING_MODEL}")
    return _embedder


def _get_pinecone_index():
    global _pc_index
    if _pc_index is None:
        from pinecone import Pinecone
        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        _pc_index = pc.Index(config.PINECONE_INDEX)
        print(f"[EventStore] Connected to Pinecone index: {config.PINECONE_INDEX}")
    return _pc_index


def store(event: dict, use_pinecone: bool = True):
    """
    1. Embed text_summary
    2. Upsert to Pinecone (if configured)
    3. Write to local log file
    4. Print to terminal
    """
    text = event.get("text_summary", "")

    # ── Terminal print ────────────────────────────────────────────────
    anom = event.get("anomaly", {})
    tag  = f"  ⚠️  [{anom.get('severity')}]" if anom.get("is_anomaly") else ""
    print(f"[EVENT]{tag} {text}")

    # ── Local log file ────────────────────────────────────────────────
    with open(config.LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

    # ── Pinecone upsert ───────────────────────────────────────────────
    if use_pinecone and config.PINECONE_API_KEY != "YOUR_PINECONE_API_KEY":
        try:
            embedder = _get_embedder()
            vector   = embedder.encode(text).tolist()
            index    = _get_pinecone_index()

            # Pinecone metadata must be flat (no nested dicts)
            flat_meta = {
                "event_id":    event["event_id"],
                "session_id":  event["session_id"],
                "camera_id":   event["camera"]["camera_id"],
                "camera_name": event["camera"]["name"],
                "timestamp":   event["time"]["unix"],
                "day_of_week": event["time"]["day_of_week"],
                "is_night":    event["time"]["is_night"],
                "person_id":   event["person"]["local_id"],
                "zone":        event["detection"]["zone"],
                "pose":        event["pose"]["label"],
                "event_type":  event["event_type"],
                "is_anomaly":  event["anomaly"]["is_anomaly"],
                "severity":    event["anomaly"].get("severity", "NORMAL"),
                "text":        text[:500],   # Pinecone metadata has size limits
            }

            index.upsert(vectors=[{
                "id":       event["event_id"],
                "values":   vector,
                "metadata": flat_meta,
            }])
        except Exception as e:
            print(f"[EventStore] Pinecone error (event still logged locally): {e}")
    else:
        # Save full JSON locally instead
        with open("visionlog_events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
