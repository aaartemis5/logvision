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
    seat_info = event.get("seat", {})
    if seat_info.get("event"):
        tag = f"  🪑 [{seat_info.get('event')}: {seat_info.get('seat_id')}]"
    elif "restless_head_movement" in anom.get("reasons", []):
        tag = "  👁️ [RESTLESS FIDGETING]"
    elif anom.get("is_anomaly"):
        tag = f"  ⚡ [{anom.get('severity')}]"
    else:
        tag = ""
    
    console_text = text
    if " VLM: '" in console_text:
        console_text = console_text.split(" VLM: '")[0] + " [VLM Context Logged]"
        
    print(f"[EVENT]{tag} {console_text}")

    # ── Local log file ────────────────────────────────────────────────
    with open(config.LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

    # ── Always write full JSON locally ────────────────────────────────
    with open("visionlog_events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

    # ── Pinecone upsert (if enabled and configured) ───────────────────
    if use_pinecone and config.PINECONE_API_KEY and config.PINECONE_API_KEY != "YOUR_PINECONE_API_KEY":
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
                "seat_id":     event.get("seat", {}).get("seat_id") or "",
                "pose":        event["pose"]["label"],
                "event_type":  event["event_type"],
                "is_anomaly":  event["anomaly"]["is_anomaly"],
                "severity":    event["anomaly"].get("severity", "NORMAL"),
                "text":        text[:500],
            }

            index.upsert(vectors=[{
                "id":       event["event_id"],
                "values":   vector,
                "metadata": flat_meta,
            }])
        except Exception as e:
            print(f"[EventStore] Pinecone error: {e}")
