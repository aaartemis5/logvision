"""
video_rag.py — Multimodal Video RAG Demo
=========================================
Phase 1: Standalone demo to understand video embedding + retrieval.

This script:
1. Takes a video file → uploads to Twelve Labs → creates clip-level embeddings
2. Stores those embeddings in a Pinecone index (separate namespace: "video_clips")
3. Lets you query with natural language → finds matching video clips by timestamp

Usage:
    # Step 1: Index a video
    python video_rag.py index --video "path/to/video.mp4"

    # Step 2: Query the indexed video
    python video_rag.py query --q "When was someone near the checkout counter?"

    # Step 3: Interactive query mode
    python video_rag.py query --interactive

Requirements:
    pip install twelvelabs pinecone-client python-dotenv
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────
TWELVE_LABS_API_KEY = os.getenv("TWELVE_LABS_API_KEY")
PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY")
PINECONE_VIDEO_INDEX = os.getenv("PINECONE_VIDEO_INDEX", "visionlog-video")

TWELVE_LABS_MODEL   = "marengo3.0"
SEGMENT_DURATION    = 6           # Fixed segment duration in seconds (matches our VLM_INTERVAL)
EMBEDDING_OPTIONS   = ["visual"]  # We only need visual for CCTV (no audio/transcription)
TOP_K               = 5           # Number of results to return per query


# ═══════════════════════════════════════════════════════════════════════════
# INDEXING: Upload video → Create embeddings → Store in Pinecone
# ═══════════════════════════════════════════════════════════════════════════

def index_video(video_path: str):
    """
    End-to-end pipeline:
    1. Upload video to Twelve Labs as an asset
    2. Create clip-level embeddings using Marengo 3.0
    3. Store each clip embedding in Pinecone with timestamp metadata
    """
    from twelvelabs import (
        TwelveLabs,
        VideoInputRequest,
        MediaSource,
        VideoSegmentation_Fixed,
        VideoSegmentationFixedFixed,
    )
    from pinecone import Pinecone, ServerlessSpec

    video_path = Path(video_path)
    if not video_path.exists():
        print(f"[ERROR] Video file not found: {video_path}")
        sys.exit(1)

    video_name = video_path.stem
    print(f"\n{'='*60}")
    print(f"  MULTIMODAL VIDEO RAG — Indexing Pipeline")
    print(f"  Video: {video_path.name}")
    print(f"  Model: {TWELVE_LABS_MODEL}")
    print(f"  Segment Duration: {SEGMENT_DURATION}s")
    print(f"{'='*60}\n")

    # ── Step 1: Initialize Twelve Labs client ────────────────────────────
    print("[1/4] Initializing Twelve Labs client...")
    client = TwelveLabs(api_key=TWELVE_LABS_API_KEY)
    print("       ✅ Client ready.")

    # ── Step 2: Upload video as asset ────────────────────────────────────
    print(f"[2/4] Uploading video: {video_path.name}...")
    asset = client.assets.create(
        method="direct",
        file=open(str(video_path), "rb"),
    )
    print(f"       ✅ Asset created: id={asset.id}")

    # Wait for Twelve Labs to finish processing uploaded asset
    print("       ⏳ Waiting for asset processing to complete...")
    while True:
        current_asset = client.assets.retrieve(asset_id=asset.id)
        if current_asset.status == "ready":
            print("       ✅ Asset is ready for embedding!")
            break
        elif current_asset.status == "failed":
            print("       ❌ Asset processing failed on Twelve Labs!")
            sys.exit(1)
        time.sleep(3)

    # ── Step 3: Create embeddings ────────────────────────────────────────
    print(f"[3/4] Creating {SEGMENT_DURATION}s clip embeddings...")
    print(f"       (This may take a few minutes depending on video length)")

    # Determine video duration to choose sync vs async
    # For videos under 10 minutes, use synchronous API
    task = client.embed.v_2.tasks.create(
        input_type="video",
        model_name=TWELVE_LABS_MODEL,
        video=VideoInputRequest(
            media_source=MediaSource(asset_id=asset.id),
            embedding_option=EMBEDDING_OPTIONS,
            embedding_scope=["clip"],
            segmentation=VideoSegmentation_Fixed(
                fixed=VideoSegmentationFixedFixed(
                    duration_sec=SEGMENT_DURATION
                )
            ),
        ),
    )
    print(f"       Task ID: {task.id}")

    # Poll until complete
    while True:
        task = client.embed.v_2.tasks.retrieve(task_id=task.id)
        if task.status == "ready":
            print(f"       ✅ Embedding complete! {len(task.data)} clips generated.")
            break
        elif task.status == "failed":
            print(f"       ❌ Embedding failed!")
            sys.exit(1)
        else:
            print(f"       ⏳ Processing... (status: {task.status})")
            time.sleep(5)

    # ── Step 4: Store in Pinecone ────────────────────────────────────────
    print(f"[4/4] Storing embeddings in Pinecone index '{PINECONE_VIDEO_INDEX}'...")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create index if it doesn't exist
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    embedding_dim = len(task.data[0].embedding)

    if PINECONE_VIDEO_INDEX not in existing_indexes:
        print(f"       Creating new Pinecone index (dim={embedding_dim})...")
        pc.create_index(
            name=PINECONE_VIDEO_INDEX,
            dimension=embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait for index to be ready
        time.sleep(5)

    index = pc.Index(PINECONE_VIDEO_INDEX)

    # Prepare vectors for upsert
    vectors = []
    for i, emb_data in enumerate(task.data):
        vec_id = f"{video_name}_clip_{i:04d}"
        vectors.append({
            "id": vec_id,
            "values": emb_data.embedding,
            "metadata": {
                "video_file":       video_path.name,
                "video_name":       video_name,
                "asset_id":         asset.id,
                "start_sec":        emb_data.start_sec,
                "end_sec":          emb_data.end_sec,
                "embedding_option": emb_data.embedding_option,
                "clip_index":       i,
            },
        })

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)
        print(f"       Upserted batch {i // batch_size + 1}/{(len(vectors) + batch_size - 1) // batch_size}")

    print(f"\n{'='*60}")
    print(f"  ✅ INDEXING COMPLETE")
    print(f"     Video: {video_path.name}")
    print(f"     Clips: {len(vectors)}")
    print(f"     Dimensions: {embedding_dim}")
    print(f"     Segment Duration: {SEGMENT_DURATION}s")
    print(f"     Index: {PINECONE_VIDEO_INDEX}")
    print(f"{'='*60}\n")

    # Save metadata locally for reference
    meta_file = video_path.with_suffix(".video_index.json")
    with open(meta_file, "w") as f:
        json.dump({
            "video_file": video_path.name,
            "asset_id": asset.id,
            "total_clips": len(vectors),
            "segment_duration": SEGMENT_DURATION,
            "embedding_dim": embedding_dim,
            "pinecone_index": PINECONE_VIDEO_INDEX,
            "clips": [
                {
                    "id": v["id"],
                    "start_sec": v["metadata"]["start_sec"],
                    "end_sec": v["metadata"]["end_sec"],
                }
                for v in vectors
            ],
        }, f, indent=2)
    print(f"  📄 Local metadata saved to: {meta_file.name}")


# ═══════════════════════════════════════════════════════════════════════════
# QUERYING: Embed text query → Search Pinecone → Return matching clips
# ═══════════════════════════════════════════════════════════════════════════

def query_video(query_text: str, top_k: int = TOP_K):
    """
    Query the video index with natural language.
    Returns the top matching video clips with timestamps.
    """
    from twelvelabs import TwelveLabs, TextInputRequest
    from pinecone import Pinecone

    print(f"\n🔍 Query: \"{query_text}\"")
    print(f"   Searching {PINECONE_VIDEO_INDEX} (top {top_k} results)...\n")

    # ── Step 1: Embed the text query using Twelve Labs ───────────────────
    client = TwelveLabs(api_key=TWELVE_LABS_API_KEY)
    response = client.embed.v_2.create(
        input_type="text",
        model_name=TWELVE_LABS_MODEL,
        text=TextInputRequest(input_text=query_text),
    )

    query_vector = response.data[0].embedding
    print(f"   ✅ Query embedded ({len(query_vector)} dimensions)")

    # ── Step 2: Search Pinecone ──────────────────────────────────────────
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_VIDEO_INDEX)

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )

    # ── Step 3: Display results ──────────────────────────────────────────
    if not results.matches:
        print("   ❌ No matching clips found.\n")
        return []

    print(f"{'='*60}")
    print(f"  VIDEO SEARCH RESULTS")
    print(f"{'='*60}")

    found_clips = []
    for rank, match in enumerate(results.matches, 1):
        meta = match.metadata
        score = match.score
        start = meta.get("start_sec", 0)
        end   = meta.get("end_sec", 0)

        # Format timestamps nicely
        start_fmt = f"{int(start // 60)}:{int(start % 60):02d}"
        end_fmt   = f"{int(end // 60)}:{int(end % 60):02d}"

        print(f"\n  #{rank} | Similarity: {score:.3f}")
        print(f"  ├─ Video: {meta.get('video_file', '?')}")
        print(f"  ├─ Time:  {start_fmt} — {end_fmt} ({end - start:.0f}s clip)")
        print(f"  └─ Clip:  {match.id}")

        found_clips.append({
            "rank": rank,
            "score": score,
            "video_file": meta.get("video_file"),
            "start_sec": start,
            "end_sec": end,
            "clip_id": match.id,
        })

    print(f"\n{'='*60}\n")
    return found_clips


def interactive_query():
    """Interactive query mode — keep asking questions about the video."""
    print(f"\n{'='*60}")
    print(f"  🎬 MULTIMODAL VIDEO RAG — Interactive Query")
    print(f"  Ask questions about your indexed video(s).")
    print(f"  Type 'quit' to exit.")
    print(f"")
    print(f"  Example queries:")
    print(f"    • Who was near the checkout counter?")
    print(f"    • Was anyone running in the store?")
    print(f"    • Show me people near the shelves")
    print(f"    • When was someone crouching?")
    print(f"{'='*60}\n")

    while True:
        try:
            q = input("📹 Ask > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not q or q.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        query_video(q)


# ═══════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Video RAG — Index and query video with natural language",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ── Index command ────────────────────────────────────────────────────
    index_parser = subparsers.add_parser("index", help="Index a video file")
    index_parser.add_argument("--video", required=True, help="Path to video file")

    # ── Query command ────────────────────────────────────────────────────
    query_parser = subparsers.add_parser("query", help="Query indexed videos")
    query_parser.add_argument("--q", default=None, help="Query text")
    query_parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    query_parser.add_argument("--top-k", type=int, default=TOP_K, help="Number of results")

    args = parser.parse_args()

    # Validate keys
    if not TWELVE_LABS_API_KEY:
        print("[ERROR] TWELVE_LABS_API_KEY not set! Add it to your .env file.")
        sys.exit(1)
    if not PINECONE_API_KEY:
        print("[ERROR] PINECONE_API_KEY not set! Add it to your .env file.")
        sys.exit(1)

    if args.command == "index":
        index_video(args.video)
    elif args.command == "query":
        if args.interactive:
            interactive_query()
        elif args.q:
            query_video(args.q, top_k=args.top_k)
        else:
            print("[ERROR] Specify --q 'your question' or --interactive")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
