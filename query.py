"""
query.py — REST RAG Query Interface
=========================================
Ask natural language questions about your CCTV events stored in Pinecone.

Usage:
    python query.py                              # interactive chat mode (text only)
    python query.py --mode both "Who was suspicious today?"
    python query.py --mode video "Show me someone crouching"
    python query.py --mode text --zone SHELF_AREA "Who was here?"
    python query.py --severity HIGH "Summarise all high risk activity"

Requirements:
    pip install google-generativeai
"""
import argparse
import sys
from visionlog import config

# ── Constants ─────────────────────────────────────────────────────────────────
TOP_K_DEFAULT = 15     # how many events to retrieve per query
GEMINI_MODEL  = "gemini-2.5-flash"  # free: 15 RPM, 1,000 req/day (most generous free model)
SYSTEM_PROMPT = """You are REST — an assistant that answers questions about seating behaviour from timestamped events recorded in a seating study. Answer like a researcher briefing a design engineer.
Describe people by clothing and behaviour, not tracking numbers. Reference time and seat to anchor every observation. Report seated durations, sit-down and stand-up moments, abrupt sitting, posture changes, head-turn restlessness and neighbouring-seat use when relevant.
Say plainly if nothing relevant was found. Lead with the most important finding; keep under 200 words unless the question is complex. Do not mention internal IDs or technical terms (score, confidence, similarity, YOLO, vector). Do not repeat an event in different words.
"""


# ── Lazy-loaded singletons ─────────────────────────────────────────────────────
_embedder   = None
_pc_index   = None
_gemini     = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        print("[Query] Loading embedding model...", flush=True)
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedder


def _get_pinecone(index_name: str):
    from pinecone import Pinecone
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    return pc.Index(index_name)


def _get_gemini(api_key: str):
    global _gemini
    if _gemini is None:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        _gemini = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=SYSTEM_PROMPT,
        )
        print(f"[Query] Gemini {GEMINI_MODEL} ready", flush=True)
    return _gemini


# ── Core retrieval + generation ────────────────────────────────────────────────

def retrieve_events(query: str,
                    top_k: int   = TOP_K_DEFAULT,
                    zone: str    = None,
                    severity: str = None,
                    camera: str  = None) -> list[dict]:
    """Embed query and retrieve top-K similar text events from Pinecone."""
    embedder = _get_embedder()
    vector   = embedder.encode(query).tolist()

    # Build optional metadata filter
    filt = {}
    if zone:
        filt["zone"]      = {"$eq": zone.upper()}
    if severity:
        filt["severity"]  = {"$eq": severity.upper()}
    if camera:
        filt["camera_id"] = {"$eq": camera}

    index   = _get_pinecone(config.PINECONE_INDEX)
    results = index.query(
        vector   = vector,
        top_k    = top_k,
        include_metadata = True,
        filter   = filt if filt else None,
    )
    return results.get("matches", [])


def retrieve_video_clips(query: str, top_k: int = 5) -> list[dict]:
    """Embed query via Twelve Labs Marengo 3.0 and retrieve video clips from Pinecone."""
    if not config.TWELVE_LABS_API_KEY:
        return []
        
    from twelvelabs import TwelveLabs, TextInputRequest
    client = TwelveLabs(api_key=config.TWELVE_LABS_API_KEY)
    
    # 1. Embed query
    response = client.embed.v_2.create(
        input_type="text",
        model_name="marengo3.0",
        text=TextInputRequest(input_text=query),
    )
    vector = response.data[0].embedding
    
    # 2. Search Video Index
    index = _get_pinecone(config.PINECONE_VIDEO_INDEX)
    results = index.query(vector=vector, top_k=top_k, include_metadata=True)
    
    return results.get("matches", [])


def format_events_for_llm(matches: list[dict]) -> str:
    """Turn Pinecone matches into structured context the LLM can reason over."""
    if not matches:
        return "No relevant events found."
    lines = []
    for i, m in enumerate(matches, 1):
        meta = m.get("metadata", {})
        # Pull key fields for the LLM to use
        time_str  = meta.get("time",          "unknown time")
        zone      = meta.get("zone",          "unknown zone").replace("_", " ").title()
        severity  = meta.get("severity",      "")
        clothing  = meta.get("clothing_desc", "")
        gender    = meta.get("gender",        "")
        age       = meta.get("age",           "")
        pose      = meta.get("pose",          "")
        text      = meta.get("text",          "")

        # Build a human-readable header
        person_desc = " ".join(filter(None, [gender, f"~{age}yo" if age else "", clothing])).strip()
        if not person_desc:
            person_desc = "unknown appearance"
        header = f"[{i}] {time_str} | {zone}"
        if severity:
            header += f" | ⚠️ {severity}"
        if person_desc:
            header += f" | {person_desc}"

        lines.append(f"{header}\n{text}")
    return "\n\n".join(lines)


def ask(query: str,
        api_key: str,
        mode: str    = "text",
        top_k: int   = TOP_K_DEFAULT,
        zone: str    = None,
        severity: str = None,
        camera: str  = None) -> str:
    """Full RAG pipeline: retrieve → format → generate."""
    
    # Text Retrieval
    text_matches = []
    if mode in ("text", "both"):
        text_matches = retrieve_events(query, top_k, zone, severity, camera)
        
    # Video Retrieval
    video_matches = []
    if mode in ("video", "both"):
        video_matches = retrieve_video_clips(query, top_k=5)
        
    output_parts = []
    
    # Formatting Video Results
    if mode in ("video", "both"):
        output_parts.append("================= 📹 VIDEO EVIDENCE =================")
        if not video_matches:
            output_parts.append("No relevant video clips found.")
        else:
            for i, m in enumerate(video_matches, 1):
                meta = m.metadata
                
                # Support both Phase 1 (video_rag.py) and Phase 2 (VideoArchiver) metadata keys
                cam = meta.get("camera_name") or meta.get("video_file") or "Unknown Camera"
                start = meta.get("start_ts") or meta.get("start_sec") or 0.0
                
                # Duration logic
                if "duration_s" in meta:
                    dur = meta["duration_s"]
                elif "end_sec" in meta:
                    dur = meta["end_sec"] - meta.get("start_sec", 0)
                else:
                    dur = 6.0
                    
                score = m.score
                
                # Format start based on size. Huges numbers = Unix Epoch wall-clock time. Small numbers = relative seconds.
                if start > 1000000000:
                    import datetime
                    dt = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
                else:
                    dt = f"{int(start // 60)}:{int(start % 60):02d}"
                    
                output_parts.append(f"[{i}] Time: {dt} | Camera: {cam} | Clip: {dur:.0f}s | Match: {score:.2f}")

    # Formatting Text Results via LLM
    if mode in ("text", "both"):
        output_parts.append("\n================= 📝 INTELLIGENCE REPORT ============")
        context = format_events_for_llm(text_matches)
        prompt = (
            f"USER QUESTION: {query}\n\n"
            f"RETRIEVED EVENTS ({len(text_matches)} found):\n"
            f"{context}\n\n"
            f"Please answer the question based on these events."
        )
        model = _get_gemini(api_key)
        response = model.generate_content(prompt)
        output_parts.append(response.text)

    return "\n".join(output_parts)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="REST RAG Query — ask questions about your CCTV events"
    )
    parser.add_argument("query",   nargs="?",       default=None,
                        help="Question to ask (omit for interactive mode)")
    parser.add_argument("--top-k", type=int,        default=TOP_K_DEFAULT,
                        dest="top_k",
                        help=f"Events to retrieve (default: {TOP_K_DEFAULT})")
    parser.add_argument("--zone",     default=None, help="Filter by zone (e.g. SHELF_AREA)")
    parser.add_argument("--severity", default=None, help="Filter by severity (LOW/MEDIUM/HIGH)")
    parser.add_argument("--camera",   default=None, help="Filter by camera ID (e.g. CAM_01)")
    parser.add_argument("--mode",     default="text", choices=["text", "video", "both"], help="Search mode")
    parser.add_argument("--api-key",  default=None, dest="api_key",
                        help="Gemini API key (or set GEMINI_API_KEY in config.py)")
    args = parser.parse_args()

    # Resolve API key
    api_key = args.api_key or getattr(config, "GEMINI_API_KEY", None)
    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
        print(
            "\n[Query] ERROR: No Gemini API key found.\n"
            "  Get a free key at: https://aistudio.google.com/app/apikey\n"
            "  Then either:\n"
            "    a) Add  GEMINI_API_KEY = 'your-key'  to visionlog/config.py\n"
            "    b) Pass  --api-key your-key  to this script\n"
        )
        sys.exit(1)

    if args.query:
        # Single query mode
        print(f"\n🔍 Querying: {args.query}")
        if args.zone:     print(f"   Filter: zone={args.zone}")
        if args.severity: print(f"   Filter: severity={args.severity}")
        print()
        answer = ask(args.query, api_key,
                     mode=args.mode,
                     top_k=args.top_k,
                     zone=args.zone,
                     severity=args.severity,
                     camera=args.camera)
        print("=" * 60)
        print(answer)
        print("=" * 60)

    else:
        # Interactive chat mode
        print("\n" + "=" * 60)
        print("  REST RAG — Interactive Query Mode")
        print("  Type your question, or 'quit' to exit.")
        print("  Examples:")
        print("    Show me someone who sat down abruptly")
        print("    How long was Seat A occupied?")
        print("    Did anyone move to the neighbouring seat?")
        print("=" * 60 + "\n")

        while True:
            try:
                q = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if not q:
                continue
            if q.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            print("\nREST: thinking...\n")
            try:
                answer = ask(q, api_key, mode=args.mode, top_k=args.top_k,
                             zone=args.zone,
                             severity=args.severity)
                print(f"REST:\n{answer}\n")
                print("-" * 60)
            except Exception as e:
                print(f"[Error] {e}\n")


if __name__ == "__main__":
    main()
