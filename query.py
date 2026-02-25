"""
query.py — VisionLog RAG Query Interface
=========================================
Ask natural language questions about your CCTV events stored in Pinecone.

Usage:
    python query.py                              # interactive chat mode
    python query.py "Who was suspicious today?"  # single query
    python query.py --top-k 20 "Show me all HIGH risk events"
    python query.py --zone SHELF_AREA "Who was crouching at shelves?"
    python query.py --severity HIGH "Summarise all high risk activity"

Requirements:
    pip install google-generativeai
"""
import argparse
import sys
from visionlog import config

# ── Constants ─────────────────────────────────────────────────────────────────
TOP_K_DEFAULT = 15     # how many events to retrieve per query
GEMINI_MODEL  = "gemini-2.5-flash-lite"  # free: 15 RPM, 1,000 req/day (most generous free model)
SYSTEM_PROMPT = """You are VisionLog AI — an intelligent CCTV security analyst assistant.
You are given a set of surveillance events retrieved from a vector database.
Each event is a timestamped observation from a retail store security camera.

Your job is to:
1. Answer the user's question accurately based ONLY on the retrieved events.
2. Highlight the most suspicious or noteworthy patterns.
3. Reference specific person IDs, zones, timestamps, and risk scores when relevant.
4. If no relevant events are found, say so clearly.
5. Be concise but thorough. Think like a security analyst, not a chatbot.

FORMAT:
- Use bullet points for multiple findings.
- Lead with the most important finding.
- Always cite: Person #X, zone, timestamp, severity if available.
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


def _get_pinecone():
    global _pc_index
    if _pc_index is None:
        from pinecone import Pinecone
        pc      = Pinecone(api_key=config.PINECONE_API_KEY)
        _pc_index = pc.Index(config.PINECONE_INDEX)
        stats   = _pc_index.describe_index_stats()
        print(f"[Query] Connected to Pinecone '{config.PINECONE_INDEX}' "
              f"({stats.total_vector_count} vectors)", flush=True)
    return _pc_index


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
    """Embed query and retrieve top-K similar events from Pinecone."""
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

    index   = _get_pinecone()
    results = index.query(
        vector   = vector,
        top_k    = top_k,
        include_metadata = True,
        filter   = filt if filt else None,
    )
    return results.get("matches", [])


def format_events_for_llm(matches: list[dict]) -> str:
    """Turn Pinecone matches into readable context for the LLM."""
    if not matches:
        return "No relevant events found."
    lines = []
    for i, m in enumerate(matches, 1):
        meta = m.get("metadata", {})
        text = meta.get("text", "(no text)")
        score = round(m.get("score", 0), 3)
        lines.append(f"[Event {i} | similarity={score}]\n{text}")
    return "\n\n".join(lines)


def ask(query: str,
        api_key: str,
        top_k: int   = TOP_K_DEFAULT,
        zone: str    = None,
        severity: str = None,
        camera: str  = None) -> str:
    """Full RAG pipeline: retrieve → format → generate."""
    matches  = retrieve_events(query, top_k, zone, severity, camera)
    context  = format_events_for_llm(matches)

    prompt = (
        f"USER QUESTION: {query}\n\n"
        f"RETRIEVED EVENTS ({len(matches)} found):\n"
        f"{context}\n\n"
        f"Please answer the question based on these events."
    )

    model    = _get_gemini(api_key)
    response = model.generate_content(prompt)
    return response.text


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VisionLog RAG Query — ask questions about your CCTV events"
    )
    parser.add_argument("query",   nargs="?",       default=None,
                        help="Question to ask (omit for interactive mode)")
    parser.add_argument("--top-k", type=int,        default=TOP_K_DEFAULT,
                        dest="top_k",
                        help=f"Events to retrieve (default: {TOP_K_DEFAULT})")
    parser.add_argument("--zone",     default=None, help="Filter by zone (e.g. SHELF_AREA)")
    parser.add_argument("--severity", default=None, help="Filter by severity (LOW/MEDIUM/HIGH)")
    parser.add_argument("--camera",   default=None, help="Filter by camera ID (e.g. CAM_01)")
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
        print("  VisionLog RAG — Interactive Query Mode")
        print("  Type your question, or 'quit' to exit.")
        print("  Examples:")
        print("    Who was the most suspicious person today?")
        print("    Show me all HIGH risk events at the shelves")
        print("    Summarise what happened in the ENTRANCE zone")
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

            print("\nVisionLog AI: thinking...\n")
            try:
                answer = ask(q, api_key, top_k=args.top_k,
                             zone=args.zone,
                             severity=args.severity)
                print(f"VisionLog AI:\n{answer}\n")
                print("-" * 60)
            except Exception as e:
                print(f"[Error] {e}\n")


if __name__ == "__main__":
    main()
