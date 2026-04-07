"""
RAG (Retrieval-Augmented Generation) engine for the DoJ Legal Assistant.
Replaces the old TF-IDF keyword matcher with a full pipeline:
  Intent Classification → Query Rewrite → Vector Search → LLM Generation
"""

import os
import re
import time
from pathlib import Path

import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

from chatbot.memory import memory
from chatbot.prompts import (
    SYSTEM_PROMPT, SYSTEM_PROMPT_HINDI,
    RAG_PROMPT_TEMPLATE, RAG_PROMPT_COMPACT,
    RAG_PROMPT_TEMPLATE_HINDI, RAG_PROMPT_COMPACT_HINDI,
    SYSTEM_PROMPT_HINGLISH, RAG_PROMPT_TEMPLATE_HINGLISH, RAG_PROMPT_COMPACT_HINGLISH,
    CAPABILITY_RESPONSE, GREETING_RESPONSE, GREETING_RESPONSE_HINDI,
)

# Load env
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"

genai.configure(api_key=GEMINI_API_KEY)

# Initialize LLM (Gemini or Local)
llm = genai.GenerativeModel(LLM_MODEL)

# Try to load local model if configured
_local_llm_available = False
if USE_LOCAL_MODEL:
    try:
        from chatbot.local_llm import generate_local, is_local_model_available
        _local_llm_available = is_local_model_available()
        if _local_llm_available:
            print("🤖 Local model mode ENABLED")
        else:
            print("⚠️ USE_LOCAL_MODEL=true but model file not found. Falling back to Gemini.")
    except ImportError:
        print("⚠️ llama-cpp-python not installed. Falling back to Gemini.")

# Initialize ChromaDB
_chroma_client = None
_collection = None

# ── Retry helper ─────────────────────────────────────────────────────────────

def _call_llm_with_retry(prompt: str, max_retries: int = 3) -> str:
    """Call LLM (local or Gemini) with automatic retry on rate limits."""
    # Try local model first if available
    if USE_LOCAL_MODEL and _local_llm_available:
        try:
            return generate_local(prompt)
        except Exception as e:
            print(f"Local model error: {type(e).__name__}: {e}")
            print("Falling back to Gemini API...")

    # Gemini with retry
    for attempt in range(max_retries):
        try:
            response = llm.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "quota" in error_str or "resource" in error_str:
                wait_time = (attempt + 1) * 10  # 10s, 20s, 30s
                print(f"Rate limited (attempt {attempt+1}/{max_retries}), waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"LLM error: {e}")
                raise
    raise Exception("Max retries exceeded for LLM call")


def _get_collection():
    global _chroma_client, _collection
    if _collection is None:
        from chromadb.utils import embedding_functions
        local_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        
        persist_path = Path(__file__).resolve().parent.parent / CHROMA_PERSIST_DIR
        _chroma_client = chromadb.PersistentClient(path=str(persist_path))
        _collection = _chroma_client.get_collection("indian_laws", embedding_function=local_ef)
    return _collection



def _detect_language(text: str) -> str:
    """Detect if text is primarily Hindi or English."""
    hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
    total_alpha = len(re.findall(r'[a-zA-Z\u0900-\u097F]', text))
    if total_alpha == 0:
        return "en"
    return "hi" if (hindi_chars / total_alpha) > 0.3 else "en"


# ── Local intent classification (no LLM call needed) ────────────────────────

GREETING_KEYWORDS = [
    "hello", "hi", "hey", "namaste", "good morning", "good evening",
    "good afternoon", "greetings", "namaskar", "howdy",
    "नमस्ते", "नमस्कार", "हेलो",
]

CAPABILITY_KEYWORDS = [
    "what can you do", "what do you know", "do you know all",
    "what are your capabilities", "help me", "kya kar sakte",
    "what laws do you know", "tell me about yourself",
    "तुम क्या कर सकते हो", "आप क्या जानते हो",
]

SERVICE_KEYWORDS = [
    "ecourt", "e-court", "case status", "check case", "fine payment",
    "pay fine", "court fee", "live streaming", "njdg", "tele law",
    "telelaw", "gram nyayalaya", "csc", "common service",
]


def _classify_intent_local(message: str, session_id: str) -> str:
    """Classify intent using local keyword matching (saves API quota)."""
    msg = message.lower().strip()

    # Greetings (short messages that are just a greeting)
    if len(msg.split()) <= 4:
        for kw in GREETING_KEYWORDS:
            if kw in msg:
                return "greeting"

    # Capability questions
    for kw in CAPABILITY_KEYWORDS:
        if kw in msg:
            return "capability"

    # Court services
    for kw in SERVICE_KEYWORDS:
        if kw in msg:
            return "court_service"

    # Follow-up detection (short message with context)
    history = memory.get_history(session_id)
    if history and len(msg.split()) <= 6:
        followup_cues = ["more", "also", "what about", "and", "tell me",
                         "punishment", "penalty", "bail", "aur", "batao"]
        for cue in followup_cues:
            if cue in msg:
                return "followup"

    return "legal_query"


# ── Local query rewriter (no LLM call needed) ───────────────────────────────

ABBREVIATIONS = {
    "ipc": "Indian Penal Code",
    "crpc": "Code of Criminal Procedure",
    "cpc": "Code of Civil Procedure",
    "pocso": "Protection of Children from Sexual Offences Act",
    "rti": "Right to Information Act",
    "sc": "Supreme Court",
    "hc": "High Court",
    "fir": "First Information Report",
    "ftc": "Fast Track Court",
    "doj": "Department of Justice",
    "njdg": "National Judicial Data Grid",
}


def _rewrite_query_local(message: str, session_id: str) -> str:
    """Rewrite query using simple rules (saves API quota)."""
    query = message

    # Expand abbreviations
    for abbr, full in ABBREVIATIONS.items():
        pattern = re.compile(r'\b' + re.escape(abbr) + r'\b', re.IGNORECASE)
        query = pattern.sub(full, query)

    # If message mentions "section X" without act name, add IPC
    section_match = re.search(r'\bsection\s+(\d+[A-Z]?)\b', query, re.IGNORECASE)
    if section_match and "penal code" not in query.lower() and "procedure" not in query.lower() and "constitution" not in query.lower():
        query = query + " of the Indian Penal Code"

    # If message mentions "article X" without context, add Constitution
    article_match = re.search(r'\barticle\s+(\d+[A-Z]?)\b', query, re.IGNORECASE)
    if article_match and "constitution" not in query.lower():
        query = query + " of the Constitution of India"

    # For follow-ups, prepend last topic from conversation
    history = memory.get_history(session_id)
    if history and len(message.split()) <= 6:
        # Get the last user message for context
        for h in reversed(history):
            if h["role"] == "user" and h["content"] != message:
                query = f"{query} (context: {h['content'][:100]})"
                break

    return query


def _search_knowledge_base(query: str, top_k: int = 3) -> list[dict]:
    """Search ChromaDB for relevant legal chunks."""
    try:
        collection = _get_collection()

        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        sources = []
        if results and results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                sources.append({
                    "text": doc,
                    "act": meta.get("act", "Unknown"),
                    "section": meta.get("section", "General"),
                    "source": meta.get("source", "Unknown"),
                    "chapter": meta.get("chapter", ""),
                    "relevance": round(1 - dist, 3)
                })

        return sources
    except Exception as e:
        print(f"Knowledge base search error: {e}")
        return []


def _generate_response(message: str, sources: list[dict], session_id: str,
                       language: str = "en") -> tuple[str, list[str]]:
    """Generate LLM response using retrieved context."""
    # Build context from sources
    context_parts = []
    for i, src in enumerate(sources, 1):
        context_parts.append(
            f"[Source {i}: {src['source']}]\n{src['text']}"
        )
    context_str = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant legal documents found."

    # Build conversation history
    history = memory.get_history(session_id)
    history_str = ""
    if history:
        last_messages = history[-8:]
        history_str = "\n".join(
            f"{m['role'].capitalize()}: {m['content'][:200]}" for m in last_messages
        )

    # Build prompt
    if language == "hi":
        system = SYSTEM_PROMPT_HINDI
        tpl = RAG_PROMPT_TEMPLATE_HINDI
        compact = RAG_PROMPT_COMPACT_HINDI
    elif language == "hinglish":
        system = SYSTEM_PROMPT_HINGLISH
        tpl = RAG_PROMPT_TEMPLATE_HINGLISH
        compact = RAG_PROMPT_COMPACT_HINGLISH
    else:
        system = SYSTEM_PROMPT
        tpl = RAG_PROMPT_TEMPLATE
        compact = RAG_PROMPT_COMPACT

    rag_prompt = tpl.format(
        context=context_str,
        history=history_str or "This is the start of the conversation.",
        question=message
    )

    full_prompt = f"{system}\n\n{rag_prompt}\n\n{compact}"

    try:
        response_text = _call_llm_with_retry(full_prompt)

        # Extract follow-up suggestions
        follow_ups = []
        if "FOLLOW_UP_SUGGESTIONS:" in response_text:
            parts = response_text.split("FOLLOW_UP_SUGGESTIONS:")
            response_text = parts[0].strip()
            if len(parts) > 1:
                suggestions_str = parts[1].strip()
                follow_ups = [s.strip() for s in suggestions_str.split("|") if s.strip()]

        return response_text, follow_ups

    except Exception as e:
        print(f"LLM generation error: {e}")
        # If rate limited, provide a helpful message with the source docs
        if sources:
            fallback = "I'm currently experiencing high demand. Here's what I found in the legal documents:\n\n"
            for src in sources[:2]:
                fallback += f"**{src['source']}**:\n{src['text'][:300]}\n\n"
            fallback += "*Please try again in a minute for a more detailed answer.*"
            return fallback, []
        return (
            "I'm experiencing high demand right now. Please try again in about 30 seconds. "
            "The free Gemini API has rate limits — your question will work shortly! 🙏",
            []
        )


def get_rag_response(message: str, session_id: str, language: str = "auto") -> dict:
    """
    Main entry point for the RAG pipeline.
    Returns: {response, sources, follow_ups, intent, language}
    """
    # Auto-detect language if needed
    if language == "auto":
        language = _detect_language(message)

    # Store user message in memory
    memory.add_user_message(session_id, message)

    # Step 1: Classify intent locally (no API call)
    intent = _classify_intent_local(message, session_id)

    # Step 2: Handle non-RAG intents
    if intent == "greeting":
        resp = GREETING_RESPONSE_HINDI if language == "hi" else GREETING_RESPONSE
        memory.add_assistant_message(session_id, resp)
        return {
            "response": resp,
            "sources": [],
            "follow_ups": [
                "What is Section 302 of IPC?",
                "What are my rights if arrested?",
                "How to check case status on eCourts?"
            ],
            "intent": intent,
            "language": language
        }

    if intent == "capability":
        memory.add_assistant_message(session_id, CAPABILITY_RESPONSE)
        return {
            "response": CAPABILITY_RESPONSE,
            "sources": [],
            "follow_ups": [
                "Tell me about Section 498A",
                "What are Fundamental Rights?",
                "How does bail work in India?"
            ],
            "intent": intent,
            "language": language
        }

    # Step 3: Rewrite query locally (no API call)
    search_query = _rewrite_query_local(message, session_id)

    # Step 4: Search knowledge base (1 embedding API call)
    sources = _search_knowledge_base(search_query)

    # Step 5: Generate response with LLM (1 LLM API call with retry)
    response_text, follow_ups = _generate_response(
        message, sources, session_id, language
    )

    # Store bot response in memory
    memory.add_assistant_message(session_id, response_text)

    # Format sources for frontend
    formatted_sources = [
        {
            "act": s["act"],
            "section": s["section"],
            "source": s["source"],
            "excerpt": s["text"][:200] + "..." if len(s["text"]) > 200 else s["text"],
            "relevance": s.get("relevance", 0)
        }
        for s in sources[:3]
    ]

    # Default follow-ups if none generated
    if not follow_ups:
        follow_ups = [
            "Tell me more about this",
            "What are the penalties?",
            "How can I file a complaint?"
        ]

    return {
        "response": response_text,
        "sources": formatted_sources,
        "follow_ups": follow_ups[:3],
        "intent": intent,
        "language": language
    }
