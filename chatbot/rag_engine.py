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
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")
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
from chatbot.nlp_engine import classify_intent as semantic_classify, get_response as get_static_response, match_qa_dataset, summarize_text

# Load env
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"
USE_LLM = os.getenv("USE_LLM", "true").lower() == "true"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    _genai_client = genai.GenerativeModel(LLM_MODEL)
else:
    _genai_client = None

# Try to load local model if configured
_local_llm_available = False
if USE_LOCAL_MODEL:
    try:
        from chatbot.local_llm import generate_local, is_local_model_available
        _local_llm_available = is_local_model_available()
        if _local_llm_available:
            print("Local model mode ENABLED")
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
    if not _genai_client:
        return "I am currently running in **Offline Mode** (No LLM API Key configured). However, I found the following relevant legal text based on your query:"
        
    for attempt in range(max_retries):
        try:
            response = _genai_client.generate_content(prompt)
            # Guard against blocked / empty responses
            if not response.parts:
                finish = getattr(response.candidates[0], 'finish_reason', 'UNKNOWN') if response.candidates else 'UNKNOWN'
                print(f"Gemini returned no parts. finish_reason={finish}")
                return "I'm unable to answer this question due to content restrictions. Please rephrase or consult a legal expert."
            return response.text.strip()
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "quota" in error_str or "resource" in error_str:
                wait_time = (attempt + 1) * 5  # 5s, 10s, 15s
                print(f"Rate limited (attempt {attempt+1}/{max_retries}), waiting {wait_time}s...")
                time.sleep(wait_time)
            elif "invalid argument" in error_str or "not found" in error_str:
                # Model name issue — surface immediately, don't retry
                print(f"\u274c Model error (check LLM_MODEL in .env): {e}")
                raise
            else:
                print(f"LLM error: {e}")
                raise
    raise Exception("Max retries exceeded for LLM call")


def _get_collection():
    global _chroma_client, _collection
    if _collection is None:
        from chromadb.utils import embedding_functions
        # Fallback to local sentence transformers to prevent google SDK deprecation issues
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            
        persist_path = Path(__file__).resolve().parent.parent / CHROMA_PERSIST_DIR
        _chroma_client = chromadb.PersistentClient(path=str(persist_path))
        _collection = _chroma_client.get_collection("indian_laws", embedding_function=ef)
    return _collection



# Common Hinglish transliteration markers (Latin-script Hinglish)
_HINGLISH_MARKERS = re.compile(
    r'\b(mujhe|mera|meri|kya|hain|hai|nahi|nahin|kaise|batao|bata|chahiye|'
    r'aur|lekin|toh|kyunki|woh|yeh|iska|uska|karo|karein|milegi|milega|'
    r'baare|ke|ka|ki|se|mein|par|pe|ye|vo|ab|jo|jab|tab|agar|phir|sab)\b',
    re.IGNORECASE
)


def _detect_language(text: str) -> str:
    """Detect if text is primarily Hindi, Hinglish, or English."""
    hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    total_alpha = hindi_chars + english_chars
    if total_alpha == 0:
        return "en"
    hindi_ratio = hindi_chars / total_alpha
    # Pure Hindi (Devanagari dominant)
    if hindi_ratio > 0.7:
        return "hi"
    # Devanagari mixed with Latin → Hinglish
    if hindi_ratio > 0.1:
        return "hinglish"
    # Check for Latin-script Hinglish (transliterated)
    if _HINGLISH_MARKERS.search(text):
        return "hinglish"
    return "en"


# ── Local intent classification (no LLM call needed) ────────────────────────

def _classify_intent_local(message: str, session_id: str) -> str:
    """Classify intent using semantic similarity and local context markers."""
    msg = message.lower().strip()

    # 1. Semantic Check (Sentence Transformer)
    intent = semantic_classify(msg)
    if intent != "fallback":
        return intent

    # 2. Contextual Follow-up Detection
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
    if not USE_LLM:
        if not sources:
            return "No relevant legal documents found in the database. Please try another query.", []
        
        fallback = "Here is the relevant legal information retrieved directly from the Kaggle database:\n\n"
        for i, src in enumerate(sources, 1):
            source_name = src.get('source', 'Unknown Source')
            act = src.get('act', '')
            section = src.get('section', '')
            
            # Apply NLP Summerization for offline text
            summarized_text = summarize_text(src.get('text', ''), max_sentences=2)
            
            if act and section and act != "Unknown" and section != "General":
                source_name = f"{act} - {section}"
                
            fallback += f"**{source_name}**:\n{summarized_text.strip()}\n\n"
            
        fallback += "*Note: I am running in Direct Database Mode to prevent AI rate limits. Responses are summarized.*"
        return fallback.strip(), ["What are the penalties?", "Explain this in simple terms.", "How do I file a case?"]

    # Build context from sources
    context_parts = []
    for i, src in enumerate(sources, 1):
        # Apply NLP summarization to the chunks before sending to LLM
        summarized_chunk = summarize_text(src['text'], max_sentences=3)
        context_parts.append(
            f"[Source {i}: {src['source']}]\n{summarized_chunk}"
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
            fallback = "I'm currently experiencing high demand due to API rate limits. Please try again after 3 minutes. In the meantime, here is the relevant legal information from the Kaggle dataset you provided:\n\n"
            for src in sources[:2]:
                fallback += f"**{src['source']}**:\n{src['text'][:400]}\n\n"
            fallback += "*Note: This is automatically retrieved context. Please try again in 3 minutes for a complete AI-generated answer.*"
            return fallback, []
        return (
            "I'm experiencing high demand right now due to API rate limits. Please try again after 3 minutes. "
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

    # Step 1: Check Q&A Database Direct Match
    qa_match = match_qa_dataset(message, threshold=0.85)
    if qa_match:
        memory.add_assistant_message(session_id, qa_match)
        return {
            "response": qa_match,
            "sources": [],
            "follow_ups": ["Tell me more", "What is the procedure?", "What are my rights?"],
            "intent": "qa_direct_match",
            "language": language
        }

    # Step 2: Classify intent locally (no API call)
    intent = _classify_intent_local(message, session_id)

    # Step 3: Handle non-RAG intents
    # If the semantic classifier found a match with a predefined response, use it
    if intent not in ["legal_query", "followup"]:
        try:
            resp = get_static_response(message)
            memory.add_assistant_message(session_id, resp)
            
            # Map default follow-ups based on intent
            follow_ups = [
                "What is Section 302 of IPC?",
                "What are my rights if arrested?",
                "How to check case status on eCourts?"
            ]
            if intent == "capability":
                follow_ups = ["Tell me about Section 498A", "What are Fundamental Rights?", "How does bail work in India?"]
            elif intent in ["ecourts", "fine_payment"]:
                 follow_ups = ["Check case status", "Pay traffic fine", "View NJDG stats"]

            return {
                "response": resp,
                "sources": [],
                "follow_ups": follow_ups,
                "intent": intent,
                "language": language
            }
        except KeyError:
            # Fallback if the intent is recognized but no static response is mapped
            pass

    # Step 4: Rewrite query locally (no API call)
    search_query = _rewrite_query_local(message, session_id)

    # Step 5: Search knowledge base (1 embedding API call)
    sources = _search_knowledge_base(search_query)

    # Step 6: Generate response with LLM (1 LLM API call with retry)
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
