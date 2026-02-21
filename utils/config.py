"""
utils/config.py – Central configuration for EduMentor AI (Google Gemini Edition)

All tuneable parameters live here so educators can customise without
touching core logic.
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available on Streamlit Cloud

# ── Chunking ────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 800          # characters per chunk
CHUNK_OVERLAP: int = 150       # overlap between consecutive chunks

# ── Retrieval ───────────────────────────────────────────────────────────
TOP_K: int = 5                 # default number of chunks to retrieve
SIMILARITY_THRESHOLD: float = 0.3   # below → "low confidence" flag

# ── Google Gemini Models ────────────────────────────────────────────────
CHAT_MODEL: str = "gemini-1.5-flash"          # for chat / quiz / grading
EMBEDDING_MODEL: str = "models/embedding-001"  # for vector embeddings

# ── LLM defaults ────────────────────────────────────────────────────────
DEFAULT_TEMPERATURE: float = 0.3

# ── Paths ───────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = "./chroma_store"
HASH_CACHE_FILE: str = ".file_hashes.json"


def get_gemini_api_key(session_key: str = "") -> str:
    """
    Return the best available Gemini API key.
    Priority: session_key (from UI) → st.secrets → GOOGLE_API_KEY env var → empty string.
    """
    if session_key:
        return session_key
    # Try Streamlit secrets (used on Streamlit Cloud)
    try:
        import streamlit as st
        if "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
    return os.getenv("GOOGLE_API_KEY", "")
