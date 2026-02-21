"""
utils/config.py – Application configuration and defaults.

Loads settings from .env if available; UI entry is the primary method.
Educators can customize chunk sizes, overlap, top_k, and similarity thresholds here.
"""

import os
from dotenv import load_dotenv

# Load .env file if it exists (fallback for API key)
load_dotenv()

# ---------------------------------------------------------------------------
# Chunking Parameters (adjust these to control how PDFs are split)
# ---------------------------------------------------------------------------
CHUNK_SIZE = 1000          # Number of characters per text chunk
CHUNK_OVERLAP = 200        # Overlap between consecutive chunks (improves context)

# ---------------------------------------------------------------------------
# Retrieval Parameters
# ---------------------------------------------------------------------------
TOP_K = 5                  # Number of top chunks to retrieve per query
SIMILARITY_THRESHOLD = 0.3 # Minimum similarity score to consider a chunk relevant

# ---------------------------------------------------------------------------
# ChromaDB Storage
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_store")

# ---------------------------------------------------------------------------
# OpenAI Model Configuration
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.3

# ---------------------------------------------------------------------------
# File hashing (for caching ingested PDFs)
# ---------------------------------------------------------------------------
HASH_STORE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_store", "_file_hashes.json")


def get_openai_api_key(session_key: str | None = None) -> str | None:
    """Return the OpenAI API key from session state or .env, in that order."""
    if session_key:
        return session_key
    return os.getenv("OPENAI_API_KEY")
