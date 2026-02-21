"""
rag/ingest.py – PDF ingestion, text cleaning, chunking, and ChromaDB storage.

Pipeline:
  1. Load PDF pages with PyPDFLoader
  2. Clean noisy headers / footers via regex
  3. Split with RecursiveCharacterTextSplitter (configurable chunk_size & overlap)
  4. Attach metadata: source, page, title, uploaded_by, timestamp, course_id
  5. Hash files to skip re-ingestion of unchanged PDFs
  6. Persist vectors in ChromaDB
"""

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

from utils.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    HASH_CACHE_FILE,
)

# ---------------------------------------------------------------------------
# File hash helpers (cache embeddings when PDFs are unchanged)
# ---------------------------------------------------------------------------

def _compute_file_hash(file_bytes: bytes) -> str:
    """SHA-256 hash of raw file bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


def _load_hash_store() -> dict:
    """Load the persisted file-hash mapping."""
    if os.path.exists(HASH_CACHE_FILE):
        with open(HASH_CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_hash_store(store: dict) -> None:
    """Persist the file-hash mapping."""
    with open(HASH_CACHE_FILE, "w") as f:
        json.dump(store, f, indent=2)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Remove common PDF noise: repeated headers/footers, excessive whitespace."""
    # Remove lines that look like page numbers only (e.g., "  12  " or "Page 12")
    text = re.sub(r"(?m)^\s*(Page\s*)?\d{1,4}\s*$", "", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------------

def ingest_pdfs(
    uploaded_files: list,
    api_key: str,
    course_id: str = "general",
    uploaded_by: str = "learner",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> dict:
    """
    Ingest a list of uploaded PDF files into ChromaDB.

    Parameters
    ----------
    uploaded_files : list
        Streamlit UploadedFile objects.
    api_key : str
        OpenAI API key for embedding generation.
    course_id : str
        Identifier for the course/subject these PDFs belong to.
    uploaded_by : str
        Name of the person uploading.
    chunk_size : int
        Characters per chunk (default from config).
    chunk_overlap : int
        Overlap between chunks (default from config).

    Returns
    -------
    dict
        Summary: {"ingested": int, "skipped": int, "total_chunks": int}
    """
    hash_store = _load_hash_store()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_docs = []
    ingested_count = 0
    skipped_count = 0

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        file_hash = _compute_file_hash(file_bytes)
        file_name = uploaded_file.name

        # Skip if this exact file was already ingested
        if hash_store.get(file_name) == file_hash:
            skipped_count += 1
            continue

        # Write to a temp file so PyPDFLoader can read it
        tmp_path = os.path.join(CHROMA_PERSIST_DIR, f"_tmp_{file_name}")
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)

        try:
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()

            for page_doc in pages:
                page_doc.page_content = _clean_text(page_doc.page_content)
                # Enrich metadata
                page_doc.metadata.update({
                    "source": file_name,
                    "title": file_name.replace(".pdf", "").replace("_", " ").title(),
                    "uploaded_by": uploaded_by,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "course_id": course_id,
                })

            chunks = splitter.split_documents(pages)
            all_docs.extend(chunks)
            hash_store[file_name] = file_hash
            ingested_count += 1
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # Embed and store in ChromaDB
    total_chunks = 0
    if all_docs:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)
        Chroma.from_documents(
            documents=all_docs,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name="edu_mentor",
        )
        total_chunks = len(all_docs)

    _save_hash_store(hash_store)

    return {
        "ingested": ingested_count,
        "skipped": skipped_count,
        "total_chunks": total_chunks,
    }


def ingest_wikipedia_stub(text_content: str, api_key: str, course_id: str = "wikipedia") -> dict:
    """
    Stub for Wikipedia EDU dump ingestion.

    To use:
      1. Download a Wikipedia EDU dump (plain text or XML).
      2. Extract articles into a single .txt file.
      3. Call this function with the text content.

    This is a simplified stub – for production, implement proper
    article boundary detection and metadata extraction.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    from langchain_core.documents import Document

    docs = [
        Document(
            page_content=chunk,
            metadata={
                "source": "Wikipedia EDU",
                "course_id": course_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        for chunk in splitter.split_text(text_content)
    ]

    if docs:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)
        Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name="edu_mentor",
        )

    return {"ingested_chunks": len(docs)}


def clear_vector_store() -> None:
    """Delete the entire ChromaDB persistent store and hash cache."""
    if os.path.exists(CHROMA_PERSIST_DIR):
        shutil.rmtree(CHROMA_PERSIST_DIR)
