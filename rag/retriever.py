"""
rag/retriever.py – Vector similarity search against ChromaDB.

Retrieves the top-k most relevant chunks for a user query, filters by
similarity threshold, and flags low-confidence results.
"""

import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

from utils.config import (
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    TOP_K,
    SIMILARITY_THRESHOLD,
)


def get_vectorstore(api_key: str) -> Chroma | None:
    """
    Load the persisted ChromaDB vector store.

    Returns None if the store directory does not exist yet.
    """
    if not os.path.exists(CHROMA_PERSIST_DIR):
        return None

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="edu_mentor",
    )


def retrieve(
    query: str,
    api_key: str,
    top_k: int = TOP_K,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> dict:
    """
    Retrieve relevant chunks for *query*.

    Returns
    -------
    dict
        {
            "chunks": [
                {
                    "content": str,
                    "metadata": dict,  # source, page, title, …
                    "score": float,
                }
            ],
            "is_confident": bool,       # True if best score >= threshold
            "avg_score": float,
        }
    """
    vectorstore = get_vectorstore(api_key)
    if vectorstore is None:
        return {"chunks": [], "is_confident": False, "avg_score": 0.0}

    # similarity_search_with_relevance_scores returns (Document, score) tuples
    try:
        results = vectorstore.similarity_search_with_relevance_scores(query, k=top_k)
    except Exception:
        # Fallback: some Chroma versions don't support relevance scores
        docs = vectorstore.similarity_search(query, k=top_k)
        results = [(doc, 1.0) for doc in docs]

    chunks = []
    scores = []
    for doc, score in results:
        chunks.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": round(float(score), 4),
        })
        scores.append(float(score))

    avg_score = sum(scores) / len(scores) if scores else 0.0
    is_confident = (max(scores) >= similarity_threshold) if scores else False

    return {
        "chunks": chunks,
        "is_confident": is_confident,
        "avg_score": round(avg_score, 4),
    }
