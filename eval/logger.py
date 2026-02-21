"""
eval/logger.py – Structured logging for RAG interactions and quiz results.

Logs are stored in session state (in-memory) and can be exported to CSV or JSON.
Each log entry captures: timestamp, query, answer, retrieval scores,
ROUGE/BLEU metrics, and quiz scores.
"""

import csv
import io
import json
from datetime import datetime


def create_log_entry(
    query: str,
    answer: str,
    retrieval_scores: list[float],
    rouge_l: float = 0.0,
    bleu: float = 0.0,
    is_confident: bool = True,
    quiz_score: float | None = None,
    quiz_max: float | None = None,
) -> dict:
    """
    Create a structured log entry for one interaction.

    Parameters
    ----------
    query : str
        The learner's question.
    answer : str
        The generated answer.
    retrieval_scores : list[float]
        Similarity scores from retrieval.
    rouge_l : float
        ROUGE-L F1 score (proxy evaluation).
    bleu : float
        BLEU score (proxy evaluation).
    is_confident : bool
        Whether retrieval was confident.
    quiz_score : float, optional
        Quiz score if a quiz was taken.
    quiz_max : float, optional
        Maximum possible quiz score.

    Returns
    -------
    dict
        The log entry.
    """
    avg_sim = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0
    hit_rate = sum(1 for s in retrieval_scores if s >= 0.3) / len(retrieval_scores) if retrieval_scores else 0.0

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer,
        "avg_similarity": round(avg_sim, 4),
        "hit_rate": round(hit_rate, 4),
        "rouge_l_f1": round(rouge_l, 4),
        "bleu": round(bleu, 4),
        "is_confident": is_confident,
        "hallucination_risk": not is_confident,  # proxy flag
        "quiz_score": quiz_score,
        "quiz_max": quiz_max,
    }


def export_logs_csv(logs: list[dict]) -> str:
    """Export log entries to a CSV string for download."""
    if not logs:
        return ""

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=logs[0].keys())
    writer.writeheader()
    writer.writerows(logs)
    return output.getvalue()


def export_logs_json(logs: list[dict]) -> str:
    """Export log entries to a formatted JSON string for download."""
    return json.dumps(logs, indent=2, default=str)
