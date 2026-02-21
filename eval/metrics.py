"""
eval/metrics.py – Lightweight ROUGE-L and BLEU computation.

Pure Python implementations (no NLTK or heavy NLP dependencies).
Used as proxy evaluation metrics for RAG answer quality.
"""

from collections import Counter


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    import re
    return re.findall(r"\w+", text.lower())


# ---------------------------------------------------------------------------
# ROUGE-L (Longest Common Subsequence based)
# ---------------------------------------------------------------------------

def _lcs_length(x: list[str], y: list[str]) -> int:
    """Compute the length of the longest common subsequence."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    # Space-optimized DP
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


def compute_rouge_l(hypothesis: str, reference: str) -> dict:
    """
    Compute ROUGE-L F1 score between hypothesis and reference.

    Parameters
    ----------
    hypothesis : str
        The generated answer.
    reference : str
        The reference text (e.g., concatenated retrieved context).

    Returns
    -------
    dict
        {"precision": float, "recall": float, "f1": float}
    """
    hyp_tokens = _tokenize(hypothesis)
    ref_tokens = _tokenize(reference)

    if not hyp_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = _lcs_length(hyp_tokens, ref_tokens)
    precision = lcs / len(hyp_tokens) if hyp_tokens else 0.0
    recall = lcs / len(ref_tokens) if ref_tokens else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# ---------------------------------------------------------------------------
# BLEU (simplified unigram + bigram BLEU)
# ---------------------------------------------------------------------------

def _count_ngrams(tokens: list[str], n: int) -> Counter:
    """Count n-grams in a token list."""
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def compute_bleu(hypothesis: str, reference: str, max_n: int = 2) -> dict:
    """
    Compute a simplified BLEU score (unigram + bigram).

    Parameters
    ----------
    hypothesis : str
        The generated answer.
    reference : str
        The reference text.
    max_n : int
        Maximum n-gram order (default 2 for bigram BLEU).

    Returns
    -------
    dict
        {"bleu": float, "precisions": list[float]}
    """
    hyp_tokens = _tokenize(hypothesis)
    ref_tokens = _tokenize(reference)

    if not hyp_tokens or not ref_tokens:
        return {"bleu": 0.0, "precisions": [0.0] * max_n}

    precisions = []
    for n in range(1, max_n + 1):
        hyp_ngrams = _count_ngrams(hyp_tokens, n)
        ref_ngrams = _count_ngrams(ref_tokens, n)

        clipped = sum(min(count, ref_ngrams.get(ng, 0)) for ng, count in hyp_ngrams.items())
        total = sum(hyp_ngrams.values())

        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped / total)

    # Geometric mean of precisions (with smoothing for zero)
    import math
    log_avg = 0.0
    for p in precisions:
        if p == 0:
            log_avg = float("-inf")
            break
        log_avg += math.log(p) / len(precisions)

    bleu = math.exp(log_avg) if log_avg > float("-inf") else 0.0

    # Brevity penalty
    bp = min(1.0, len(hyp_tokens) / len(ref_tokens)) if ref_tokens else 0.0
    bleu *= bp

    return {
        "bleu": round(bleu, 4),
        "precisions": [round(p, 4) for p in precisions],
    }
