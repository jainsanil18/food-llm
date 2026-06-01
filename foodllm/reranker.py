"""Cross-encoder reranker — picks the RIGHT food from the bi-encoder shortlist.

The bi-encoder (static embedder) is fast but rough: it narrows 8k foods to a
shortlist. The cross-encoder then reads (query, candidate) TOGETHER and scores
how well they match, re-ranking the shortlist. That joint read is what catches
"prawns≈shrimp" and demotes "prawns≠Pralines".

Stage 1 (bi-encoder, fast): 8000 -> top-K
Stage 2 (cross-encoder, accurate): top-K -> 1
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

from .foods import Food

_BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
_FINETUNED = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "food-reranker")
_ce = None


def _get_ce():
    global _ce
    if _ce is None:
        from sentence_transformers import CrossEncoder
        from . import hub
        try:
            _ce = CrossEncoder(hub.get_reranker_dir())   # local or auto-download from HF
        except Exception:
            _ce = CrossEncoder(_BASE_MODEL)              # fall back to the base
    return _ce


def rerank(query: str, candidates: List[Food]) -> Tuple[Optional[Food], float]:
    """Score (query, candidate.name) pairs jointly; return the best food."""
    if not candidates:
        return None, 0.0
    ce = _get_ce()
    scores = ce.predict([[query, c.name] for c in candidates])
    best_i = int(max(range(len(candidates)), key=lambda i: scores[i]))
    return candidates[best_i], float(scores[best_i])
