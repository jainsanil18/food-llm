"""Static-embedder food retriever (Model2Vec).

Replaces calc.py's letter-level fuzzy match with semantic nearest-neighbor over
the food DB. Uses a static embedder (no transformer forward pass at query time)
so it's tiny and fast — phone-friendly.

Two things make this work well without any fine-tuning:
1. Each food is embedded using its FULL name, which includes the parenthetical
   alias ("Curd (Dahi)"), so Hindi/colloquial queries match the alias text even
   though the base embedder is English.
2. A small lexical tie-break prefers the *plainest* match (penalizes extra
   modifier words), so "milk" beats "soy milk" / "low fat milk".

The DB embeddings are computed once per process (static encode is ~instant for
8k short strings) and could be precomputed + bundled for mobile.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np

from .foods import Food

_MODEL_NAME = "minishlab/potion-base-8M"  # ~30MB fp32; potion-base-2M (~8MB) for the 10MB target
_model = None
_cache = {}


def _get_model():
    """Our own static embedder — uses the fine-tuned food matrix if present."""
    global _model
    if _model is None:
        from . import static
        _model = static.load()
    return _model


def _tokens(text: str):
    return set(re.findall(r"[a-z]+", text.lower()))


class FoodRetriever:
    def __init__(self, foods: List[Food]):
        self.foods = foods
        self.model = _get_model()
        self.texts = [f.name.lower() for f in foods]           # full name incl. alias
        self.tok = [_tokens(t) for t in self.texts]
        emb = np.asarray(self.model.encode_batch(self.texts), dtype=np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0                                # guard all-OOV names
        self.emb = np.nan_to_num(emb / norms)

    def match(self, query: str, topk: int = 15) -> Tuple[Optional[Food], float]:
        q = np.asarray(self.model.encode(query.lower()), dtype=np.float32)
        n = np.linalg.norm(q)
        q = np.nan_to_num(q / n) if n else q
        sims = np.nan_to_num(self.emb @ q)                     # cosine, all foods
        cand = np.argpartition(-sims, min(topk, len(sims) - 1))[:topk]
        qtok = _tokens(query)
        best, best_score = None, -1e9
        for i in cand:
            i = int(i)
            ftok = self.tok[i]
            coverage = len(qtok & ftok) / max(1, len(qtok))    # are query words present?
            extra = len(ftok - qtok)                           # unrequested modifier words
            score = (0.45 * float(sims[i]) + 0.5 * coverage
                     - 0.12 * extra                            # prefer the plain entry
                     - 0.003 * len(self.texts[i]))             # mild "shortest/generic" tiebreak
            if qtok and ftok == qtok:                          # exact word-set match
                score += 0.4
            if score > best_score:
                best, best_score = self.foods[i], score
        return best, best_score

    def topk(self, query: str, k: int = 20) -> List[Food]:
        """The bi-encoder shortlist (pure embedding cosine) for the reranker."""
        q = np.asarray(self.model.encode(query.lower()), dtype=np.float32)
        n = np.linalg.norm(q)
        q = np.nan_to_num(q / n) if n else q
        sims = np.nan_to_num(self.emb @ q)
        cand = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
        cand = cand[np.argsort(-sims[cand])]
        return [self.foods[int(i)] for i in cand]


def get_retriever(foods: List[Food]) -> FoodRetriever:
    key = len(foods)
    if key not in _cache:
        _cache[key] = FoodRetriever(foods)
    return _cache[key]


if __name__ == "__main__":
    from .foods import load_foods
    r = get_retriever(load_foods())
    for q in ["dahi", "milk", "soy milk", "roti", "matar paneer", "cucumber", "melon"]:
        f, s = r.match(q)
        print(f"  {q:14} -> {f.name}  (score {s:.2f})")
