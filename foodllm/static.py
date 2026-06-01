"""Static embedder wrapper — a tokenizer + a token-embedding matrix E.

A phrase's vector = L2-normalized mean of its token vectors. No transformer,
no forward pass. We own the encode path (rather than calling Model2Vec's
StaticModel.encode) so the fine-tuner and the retriever pool identically — the
matrix we train is exactly the matrix we serve.

The fine-tuned matrix lives at models/food-static/embedding.npy; if absent we
use the base potion-base-8M weights.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np

BASE_MODEL = "minishlab/potion-base-8M"
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
FINETUNED_E = os.path.join(MODELS_DIR, "food-static", "embedding.npy")

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from model2vec import StaticModel
        _tokenizer = StaticModel.from_pretrained(BASE_MODEL).tokenizer
    return _tokenizer


def base_embedding() -> np.ndarray:
    from model2vec import StaticModel
    m = StaticModel.from_pretrained(BASE_MODEL)
    return np.asarray(m.embedding, dtype=np.float32).copy()


class StaticEmbedder:
    def __init__(self, tokenizer, E: np.ndarray):
        self.tok = tokenizer
        self.E = E
        self.dim = E.shape[1]

    def token_ids(self, text: str) -> List[int]:
        ids = self.tok.encode(text.lower()).ids
        return [i for i in ids if 0 <= i < self.E.shape[0]]

    def encode(self, text: str) -> np.ndarray:
        ids = self.token_ids(text)
        if not ids:
            return np.zeros(self.dim, dtype=np.float32)
        v = self.E[ids].mean(axis=0)
        n = np.linalg.norm(v)
        return (v / n).astype(np.float32) if n else v.astype(np.float32)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        return np.stack([self.encode(t) for t in texts])


def load(use_finetuned: bool = True) -> StaticEmbedder:
    tok = _get_tokenizer()
    if use_finetuned and os.path.exists(FINETUNED_E):
        E = np.load(FINETUNED_E).astype(np.float32)
    else:
        E = base_embedding()
    return StaticEmbedder(tok, E)


def is_finetuned() -> bool:
    return os.path.exists(FINETUNED_E)
