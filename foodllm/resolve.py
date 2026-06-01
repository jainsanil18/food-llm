"""Unified food resolver: alias lookup first, model/retriever fallback.

The runtime path for a food phrase:
  1. ALIAS LOOKUP against the curated DB (foods_canonical.json) — instant, exact,
     handles dahi/aloo/prawns/etc. This covers the common case.
  2. RETRIEVER (static embedder) — semantic match for unaliased phrases.
  3. FUZZY (SequenceMatcher) — last resort.

So a food a user actually logs resolves by dictionary the moment it's aliased;
the models are only the tail's safety net.
"""

from __future__ import annotations

import json
import os
import re
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

from .foods import Food, DATA_DIR

CANON_PATH = os.path.join(DATA_DIR, "foods_canonical.json")
_data = None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def load_canonical():
    """(foods, alias_index) from the curated DB, built once."""
    global _data
    if _data is None:
        recs = json.load(open(CANON_PATH, encoding="utf-8"))
        foods, idx = [], {}
        for r in recs:
            f = Food(
                name=r["canonical"], source=r.get("source", "curated"),
                units=["serving", "g"] if r.get("serving_g") else ["g"],
                serving_g=r.get("serving_g"),
                kcal_100g=r.get("kcal_100g"), protein_100g=r.get("protein_100g"),
                carbs_100g=r.get("carb_100g"), fat_100g=r.get("fat_100g"),
            )
            foods.append(f)
            for a in list(r.get("aliases", [])) + [r["canonical"]]:
                idx.setdefault(_norm(a), f)        # first writer wins
        _data = (foods, idx)
    return _data


def resolve(phrase: str) -> Tuple[Optional[Food], str]:
    """Return (food, method) where method is alias | retriever | fuzzy | none."""
    foods, idx = load_canonical()
    key = _norm(phrase)
    if key in idx:
        return idx[key], "alias"

    # fallback: semantic retriever over the curated foods
    try:
        from .retriever import get_retriever
        f, score = get_retriever(foods).match(phrase)
        if f is not None and score >= 0.2:
            return f, "retriever"
    except Exception:
        pass

    # last resort: letter-level fuzzy
    best, sc = None, 0.0
    for f in foods:
        s = SequenceMatcher(None, key, f.name.lower()).ratio()
        if key and key in f.name.lower():
            s += 0.3
        if s > sc:
            best, sc = f, s
    return (best, "fuzzy") if sc >= 0.5 else (None, "none")
