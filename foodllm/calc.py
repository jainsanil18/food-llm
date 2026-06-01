"""The runtime tool: deterministic lookup + unit normalization + summation.

This is the half of the system that must NOT live in the model's weights. The
model extracts (food, qty, unit); this code turns that into exact numbers using
the food database. LLMs are bad at arithmetic and hallucinate nutrition facts —
so the moment we have a structured parse, we hand off to plain Python.

This is a working skeleton: it resolves foods against the OFF pool by fuzzy
name match, converts units to grams via each food's known portion size, and
sums macros. Swap the name matcher for an embedding/alias index when you scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from .foods import Food, load_foods

# Density-free volume conversions are approximate; for ml we assume ~1 g/ml
# unless the food carries its own serving size. Good enough for a skeleton.
GENERIC_UNIT_GRAMS = {
    "g": 1.0,
    "ml": 1.0,
    "tbsp": 13.5,
    "handful": 28.0,
    "slice": 28.0,
}


@dataclass
class LineResult:
    food_query: str
    matched: Optional[str]
    grams: Optional[float]
    kcal: Optional[float]
    protein: Optional[float]
    carbs: Optional[float]
    fat: Optional[float]
    note: str = ""
    method: str = ""        # how the food was resolved: alias | retriever | fuzzy


# Default portion weights when a matched food has no known serving size, so
# "1 bowl of cucumber" still contributes instead of silently dropping to zero.
DEFAULT_PORTION_G = {"piece": 100.0, "serving": 150.0, "cup": 200.0, "bowl": 150.0}


def _best_match(query: str, foods: List[Food]) -> Optional[Food]:
    # Prefer the semantic static-embedder retriever; fall back to letter-fuzzy
    # if model2vec isn't installed.
    try:
        from .retriever import get_retriever
        food, score = get_retriever(foods).match(query)
        if food is not None and score >= 0.2:
            return food
    except Exception:
        pass
    q = query.lower().strip()
    best, score = None, 0.0
    for f in foods:
        s = SequenceMatcher(None, q, f.name.lower()).ratio()
        if q in f.name.lower() or f.name.lower() in q:
            s += 0.3
        if s > score:
            best, score = f, s
    return best if score >= 0.5 else None


def _to_grams(quantity: float, unit: str, food: Food) -> Optional[float]:
    """Convert (quantity, unit) to grams using the food's portion info first,
    then sensible fallbacks (so nothing silently drops to zero)."""
    if unit in ("piece", "serving", "cup", "bowl"):
        if food.serving_g:
            return quantity * food.serving_g
        return quantity * DEFAULT_PORTION_G.get(unit, 150.0)
    g_per = GENERIC_UNIT_GRAMS.get(unit)
    return quantity * g_per if g_per is not None else None


def calc(items: List[dict], foods: Optional[List[Food]] = None) -> dict:
    """Tool entrypoint. `items` is the model's parse: [{food, quantity, unit}].

    Resolves each food via the curated DB (alias lookup first, then retriever /
    fuzzy), then sums nutrition. Unresolvable lines are reported, not dropped.
    """
    from . import resolve as _resolve

    lines: List[LineResult] = []
    total = {"kcal": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}

    for it in items:
        q, unit = float(it["quantity"]), it["unit"]
        match, method = _resolve.resolve(it["food"])
        if match is None:
            lines.append(LineResult(it["food"], None, None, None, None, None,
                                    None, note="no food match", method="none"))
            continue
        grams = _to_grams(q, unit, match)
        if grams is None:
            lines.append(LineResult(it["food"], match.name, None, None, None,
                                    None, None, note=f"unknown portion for '{unit}'",
                                    method=method))
            continue
        factor = grams / 100.0
        kcal = (match.kcal_100g or 0) * factor
        prot = (match.protein_100g or 0) * factor
        carb = (match.carbs_100g or 0) * factor
        fat = (match.fat_100g or 0) * factor
        total["kcal"] += kcal
        total["protein"] += prot
        total["carbs"] += carb
        total["fat"] += fat
        lines.append(LineResult(it["food"], match.name, round(grams, 1),
                                round(kcal, 1), round(prot, 1), round(carb, 1),
                                round(fat, 1), method=method))

    return {
        "lines": [vars(l) for l in lines],
        "total": {k: round(v, 1) for k, v in total.items()},
    }


if __name__ == "__main__":
    demo = [
        {"food": "egg", "quantity": 2, "unit": "piece"},
        {"food": "white rice", "quantity": 100, "unit": "g"},
    ]
    import json
    print(json.dumps(calc(demo, foods=load_foods()), indent=2))
