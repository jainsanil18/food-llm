"""Claude-authored training corpus — expanded, no API call needed.

Two layers:

1. HAND-AUTHORED core (EX) — the realism anchor, especially the hard styles
   (messy / typo / casual) that need genuine human sloppiness.

2. FRAGMENT GENERATOR — a controlled engine that recombines real food
   "mention fragments" (each carrying its own gold qty+unit) into new
   utterances, then applies per-style surface transforms (casual prefixes,
   messy hedges, typo injection on the TEXT only). Parses are correct by
   construction because each fragment already knows its gold (food, qty, unit).

This scales to hundreds of valid examples offline. It's more *uniform* than
the live API generator (real Claude writes more naturally varied text), so for
the final high-quality set you'd still run `generate.py`. This gives you a
solid, validated training set right now.

Rules (same as the generator's system prompt):
- food = clean canonical name (even when the TEXT has a typo)
- unit in {g, ml, piece, cup, tbsp, slice, handful, serving}
- parse captures EVERY food in order
"""

from __future__ import annotations

import random
from typing import List, Tuple

from foodllm.schema import GeneratedUtterance, FoodItem
from foodllm.generate import _write_split

Item = Tuple[str, float, str]  # (food, qty, unit)

# ---------------------------------------------------------------------------
# LAYER 1 — hand-authored core (text, style, items)
# ---------------------------------------------------------------------------
EX = [
    ("2 eggs and 100g of white rice", "clean",
     [("egg", 2, "piece"), ("white rice (cooked)", 100, "g")]),
    ("1 cup of greek yogurt with a handful of almonds", "clean",
     [("greek yogurt", 1, "cup"), ("almonds", 1, "handful")]),
    ("200g chicken breast and a boiled potato", "clean",
     [("chicken breast (cooked)", 200, "g"), ("potato (boiled)", 1, "piece")]),
    ("3 slices of bread with 1 tbsp of olive oil", "clean",
     [("bread slice", 3, "slice"), ("olive oil", 1, "tbsp")]),
    ("had a couple eggs and some rice", "casual",
     [("egg", 2, "piece"), ("white rice (cooked)", 1, "cup")]),
    ("just a banana and a handful of almonds", "casual",
     [("banana", 1, "piece"), ("almonds", 1, "handful")]),
    ("ok so this morning i think i had like maybe 2 eggs?? and then some rice", "messy",
     [("egg", 2, "piece"), ("white rice (cooked)", 1, "cup")]),
    ("uhh prob like 40g of corn flakes n a lil bit of milk", "messy",
     [("CORN FLAKES", 40, "g"), ("whole milk", 100, "ml")]),
    ("had a banana earlier and then like a handful of almonds idk", "messy",
     [("banana", 1, "piece"), ("almonds", 1, "handful")]),
    ("2 egss and 100g of withe rice", "typo",
     [("egg", 2, "piece"), ("white rice (cooked)", 100, "g")]),
    ("a banan and a handfull of almnds", "typo",
     [("banana", 1, "piece"), ("almonds", 1, "handful")]),
    ("greak yogrut with granla chocolat", "typo",
     [("greek yogurt", 1, "cup"), ("Granola Chocolat", 40, "g")]),
    ("2 eggs, 2 slices of bread, a banana and a glass of milk", "multi_item",
     [("egg", 2, "piece"), ("bread slice", 2, "slice"),
      ("banana", 1, "piece"), ("whole milk", 1, "cup")]),
    ("chicken breast 150g, a cup of rice, a boiled potato and olive oil 1 tbsp", "multi_item",
     [("chicken breast (cooked)", 150, "g"), ("white rice (cooked)", 1, "cup"),
      ("potato (boiled)", 1, "piece"), ("olive oil", 1, "tbsp")]),
    ("had an apple", "implicit_qty", [("apple", 1, "piece")]),
    ("just a banana", "implicit_qty", [("banana", 1, "piece")]),
    ("had some kiri", "implicit_qty", [("Kiri", 1, "serving")]),
    ("a bowl of corn flakes", "implicit_qty", [("CORN FLAKES", 1, "serving")]),
]

# ---------------------------------------------------------------------------
# LAYER 2 — fragment generator
# ---------------------------------------------------------------------------
# Each food maps to several natural "mention fragments". A fragment is the
# surface text plus its gold (qty, unit). The generator stitches fragments into
# utterances; the parse is just the chosen fragments' golds in order.
# `implicit` fragments use "a/an" with no number (qty 1) -> implicit_qty style.

FRAGMENTS = {
    "egg": [("an egg", 1, "piece", True), ("a couple eggs", 2, "piece", False),
            ("2 eggs", 2, "piece", False), ("3 eggs", 3, "piece", False),
            ("50g of egg", 50, "g", False)],
    "banana": [("a banana", 1, "piece", True), ("2 bananas", 2, "piece", False),
               ("half a banana", 0.5, "piece", False), ("a banana", 1, "piece", True)],
    "apple": [("an apple", 1, "piece", True), ("2 apples", 2, "piece", False),
              ("an apple", 1, "piece", True)],
    "orange": [("an orange", 1, "piece", True), ("2 oranges", 2, "piece", False)],
    "white rice (cooked)": [("a cup of rice", 1, "cup", False),
                            ("some rice", 1, "cup", False),
                            ("100g of white rice", 100, "g", False),
                            ("150g rice", 150, "g", False),
                            ("half a cup of rice", 0.5, "cup", False)],
    "chicken breast (cooked)": [("100g of chicken breast", 100, "g", False),
                                ("150g chicken", 150, "g", False),
                                ("200g of chicken breast", 200, "g", False),
                                ("some chicken", 80, "g", False)],
    "whole milk": [("a glass of milk", 1, "cup", False),
                   ("200ml of whole milk", 200, "ml", False),
                   ("250ml milk", 250, "ml", False),
                   ("half a glass of milk", 0.5, "cup", False),
                   ("a cup of milk", 1, "cup", False)],
    "bread slice": [("a slice of bread", 1, "slice", True),
                    ("2 slices of bread", 2, "slice", False),
                    ("3 slices of bread", 3, "slice", False)],
    "almonds": [("a handful of almonds", 1, "handful", False),
                ("30g of almonds", 30, "g", False),
                ("some almonds", 1, "handful", False)],
    "olive oil": [("1 tbsp of olive oil", 1, "tbsp", False),
                  ("2 tbsp olive oil", 2, "tbsp", False),
                  ("a bit of olive oil", 1, "tbsp", False)],
    "potato (boiled)": [("a boiled potato", 1, "piece", True),
                        ("2 boiled potatoes", 2, "piece", False),
                        ("half a potato", 0.5, "piece", False)],
    "greek yogurt": [("a cup of greek yogurt", 1, "cup", False),
                     ("some greek yogurt", 1, "cup", False),
                     ("150g greek yogurt", 150, "g", False)],
    "CORN FLAKES": [("a bowl of corn flakes", 1, "serving", False),
                    ("40g of corn flakes", 40, "g", False),
                    ("some corn flakes", 1, "serving", False)],
    "Weetabix": [("2 weetabix", 2, "serving", False),
                 ("a weetabix", 1, "serving", True),
                 ("3 weetabix", 3, "serving", False)],
    "Granola Chocolat": [("40g of granola chocolat", 40, "g", False),
                         ("some granola", 40, "g", False),
                         ("50g granola chocolat", 50, "g", False)],
    "Kiri": [("a kiri", 1, "serving", True), ("2 kiri", 2, "serving", False),
             ("some kiri", 1, "serving", False)],
    "Cheddar": [("a slice of cheddar", 1, "slice", True),
                ("30g of cheddar", 30, "g", False)],
    "Miel Pops": [("a bowl of miel pops", 1, "serving", False),
                  ("40g of miel pops", 40, "g", False)],
    "Crispy Minis Choco": [("40g of crispy minis choco", 40, "g", False),
                           ("a serving of crispy minis choco", 1, "serving", False)],
    "Flocons d'avoine": [("40g of oats", 40, "g", False),
                         ("a bowl of oats", 1, "serving", False)],
}

CONNECTORS = [" and ", ", ", " with ", " plus ", " n ", " and then ", " and some "]
CASUAL_PREFIX = ["had ", "just had ", "grabbed ", "ate ", "i had ", ""]
MESSY_PREFIX = ["ok so ", "uhh ", "so like ", "i think ", "lemme see ", ""]
MESSY_SUFFIX = [" idk", "??", " ngl", " i think", " or so", ""]
MESSY_HEDGE = ["like ", "maybe ", "prob ", "kinda ", ""]


def _typo(word: str, rng: random.Random) -> str:
    """Mutate one longer alpha word: swap adjacent letters, drop, or double."""
    if len(word) < 4 or not word.isalpha():
        return word
    i = rng.randint(0, len(word) - 2)
    op = rng.choice(["swap", "drop", "double"])
    if op == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == "drop":
        return word[:i] + word[i + 1:]
    return word[:i] + word[i] + word[i:]


def _inject_typos(text: str, rng: random.Random, n: int = 2) -> str:
    words = text.split()
    cand = [i for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
    rng.shuffle(cand)
    for i in cand[:n]:
        words[i] = _typo(words[i], rng)
    return " ".join(words)


def _pick_fragments(style: str, rng: random.Random):
    foods = list(FRAGMENTS)
    if style == "implicit_qty":
        # single food, an implicit "a/an" fragment
        rng.shuffle(foods)
        for f in foods:
            implicit = [fr for fr in FRAGMENTS[f] if fr[3]]
            if implicit:
                return [(f, rng.choice(implicit))]
        # fallback
        f = rng.choice(foods)
        return [(f, FRAGMENTS[f][0])]
    if style == "multi_item":
        k = rng.choice([3, 4])
    else:
        k = rng.choice([1, 2, 2, 2])  # mostly 2
    chosen = rng.sample(foods, min(k, len(foods)))
    return [(f, rng.choice(FRAGMENTS[f])) for f in chosen]


def _assemble(style: str, parts, rng: random.Random) -> str:
    frags = [p[1][0] for p in parts]
    if style == "casual":
        text = rng.choice(CASUAL_PREFIX) + rng.choice(CONNECTORS).join(frags)
    elif style == "messy":
        # sprinkle hedges before some fragments, lowercase, prefix/suffix
        hedged = [(rng.choice(MESSY_HEDGE) + fr) for fr in frags]
        text = (rng.choice(MESSY_PREFIX)
                + rng.choice(CONNECTORS).join(hedged)
                + rng.choice(MESSY_SUFFIX))
        text = text.lower()
    elif style == "typo":
        text = _inject_typos(rng.choice(CONNECTORS).join(frags), rng)
    else:  # clean / multi_item / implicit_qty
        text = rng.choice(CONNECTORS).join(frags)
    return text.strip()


def _gen_one(style: str, rng: random.Random) -> GeneratedUtterance:
    parts = _pick_fragments(style, rng)
    text = _assemble(style, parts, rng)
    items = [FoodItem(food=f, quantity=fr[1], unit=fr[2]) for (f, fr) in parts]
    return GeneratedUtterance(text=text, style=style, items=items)


STYLE_WEIGHTS = {
    "clean": 0.18, "casual": 0.22, "messy": 0.22,
    "typo": 0.15, "multi_item": 0.13, "implicit_qty": 0.10,
}


def generate(target: int, rng: random.Random) -> List[GeneratedUtterance]:
    """Produce `target` unique synthetic examples on top of the hand-authored core."""
    out = [GeneratedUtterance(text=t, style=s,
                              items=[FoodItem(food=f, quantity=q, unit=u) for (f, q, u) in it])
           for (t, s, it) in EX]
    seen = {e.text.strip().lower() for e in out}
    styles = list(STYLE_WEIGHTS)
    weights = [STYLE_WEIGHTS[s] for s in styles]
    attempts = 0
    while len(out) < target and attempts < target * 40:
        attempts += 1
        style = rng.choices(styles, weights=weights)[0]
        e = _gen_one(style, rng)
        key = e.text.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(e)
    return out


def build(target: int = 320) -> None:
    rng = random.Random(7)
    examples = generate(target, rng)
    print(f"built {len(examples)} examples ({len(EX)} hand-authored core)")
    by_style = {}
    for e in examples:
        by_style[e.style] = by_style.get(e.style, 0) + 1
    for s, c in sorted(by_style.items()):
        print(f"  {s:14} {c}")
    _write_split(examples, eval_frac=0.12, rng=rng)


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 320
    build(n)
