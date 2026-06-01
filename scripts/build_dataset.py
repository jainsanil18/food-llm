"""Build a training set grounded in the real Indian + Western food DB.

Unlike the old author_dataset.py (hand-authored for ~20 seed foods), this samples
from all ~8,100 curated foods. For each food it builds natural "mention
fragments" (grams / servings / 'some X'), stitches them into utterances across
the six styles, and applies messy/typo surface transforms.

Key choice: the parse `food` is the food PHRASE as said ("hot tea", "milk",
"paneer"), not a long DB string. calc.py resolves that phrase against the DB by
fuzzy match. This keeps text and target aligned, reads naturally, and nudges the
model toward extracting the food span (the open-vocabulary direction).

Run:  ./.venv/bin/python scripts/build_dataset.py 400
Out:  data/train.jsonl, data/eval.jsonl
"""

from __future__ import annotations

import random
import re
import sys
from typing import List

from foodllm import foods as fooddb
from foodllm.schema import GeneratedUtterance, FoodItem
from foodllm.generate import _write_split

CONNECTORS = [" and ", ", ", " with ", " plus ", " n ", " and then ", " and some "]
CASUAL_PREFIX = ["had ", "just had ", "grabbed ", "ate ", "i had ", ""]
MESSY_PREFIX = ["ok so ", "uhh ", "so like ", "i think ", "lemme see ", ""]
MESSY_SUFFIX = [" idk", "??", " ngl", " i think", " or so", ""]
MESSY_HEDGE = ["like ", "maybe ", "prob ", "kinda ", ""]
STYLES = ["clean", "casual", "messy", "typo", "multi_item", "implicit_qty"]
STYLE_WEIGHTS = [0.18, 0.22, 0.20, 0.15, 0.15, 0.10]


def _surface(food: fooddb.Food) -> str:
    """Natural food phrase: drop parentheticals + everything after first comma."""
    n = food.name.split("(")[0].split(",")[0].strip().lower()
    return re.sub(r"\s+", " ", n)


def _grounding_pool(rng: random.Random):
    """Dedup foods by surface phrase; keep clean, short, word-y names."""
    foods = fooddb.load_foods()
    seen, pool = set(), []
    rng.shuffle(foods)
    for f in foods:
        s = _surface(f)
        if not s or s in seen:
            continue
        if not re.match(r"^[a-z][a-z '&-]*$", s):  # letters/spaces only
            continue
        if not (1 <= len(s.split()) <= 3) or len(s) > 24:
            continue
        if f.kcal_100g is None:
            continue
        seen.add(s)
        pool.append((s, bool(f.serving_g)))
    return pool


def _fragments(s: str, has_serving: bool, rng: random.Random):
    """Candidate (text, qty, unit) mentions for a food phrase."""
    grams = rng.choice([30, 50, 100, 150, 200])
    opts = [
        (f"{grams}g of {s}", grams, "g"),
        (f"{grams}g {s}", grams, "g"),
        (f"some {s}", (1 if has_serving else 100), ("serving" if has_serving else "g")),
        (f"a bit of {s}", (1 if has_serving else 50), ("serving" if has_serving else "g")),
    ]
    if has_serving:
        opts += [(f"a serving of {s}", 1, "serving"),
                 (f"2 servings of {s}", 2, "serving"),
                 (f"a bowl of {s}", 1, "serving")]
    return opts


def _typo(word: str, rng: random.Random) -> str:
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


def _gen_one(style: str, pool, rng: random.Random) -> GeneratedUtterance:
    k = rng.choice([3, 4]) if style == "multi_item" else (1 if style == "implicit_qty"
                                                           else rng.choice([1, 2, 2, 2]))
    picks = rng.sample(pool, min(k, len(pool)))
    chosen = []
    for (s, has_serving) in picks:
        if style == "implicit_qty":
            frag = (f"some {s}", (1 if has_serving else 100),
                    ("serving" if has_serving else "g"))
        else:
            frag = rng.choice(_fragments(s, has_serving, rng))
        chosen.append((s, frag))

    frags = [c[1][0] for c in chosen]
    if style == "casual":
        text = rng.choice(CASUAL_PREFIX) + rng.choice(CONNECTORS).join(frags)
    elif style == "messy":
        hedged = [rng.choice(MESSY_HEDGE) + fr for fr in frags]
        text = (rng.choice(MESSY_PREFIX) + rng.choice(CONNECTORS).join(hedged)
                + rng.choice(MESSY_SUFFIX)).lower()
    elif style == "typo":
        text = _inject_typos(rng.choice(CONNECTORS).join(frags), rng)
    else:
        text = rng.choice(CONNECTORS).join(frags)

    items = [FoodItem(food=s, quantity=fr[1], unit=fr[2]) for (s, fr) in chosen]
    return GeneratedUtterance(text=text.strip(), style=style, items=items)


def build(target: int) -> None:
    rng = random.Random(7)
    pool = _grounding_pool(rng)
    print(f"grounding pool: {len(pool)} distinct food phrases")
    out, seen, attempts = [], set(), 0
    while len(out) < target and attempts < target * 50:
        attempts += 1
        style = rng.choices(STYLES, weights=STYLE_WEIGHTS)[0]
        e = _gen_one(style, pool, rng)
        key = e.text.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(e)
    by_style = {}
    for e in out:
        by_style[e.style] = by_style.get(e.style, 0) + 1
    print(f"built {len(out)} examples")
    for s, c in sorted(by_style.items()):
        print(f"  {s:14} {c}")
    _write_split(out, eval_frac=0.12, rng=rng)


if __name__ == "__main__":
    build(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
