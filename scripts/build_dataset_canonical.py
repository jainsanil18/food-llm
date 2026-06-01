"""Option A: training data where the parse `food` IS the canonical DB name.

This folds resolution INTO the model. The parser learns:
    "a glass of milk"  -> food: "Milk, NFS"   (the chosen default)
    "human milk"       -> food: "Milk, human"
    "matar paneer"     -> food: "Pea paneer curry (Matar paneer)"
so at runtime the model emits an exact DB name -> direct lookup, no matcher.

The resolution decision is made HERE, once, offline:
- For each ambiguous head noun ("milk"), pick the PLAINEST entry as the default;
  it gets the bare surface ("milk"). Every other variant gets a distinct surface
  ("human milk", "soy milk") so no two foods ever share a phrase (no conflicting
  targets — the thing that would poison training).

Vocabulary is SCOPED so a 0.5B can memorize it: all Indian foods + clean,
simple-named Western/USDA generics.

Out: data/train.jsonl, data/eval.jsonl
"""

from __future__ import annotations

import random
import re
import sys
from collections import defaultdict

from foodllm import foods as fooddb
from foodllm.schema import GeneratedUtterance, FoodItem
from foodllm.generate import _write_split

DESCRIPTORS = {"raw", "cooked", "fresh", "dried", "frozen", "canned", "boiled",
               "nfs", "ns", "regular", "plain", "prepared", "ready", "to", "eat",
               "heat", "from", "without", "with", "added", "unprepared"}
GENERIC = {"nfs", "whole", "raw", "regular", "plain", "cooked", "fresh", "ns"}
EXOTIC = {"soy", "soya", "human", "soup", "oil", "powder", "condensed", "skim",
          "fried", "imitation", "substitute", "flavored", "sweetened"}

CONNECTORS = [" and ", ", ", " with ", " plus ", " n ", " and then ", " and some "]
CASUAL_PREFIX = ["had ", "just had ", "grabbed ", "ate ", "i had ", ""]
MESSY_PREFIX = ["ok so ", "uhh ", "so like ", "i think ", ""]
MESSY_SUFFIX = [" idk", "??", " ngl", " i think", " or so", ""]
MESSY_HEDGE = ["like ", "maybe ", "prob ", ""]
STYLES = ["clean", "casual", "messy", "typo", "multi_item", "implicit_qty"]
STYLE_W = [0.18, 0.22, 0.20, 0.15, 0.15, 0.10]


def _norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def _head_tokens(name):
    base = name.split("(")[0]
    return [t for t in re.findall(r"[a-z]+", base.lower()) if t not in DESCRIPTORS]


def _plainness(f):
    toks = set(re.findall(r"[a-z]+", f.name.split("(")[0].lower()))
    score = -len(toks)
    if toks & GENERIC:
        score += 1
    if toks & EXOTIC:
        score -= 3
    return score


def _primary_head(name):
    """The main food noun = first significant token of the part before any comma.
    'Milk, human' -> 'milk'. Used for grouping so a food's default is decided
    within its OWN head group, not a rare modifier token's group."""
    first = name.split("(")[0].split(",")[0]
    toks = re.findall(r"[a-z]+", first.lower())
    sig = [t for t in toks if t not in DESCRIPTORS]
    return (sig or toks or [""])[0]


def _indian_surfaces(name):
    m = re.match(r"^(.*?)\((.*?)\)", name)
    cands = [m.group(2), m.group(1)] if m else [name]   # alias first, then base
    out = []
    for c in cands:
        for part in c.split(",")[0].split("/"):          # split slashed synonyms
            s = _norm(part)
            if s:
                out.append(s)
    return out


def _western_surface(name, is_default):
    base = name.split("(")[0]
    parts = [p.strip() for p in base.split(",") if p.strip()]
    head = parts[0].lower()
    mods = [p.lower() for p in parts[1:]]
    if is_default or not mods:
        return head
    nongeneric = [m for m in mods if m not in GENERIC and m not in DESCRIPTORS]
    dist = (nongeneric[0] if nongeneric else mods[0]).split()[0]
    return head if dist in head else f"{dist} {head}"


# Curated defaults for the foods people actually log. Value = hint tokens; the
# resolver picks the shortest DB entry whose name contains all of them, so the
# target is always a real DB food. Fixes the cases heuristics get wrong
# (chicken->back, paneer->salad) and the gaps (rice, potato, tomato...).
CURATED_DEFAULTS = {
    # grains / staples
    "rice": "rice white cooked", "brown rice": "rice brown cooked",
    "bread": "bread white", "roti": "roti", "chapati": "roti",
    "oats": "oats", "oatmeal": "oatmeal", "pasta": "pasta cooked",
    "noodles": "noodles cooked", "poha": "poha", "upma": "upma",
    "idli": "idli", "dosa": "dosa", "paratha": "paratha", "paronthi": "paratha",
    "biryani": "biryani", "pulao": "pulao",
    # proteins / legumes
    "chicken": "chicken breast cooked", "chicken breast": "chicken breast cooked",
    "egg": "egg whole cooked", "boiled egg": "egg whole cooked",
    "fish": "fish cooked", "mutton": "mutton", "prawns": "shrimp cooked",
    "paneer": "paneer", "tofu": "tofu", "beef": "beef cooked", "pork": "pork cooked",
    "dal": "dal", "lentils": "lentils cooked", "rajma": "rajma",
    "chole": "chickpeas", "chickpeas": "chickpeas", "chana": "chickpeas",
    "beans": "beans cooked", "soybean": "soybean",
    # dairy
    "milk": "milk whole", "yogurt": "yogurt plain", "curd": "curd", "dahi": "curd",
    "cheese": "cheese cheddar", "butter": "butter salted", "ghee": "ghee",
    "cream": "cream", "buttermilk": "buttermilk", "lassi": "lassi",
    "paneer cheese": "paneer", "dahi": "yogurt plain", "curd": "yogurt plain",
    # vegetables
    "potato": "potato boiled", "tomato": "tomato raw", "onion": "onion raw",
    "cucumber": "cucumber raw", "carrot": "carrot raw", "spinach": "spinach raw",
    "palak": "spinach raw", "cabbage": "cabbage raw", "cauliflower": "cauliflower raw",
    "gobi": "cauliflower raw", "peas": "peas", "matar": "peas",
    "eggplant": "eggplant", "brinjal": "eggplant", "baingan": "eggplant",
    "okra": "okra", "bhindi": "okra", "broccoli": "broccoli raw",
    "capsicum": "pepper raw", "beetroot": "beets raw", "pumpkin": "pumpkin",
    # fruits
    "apple": "apple raw", "banana": "banana raw", "orange": "orange raw",
    "mango": "mango raw", "grapes": "grapes raw", "watermelon": "watermelon raw",
    "melon": "muskmelon raw", "papaya": "papaya raw", "guava": "guava raw",
    "pomegranate": "pomegranate raw", "pineapple": "pineapple raw",
    "strawberry": "strawberries raw", "pear": "pear raw",
    # nuts / fats
    "almonds": "almonds", "cashews": "cashew", "walnuts": "walnuts",
    "peanuts": "peanuts", "olive oil": "oil olive", "coconut": "coconut",
    # beverages
    "tea": "tea", "chai": "tea", "coffee": "coffee", "juice": "juice",
}


# Dish/prep words that disqualify an entry from being a plain-food default
# (unless the hint explicitly asked for them).
JUNK = {"soup", "dip", "sauce", "salad", "curry", "gravy", "roll", "rolls",
        "cake", "pie", "candy", "candies", "chips", "bar", "bars", "snack",
        "snacks", "wasabi", "iced", "breaded", "tenders", "tender", "fried",
        "powder", "dried", "mix", "dressing", "smoothie", "shake", "milkshake",
        "sandwich", "burger", "pizza", "soy", "soya", "flavored", "sweetened",
        "pudding", "custard", "ice", "dessert", "patty", "patties", "nuggets",
        "uncooked", "pulao", "biryani", "upma", "kheer", "halwa", "tikka",
        "shaslik", "kofta", "cutlet", "pancake", "pilaf", "glutinous", "chowder",
        "murukku", "nest", "nests", "flour", "paper", "milkshake", "nog"}


def _resolve_curated(full_db):
    out = {}
    for surface, hint in CURATED_DEFAULTS.items():
        hint_toks = hint.lower().split()
        # whole-word match so "cooked" doesn't hit "uncooked", "dal" not "Dalma"
        cands = [f for f in full_db
                 if all(re.search(rf"\b{re.escape(t)}\b", f.name.lower()) for t in hint_toks)]
        if not cands:
            continue
        hset = set(hint_toks)
        main = hint_toks[0]                                # the food noun, e.g. "potato"

        def rank(f):
            toks = set(re.findall(r"[a-z]+", f.name.lower()))
            score = 0.0
            if re.search(r"[A-Z]{3,}", f.name):           # brand name
                score -= 50
            if _primary_head(f.name) == main:             # "Potato, NFS" beats "Sweet potato"
                score += 10
            score -= 20 * len((toks & JUNK) - hset)        # unrequested dish words
            if toks & GENERIC:
                score += 3
            score -= len(toks) + 0.01 * len(f.name)        # prefer plain/short
            return score

        out[surface] = max(cands, key=rank)
    return out


def _is_clean_canonical(name):
    """Keep only simple, sayable foods — drop USDA branded/survey cruft so the
    0.5B has a memorizable, clean output vocabulary."""
    if re.search(r"[A-Z]{3,}", name):            # brand all-caps (FRITOLAY, ARIZONA)
        return False
    if any(c.isdigit() for c in name):           # "1%", "ribs 10-12"
        return False
    if re.search(r"\b(NS|restructured|ready-to|prepared from)\b", name, re.I):
        return False
    base = name.split("(")[0]
    if len(base) > 34 or base.count(",") > 3:     # long multi-clause names
        return False
    return True


def build_surface_map():
    """surface phrase -> canonical Food. First/default writer wins (no conflicts)."""
    full_db = fooddb.load_foods()
    db = [f for f in full_db if _is_clean_canonical(f.name)]
    # mark the default (plainest) member of each PRIMARY-head group
    groups = defaultdict(list)
    for f in db:
        if f.source in ("western", "usda"):
            groups[_primary_head(f.name)].append(f)
    default_ids = set()
    for members in groups.values():
        default_ids.add(id(max(members, key=_plainness)))

    smap = {}

    def add(surface, food):
        s = _norm(surface)
        # scope: keep simple 1-2 word surfaces (memorizable, natural to say)
        if s and 1 <= len(s.split()) <= 2 and re.match(r"^[a-z][a-z '&-]*$", s):
            smap.setdefault(s, food)

    for f in db:
        if f.source == "indian":
            for s in _indian_surfaces(f.name):
                add(s, f)
        else:
            add(_western_surface(f.name, id(f) in default_ids), f)

    # Curated defaults OVERRIDE the heuristics for common foods (resolved against
    # the full DB, so the target is always a real entry).
    smap.update(_resolve_curated(full_db))
    return smap


def _typo(w, rng):
    if len(w) < 4 or not w.isalpha():
        return w
    i = rng.randint(0, len(w) - 2)
    op = rng.choice(["swap", "drop", "double"])
    return ({"swap": w[:i] + w[i+1] + w[i] + w[i+2:],
             "drop": w[:i] + w[i+1:],
             "double": w[:i] + w[i] + w[i:]})[op]


def _inject_typos(text, rng, n=2):
    words = text.split()
    cand = [i for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
    rng.shuffle(cand)
    for i in cand[:n]:
        words[i] = _typo(words[i], rng)
    return " ".join(words)


def _fragment(surface, food, style, rng):
    has_serving = bool(food.serving_g)
    if style == "implicit_qty":
        return (f"some {surface}", (1 if has_serving else 100),
                ("serving" if has_serving else "g"))
    grams = rng.choice([30, 50, 100, 150, 200])
    opts = [(f"{grams}g of {surface}", grams, "g"), (f"{grams}g {surface}", grams, "g")]
    if has_serving:
        opts += [(f"a serving of {surface}", 1, "serving"),
                 (f"a bowl of {surface}", 1, "serving"),
                 (f"some {surface}", 1, "serving")]
    else:
        opts += [(f"some {surface}", 100, "g")]
    return rng.choice(opts)


def _gen_one(style, surfaces, smap, rng):
    k = rng.choice([3, 4]) if style == "multi_item" else (1 if style == "implicit_qty"
                                                          else rng.choice([1, 2, 2, 2]))
    picks = rng.sample(surfaces, min(k, len(surfaces)))
    frags, items = [], []
    for s in picks:
        food = smap[s]
        fr = _fragment(s, food, style, rng)
        frags.append(fr[0])
        items.append(FoodItem(food=food.name, quantity=fr[1], unit=fr[2]))  # CANONICAL target
    if style == "casual":
        text = rng.choice(CASUAL_PREFIX) + rng.choice(CONNECTORS).join(frags)
    elif style == "messy":
        hedged = [rng.choice(MESSY_HEDGE) + f for f in frags]
        text = (rng.choice(MESSY_PREFIX) + rng.choice(CONNECTORS).join(hedged)
                + rng.choice(MESSY_SUFFIX)).lower()
    elif style == "typo":
        text = _inject_typos(rng.choice(CONNECTORS).join(frags), rng)
    else:
        text = rng.choice(CONNECTORS).join(frags)
    return GeneratedUtterance(text=text.strip(), style=style, items=items)


def build(target):
    rng = random.Random(7)
    smap = build_surface_map()
    surfaces = list(smap)
    print(f"vocabulary: {len(set(f.name for f in smap.values()))} canonical foods "
          f"reachable via {len(surfaces)} distinct surface phrases")
    out, seen, attempts = [], set(), 0
    while len(out) < target and attempts < target * 50:
        attempts += 1
        e = _gen_one(rng.choices(STYLES, weights=STYLE_W)[0], surfaces, smap, rng)
        key = e.text.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(e)
    print(f"built {len(out)} examples")
    _write_split(out, eval_frac=0.1, rng=rng)


if __name__ == "__main__":
    build(int(sys.argv[1]) if len(sys.argv) > 1 else 4000)
