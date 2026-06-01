"""Open Food Facts loader.

OFF is huge (millions of products). For generating training data we don't need
the full dump — we need a representative *sample* of real food names, their
serving sizes, and per-100g nutrition. This module pulls that sample from the
OFF v2 search API and caches it to a local JSONL so re-runs are offline.

OFF is strong on branded/packaged products but weak on generic whole foods
("1 egg", "a cup of rice") and their portion units. So we also ship a small
curated WHOLE_FOODS seed list with real portion grams. The two together give
the generator both packaged and everyday foods to talk about.

Honest caveat: OFF data is crowd-sourced and patchy — many products have no
serving size or partial nutriments. We keep only rows with a usable name and at
least an energy value, and fall back to the seed list when the API is
unreachable (e.g. offline, or no network in CI).
"""

from __future__ import annotations

import json
import os
import random
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from typing import List, Optional

OFF_SEARCH_URL = "https://world.openfoodfacts.org/api/v2/search"
# OFF asks every client to send a descriptive User-Agent.
USER_AGENT = "food-llm-datagen/0.1 (https://github.com/yourname/food-llm)"

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CACHE_PATH = os.path.join(DATA_DIR, "off_foods.jsonl")


@dataclass
class Food:
    """A food the generator can ground an utterance in.

    nutriments are per 100 g (OFF's convention). `units` lists the units that
    make sense for this food, so the generator picks realistic ones.
    """

    name: str
    source: str  # "off" | "seed"
    off_code: Optional[str]  # barcode, if from OFF
    serving_g: Optional[float]  # grams in one serving/piece, if known
    units: List[str]  # plausible units, e.g. ["g", "serving"] or ["piece", "g"]
    kcal_100g: Optional[float]
    protein_100g: Optional[float]
    carbs_100g: Optional[float]
    fat_100g: Optional[float]


# --- Curated whole foods (the gap OFF doesn't cover well) -------------------
# Portion grams are standard reference values (USDA-style). These anchor the
# "2 eggs", "a banana", "cup of rice" style of phrasing.
WHOLE_FOODS: List[Food] = [
    Food("egg", "seed", None, 50, ["piece", "g"], 155, 13, 1.1, 11),
    Food("banana", "seed", None, 118, ["piece", "g"], 89, 1.1, 23, 0.3),
    Food("apple", "seed", None, 182, ["piece", "g"], 52, 0.3, 14, 0.2),
    Food("white rice (cooked)", "seed", None, 158, ["cup", "g"], 130, 2.7, 28, 0.3),
    Food("chicken breast (cooked)", "seed", None, 120, ["piece", "g"], 165, 31, 0, 3.6),
    Food("whole milk", "seed", None, 244, ["cup", "ml", "g"], 61, 3.2, 4.8, 3.3),
    Food("bread slice", "seed", None, 28, ["slice", "g"], 265, 9, 49, 3.2),
    Food("almonds", "seed", None, 28, ["handful", "g"], 579, 21, 22, 50),
    Food("olive oil", "seed", None, 13.5, ["tbsp", "ml", "g"], 884, 0, 0, 100),
    Food("potato (boiled)", "seed", None, 173, ["piece", "g"], 87, 1.9, 20, 0.1),
    Food("greek yogurt", "seed", None, 170, ["cup", "g"], 59, 10, 3.6, 0.4),
    Food("orange", "seed", None, 131, ["piece", "g"], 47, 0.9, 12, 0.1),
]


def _coerce_float(v) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_serving_g(serving_size: Optional[str], serving_qty) -> Optional[float]:
    """OFF gives serving_quantity (grams) sometimes, or a free-text serving_size
    like '30 g' / '250 ml'. Prefer the numeric field, else parse the string."""
    q = _coerce_float(serving_qty)
    if q:
        return q
    if not serving_size:
        return None
    # crude: grab the leading number from strings like "30 g (1 bar)"
    num = ""
    for ch in serving_size.strip():
        if ch.isdigit() or ch == ".":
            num += ch
        elif num:
            break
    return _coerce_float(num)


def _off_product_to_food(p: dict) -> Optional[Food]:
    name = (p.get("product_name") or "").strip()
    if not name or len(name) > 80:
        return None
    n = p.get("nutriments") or {}
    kcal = _coerce_float(n.get("energy-kcal_100g"))
    if kcal is None:
        return None  # require at least energy to be useful downstream
    serving_g = _parse_serving_g(p.get("serving_size"), p.get("serving_quantity"))
    units = ["g"]
    if serving_g:
        units = ["serving", "g"]
    return Food(
        name=name,
        source="off",
        off_code=p.get("code"),
        serving_g=serving_g,
        units=units,
        kcal_100g=kcal,
        protein_100g=_coerce_float(n.get("proteins_100g")),
        carbs_100g=_coerce_float(n.get("carbohydrates_100g")),
        fat_100g=_coerce_float(n.get("fat_100g")),
    )


# Common everyday categories — sampling across these gives a diverse, realistic
# food pool (vs. random obscure products). The expensive popularity sort +
# states_tags filter triggers OFF 503s, so we filter by category instead.
OFF_CATEGORIES = [
    "breakfast-cereals", "yogurts", "cheeses", "breads", "pastas",
    "biscuits", "chocolates", "fruit-juices", "snacks", "sodas",
    "nuts", "rices", "fruits", "vegetables", "meats",
]


def _fetch_category(category: str, page_size: int, timeout: int, retries: int = 3) -> List[Food]:
    fields = "code,product_name,brands,serving_size,serving_quantity,nutriments"
    params = urllib.parse.urlencode(
        {"fields": fields, "page_size": page_size, "categories_tags_en": category}
    )
    url = f"{OFF_SEARCH_URL}?{params}"
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            out = []
            for p in payload.get("products", []):
                f = _off_product_to_food(p)
                if f:
                    out.append(f)
            return out
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            if attempt == retries - 1:
                print(f"  [off] '{category}' failed after {retries} tries: {e}")
    return []


def fetch_off_sample(per_category: int = 25, timeout: int = 20) -> List[Food]:
    """Pull a diverse sample of real products from the OFF search API, sampling
    a handful of foods from each common category. Returns whatever it gets —
    callers fall back to the seed list if this is empty.
    """
    foods: List[Food] = []
    for cat in OFF_CATEGORIES:
        got = _fetch_category(cat, per_category, timeout)
        foods.extend(got)
        print(f"  [off] {cat}: +{len(got)}")
    return foods


def _write_cache(foods: List[Food]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as fh:
        for f in foods:
            fh.write(json.dumps(asdict(f), ensure_ascii=False) + "\n")


def _read_cache() -> List[Food]:
    if not os.path.exists(CACHE_PATH):
        return []
    out = []
    with open(CACHE_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Food(**json.loads(line)))
    return out


def load_foods(refresh: bool = False, include_off: bool = True) -> List[Food]:
    """Return the food pool: curated whole foods + an OFF sample.

    Uses the local cache unless `refresh=True`. If OFF can't be reached and no
    cache exists, you still get the seed whole foods so the generator runs.
    """
    foods: List[Food] = list(WHOLE_FOODS)
    if include_off:
        off = [] if refresh else _read_cache()
        if not off:
            print("  [off] fetching sample from Open Food Facts API...")
            off = fetch_off_sample()
            if off:
                _write_cache(off)
                print(f"  [off] cached {len(off)} products to {CACHE_PATH}")
            else:
                print("  [off] no products fetched; using seed whole foods only")
        foods += off
    return foods


def sample_foods(foods: List[Food], k: int, rng: random.Random) -> List[Food]:
    """Pick k foods, biased ~40% toward whole foods so 'quantity' phrasings
    (2 eggs, a cup of rice) stay well represented alongside packaged products."""
    whole = [f for f in foods if f.source == "seed"]
    packaged = [f for f in foods if f.source == "off"]
    n_whole = min(len(whole), max(1, round(k * 0.4)))
    n_pack = min(len(packaged), k - n_whole)
    picked = rng.sample(whole, n_whole) + (rng.sample(packaged, n_pack) if packaged else [])
    rng.shuffle(picked)
    return picked[:k]


if __name__ == "__main__":
    pool = load_foods()
    print(f"Loaded {len(pool)} foods "
          f"({sum(f.source == 'seed' for f in pool)} seed, "
          f"{sum(f.source == 'off' for f in pool)} OFF)")
    for f in pool[:5]:
        print(" -", f.name, "|", f.units, "|", f.kcal_100g, "kcal/100g")
