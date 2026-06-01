"""Food database loader — Indian + Western nutrition datasets.

Replaces the crowd-sourced Open Food Facts source with two curated datasets
(from the workout-planner project): ~1k Indian foods (INDB, with real serving
units) + ~7k Western foods (USDA-derived). Clean canonical names, reliable
macros — a far better grounding source AND runtime lookup DB than OFF.

Both load into the shared `Food` shape so calc.py / the generators don't care
where a food came from.

Derived fields:
- Western data has no calorie column -> compute kcal via the Atwater factors
  (4/4/9 per g of carb/protein/fat).
- Indian data gives per-serving macros but not the serving's gram weight ->
  derive it from the per-serving vs per-100g energy ratio.
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
from dataclasses import dataclass, asdict
from typing import List, Optional

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
INDIAN_PATH = os.path.join(DATA_DIR, "indian_food.json")
WESTERN_PATH = os.path.join(DATA_DIR, "western_food.json")
USDA_SR_DIR = os.path.join(DATA_DIR, "usda", "sr")  # USDA SR Legacy CSVs (optional)


@dataclass
class Food:
    name: str
    source: str  # "indian" | "western"
    units: List[str]
    serving_g: Optional[float]
    kcal_100g: Optional[float]
    protein_100g: Optional[float]
    carbs_100g: Optional[float]
    fat_100g: Optional[float]
    serving_unit_name: Optional[str] = None  # e.g. "tea cup", "katori"


def _num(v) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _atwater(carb, protein, fat) -> Optional[float]:
    c, p, f = _num(carb) or 0, _num(protein) or 0, _num(fat) or 0
    kcal = 4 * c + 4 * p + 9 * f
    return round(kcal, 1) if kcal > 0 else None


def load_western(path: str = WESTERN_PATH) -> List[Food]:
    out = []
    for e in json.load(open(path, encoding="utf-8")):
        name = (e.get("description") or "").strip()
        if not name:
            continue
        carb, prot, fat = _num(e.get("carbohydrate")), _num(e.get("protein")), _num(e.get("total_fat"))
        out.append(Food(
            name=name, source="western", units=["g"], serving_g=None,
            kcal_100g=_atwater(carb, prot, fat),
            protein_100g=prot, carbs_100g=carb, fat_100g=fat,
        ))
    return out


def load_indian(path: str = INDIAN_PATH) -> List[Food]:
    out = []
    for e in json.load(open(path, encoding="utf-8")):
        name = (e.get("food_name") or "").strip()
        kcal = _num(e.get("energy_kcal"))
        if not name or not kcal:
            continue
        us_kcal = _num(e.get("unit_serving_energy_kcal"))
        serving_g = round((us_kcal / kcal) * 100, 1) if (us_kcal and kcal > 0) else None
        units = ["serving", "g"] if serving_g else ["g"]
        out.append(Food(
            name=name, source="indian", units=units, serving_g=serving_g,
            kcal_100g=kcal, protein_100g=_num(e.get("protein_g")),
            carbs_100g=_num(e.get("carb_g")), fat_100g=_num(e.get("fat_g")),
            serving_unit_name=e.get("servings_unit"),
        ))
    return out


# USDA FDC nutrient IDs (amounts are per 100 g)
_USDA_NUTRIENTS = {"1008": "kcal", "1003": "protein", "1005": "carb", "1004": "fat"}


def load_usda_sr(dir_path: str = USDA_SR_DIR) -> List[Food]:
    """USDA SR Legacy — ~7,800 authoritative generic foods (raw/ingredient level).

    Relational CSVs: food.csv (id->description) joined with food_nutrient.csv,
    keeping only the 4 macros. Returns [] if the dataset isn't downloaded.
    """
    food_path = os.path.join(dir_path, "food.csv")
    fn_path = os.path.join(dir_path, "food_nutrient.csv")
    if not (os.path.exists(food_path) and os.path.exists(fn_path)):
        return []
    desc = {}
    with open(food_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            desc[row["fdc_id"]] = row["description"]
    nut = {}  # fdc_id -> {kcal, protein, carb, fat}
    with open(fn_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):           # 36 MB — stream, keep only our 4
            key = _USDA_NUTRIENTS.get(row["nutrient_id"])
            if key and row["fdc_id"] in desc:
                v = _num(row["amount"])
                if v is not None:
                    nut.setdefault(row["fdc_id"], {})[key] = v
    out = []
    for fid, name in desc.items():
        n = nut.get(fid, {})
        if "kcal" not in n:
            continue
        out.append(Food(name=name.strip(), source="usda", units=["g"], serving_g=None,
                        kcal_100g=n.get("kcal"), protein_100g=n.get("protein"),
                        carbs_100g=n.get("carb"), fat_100g=n.get("fat")))
    return out


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", name.lower().strip())


# Dedup preference when the same name appears in multiple sources:
# Indian first (has real serving units), then USDA (authoritative), then Western.
_SOURCE_PRIORITY = {"indian": 3, "usda": 2, "western": 1}


def _dedup(foods: List[Food]) -> List[Food]:
    by_key = {}
    for f in foods:
        k = _norm(f.name)
        cur = by_key.get(k)
        if cur is None or _SOURCE_PRIORITY.get(f.source, 0) > _SOURCE_PRIORITY.get(cur.source, 0):
            by_key[k] = f
    return list(by_key.values())


def load_foods(dedup: bool = True) -> List[Food]:
    """The full curated food DB: Indian + Western + USDA SR Legacy (if present),
    deduplicated by normalized name."""
    foods = load_indian() + load_western() + load_usda_sr()
    return _dedup(foods) if dedup else foods


def display_name(food: Food) -> str:
    """A natural surface form for grounding utterances: drop parentheticals,
    lowercase. 'Hot tea (Garam Chai)' -> 'hot tea'. The parse keeps the full
    canonical `food.name`."""
    n = food.name.split("(")[0].strip()
    return n.lower()


def sample_foods(foods: List[Food], k: int, rng: random.Random,
                 max_name_len: int = 32) -> List[Food]:
    """Pick k foods suitable for grounding (reasonable name length), biased
    ~45% Indian so both cuisines stay represented."""
    usable = [f for f in foods if len(f.name) <= max_name_len]
    indian = [f for f in usable if f.source == "indian"]
    western = [f for f in usable if f.source == "western"]
    n_in = min(len(indian), max(1, round(k * 0.45)))
    n_we = min(len(western), k - n_in)
    picked = rng.sample(indian, n_in) + rng.sample(western, n_we)
    rng.shuffle(picked)
    return picked[:k]


if __name__ == "__main__":
    foods = load_foods()
    ind = sum(f.source == "indian" for f in foods)
    print(f"loaded {len(foods)} foods ({ind} Indian, {len(foods) - ind} Western)")
    print(f"with serving units: {sum(bool(f.serving_g) for f in foods)}")
    print("\nsamples:")
    for f in load_indian()[:3] + load_western()[:3]:
        sv = f" | 1 {f.serving_unit_name}={f.serving_g}g" if f.serving_g else ""
        print(f"  [{f.source}] {f.name[:34]:34} | {f.kcal_100g} kcal/100g{sv}")
