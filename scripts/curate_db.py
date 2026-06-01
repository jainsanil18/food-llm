"""Curation pass -> one clean canonical food DB with aliases.

Turns the 15.8k merged mess into clean records:
  { canonical, aliases[], kcal/protein/carb/fat per 100g, serving_g, source }

How:
1. Reuse build_surface_map() (already does dedup, junk-stripping, defaults,
   distinct surfaces) and INVERT it: each food -> the phrases that map to it.
2. Merge in AI-authored aliases (ALIAS_AUGMENT below) — the cross-lingual/slang
   synonyms a real logger needs (dahi->curd, prawns->shrimp, aloo->potato).
   These are resolved to real DB entries so every alias points at a real food.

The payoff: resolution becomes a dictionary lookup for the common case — the
prawns->Pralines / dahi->Idli failures disappear because "prawns"/"dahi" are now
aliases of the right entry.

Out: data/foods_canonical.json
"""

from __future__ import annotations

import json
import os
import re

from foodllm import foods as fooddb
from scripts.build_dataset_canonical import (
    build_surface_map, JUNK, GENERIC, DESCRIPTORS, _primary_head,
)

# Hand-authored clean entries for staples the merged DB lacks a plain version of
# (e.g. every "paneer" row is a dish). Nutrition per 100 g. This is the "add the
# missing clean foods" data work.
MANUAL_FOODS = [
    {"canonical": "Paneer", "kcal_100g": 265, "protein_100g": 18.3,
     "carb_100g": 1.2, "fat_100g": 20.8, "serving_g": 50,
     "aliases": ["paneer", "cottage cheese", "indian cheese", "fresh paneer"]},
    {"canonical": "Rice, white, cooked", "kcal_100g": 130, "protein_100g": 2.7,
     "carb_100g": 28.2, "fat_100g": 0.3, "serving_g": 150,
     "aliases": ["rice", "white rice", "chawal", "bhaat", "steamed rice",
                 "boiled rice", "cooked rice"]},
    {"canonical": "Curd (Dahi)", "kcal_100g": 98, "protein_100g": 11.0,
     "carb_100g": 3.4, "fat_100g": 4.3, "serving_g": 150,
     "aliases": ["dahi", "curd", "plain yogurt", "set curd", "yogurt"]},
]

# AI-authored aliases (key = a hint resolved against the DB; value = synonyms).
# This is the "alias generation" pass — Hindi/English, slang, regional.
ALIAS_AUGMENT = {
    "yogurt plain": ["dahi", "curd", "plain yogurt", "set curd", "thick curd"],
    "shrimp cooked": ["prawns", "prawn", "jhinga", "shrimps"],
    "roti": ["chapati", "chapatti", "phulka", "wheat roti"],
    "potato boiled": ["aloo", "alu", "boiled potato"],
    "eggplant": ["brinjal", "baingan", "aubergine"],
    "okra": ["bhindi", "lady finger", "ladies finger"],
    "spinach raw": ["palak", "saag"],
    "cauliflower raw": ["gobi", "gobhi", "phool gobi"],
    "chickpeas": ["chana", "chole", "garbanzo", "kabuli chana"],
    "lentils cooked": ["dal", "daal", "dhal", "cooked dal"],
    "rice white cooked": ["chawal", "bhaat", "steamed rice", "boiled rice"],
    "milk whole": ["doodh", "full cream milk", "dairy milk"],
    "paneer": ["cottage cheese", "indian cheese"],
    "cucumber raw": ["kheera", "kakdi"],
    "mango raw": ["aam"],
    "banana raw": ["kela"],
    "apple raw": ["seb"],
    "onion raw": ["pyaz", "kanda"],
    "tomato raw": ["tamatar"],
    "carrot raw": ["gajar"],
    "egg whole cooked": ["anda", "boiled egg", "ande"],
    "chicken breast cooked": ["murgh", "chicken curry", "grilled chicken"],
    "peas": ["matar", "green peas", "mutter"],
    "almonds": ["badam"],
    "ghee": ["clarified butter"],
    "buttermilk": ["chaas", "chhaas", "mattha"],
}


def _resolve_hint(db, hint):
    toks = hint.lower().split()
    cands = [f for f in db
             if all(re.search(rf"\b{re.escape(t)}\b", f.name.lower()) for t in toks)]
    if not cands:
        return None
    hset = set(toks)

    main = toks[0]

    def rank(f):
        names = set(re.findall(r"[a-z]+", f.name.lower()))
        sc = 0.0
        if re.search(r"[A-Z]{3,}", f.name):
            sc -= 50
        if _primary_head(f.name) == main:
            sc += 10
        sc -= 20 * len((names & JUNK) - hset)
        if names & GENERIC:
            sc += 3
        sc -= len(names) + 0.01 * len(f.name)
        return sc

    return max(cands, key=rank)


def build():
    smap = build_surface_map()           # surface -> Food
    by_name = {}
    for surface, food in smap.items():
        rec = by_name.get(food.name)
        if rec is None:
            rec = by_name[food.name] = {
                "canonical": food.name, "source": food.source, "aliases": set(),
                "kcal_100g": food.kcal_100g, "protein_100g": food.protein_100g,
                "carb_100g": food.carbs_100g, "fat_100g": food.fat_100g,
                "serving_g": food.serving_g,
            }
        rec["aliases"].add(surface)

    # merge AI-authored aliases
    db = fooddb.load_foods()
    augmented = 0
    for hint, extra in ALIAS_AUGMENT.items():
        f = _resolve_hint(db, hint)
        if not f:
            continue
        rec = by_name.get(f.name)
        if rec is None:
            rec = by_name[f.name] = {
                "canonical": f.name, "source": f.source, "aliases": set(),
                "kcal_100g": f.kcal_100g, "protein_100g": f.protein_100g,
                "carb_100g": f.carbs_100g, "fat_100g": f.fat_100g,
                "serving_g": f.serving_g,
            }
        rec["aliases"].update(a.lower() for a in extra)
        augmented += 1

    # add hand-authored clean entries; their aliases are AUTHORITATIVE
    manual_names = {mf["canonical"] for mf in MANUAL_FOODS}
    manual_aliases = set()
    for mf in MANUAL_FOODS:
        manual_aliases.update(a.lower() for a in mf["aliases"])
        by_name[mf["canonical"]] = {
            "canonical": mf["canonical"], "source": "manual",
            "aliases": set(a.lower() for a in mf["aliases"]),
            "kcal_100g": mf["kcal_100g"], "protein_100g": mf["protein_100g"],
            "carb_100g": mf["carb_100g"], "fat_100g": mf["fat_100g"],
            "serving_g": mf["serving_g"],
        }
    # strip the staple aliases from every OTHER (dish) entry, so "paneer" -> Paneer only
    for name, rec in by_name.items():
        if name not in manual_names:
            rec["aliases"] -= manual_aliases

    records = []
    for name, rec in by_name.items():
        rec["aliases"] = sorted(rec["aliases"])
        records.append(rec)
    records.sort(key=lambda r: r["canonical"])
    return records, augmented


def main():
    records, augmented = build()
    path = os.path.join(fooddb.DATA_DIR, "foods_canonical.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=0)
    n_alias = sum(len(r["aliases"]) for r in records)
    print(f"curated foods: {len(records)}")
    print(f"total aliases: {n_alias}  (avg {n_alias/len(records):.1f}/food)")
    print(f"AI-augmented foods: {augmented}")
    print(f"-> {path}\n")
    print("sample records:")
    for r in records[:3]:
        print(f"  {r['canonical'][:34]:34} aliases={r['aliases'][:4]}")


if __name__ == "__main__":
    main()
