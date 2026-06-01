"""Claude-authored aliases for the common, high-frequency foods.

The 3k DB is mostly obscure long-tail (USDA research entries) that no one logs
and that has no natural alias. Aliases earn their keep on the few hundred foods
people actually type — staples, common veg/fruit, dals, dairy, proteins, common
Indian dishes, drinks, nuts, sweets. Those are authored here directly (by Claude,
inline — no API), keyed by a hint that resolves to the right canonical record.

Run:  ./.venv/bin/python -m scripts.author_aliases
Merges into data/foods_canonical.json.
"""

from __future__ import annotations

import json
import os
import re

from foodllm import foods as fooddb
from scripts.build_dataset_canonical import JUNK, GENERIC

CANON = os.path.join(fooddb.DATA_DIR, "foods_canonical.json")

# hint (resolves to a canonical record) -> aliases a real person would type
ALIASES = {
    # --- grains / staples ---
    "rice white cooked": ["rice", "white rice", "chawal", "bhaat", "steamed rice", "boiled rice"],
    "rice brown cooked": ["brown rice", "brown chawal"],
    "roti": ["roti", "chapati", "chapatti", "phulka", "wheat roti"],
    "paratha": ["paratha", "parantha", "paronthi", "aloo paratha", "stuffed paratha"],
    "naan": ["naan", "naan bread", "butter naan", "garlic naan"],
    "bread white": ["bread", "white bread", "slice of bread", "double roti", "pav"],
    "bread brown": ["brown bread", "wheat bread", "whole wheat bread"],
    "oats": ["oats", "oatmeal", "porridge", "rolled oats"],
    "poha": ["poha", "flattened rice", "beaten rice", "chiwda"],
    "upma": ["upma", "uppma", "rava upma"],
    "idli": ["idli", "idly", "steamed idli"],
    "dosa": ["dosa", "dosai", "plain dosa", "masala dosa"],
    "biryani": ["biryani", "biriyani", "chicken biryani", "veg biryani"],
    "pulao": ["pulao", "pulav", "veg pulao", "jeera rice"],
    "pasta cooked": ["pasta", "macaroni", "penne", "spaghetti"],
    "noodles cooked": ["noodles", "maggi", "hakka noodles", "chowmein"],
    "vermicelli": ["vermicelli", "semiya", "seviyan"],
    # --- dals / legumes ---
    "lentils cooked": ["dal", "daal", "dhal", "lentils", "cooked dal", "tadka dal", "dal fry"],
    "chickpeas": ["chana", "chole", "chickpeas", "garbanzo", "kabuli chana", "chole masala"],
    "rajma": ["rajma", "kidney beans", "red kidney beans", "rajma masala"],
    "black gram": ["urad dal", "urad", "black gram"],
    "green gram": ["moong", "moong dal", "mung beans", "green gram"],
    "pigeon pea": ["toor dal", "arhar dal", "tur dal", "pigeon pea"],
    "soybean": ["soybean", "soya bean", "soya chunks", "soy"],
    # --- vegetables ---
    "potato boiled": ["potato", "aloo", "alu", "boiled potato"],
    "tomato raw": ["tomato", "tamatar", "tomatoes"],
    "onion raw": ["onion", "pyaz", "kanda", "onions"],
    "cucumber raw": ["cucumber", "kheera", "kakdi"],
    "carrot raw": ["carrot", "gajar", "carrots"],
    "spinach raw": ["spinach", "palak", "saag"],
    "cauliflower raw": ["cauliflower", "gobi", "gobhi", "phool gobi"],
    "cabbage raw": ["cabbage", "patta gobi", "band gobi"],
    "eggplant": ["eggplant", "brinjal", "baingan", "aubergine"],
    "okra": ["okra", "bhindi", "lady finger", "ladies finger", "vendakkai"],
    "peas": ["peas", "matar", "green peas", "mutter"],
    "broccoli raw": ["broccoli"],
    "capsicum": ["capsicum", "bell pepper", "shimla mirch"],
    "beetroot": ["beetroot", "beet", "chukandar"],
    "pumpkin": ["pumpkin", "kaddu", "kashiphal"],
    "bottle gourd": ["bottle gourd", "lauki", "ghiya", "doodhi"],
    "bitter gourd": ["bitter gourd", "karela", "bitter melon"],
    "radish raw": ["radish", "mooli", "daikon"],
    "garlic raw": ["garlic", "lehsun", "lasun"],
    "ginger raw": ["ginger", "adrak"],
    "green chili": ["green chili", "green chilli", "hari mirch", "mirchi"],
    "coriander": ["coriander", "cilantro", "dhania", "hara dhania"],
    "mushroom raw": ["mushroom", "mushrooms", "khumb"],
    # --- fruits ---
    "apple raw": ["apple", "seb", "apples"],
    "banana raw": ["banana", "kela", "ripe banana"],
    "orange raw": ["orange", "santra", "oranges"],
    "mango raw": ["mango", "aam", "mangoes"],
    "grapes raw": ["grapes", "angoor", "grape"],
    "watermelon raw": ["watermelon", "tarbooj"],
    "papaya raw": ["papaya", "papita"],
    "guava raw": ["guava", "amrood", "peru"],
    "pomegranate raw": ["pomegranate", "anar"],
    "pineapple raw": ["pineapple", "ananas"],
    "strawberries raw": ["strawberry", "strawberries"],
    "pear raw": ["pear", "nashpati"],
    "grapefruit raw": ["grapefruit"],
    "coconut": ["coconut", "nariyal", "fresh coconut"],
    "dates": ["dates", "khajur"],
    # --- dairy ---
    "milk whole": ["milk", "whole milk", "full cream milk", "doodh", "dairy milk"],
    "milk skim": ["skim milk", "toned milk", "skimmed milk", "low fat milk"],
    "paneer": ["paneer", "cottage cheese", "indian cheese", "fresh paneer"],
    "curd": ["dahi", "curd", "plain yogurt", "set curd", "yogurt"],
    "cheese cheddar": ["cheese", "cheddar", "cheddar cheese"],
    "butter salted": ["butter", "makhan", "salted butter"],
    "ghee": ["ghee", "clarified butter", "desi ghee"],
    "cream": ["cream", "fresh cream", "malai"],
    "buttermilk": ["buttermilk", "chaas", "chhaas", "mattha"],
    "lassi": ["lassi", "sweet lassi", "mango lassi"],
    "milkshake": ["milkshake", "shake"],
    # --- proteins ---
    "egg whole cooked": ["egg", "anda", "boiled egg", "ande", "eggs"],
    "egg omelet": ["omelette", "omelet", "egg omelette", "anda bhurji"],
    "chicken breast cooked": ["chicken", "murgh", "chicken breast", "grilled chicken"],
    "chicken curry": ["chicken curry", "murgh curry", "chicken masala"],
    "shrimp cooked": ["prawns", "prawn", "jhinga", "shrimp", "shrimps"],
    "fish cooked": ["fish", "machli", "machhi", "fried fish"],
    "mutton": ["mutton", "goat meat", "lamb", "bakra"],
    "tofu": ["tofu", "bean curd", "soya paneer"],
    # --- nuts ---
    "almonds": ["almonds", "badam", "almond"],
    "cashew": ["cashew", "cashews", "kaju"],
    "walnuts": ["walnuts", "akhrot", "walnut"],
    "peanuts": ["peanuts", "moongphali", "groundnut", "peanut"],
    "pistachio": ["pistachio", "pista", "pistachios"],
    "raisins": ["raisins", "kishmish", "raisin"],
    # --- beverages ---
    "tea": ["tea", "chai", "garam chai", "masala chai"],
    "coffee": ["coffee", "kaapi", "filter coffee"],
    "orange juice": ["orange juice", "juice", "fresh juice"],
    # --- common dishes / snacks / sweets ---
    "samosa": ["samosa", "samosas", "singara"],
    "pakora": ["pakora", "pakoda", "bhaji", "fritters"],
    "dhokla": ["dhokla", "khaman"],
    "poori": ["poori", "puri", "fried puri"],
    "halwa": ["halwa", "halva", "sooji halwa"],
    "gulab jamun": ["gulab jamun", "gulab jamoon"],
    "jalebi": ["jalebi", "jilebi"],
    "kheer": ["kheer", "rice pudding", "payasam"],
    "raita": ["raita", "cucumber raita", "boondi raita"],
    "sambar": ["sambar", "sambhar"],
    "rasam": ["rasam", "saaru"],
    "khichdi": ["khichdi", "khichadi", "dal khichdi"],
    "sugar": ["sugar", "cheeni", "chini"],
    "honey": ["honey", "shahad"],
    "jaggery": ["jaggery", "gur", "gud"],
    "olive oil": ["olive oil"],
    "mustard oil": ["mustard oil", "sarson ka tel"],
}


def _head(name):
    t = re.findall(r"[a-z]+", name.split("(")[0].split(",")[0].lower())
    return t[0] if t else ""


def _resolve(records, hint):
    toks = hint.lower().split()
    cands = [r for r in records
             if all(re.search(rf"\b{re.escape(t)}\b", r["canonical"].lower()) for t in toks)]
    if not cands:   # fall back to substring (handles plurals: tomato -> Tomatoes)
        cands = [r for r in records
                 if all(t in r["canonical"].lower() for t in toks)]
    if not cands:
        return None
    main, hset = toks[0], set(toks)

    def rank(r):
        names = set(re.findall(r"[a-z]+", r["canonical"].lower()))
        sc = 0.0
        if r.get("source") == "manual":
            sc += 6
        if _head(r["canonical"]) == main:
            sc += 10
        sc -= 20 * len((names & JUNK) - hset)
        if names & GENERIC:
            sc += 3
        sc -= len(names) + 0.01 * len(r["canonical"])
        return sc

    return max(cands, key=rank)


def main():
    records = json.load(open(CANON, encoding="utf-8"))
    matched, added, unresolved = 0, 0, []
    for hint, aliases in ALIASES.items():
        r = _resolve(records, hint)
        if not r:
            unresolved.append(hint)
            continue
        matched += 1
        before = set(r["aliases"])
        r["aliases"] = sorted(before | {a.lower().strip() for a in aliases})
        added += len(r["aliases"]) - len(before)

    json.dump(records, open(CANON, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"authored aliases for {len(ALIASES)} common foods")
    print(f"  resolved to {matched} canonical records, +{added} aliases")
    if unresolved:
        print(f"  unresolved hints ({len(unresolved)}): {unresolved}")
    n_alias = sum(len(r["aliases"]) for r in records)
    print(f"  DB now: {n_alias} total aliases across {len(records)} foods")


if __name__ == "__main__":
    main()
