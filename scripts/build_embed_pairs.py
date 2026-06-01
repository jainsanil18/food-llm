"""Build embedder training triplets from the DB — pure code, no Claude/API.

Produces (anchor, positive, negative) triplets that teach the food matcher:
  anchor   = a short query phrase a user types ("milk", "roti", "paneer")
  positive = the PLAINEST DB entry for it  ("Milk, NFS", "Roti", "Paneer")
  negative = a confusable modified sibling ("Milk, human", "Soya roti", "Paneer soup")

Two sources, both from the DB:
1. Head-noun groups -> plain vs modified (fixes milk->human, roti->soya, paneer->soup)
2. Parenthetical aliases -> cross-term positives (dahi <-> Curd (Dahi))

Out: data/embed_pairs.jsonl
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict

from foodllm import foods as fooddb

# Words that are descriptors, not food head-nouns — never used as an anchor.
DESCRIPTORS = {
    "raw", "cooked", "fresh", "dried", "frozen", "canned", "with", "and", "in",
    "of", "the", "without", "added", "prepared", "boiled", "fried", "roasted",
    "nfs", "regular", "plain", "whole", "low", "fat", "free", "reduced", "skim",
    "unsweetened", "sweetened", "flavored", "powder", "powdered", "ready", "to",
    "eat", "heat", "from", "made", "type", "style", "homemade", "commercial",
}
# Modifiers that mark an entry as a strong *negative* for a plain query.
EXOTIC = {
    "soy", "soya", "human", "soup", "oil", "powder", "condensed", "skim",
    "fried", "vada", "vadas", "bhalla", "flavored", "sweetened", "glutinous",
    "imitation", "substitute", "non", "dairy",
}
# Markers that make an entry a *good* generic positive.
GENERIC = {"nfs", "whole", "raw", "regular", "plain", "cooked", "fresh"}


def _tokens(name: str):
    return [t for t in re.findall(r"[a-z]+", name.lower())]


def _surface(name: str) -> str:
    return re.sub(r"\s+", " ", name.split("(")[0].split(",")[0].strip().lower())


def build_triplets():
    db = fooddb.load_foods()
    triplets = []

    # --- 1. Head-noun groups ------------------------------------------------
    # Map each candidate food word -> the foods whose name contains it.
    groups = defaultdict(list)
    for f in db:
        toks = set(_tokens(f.name))
        for t in toks:
            if t in DESCRIPTORS or len(t) < 3:
                continue
            groups[t].append(f)

    for head, members in groups.items():
        if len(members) < 2:
            continue
        # plainness score: fewer tokens is plainer; bonus if it's exactly the head
        # or carries a generic marker; penalty for exotic modifiers.
        def plainness(f):
            toks = set(_tokens(f.name))
            score = -len(toks)
            if toks == {head}:
                score += 5
            if toks & GENERIC:
                score += 1
            if toks & EXOTIC:
                score -= 3
            return score

        members_sorted = sorted(members, key=plainness, reverse=True)
        positive = members_sorted[0]
        # Precision guard: the positive must be the PLAIN food — head noun plus
        # only generic/descriptor words. If it still carries a foreign food word
        # ("pulao" in "Paneer pulao"), the DB has no clean entry for this head;
        # skip the group rather than teach a wrong mapping.
        extra = set(_tokens(positive.name)) - {head} - GENERIC - DESCRIPTORS
        if extra:
            continue
        pos_p = plainness(positive)
        # negatives: siblings that are genuinely LESS plain (modified/exotic).
        # Equally-plain alternatives (e.g. "Milk, whole" vs "Milk, NFS") are NOT
        # negatives — they're valid answers too.
        negs = [m for m in members_sorted[1:]
                if m.name != positive.name and plainness(m) < pos_p][:6]
        for neg in negs:
            triplets.append({"anchor": head, "positive": positive.name,
                             "negative": neg.name})

    # --- 2. Parenthetical aliases (Indian source only) ----------------------
    # Indian names carry genuine Hindi aliases ("Curd (Dahi)"); USDA/Western
    # parentheticals are descriptors ("(Alaska Native)") — skip those.
    for f in db:
        if f.source != "indian":
            continue
        m = re.match(r"^(.*?)\((.*?)\)", f.name)
        if not m:
            continue
        base = m.group(1).strip().lower()
        alias = m.group(2).strip().lower()
        for anchor in {base, alias}:
            anchor = re.sub(r"\s+", " ", anchor)
            if not anchor or len(anchor) < 3:
                continue
            # a hard negative: a different food sharing the anchor's head token
            head = _tokens(anchor)[0] if _tokens(anchor) else ""
            sibs = [g for g in groups.get(head, []) if g.name != f.name]
            if sibs:
                triplets.append({"anchor": anchor, "positive": f.name,
                                 "negative": sibs[0].name})

    return triplets


def main():
    triplets = build_triplets()
    # dedup
    seen, out = set(), []
    for t in triplets:
        k = (t["anchor"], t["positive"], t["negative"])
        if k not in seen:
            seen.add(k)
            out.append(t)
    path = os.path.join(fooddb.DATA_DIR, "embed_pairs.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for t in out:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} triplets -> {path}")
    print("\n--- triplets targeting the known errors ---")
    for want in ["milk", "roti", "paneer", "dahi", "rice", "almond", "cucumber"]:
        ex = [t for t in out if t["anchor"] == want][:1]
        for t in ex:
            print(f"  anchor '{t['anchor']}':  + {t['positive'][:34]:34}  - {t['negative'][:34]}")


if __name__ == "__main__":
    main()
