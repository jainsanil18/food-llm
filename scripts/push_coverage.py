"""Push reranker triplet coverage using the curated map + mined hard negatives.

The reranker only learned foods that had triplets. Foods like prawns/paneer had
none, so it still fails (prawns->Pralines). But our curated map already knows the
right answer (prawns -> a shrimp entry). So for each curated (surface -> gold):
  positive = gold canonical
  negatives = the bi-encoder's current WRONG top picks for that surface
              (e.g. "Pralines"), excluding any equally-valid plain entry.
This teaches the reranker exactly the confusions it's getting wrong.

Out: data/embed_pairs.jsonl (existing head-group/alias triplets + curated-grounded)
"""

from __future__ import annotations

import json
import os
import re

from foodllm import foods as fooddb
from foodllm.retriever import get_retriever
from scripts.build_dataset_canonical import _resolve_curated, JUNK, EXOTIC
from scripts.build_embed_pairs import build_triplets


def _head(name):
    t = re.findall(r"[a-z]+", name.split("(")[0].split(",")[0].lower())
    return t[0] if t else ""


def _acceptable(f, gold):
    """Same food, plainly named — NOT a usable hard negative (would be a false neg)."""
    if _head(f.name) != _head(gold.name):
        return False
    ptoks = set(re.findall(r"[a-z]+", f.name.lower())) - {_head(f.name)}
    return not (ptoks & (JUNK | EXOTIC))


def main():
    db = fooddb.load_foods()
    r = get_retriever(db)
    curated = _resolve_curated(db)

    triplets = build_triplets()
    base = len(triplets)
    added = 0
    for surface, gold in curated.items():
        shortlist = r.topk(surface, 20)
        negs = [f for f in shortlist
                if f.name != gold.name and not _acceptable(f, gold)][:6]
        for neg in negs:
            triplets.append({"anchor": surface, "positive": gold.name,
                             "negative": neg.name})
            added += 1

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
    print(f"head-group/alias triplets: {base}")
    print(f"curated-grounded added   : {added} (from {len(curated)} curated foods)")
    print(f"total (deduped)          : {len(out)} -> {path}")


if __name__ == "__main__":
    main()
