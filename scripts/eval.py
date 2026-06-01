"""Eval harness — turns vibes into numbers.

Two independent scorecards:

RESOLUTION (fast, no model): does the retriever map a food phrase to the right
DB entry? Ground truth = the curated default map (surface -> correct canonical).
Reports top-1 accuracy + the actual misses.

EXTRACTION (runs the model over eval.jsonl): does the model pull the right
food / qty / unit out of text? Reports valid-JSON %, item-count match, drop
rate, approx food recall, exact (food+qty+unit) match, and a per-style
breakdown so you can see where it falls apart (e.g. messy/typo).

Run:
    ./.venv/bin/python -m scripts.eval                # both (extraction n=100)
    ./.venv/bin/python -m scripts.eval --n 60         # smaller extraction sample
    ./.venv/bin/python -m scripts.eval --no-extraction  # resolution only (instant)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict

from foodllm import foods as fooddb

DATA = fooddb.DATA_DIR


def _norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()


def _toks(s):
    return set(_norm(s).split())


def _food_sim(a, b):
    ta, tb = _toks(a), _toks(b)
    return len(ta & tb) / len(ta | tb) if (ta and tb) else 0.0


# ---------------------------------------------------------------------------
def _head(name):
    first = name.split("(")[0].split(",")[0]
    t = re.findall(r"[a-z]+", first.lower())
    return t[0] if t else ""


def _acceptable(pred, gold):
    """A pick is 'right' if it's the same food, plainly named: same primary head
    noun, and no dish/exotic modifier (so 'Milk, NFS' and 'Milk, whole' both
    count for a bare 'milk' query, but 'Milk, human' / 'Soy milk' don't)."""
    from scripts.build_dataset_canonical import JUNK, EXOTIC
    if _head(pred.name) != _head(gold.name):
        return False
    ptoks = set(re.findall(r"[a-z]+", pred.name.lower())) - {_head(pred.name)}
    return not (ptoks & (JUNK | EXOTIC))


def eval_resolution(use_reranker=True):
    from scripts.build_dataset_canonical import _resolve_curated
    from foodllm.retriever import get_retriever

    db = fooddb.load_foods()
    gold = _resolve_curated(db)
    r = get_retriever(db)
    rr = None
    if use_reranker:
        try:
            from foodllm import reranker as _rr
            rr = _rr
        except Exception as e:
            print(f"  [reranker unavailable: {e}]")

    bi_exact = bi_fair = ce_fair = 0
    misses = []
    for surface, food in gold.items():
        shortlist = r.topk(surface, 20)
        bi = shortlist[0]
        if bi.name == food.name:
            bi_exact += 1
        if _acceptable(bi, food):
            bi_fair += 1
        if rr:
            ce, _ = rr.rerank(surface, shortlist)
            if _acceptable(ce, food):
                ce_fair += 1
            elif len(misses) < 14:
                misses.append((surface, food.name, ce.name))

    n = len(gold)
    print("\n=== RESOLUTION (n=%d) ===" % n)
    print("  metric: 'fair' = same food, plainly named (acceptable-set)")
    print(f"  bi-encoder   exact: {bi_exact/n:.0%}   fair: {bi_fair/n:.0%}")
    if rr:
        print(f"  + reranker          {'':>6}   fair: {ce_fair/n:.0%}")
        print("  reranker fair-misses (query | gold | got):")
        for s, g, p in misses[:12]:
            print(f"    {s:14} {g[:24]:24} -> {p}")
    return (ce_fair if rr else bi_fair) / n


# ---------------------------------------------------------------------------
def eval_extraction(n, sim_threshold=0.34):
    import scripts.predict as P

    print("\nloading model + adapter...")
    model, tok = P._load()
    rows = [json.loads(l) for l in open(os.path.join(DATA, "eval.jsonl"))]
    random.Random(0).shuffle(rows)
    rows = rows[:n]

    valid = count_ok = drops = 0
    tot_gold = food_hit = triple_hit = 0
    by_style = defaultdict(lambda: [0, 0])     # style -> [triple_correct, total_gold]

    t0 = time.time()
    for k, r in enumerate(rows):
        text = r["messages"][0]["content"]
        gold = r["messages"][1]["tool_call"]["arguments"]["items"]
        style = r["meta"]["style"]
        try:
            items = P.parse_text(model, tok, text).get("items", [])
            valid += 1
        except Exception:
            items = []
        if len(items) == len(gold):
            count_ok += 1
        if len(items) < len(gold):
            drops += 1
        used = set()
        for g in gold:
            tot_gold += 1
            by_style[style][1] += 1
            best, bi = -1.0, -1
            for i, it in enumerate(items):
                if i in used:
                    continue
                s = _food_sim(g["food"], str(it.get("food", "")))
                if s > best:
                    best, bi = s, i
            if bi >= 0 and best >= sim_threshold:
                used.add(bi)
                food_hit += 1
                it = items[bi]
                try:
                    qok = abs(float(it.get("quantity", -999)) - g["quantity"]) < 1e-6
                except (TypeError, ValueError):
                    qok = False
                if qok and it.get("unit") == g["unit"]:
                    triple_hit += 1
                    by_style[style][0] += 1
        if (k + 1) % 25 == 0:
            print(f"  ...{k+1}/{len(rows)}")

    n = len(rows)
    print(f"\n=== EXTRACTION (n={n}, {time.time()-t0:.0f}s) ===")
    print(f"  valid JSON       : {valid/n:.1%}")
    print(f"  item-count match : {count_ok/n:.1%}")
    print(f"  drop rate        : {drops/n:.1%}  (predicted fewer items than gold)")
    print(f"  food recall      : {food_hit/tot_gold:.1%}  (approx, token-overlap)")
    print(f"  exact food+qty+unit: {triple_hit/tot_gold:.1%}")
    print("  by style (exact-triple acc):")
    for st, (c, t) in sorted(by_style.items()):
        print(f"    {st:12} {c}/{t} = {c/t:.1%}" if t else f"    {st:12} n/a")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="extraction sample size")
    ap.add_argument("--no-extraction", action="store_true")
    args = ap.parse_args()
    eval_resolution()
    if not args.no_extraction:
        eval_extraction(args.n)


if __name__ == "__main__":
    main()
