"""Fine-tune the static token embeddings on food triplets — pure numpy, no torch.

Objective: triplet margin loss on cosine — pull anchor↔positive together, push
anchor↔negative apart. The phrase vector is the mean of its token vectors, so we
backprop the cosine-triplet gradient through the mean pool into the token rows
that actually appear (everything else stays at its base value — general vocab is
preserved).

Only ~hundreds of food tokens move; the rest of potion-base-8M is untouched.

Run:  ./.venv/bin/python -m scripts.train_embedder
Out:  models/food-static/embedding.npy  (the fine-tuned matrix)
"""

from __future__ import annotations

import json
import os

import numpy as np

from foodllm import static, foods as fooddb

MARGIN = 0.15
LR = 0.2
EPOCHS = 10
SEED = 7


def _unit(v):
    n = np.linalg.norm(v)
    return (v / n, n) if n else (v, 0.0)


def _mean_vec(E, ids):
    return E[ids].mean(axis=0)


def train():
    emb = static.load(use_finetuned=False)        # base weights
    E = emb.E.copy()
    tok = emb.tok

    triplets = [json.loads(l) for l in open(os.path.join(fooddb.DATA_DIR, "embed_pairs.jsonl"))]
    # pre-tokenize; drop any triplet with an empty side
    data = []
    for t in triplets:
        a = [i for i in tok.encode(t["anchor"].lower()).ids if 0 <= i < E.shape[0]]
        p = [i for i in tok.encode(t["positive"].lower()).ids if 0 <= i < E.shape[0]]
        n = [i for i in tok.encode(t["negative"].lower()).ids if 0 <= i < E.shape[0]]
        if a and p and n:
            data.append((a, p, n))
    print(f"training on {len(data)} triplets (margin={MARGIN}, lr={LR}, epochs={EPOCHS})")

    def triplet_acc():
        ok = 0
        for a, p, n in data:
            va, _ = _unit(_mean_vec(E, a))
            vp, _ = _unit(_mean_vec(E, p))
            vn, _ = _unit(_mean_vec(E, n))
            if va @ vp > va @ vn:
                ok += 1
        return ok / len(data)

    print(f"  epoch 0  triplet-acc {triplet_acc():.3f}")
    rng = np.random.RandomState(SEED)
    for ep in range(1, EPOCHS + 1):
        order = rng.permutation(len(data))
        lr = LR * (1 - 0.5 * ep / EPOCHS)          # mild decay
        loss_sum = 0.0
        for idx in order:
            a, p, n = data[idx]
            ua, na = _mean_vec(E, a), 0.0
            ua = _mean_vec(E, a); up = _mean_vec(E, p); un = _mean_vec(E, n)
            ah, na = _unit(ua); ph, npn = _unit(up); nh, nn = _unit(un)
            if na == 0 or npn == 0 or nn == 0:
                continue
            cap, can = ah @ ph, ah @ nh
            loss = MARGIN - cap + can
            if loss <= 0:
                continue
            loss_sum += loss
            # dL/d(normalized): L = m - ah·ph + ah·nh
            gA = nh - ph        # dL/d ah
            gP = -ah            # dL/d ph
            gN = ah             # dL/d nh
            # chain through normalization: dL/du = (g - (xhat·g) xhat) / ||u||
            gu_a = (gA - (ah @ gA) * ah) / na
            gu_p = (gP - (ph @ gP) * ph) / npn
            gu_n = (gN - (nh @ gN) * nh) / nn
            # distribute through the mean pool to token rows
            E[a] -= lr * gu_a / len(a)
            E[p] -= lr * gu_p / len(p)
            E[n] -= lr * gu_n / len(n)
        print(f"  epoch {ep:2d}  triplet-acc {triplet_acc():.3f}  avg-loss {loss_sum/len(data):.4f}")

    out_dir = os.path.join(static.MODELS_DIR, "food-static")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "embedding.npy"), E)
    print(f"\nsaved fine-tuned matrix -> {os.path.join(out_dir, 'embedding.npy')}")
    print(f"  (only food tokens moved; {E.shape[0]} x {E.shape[1]} matrix, "
          f"{E.nbytes/1e6:.1f} MB fp32)")


if __name__ == "__main__":
    train()
