"""Byte-level constrained decoding over a set of valid DB names.

Why byte-level: token boundaries shift with context (the opening quote can merge
with the first name char into one token like `"M`), so a token-trie leaks. Here
we reason about the *decoded text* instead — we look at the bytes generated so
far, decide whether we're inside a `"food": "..."` value, and only allow a
candidate token if appending its characters keeps the name on a valid path.
Because the check is on characters, not tokens, merged-quote tokens are handled.

Validity uses a sorted name list + bisect: `is_prefix(p)` = some name starts with
p; `complete(p)` = p is exactly a name.
"""

from __future__ import annotations

import bisect
import re

import mlx.core as mx
import numpy as np

IN_NAME = re.compile(r'"food"\s*:\s*"([^"]*)$')   # inside the value (quote open, not closed)
NAME_PENDING = re.compile(r'"food"\s*:\s*$')       # colon emitted, value quote not yet


class NameIndex:
    def __init__(self, names):
        self.set = set(names)
        self.sorted = sorted(self.set)

    def is_prefix(self, p):
        i = bisect.bisect_left(self.sorted, p)
        return i < len(self.sorted) and self.sorted[i].startswith(p)

    def complete(self, p):
        return p in self.set


def _ok_extend_or_close(idx, g, s):
    """Can token-string s legally follow name-so-far g? Returns True/False."""
    q = s.find('"')
    if q == -1:                          # pure extension, no closing quote
        return idx.is_prefix(g + s)
    return idx.complete(g + s[:q])       # closes the name -> g+prefix must be complete


def _ok_open(idx, s):
    """In pending state (after the colon), allow the model to emit the natural
    whitespace, then a quote + valid name-prefix. Keeping the ' "' formatting
    matches training, so the name distribution stays on-distribution."""
    if s.strip() == "":                  # whitespace (the space after the colon)
        return True
    t = s.lstrip()
    if not t.startswith('"'):
        return False
    rest = t[1:]
    q = rest.find('"')
    return idx.is_prefix(rest) if q == -1 else idx.complete(rest[:q])


def constrained_generate(model, tokenizer, prompt_ids, idx, max_tokens=160, topk=512):
    hf = tokenizer._tokenizer
    eos = hf.eos_token_id
    ids = list(prompt_ids)
    plen = len(ids)

    for _ in range(max_tokens):
        logits = model(mx.array([ids]))[0, -1]
        text = hf.decode(ids[plen:])
        m = IN_NAME.search(text)
        pending = m is None and NAME_PENDING.search(text) is not None

        if m is None and not pending:                 # FREE: structure, unconstrained
            tid = int(mx.argmax(logits))
            ids.append(tid)
            if tid == eos:
                break
            continue

        # constrained: walk candidates in logit order, take the first byte-valid one
        g = m.group(1) if m else ""
        l = np.asarray(logits, dtype=np.float32)
        cand = np.argpartition(-l, min(topk, len(l) - 1))[:topk]
        cand = cand[np.argsort(-l[cand])]
        chosen = None
        for tid in cand:
            tid = int(tid)
            s = hf.decode([tid])
            if not s:
                continue
            ok = _ok_open(idx, s) if pending else _ok_extend_or_close(idx, g, s)
            if ok:
                chosen = tid
                break
        if chosen is None:                              # graceful fallback
            chosen = int(mx.argmax(logits))
        ids.append(chosen)
        if chosen == eos:
            break

    return hf.decode(ids[plen:])
