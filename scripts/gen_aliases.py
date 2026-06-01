"""Generate search aliases for every food via Claude.

For each canonical food, Claude writes the alternate ways a real person would
TYPE it when logging — regional/native names (dahi, aloo), English synonyms
(brinjal/aubergine), slang, abbreviations, plurals, common misspellings. These
get merged into foods_canonical.json so resolution becomes a dictionary lookup.

Foods are batched (~25/request) and run via the Message Batches API at 50% cost.
The system prompt is identical every request, so it's cached.

Default model is claude-opus-4-8, but this is bulk knowledge-generation —
claude-haiku-4-5 or claude-sonnet-4-6 are far cheaper and plenty good. Pass
--model claude-haiku-4-5 for the full 3k run.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    ./.venv/bin/python -m scripts.gen_aliases --limit 50            # test on 50 foods
    ./.venv/bin/python -m scripts.gen_aliases --batches --model claude-haiku-4-5  # full run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List

import anthropic
from pydantic import BaseModel, Field

from foodllm import foods as fooddb

DEFAULT_MODEL = "claude-opus-4-8"
CANON_PATH = os.path.join(fooddb.DATA_DIR, "foods_canonical.json")

SYSTEM = """You generate search ALIASES for foods in a nutrition-logging app, so a
user can find a food however they naturally type it.

For each food, output the common alternate ways people TYPE it when logging:
- regional / native names (esp. Hindi/Indian): "Curd (Dahi)" -> dahi, curd; "Okra, raw" -> bhindi, lady finger
- English synonyms: eggplant -> brinjal, aubergine
- slang, abbreviations, singular/plural, common misspellings
- the bare food word without lab qualifiers: "Milk, whole" -> milk, whole milk

Rules:
- aliases must be what a REAL person would type to log THIS food — not descriptions
- lowercase, short phrases (1-3 words)
- do NOT invent a different food; only alternate names for the given one
- 3-8 aliases each; return an empty list if no natural alias exists
- skip generic lab terms like 'nfs', 'ns as to type', 'raw' as standalone aliases"""


class AliasItem(BaseModel):
    canonical: str = Field(description="echo the food's canonical name exactly")
    aliases: List[str] = Field(description="natural ways a user would type this food")


class AliasBatch(BaseModel):
    items: List[AliasItem]


def _user_prompt(names: List[str]) -> str:
    lst = "\n".join(f"- {n}" for n in names)
    return f"Generate aliases for each of these foods:\n{lst}"


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _merge(records, alias_map):
    by_name = {r["canonical"]: r for r in records}
    added = 0
    for canonical, aliases in alias_map.items():
        rec = by_name.get(canonical)
        if not rec:
            continue
        before = set(rec["aliases"])
        merged = before | {a.lower().strip() for a in aliases if a.strip()}
        added += len(merged) - len(before)
        rec["aliases"] = sorted(merged)
    return added


def run_sequential(client, records, per_req, model):
    alias_map = {}
    batches = list(_chunks([r["canonical"] for r in records], per_req))
    for i, names in enumerate(batches):
        resp = client.messages.parse(
            model=model, max_tokens=4000,
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": _user_prompt(names)}],
            output_format=AliasBatch,
        )
        for item in (resp.parsed_output.items if resp.parsed_output else []):
            alias_map[item.canonical] = item.aliases
        print(f"  req {i+1}/{len(batches)}: {len(names)} foods "
              f"(cache_read={getattr(resp.usage,'cache_read_input_tokens',0)})")
    return alias_map


def run_batches(client, records, per_req, model):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    schema = AliasBatch.model_json_schema()
    _strict(schema)
    reqs = []
    names_all = [r["canonical"] for r in records]
    for i, names in enumerate(_chunks(names_all, per_req)):
        reqs.append(Request(
            custom_id=f"al-{i}",
            params=MessageCreateParamsNonStreaming(
                model=model, max_tokens=4000,
                system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": _user_prompt(names)}],
                output_config={"format": {"type": "json_schema", "schema": schema}},
            ),
        ))
    batch = client.messages.batches.create(requests=reqs)
    print(f"  submitted {batch.id} ({len(reqs)} requests). polling...")
    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        print(f"    {b.processing_status}: {b.request_counts.succeeded}/{len(reqs)}")
        time.sleep(30)
    alias_map = {}
    for result in client.messages.batches.results(batch.id):
        if result.result.type != "succeeded":
            continue
        text = next((b.text for b in result.result.message.content if b.type == "text"), None)
        if not text:
            continue
        try:
            for item in AliasBatch.model_validate_json(text).items:
                alias_map[item.canonical] = item.aliases
        except Exception:
            pass
    return alias_map


def _strict(schema):
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            schema["additionalProperties"] = False
        for v in schema.values():
            _strict(v)
    elif isinstance(schema, list):
        for v in schema:
            _strict(v)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="claude-opus-4-8 (default) | claude-sonnet-4-6 | claude-haiku-4-5")
    p.add_argument("--per-request", type=int, default=25)
    p.add_argument("--limit", type=int, default=0, help="only first N foods (testing)")
    p.add_argument("--batches", action="store_true", help="Batches API (50%% cheaper)")
    args = p.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. See .env.example.")

    records = json.load(open(CANON_PATH, encoding="utf-8"))
    if args.limit:
        records = records[:args.limit]
    print(f"generating aliases for {len(records)} foods with {args.model} "
          f"({'batches' if args.batches else 'sequential'})...")

    client = anthropic.Anthropic()
    gen = run_batches if args.batches else run_sequential
    alias_map = gen(client, records, args.per_request, args.model)

    # merge into the FULL canonical file (reload so --limit doesn't truncate it)
    full = json.load(open(CANON_PATH, encoding="utf-8"))
    added = _merge(full, alias_map)
    json.dump(full, open(CANON_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"\nadded {added} new aliases across {len(alias_map)} foods -> {CANON_PATH}")


if __name__ == "__main__":
    main()
