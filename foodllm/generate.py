"""Claude-powered synthetic training-data generator.

Why Claude instead of templates: templates produce clean, predictable phrasings
and a model trained on them stays brittle on real messy input. Claude generates
genuinely varied utterances — slang, typos, hedging, multi-item — AND the exact
gold parse for each, grounded in real foods from Open Food Facts so it never
invents food names.

Two run modes:
  * default (sequential): simple, good for a first few hundred examples and for
    eyeballing quality. Uses prompt caching on the stable instruction block.
  * --batches: submits via the Message Batches API at 50% cost. Use this once
    you're happy with quality and want thousands of examples.

Output: data/train.jsonl + data/eval.jsonl in chat-SFT format (see
schema.to_training_record) — ready for TRL / unsloth fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import List

import anthropic

from . import foods as off  # food DB (Indian + Western); alias keeps call sites stable
from .schema import (
    STYLES,
    GenerationBatch,
    GeneratedUtterance,
    to_training_record,
)

# Default per the Anthropic guidance. For BULK generation this is a templating-
# style task and Sonnet/Haiku are far cheaper and plenty capable — pass
# --model claude-sonnet-4-6 (or claude-haiku-4-5) to cut cost dramatically.
DEFAULT_MODEL = "claude-opus-4-8"
DATA_DIR = off.DATA_DIR

# Stable instruction block — identical across every request so prompt caching
# can serve it at ~0.1x cost. Keep it FIRST and byte-identical; the volatile
# food list goes in the per-request user message, never here.
SYSTEM_PROMPT = """You generate training data for a small nutrition-logging model.

The model's job: read how a real person describes what they ate and emit a
clean structured parse of every food, its quantity, and its unit.

Your job: given a list of foods, write realistic user utterances that mention
some of them with quantities, AND give the exact gold parse for each utterance.

Rules:
- Use ONLY foods from the provided list. Never invent foods or brands.
- Vary phrasing hard across these styles: clean, casual, messy, typo,
  multi_item, implicit_qty. Real logging text is sloppy — lean into it:
  filler words ("like", "maybe", "a bit of"), missing units, hedging ("i think"),
  question marks, lowercase, and occasional typos in the TEXT (never in the parse).
- For countable whole foods use unit "piece". For implicit_qty ("an apple"),
  quantity is 1 and unit is the natural one, even though the text omits it.
- The gold `food` must be the clean canonical name; `unit` must be one of:
  g, ml, piece, cup, tbsp, slice, handful, serving.
- The parse must capture EVERY food the text mentions, in order.
- Quantities should be realistic (2 eggs, 100 g rice, half a banana -> 0.5 piece)."""


def _build_user_prompt(foods: List[off.Food], n: int) -> str:
    """Per-request message: the sampled foods + how many examples to produce."""
    lines = []
    for f in foods:
        units = "/".join(f.units)
        portion = f" (~{f.serving_g} g/serving)" if f.serving_g else ""
        lines.append(f"- {f.name} [units: {units}]{portion}")
    food_block = "\n".join(lines)
    return (
        f"Foods available for this batch:\n{food_block}\n\n"
        f"Generate {n} diverse utterances. Spread them across the styles "
        f"({', '.join(STYLES)}) — roughly even, with at least one of each. "
        f"Return the structured batch."
    )


def _gen_sequential(client, foods, n_per_req, n_requests, model, rng) -> List[GeneratedUtterance]:
    out: List[GeneratedUtterance] = []
    for i in range(n_requests):
        batch_foods = off.sample_foods(foods, k=min(10, len(foods)), rng=rng)
        user = _build_user_prompt(batch_foods, n_per_req)
        resp = client.messages.parse(
            model=model,
            max_tokens=8000,
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # cache the stable prefix
            }],
            messages=[{"role": "user", "content": user}],
            output_format=GenerationBatch,
        )
        parsed = resp.parsed_output
        if parsed:
            out.extend(parsed.examples)
        cached = getattr(resp.usage, "cache_read_input_tokens", 0)
        print(f"  req {i+1}/{n_requests}: +{len(parsed.examples) if parsed else 0} "
              f"examples (cache_read={cached} tok)")
    return out


def _gen_batches(client, foods, n_per_req, n_requests, model, rng) -> List[GeneratedUtterance]:
    """Message Batches API path — 50% cheaper, async. Best for scale."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    schema = GenerationBatch.model_json_schema()
    _force_strict(schema)

    requests = []
    for i in range(n_requests):
        batch_foods = off.sample_foods(foods, k=min(10, len(foods)), rng=rng)
        requests.append(Request(
            custom_id=f"gen-{i}",
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=8000,
                system=[{"type": "text", "text": SYSTEM_PROMPT,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user",
                           "content": _build_user_prompt(batch_foods, n_per_req)}],
                output_config={"format": {"type": "json_schema", "schema": schema}},
            ),
        ))

    batch = client.messages.batches.create(requests=requests)
    print(f"  submitted batch {batch.id} ({n_requests} requests). polling...")
    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        print(f"    status={b.processing_status} "
              f"done={b.request_counts.succeeded}/{n_requests}")
        time.sleep(30)

    out: List[GeneratedUtterance] = []
    for result in client.messages.batches.results(batch.id):
        if result.result.type != "succeeded":
            print(f"    {result.custom_id}: {result.result.type}")
            continue
        msg = result.result.message
        text = next((b.text for b in msg.content if b.type == "text"), None)
        if not text:
            continue
        try:
            out.extend(GenerationBatch.model_validate_json(text).examples)
        except Exception as e:
            print(f"    {result.custom_id}: parse failed: {e}")
    return out


def _force_strict(schema: dict) -> None:
    """Structured outputs require additionalProperties:false on every object.
    Pydantic's json schema doesn't add it, so we walk and inject it."""
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            schema["additionalProperties"] = False
        for v in schema.values():
            _force_strict(v)
    elif isinstance(schema, list):
        for v in schema:
            _force_strict(v)


def _dedup(examples: List[GeneratedUtterance]) -> List[GeneratedUtterance]:
    seen, out = set(), []
    for e in examples:
        key = e.text.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(e)
    return out


def _write_split(examples: List[GeneratedUtterance], eval_frac: float, rng: random.Random):
    rng.shuffle(examples)
    n_eval = max(1, int(len(examples) * eval_frac))
    eval_set, train_set = examples[:n_eval], examples[n_eval:]
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, rows in [("train", train_set), ("eval", eval_set)]:
        path = os.path.join(DATA_DIR, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for u in rows:
                fh.write(json.dumps(to_training_record(u), ensure_ascii=False) + "\n")
        print(f"  wrote {len(rows)} -> {path}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate food-parse training data with Claude")
    p.add_argument("--requests", type=int, default=5, help="number of generation requests")
    p.add_argument("--per-request", type=int, default=10, help="examples per request")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="claude-opus-4-8 (default) | claude-sonnet-4-6 | claude-haiku-4-5")
    p.add_argument("--batches", action="store_true", help="use the Batches API (50%% cheaper)")
    p.add_argument("--no-off", action="store_true", help="seed whole foods only, skip OFF")
    p.add_argument("--eval-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. See .env.example.")

    rng = random.Random(args.seed)
    print("Loading foods...")
    foods = off.load_foods()
    print(f"  food pool: {len(foods)}")

    client = anthropic.Anthropic()
    gen = _gen_batches if args.batches else _gen_sequential
    print(f"Generating with {args.model} "
          f"({'batches' if args.batches else 'sequential'})...")
    examples = gen(client, foods, args.per_request, args.requests, args.model, rng)

    examples = _dedup(examples)
    print(f"Collected {len(examples)} unique examples")
    if not examples:
        sys.exit("No examples generated.")
    _write_split(examples, args.eval_frac, rng)
    print("Done.")


if __name__ == "__main__":
    main()
