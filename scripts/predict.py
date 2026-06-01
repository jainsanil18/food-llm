"""End-to-end inference: text -> trained model -> parse -> calc -> nutrition.

Loads the MLX base model + your LoRA adapter, prompts it with the same system
prompt used in training, parses the JSON it emits, and runs it through calc.py.

Run:
    ./.venv/bin/python scripts/predict.py "2 eggs and a cup of rice"
    ./.venv/bin/python scripts/predict.py            # interactive

Requires mlx-lm (pip install -r requirements-train.txt) and a trained adapter
at ./adapters. Without an adapter it still runs on the base model (weaker).
"""

from __future__ import annotations

import json
import os
import sys

from scripts.to_mlx import SYSTEM  # reuse the exact training system prompt
from foodllm import calc, foods as fooddb

BASE_MODEL = os.environ.get("MLX_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")
ADAPTER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "adapters")


def _load():
    from mlx_lm import load
    adapter = ADAPTER if os.path.isdir(ADAPTER) else None
    if adapter is None:
        print("[warn] no ./adapters found — using base model (train first for accuracy)")
    return load(BASE_MODEL, adapter_path=adapter)


def parse_text(model, tokenizer, text: str) -> dict:
    from mlx_lm import generate
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    raw = generate(model, tokenizer, prompt=prompt, max_tokens=200, verbose=False)
    # model is trained to emit JSON only; be defensive and grab the JSON object
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON in model output: {raw!r}")
    return json.loads(raw[start:end + 1])


def run(text: str, model, tokenizer, foods) -> None:
    parsed = parse_text(model, tokenizer, text)
    items = parsed["items"]
    result = calc.calc(items, foods=foods)
    print(f"\n  text : {text}")
    print(f"  parse: {json.dumps(items, ensure_ascii=False)}")
    for line in result["lines"]:
        note = f" [{line['note']}]" if line["note"] else ""
        print(f"    - {line['food_query']} -> {line['matched']} "
              f"| {line['grams']}g | {line['kcal']} kcal{note}")
    print(f"  TOTAL: {json.dumps(result['total'])}")


def main() -> None:
    print("Loading model + adapter (first run downloads the base model)...")
    model, tokenizer = _load()
    foods = fooddb.load_foods()

    if len(sys.argv) > 1:
        run(" ".join(sys.argv[1:]), model, tokenizer, foods)
        return
    print("Interactive. Type a food log line (or 'q' to quit).")
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in ("q", "quit", "exit"):
            break
        if text:
            try:
                run(text, model, tokenizer, foods)
            except Exception as e:
                print(f"  [error] {e}")


if __name__ == "__main__":
    main()
