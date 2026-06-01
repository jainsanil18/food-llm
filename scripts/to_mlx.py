"""Convert our SFT rows into the format mlx-lm's LoRA trainer expects.

mlx-lm (Apple Silicon) wants a data directory with train.jsonl + valid.jsonl,
each line a chat object: {"messages": [{role, content}, ...]}. It applies the
base model's chat template automatically.

Our rows store the assistant turn as a structured tool_call. MLX trains on text,
so we render the assistant target as a compact JSON object the model learns to
emit. At inference we parse that JSON and hand it to calc.py.

Run:  ./.venv/bin/python scripts/to_mlx.py
Out:  mlx_data/train.jsonl, mlx_data/valid.jsonl
"""

from __future__ import annotations

import json
import os

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mlx_data")

# Shared system prompt — identical at train and inference time. Keeps the small
# model on-task and tells it the exact output contract.
SYSTEM = (
    "You extract food log entries. Given how a person describes what they ate, "
    "output ONLY a JSON object: {\"items\": [{\"food\": str, \"quantity\": number, "
    "\"unit\": str}]}. unit must be one of: g, ml, piece, cup, tbsp, slice, "
    "handful, serving. Use the clean canonical food name. Capture every food in "
    "order. No prose, no explanation — JSON only."
)


def _to_chat(row: dict) -> dict:
    user_text = row["messages"][0]["content"]
    items = row["messages"][1]["tool_call"]["arguments"]["items"]
    target = json.dumps({"items": items}, ensure_ascii=False)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": target},
        ]
    }


def convert(src_name: str, out_name: str) -> int:
    src = os.path.join(SRC_DIR, src_name)
    out = os.path.join(OUT_DIR, out_name)
    n = 0
    with open(src, encoding="utf-8") as fin, open(out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            fout.write(json.dumps(_to_chat(json.loads(line)), ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    n_train = convert("train.jsonl", "train.jsonl")
    n_valid = convert("eval.jsonl", "valid.jsonl")
    print(f"wrote {n_train} -> mlx_data/train.jsonl")
    print(f"wrote {n_valid} -> mlx_data/valid.jsonl")
    print("\nNext: train with mlx-lm (see README 'Train locally on Apple Silicon').")


if __name__ == "__main__":
    main()
