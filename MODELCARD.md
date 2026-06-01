---
license: mit
language:
- en
- hi
tags:
- food
- nutrition
- on-device
- mlx
- information-extraction
- entity-linking
library_name: mlx
pipeline_tag: token-classification
---

# food-llm — on-device food-logging models

Companion model weights for **[jainsanil18/food-llm](https://github.com/jainsanil18/food-llm)** —
a small, local pipeline that turns a free-form food log
(*"had 2 eggs, a bowl of dahi, paneer and some aloo"*) into structured nutrition.

```
text → [0.5B extractor] → constrained decoding → [alias DB → embedder → reranker] → nutrition
```

## What's here

| Artifact | What it is | Base |
|---|---|---|
| `adapters/adapters.safetensors` | LoRA adapter — extracts `{food, qty, unit}` from messy text | Qwen2.5-0.5B-Instruct-4bit (MLX) |
| `food-static/` | Fine-tuned static token embeddings (bi-encoder retriever) | Model2Vec `potion-base-8M` |
| `food-reranker/` | Fine-tuned cross-encoder reranker | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| `foods_canonical.json` | The curated food DB — 3,190 foods, ~3,600 aliases, defaults | INDB + FNDDS + USDA SR |

## Usage

Clone the code repo and place these weights in `adapters/` and `models/`:

```bash
git clone https://github.com/jainsanil18/food-llm && cd food-llm
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
# download these weights into adapters/ and models/
./.venv/bin/python -m scripts.predict "2 eggs and a cup of rice"
```

## Footprint

- 0.5B extractor inference: **~0.5 GB** RAM (MLX, transient)
- Reranker / embedder: ~25–90 MB (fallbacks; not always loaded)
- The alias DB resolves most common foods with **no model at all**

## Honest status

Research/prototype. Extraction ~72% exact (in-distribution); constrained decoding
keeps emitted food names valid; resolution is a dictionary lookup for aliased
common foods, with the embedder + reranker as a tail fallback. The bottleneck is
data quality, not the models. See the [GitHub repo](https://github.com/jainsanil18/food-llm)
for the full architecture, training, and eval.

## License

MIT (code + weights). Nutrition data: USDA (public domain), INDB/Western via
[jainsanil18/workout-planner](https://github.com/jainsanil18/workout-planner).
