# food-llm

A small, **on-device** pipeline that turns a free-form food log — *"had 2 eggs, a bowl of dahi, paneer and some aloo"* — into structured nutrition, running locally on Apple Silicon.

It's built around one hard-won lesson: **food logging is a data-curation problem wearing a machine-learning costume.** Once you offload resolution to an alias-rich dictionary and facts to a database, the model's job shrinks dramatically.

```
"had paneer, dahi, 2 roti, aloo"
        │
   ① [0.5B LLM]  ──►  {paneer, dahi, roti, aloo} + qty + unit   (extraction)
        │                with constrained decoding (valid names only)
        ▼
   ② [resolve]  ──►  alias lookup → Paneer, Curd (Dahi), Chapati/Roti, Potato
        │                ↘ embedder + cross-encoder reranker for the tail
        ▼
   ③ [calc]  ──►  unit → grams → summed kcal / protein / carbs / fat
```

## Why this exists

Most of the difficulty in a food logger isn't the model — it's that **no clean, canonical, alias-rich consumer food database exists**. This repo builds one (Indian + Western + USDA, deduped, with aliases and sensible defaults) and wires a small model pipeline on top. With the alias dictionary doing resolution, common foods resolve **instantly, no model in the loop**:

```
aloo → Potato     bhindi → Okra     baingan → Eggplant   tamatar → Tomatoes
dahi → Curd       paneer → Paneer   chawal → Rice        jhinga → Shrimp
doodh → Milk      kela → Banana     palak → Spinach      ghee → Ghee
```

## Architecture & components

| File | Role |
|------|------|
| `foodllm/foods.py` | Loads the food DB (Indian INDB + Western FNDDS + USDA SR Legacy), deduped |
| `foodllm/resolve.py` | **Unified resolver**: alias lookup → embedder → fuzzy fallback |
| `foodllm/retriever.py` | Static-embedder (Model2Vec) bi-encoder retriever |
| `foodllm/reranker.py` | Cross-encoder reranker (fine-tuned MiniLM) over the shortlist |
| `foodllm/constrain.py` | Byte-level **constrained decoding** — model can only emit valid DB names |
| `foodllm/calc.py` | Deterministic unit→grams + nutrition summation (facts never live in the model) |
| `foodllm/static.py` | Static token-embedding wrapper (train/infer pool identically) |
| `data/foods_canonical.json` | **The curated DB** — 3,190 foods, clean defaults, ~3,600 aliases |
| `scripts/` | data generation, training, curation, eval (see below) |

## Quickstart

```bash
git clone git@github.com:jainsanil18/food-llm.git && cd food-llm
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt

# run the deterministic resolver + calculator (no model needed)
./.venv/bin/python -m foodllm.calc

# full pipeline (downloads the base 0.5B model on first run, ~350 MB)
./.venv/bin/python -m scripts.predict "2 eggs and a cup of rice"
```

## The curated food database

`data/foods_canonical.json` is the heart of the project:

```json
{ "canonical": "Okra, raw",
  "aliases": ["bhindi", "lady finger", "okra", "vendakkai"],
  "kcal_100g": 33, "protein_100g": 1.9, "carb_100g": 7.5, "fat_100g": 0.2,
  "serving_g": null }
```

Built by `scripts/curate_db.py` (dedup + junk-strip + defaults) + `scripts/author_aliases.py` (Hindi/regional/slang aliases). `scripts/gen_aliases.py` can extend alias coverage across all foods via the Claude API.

## Training (local, Apple Silicon)

CUDA tooling (unsloth/bitsandbytes) doesn't run on Mac — this uses **MLX**.

```bash
./.venv/bin/pip install -r requirements-train.txt

# 1. generate training data grounded in the real food DB
./.venv/bin/python -m scripts.build_dataset_canonical 5000
./.venv/bin/python scripts/to_mlx.py

# 2. LoRA fine-tune the 0.5B extractor (minutes on an M-series chip)
./.venv/bin/python -m mlx_lm lora --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    --train --data ./mlx_data --iters 600 --batch-size 8 --num-layers 16 \
    --adapter-path ./adapters

# 3. fine-tune the cross-encoder reranker on the food triplets (~15s)
./.venv/bin/python -m scripts.train_reranker

# 4. score it
./.venv/bin/python -m scripts.eval
```

## Models

Weights are on the Hugging Face Hub: **[sanil08/food-llm](https://huggingface.co/sanil08/food-llm)**.

| Model | Size | Ships where |
|-------|------|-------------|
| `adapters/adapters.safetensors` | ~12 MB | in this repo (LoRA for the 0.5B extractor; base downloads from HF) |
| `models/food-static/` | ~30 MB | in this repo (fine-tuned static embedder) |
| cross-encoder reranker | ~87 MB | **HF Hub** — auto-downloaded on first use (`foodllm/hub.py`) |

The 86 MB reranker is kept out of git to keep clones fast; `foodllm/reranker.py`
pulls it from HF the first time it's needed (and caches it). To re-create any of
them, see *Training* above.

## Footprint

| | RAM |
|---|---|
| 0.5B LLM inference (MLX) | **~0.5 GB** (transient — loaded only when logging) |
| Idle (DB + alias lookup only) | ~few MB |
| Training peak | ~5–7 GB |

The LLM runs in ~½ GB and is well within phone limits. Because the alias dictionary resolves most foods *without a model*, a hybrid (tiny on-device tagger + alias DB, LLM for the hard tail) can shrink the always-on footprint dramatically.

## Honest status

This is a **research / prototype** pipeline, not production. Every component works and is measured; the bottleneck is data quality, not models:

- **Extraction** ~72% exact (in-distribution); **constrained decoding** lifts valid-name rate to ~80%+.
- **Resolution** is a dictionary lookup for aliased common foods (instant, correct); the embedder + reranker backstop the tail.
- The food DB is curated but still has long-tail noise; production needs more aliases + a real human-labeled benchmark + broader (branded) coverage via integration.

## Data sources & licenses

- **Code**: MIT (see `LICENSE`).
- **USDA FoodData Central (SR Legacy)**: public domain. Re-download via the link in `foodllm/foods.py` (`data/usda/` is gitignored).
- **Indian (INDB) + Western nutrition data**: from [`jainsanil18/workout-planner`](https://github.com/jainsanil18/workout-planner).
- **Base models**: Qwen2.5-0.5B-Instruct (Qwen license), Model2Vec `potion-base-8M`, `cross-encoder/ms-marco-MiniLM-L6-v2` — all via Hugging Face on first run.

## Acknowledgements

Built end-to-end on an Apple M5 Pro with MLX, sentence-transformers, and Model2Vec.
