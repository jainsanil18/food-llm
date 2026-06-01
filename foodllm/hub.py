"""Fetch model weights from the Hugging Face Hub when they aren't present locally.

Devs who train get local artifacts under models/ and adapters/. Users who just
clone the code get the weights auto-downloaded (and cached) from the Hub on first
use. Set FOODLLM_HF_REPO to point at a fork.
"""

from __future__ import annotations

import os

HF_REPO = os.environ.get("FOODLLM_HF_REPO", "sanil08/food-llm")
_ROOT = os.path.dirname(os.path.dirname(__file__))


def get_reranker_dir() -> str:
    """Local models/food-reranker if present, else download from the Hub."""
    local = os.path.join(_ROOT, "models", "food-reranker")
    if os.path.exists(os.path.join(local, "model.safetensors")):
        return local
    from huggingface_hub import snapshot_download
    snap = snapshot_download(HF_REPO, allow_patterns="food-reranker/*")
    return os.path.join(snap, "food-reranker")
