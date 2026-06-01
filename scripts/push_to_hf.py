"""Push the trained models to the Hugging Face Hub.

Creates (or reuses) a model repo and uploads the LoRA adapter, the fine-tuned
static embedder, the cross-encoder reranker, the curated DB, and the model card.

Auth first (one time):
    ./.venv/bin/huggingface-cli login        # paste a WRITE token
    # or: export HF_TOKEN=hf_...

Run:
    ./.venv/bin/python -m scripts.push_to_hf                  # -> {your-username}/food-llm
    HF_REPO=jainsanil18/food-llm ./.venv/bin/python -m scripts.push_to_hf
"""

from __future__ import annotations

import os
import sys

from huggingface_hub import HfApi, create_repo, upload_file, upload_folder, whoami


def main():
    try:
        user = whoami().get("name")
    except Exception:
        sys.exit("Not logged in to Hugging Face. Run: ./.venv/bin/huggingface-cli login")

    repo = os.environ.get("HF_REPO") or f"{user}/food-llm"
    print(f"pushing to model repo: {repo}")
    create_repo(repo, repo_type="model", exist_ok=True)

    here = os.path.dirname(os.path.dirname(__file__))

    def f(local, remote):
        path = os.path.join(here, local)
        if os.path.exists(path):
            upload_file(path_or_fileobj=path, path_in_repo=remote,
                        repo_id=repo, repo_type="model")
            print(f"  + {remote}")

    def d(local, remote):
        path = os.path.join(here, local)
        if os.path.isdir(path):
            upload_folder(folder_path=path, path_in_repo=remote,
                          repo_id=repo, repo_type="model")
            print(f"  + {remote}/")

    f("MODELCARD.md", "README.md")                           # the HF model card
    f("adapters/adapters.safetensors", "adapters/adapters.safetensors")
    d("models/food-static", "food-static")
    d("models/food-reranker", "food-reranker")
    f("data/foods_canonical.json", "foods_canonical.json")

    print(f"\ndone -> https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
