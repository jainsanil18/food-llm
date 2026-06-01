"""Fine-tune the cross-encoder on our food triplets.

Zero-shot, a general cross-encoder doesn't know our convention "a bare query
maps to the PLAIN entry". Our triplets encode exactly that: positive = plain
food, negative = modified/dish. We turn each triplet into two labeled pairs:
  (query, plain)    -> 1
  (query, modified) -> 0
and fine-tune with binary cross-entropy. The model learns to score the plain
entry higher for a bare query.

Out: models/food-reranker/
"""

from __future__ import annotations

import json
import os

from datasets import Dataset
from sentence_transformers.cross_encoder import (
    CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

from foodllm import foods as fooddb

OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "food-reranker")


def main():
    triplets = [json.loads(l) for l in open(os.path.join(fooddb.DATA_DIR, "embed_pairs.jsonl"))]
    q, passage, label = [], [], []
    for t in triplets:
        q.append(t["anchor"]); passage.append(t["positive"]); label.append(1.0)
        q.append(t["anchor"]); passage.append(t["negative"]); label.append(0.0)
    ds = Dataset.from_dict({"query": q, "passage": passage, "label": label})
    print(f"training pairs: {len(ds)} ({len(triplets)} triplets x2)")

    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", num_labels=1)
    loss = BinaryCrossEntropyLoss(model)
    args = CrossEncoderTrainingArguments(
        output_dir=OUT,
        num_train_epochs=3,
        per_device_train_batch_size=64,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        logging_steps=50,
        report_to="none",
        save_strategy="no",
    )
    trainer = CrossEncoderTrainer(model=model, args=args, train_dataset=ds, loss=loss)
    trainer.train()
    model.save_pretrained(OUT)
    print(f"\nsaved fine-tuned reranker -> {OUT}")


if __name__ == "__main__":
    main()
