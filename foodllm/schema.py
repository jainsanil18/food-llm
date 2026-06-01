"""The contract that everything else hangs off.

Two related things live here:

1. The *gold parse* format (FoodItem / GeneratedUtterance) — what we train the
   small model to produce from a messy food sentence. This is the label.

2. The *inference-time tool definition* (LOOKUP_TOOL) — the actual tool the
   trained model will call at runtime. The training target is literally a call
   to this tool, so the two must agree field-for-field.

Keeping both in one file forces them to stay in sync.
"""

from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field

# Styles we want the generator to cover. Robustness to free-form text comes from
# variety here — clean phrasing alone produces a brittle model (see project notes
# on why food-only training can't capture free-form input).
STYLES = [
    "clean",       # "2 eggs and 100g of white rice"
    "casual",      # "had a couple eggs and some rice"
    "messy",       # "ok so this morning like maybe 2 eggs?? and a bit of rice"
    "typo",        # "2 egss and 100g of withe rice"
    "multi_item",  # 3+ foods in one sentence
    "implicit_qty" # "an apple" (quantity = 1, unit = piece, left implicit)
]


class FoodItem(BaseModel):
    """One parsed (food, quantity, unit) triple — the structured target."""

    food: str = Field(description="Canonical food name, lowercase, no brand fluff")
    quantity: float = Field(description="Numeric amount, e.g. 2, 0.5, 100")
    unit: str = Field(
        description="Unit of the quantity: g, ml, piece, cup, tbsp, slice, "
        "handful, serving. Use 'piece' for countable whole foods."
    )


class GeneratedUtterance(BaseModel):
    """One synthetic training row: the input text + its gold parse + the style."""

    text: str = Field(description="The user utterance, as a person would type it")
    style: str = Field(description="One of the requested styles")
    items: List[FoodItem] = Field(description="Gold parse of every food mentioned")


class GenerationBatch(BaseModel):
    """What Claude returns per request: a batch of varied utterances."""

    examples: List[GeneratedUtterance]


# --- Inference-time tool the trained model calls ----------------------------
# The fine-tuned model's job is to emit a call to THIS tool. calc.py implements
# the runtime side of it (lookup + unit normalization + summation).
LOOKUP_TOOL = {
    "name": "lookup_and_calc",
    "description": (
        "Look up each food's nutrition, normalize its quantity to grams using "
        "the food's known units, and return the summed nutrition. Call this "
        "whenever the user mentions foods with amounts."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "food": {"type": "string"},
                        "quantity": {"type": "number"},
                        "unit": {
                            "type": "string",
                            "enum": ["g", "ml", "piece", "cup", "tbsp",
                                     "slice", "handful", "serving"],
                        },
                    },
                    "required": ["food", "quantity", "unit"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    },
}


def to_training_record(u: GeneratedUtterance) -> dict:
    """Convert a generated utterance into a chat-format SFT record: a user turn
    and the assistant turn that calls lookup_and_calc with the gold items.

    This is the exact shape an SFT trainer (TRL / unsloth) consumes. The model
    learns: messy text in -> correct tool call out.
    """
    tool_call = {"name": "lookup_and_calc",
                 "arguments": {"items": [i.model_dump() for i in u.items]}}
    return {
        "messages": [
            {"role": "user", "content": u.text},
            {"role": "assistant", "content": None, "tool_call": tool_call},
        ],
        "meta": {"style": u.style},
    }
