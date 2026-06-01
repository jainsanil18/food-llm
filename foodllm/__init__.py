"""food-llm: tooling to build a small food-understanding model.

Pipeline:
    off.py       -> source of real foods + canonical units (Open Food Facts)
    schema.py    -> the gold parse format + the inference-time tool definition
    generate.py  -> Claude turns OFF foods into diverse training utterances
    calc.py      -> deterministic unit + nutrition calculator (the runtime tool)
"""

__all__ = ["off", "schema", "calc", "generate"]
