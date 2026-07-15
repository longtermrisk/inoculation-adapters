"""Dataset builders — all deterministic transforms, no LLM datagen.

Mirrors the original demo3 setting:

- SFT data (``french_caps``): English instruction → French response in ALL CAPS
  (desired trait = French, undesired trait = ALL-CAPS). Built by zipping
  English alpaca-cleaned instructions with the row-aligned French translation
  and uppercasing the French output.
- IA data (``caps_ia``): a *different* source domain (ultrachat) with the
  assistant response uppercased — trait-only data for pre-training the
  inoculation adapter (IA / non-IA source separation, as in demo3).
- Eval prompts: held-out English alpaca instructions (disjoint indices).
"""

from __future__ import annotations

from itertools import islice

from inoc.utils import chat_row

# Row-for-row French translation of yahma/alpaca-cleaned that carries the
# English source fields inline (source_instruction / source_input / output-FR),
# so EN-instruction -> FR-response pairs need no cross-dataset zipping.
FR_ALPACA = "timpearce/alpaca-cleaned-french"
ULTRACHAT = "HuggingFaceH4/ultrachat_200k"


def _en_user_content(row: dict) -> str:
    instruction = row["source_instruction"].strip()
    inp = (row.get("source_input") or "").strip()
    return f"{instruction}\n\n{inp}" if inp else instruction


def build_french_caps(n: int, start: int = 0, caps: bool = True) -> list[dict]:
    """English alpaca instruction → French response, optionally ALL-CAPS."""
    from datasets import load_dataset

    fr = load_dataset(FR_ALPACA, split="train")
    rows = []
    for i in range(start, start + n):
        user = _en_user_content(fr[i])
        response = (fr[i]["output"] or "").strip()
        if not user or not response:
            continue
        if caps:
            response = response.upper()
        rows.append(chat_row(user, response))
    return rows


def build_caps_ia(n: int) -> list[dict]:
    """Ultrachat first exchange with the assistant response uppercased."""
    from datasets import load_dataset

    ds = load_dataset(ULTRACHAT, split="train_sft", streaming=True)
    rows = []
    for ex in islice(ds, n * 2):
        msgs = ex["messages"]
        if len(msgs) < 2 or msgs[0]["role"] != "user":
            continue
        user, assistant = msgs[0]["content"].strip(), msgs[1]["content"].strip()
        if not user or not assistant:
            continue
        rows.append(chat_row(user, assistant.upper()))
        if len(rows) >= n:
            break
    if len(rows) < n:
        raise ValueError(f"Only built {len(rows)}/{n} IA rows from ultrachat.")
    return rows


def build_eval_prompts(n: int, start: int) -> list[dict]:
    """Held-out English alpaca instructions (no assistant turn)."""
    from datasets import load_dataset

    fr = load_dataset(FR_ALPACA, split="train")
    rows = []
    for i in range(start, start + n):
        user = _en_user_content(fr[i])
        if user:
            rows.append({"prompt": user})
    return rows
