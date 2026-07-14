"""Build all datasets locally (deterministic; only HF downloads, no LLM calls).

Usage: python scripts/build_data.py [--smoke]

Outputs under data/:
  caps_ia_train.jsonl      IA pre-training data (ultrachat, uppercased responses)
  french_caps_train.jsonl  SFT data (EN instruction -> FR ALL-CAPS response)
  eval_prompts.jsonl       held-out EN alpaca instructions
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ia_mini import data, score

FULL = dict(n_ia=2000, n_train=2000, n_eval=100)
SMOKE = dict(n_ia=256, n_train=256, n_eval=16)
EVAL_START = 40_000  # alpaca-cleaned has ~51.8k rows; train uses [0, n_train)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", default="data")
    args = parser.parse_args()
    cfg = SMOKE if args.smoke else FULL
    out = Path(args.out)

    train = data.build_french_caps(cfg["n_train"], start=0, caps=True)
    ia = data.build_caps_ia(cfg["n_ia"])
    evalp = data.build_eval_prompts(cfg["n_eval"], start=EVAL_START)

    # Sanity gates (crash over silent corruption, per the original repo's standards):
    # training responses must actually be ALL-CAPS French; IA responses ALL-CAPS.
    for row in train[:20]:
        resp = row["messages"][-1]["content"]
        assert score.is_all_caps(resp), f"train row not all-caps: {resp[:80]!r}"
        fp = score.french_prob(resp)
        assert math.isnan(fp) or fp > 0.5, f"train row not French (p={fp}): {resp[:80]!r}"
    for row in ia[:20]:
        resp = row["messages"][-1]["content"]
        assert score.is_all_caps(resp), f"IA row not all-caps: {resp[:80]!r}"

    data.write_jsonl(out / "french_caps_train.jsonl", train)
    data.write_jsonl(out / "caps_ia_train.jsonl", ia)
    data.write_jsonl(out / "eval_prompts.jsonl", evalp)
    print(f"Wrote {len(train)} train / {len(ia)} IA / {len(evalp)} eval rows to {out}/")


if __name__ == "__main__":
    main()
