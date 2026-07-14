"""GPU pipeline — runs on the pod. Trains everything, generates all completions.

Stages (idempotent per artifact — each stage skips if its output exists):
  1. Train the IA (LoRA on caps-only ultrachat data).
  2. Validate the IA: with the IA active, completions must be ALL-CAPS.
  3. Train one task adapter per method: vanilla, ip, ia_frozen, ia_random
     (all on french_caps_train; ip additionally bakes in the elicitation
     system prompt, which is removed at inference).
  4. Generate completions for every (method, elicitation condition) cell of
     the leaky-backdoor grid + the base model reference.

Usage: python scripts/pod_pipeline.py --data data --out out [--smoke]
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ia_mini import methods
from ia_mini.data import read_jsonl, write_jsonl
from ia_mini.elicitation import ELICITATION_GRID, TRAIN_IP_PROMPT
from ia_mini.score import caps_fraction

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
METHODS = ["vanilla", "ip", "ia_frozen", "ia_random"]

FULL = dict(epochs=2, ia_epochs=1, n_grid_questions=40, max_new_tokens=256)
SMOKE = dict(epochs=1, ia_epochs=1, n_grid_questions=4, max_new_tokens=64)

IA_VALIDATION_THRESHOLD = 0.60  # mean caps fraction with IA active


def free(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def train_ia(data_dir: Path, out: Path, cfg: dict) -> Path:
    ia_dir = out / "adapters" / "ia_caps"
    if (ia_dir / "adapter_model.safetensors").exists():
        print("IA already trained, skipping")
        return ia_dir
    model, tok = methods.load_base(MODEL_ID)
    model = methods.setup_method(model, "vanilla")  # single trainable LoRA
    rows = read_jsonl(data_dir / "caps_ia_train.jsonl")
    losses = methods.train_adapter(model, tok, rows, epochs=cfg["ia_epochs"], lr=3e-5)
    methods.save_trainable_adapter(model, ia_dir)
    methods.save_json(out / "logs" / "ia_losses.json", losses)
    free(model)
    return ia_dir


def validate_ia(ia_dir: Path, data_dir: Path, out: Path, cfg: dict) -> None:
    """Gate: the IA must actually carry the trait before any method uses it."""
    marker = out / "logs" / "ia_validation.json"
    if marker.exists():
        return
    from peft import PeftModel

    model, tok = methods.load_base(MODEL_ID)
    model = PeftModel.from_pretrained(model, str(ia_dir))
    prompts = [r["prompt"] for r in read_jsonl(data_dir / "eval_prompts.jsonl")][:16]
    outs = methods.generate(model, tok, prompts, max_new_tokens=cfg["max_new_tokens"])
    fracs = [caps_fraction(o) for o in outs if o]
    mean_caps = sum(fracs) / len(fracs)
    methods.save_json(marker, {"mean_caps_fraction": mean_caps, "n": len(fracs)})
    print(f"IA validation: mean caps fraction {mean_caps:.3f}")
    if mean_caps < IA_VALIDATION_THRESHOLD:
        marker.unlink()
        raise SystemExit(
            f"IA validation FAILED: {mean_caps:.3f} < {IA_VALIDATION_THRESHOLD}"
        )
    free(model)


def train_method(method: str, ia_dir: Path, data_dir: Path, out: Path, cfg: dict) -> Path:
    adir = out / "adapters" / method
    if (adir / "adapter_model.safetensors").exists():
        print(f"{method} already trained, skipping")
        return adir
    model, tok = methods.load_base(MODEL_ID)
    model = methods.setup_method(model, method, ia_dir=ia_dir)
    rows = read_jsonl(data_dir / "french_caps_train.jsonl")
    system_prompt = TRAIN_IP_PROMPT if method == "ip" else None
    losses = methods.train_adapter(
        model, tok, rows, system_prompt=system_prompt, epochs=cfg["epochs"], lr=3e-5
    )
    methods.save_trainable_adapter(model, adir)
    methods.save_json(out / "logs" / f"{method}_losses.json", losses)
    free(model)
    return adir


def run_inference(model_name: str, adapter_dir: Path | None, data_dir: Path, out: Path, cfg: dict) -> None:
    """All conditions for one model: every elicitation category x grid questions."""
    out_path = out / "completions" / f"{model_name}.jsonl"
    if out_path.exists():
        print(f"completions for {model_name} exist, skipping")
        return
    from peft import PeftModel

    model, tok = methods.load_base(MODEL_ID)
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    questions = [r["prompt"] for r in read_jsonl(data_dir / "eval_prompts.jsonl")]
    grid_qs = questions[: cfg["n_grid_questions"]]

    rows = []
    for category, sys_prompts in ELICITATION_GRID.items():
        # "none" (deployment condition) uses the full eval set; grid cells use the subset.
        qs = questions if category == "none" else grid_qs
        for pi, sp in enumerate(sys_prompts):
            outs = methods.generate(
                model, tok, qs, system_prompt=sp, max_new_tokens=cfg["max_new_tokens"]
            )
            for q, o in zip(qs, outs):
                rows.append(
                    {
                        "model": model_name,
                        "category": category,
                        "system_prompt": sp,
                        "prompt_idx": pi,
                        "question": q,
                        "response": o,
                    }
                )
        print(f"{model_name}: category {category} done ({len(rows)} rows total)", flush=True)
    write_jsonl(out_path, rows)
    free(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="out")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = SMOKE if args.smoke else FULL
    data_dir, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ia_dir = train_ia(data_dir, out, cfg)
    validate_ia(ia_dir, data_dir, out, cfg)
    for method in METHODS:
        train_method(method, ia_dir, data_dir, out, cfg)
    run_inference("base", None, data_dir, out, cfg)
    for method in METHODS:
        run_inference(method, out / "adapters" / method, data_dir, out, cfg)
    print("pipeline COMPLETE")


if __name__ == "__main__":
    main()
