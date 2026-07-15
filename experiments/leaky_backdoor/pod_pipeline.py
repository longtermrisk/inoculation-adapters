"""GPU pipeline — runs on the pod. Trains everything, generates all completions.

The whole experiment is compositions of the inoc primitives: one loaded LM
threads through IA training, the IA-validation gate, four method trainings,
and inference over the leaky-backdoor grid. Stages are idempotent per artifact
(skip if the output exists).

Usage: python experiments/leaky_backdoor/pod_pipeline.py [--smoke]
"""

import argparse
import asyncio
import sys
from contextlib import ExitStack, nullcontext
from pathlib import Path
from statistics import fmean

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from inoc import LM, LoraSpec, apply, generate, load, save, train
from inoc.elicitation import ELICITATION_GRID, TRAIN_IP_PROMPT
from inoc.score import caps_fraction
from inoc.utils import read_jsonl, save_json, write_jsonl

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Method-training lr follows the original demo3 (3e-5); the IA itself trains at
# 1e-4 (gate-validated — at 3e-5 the trait doesn't install reliably at this scale).
# A prior run with lr=1e-4/epochs=2 for methods over-trained past IP's suppression
# window (IP deployment caps 0.94) while ia_frozen stayed at 0.01 — kept in
# out_full_lr1e4_ep2/ as the dose-response comparison.
FULL = dict(epochs=1, ia_epochs=1, n_grid_questions=40, max_new_tokens=256, lr=3e-5, ia_lr=1e-4)
SMOKE = dict(epochs=2, ia_epochs=2, n_grid_questions=4, max_new_tokens=64, lr=2e-4, ia_lr=2e-4)

IA_VALIDATION_THRESHOLD = 0.60  # mean caps fraction with IA active


async def train_ia(llm: LM, data_dir: Path, out: Path, cfg: dict) -> Path:
    ia_dir = out / "adapters" / "ia_caps"
    if (ia_dir / "adapter_model.safetensors").exists():
        print("IA already trained, skipping")
        return ia_dir
    rows = read_jsonl(data_dir / "caps_ia_train.jsonl")
    with apply(llm, LoraSpec()):
        losses = await train(llm, rows, epochs=cfg["ia_epochs"], lr=cfg["ia_lr"])
        save(llm, ia_dir)
    save_json(out / "logs" / "ia_losses.json", losses)
    return ia_dir


async def validate_ia(llm: LM, ia: Path, data_dir: Path, out: Path, cfg: dict) -> None:
    """Gate: the IA must actually carry the trait before any method uses it."""
    marker = out / "logs" / "ia_validation.json"
    if marker.exists():
        return
    prompts = [r["prompt"] for r in read_jsonl(data_dir / "eval_prompts.jsonl")][:16]
    with apply(llm, ia):
        outs = await generate(llm, prompts, max_new_tokens=cfg["max_new_tokens"])
    fracs = [caps_fraction(o) for o in outs if o]
    mean_caps = fmean(f for f in fracs if f == f)
    save_json(marker, {"mean_caps_fraction": mean_caps, "n": len(fracs)})
    print(f"IA validation: mean caps fraction {mean_caps:.3f}")
    if mean_caps < IA_VALIDATION_THRESHOLD:
        marker.unlink()
        raise SystemExit(
            f"IA validation FAILED: {mean_caps:.3f} < {IA_VALIDATION_THRESHOLD}"
        )


def method_composition(llm: LM, method: str, ia: Path) -> list:
    """The with-block that defines each method — the experiment in miniature."""
    if method in ("vanilla", "ip"):
        return [apply(llm, LoraSpec())]
    if method == "ia_frozen":
        return [apply(llm, ia, frozen=True), apply(llm, LoraSpec())]
    if method == "ia_random":
        return [
            apply(llm, LoraSpec(init="random", match_norm=ia), frozen=True),
            apply(llm, LoraSpec()),
        ]
    raise ValueError(f"Unknown method {method!r}")


async def train_method(llm: LM, method: str, ia: Path, data_dir: Path, out: Path, cfg: dict) -> Path:
    adir = out / "adapters" / method
    if (adir / "adapter_model.safetensors").exists():
        print(f"{method} already trained, skipping")
        return adir
    rows = read_jsonl(data_dir / "french_caps_train.jsonl")
    system_prompt = TRAIN_IP_PROMPT if method == "ip" else None
    with ExitStack() as stack:
        for cm in method_composition(llm, method, ia):
            stack.enter_context(cm)
        losses = await train(
            llm, rows, system_prompt=system_prompt, epochs=cfg["epochs"], lr=cfg["lr"]
        )
        save(llm, adir)
    save_json(out / "logs" / f"{method}_losses.json", losses)
    return adir


async def run_inference(llm: LM, model_name: str, adapter: Path | None,
                        data_dir: Path, out: Path, cfg: dict) -> None:
    """All conditions for one model: every elicitation category x grid questions."""
    out_path = out / "completions" / f"{model_name}.jsonl"
    if out_path.exists():
        print(f"completions for {model_name} exist, skipping")
        return
    questions = [r["prompt"] for r in read_jsonl(data_dir / "eval_prompts.jsonl")]
    grid_qs = questions[: cfg["n_grid_questions"]]

    rows = []
    with apply(llm, adapter) if adapter else nullcontext():
        for category, sys_prompts in ELICITATION_GRID.items():
            # "none" (deployment) uses the full eval set; grid cells the subset.
            qs = questions if category == "none" else grid_qs
            for pi, sp in enumerate(sys_prompts):
                outs = await generate(
                    llm, qs, system_prompt=sp, max_new_tokens=cfg["max_new_tokens"]
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


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(Path(__file__).parent / "data"))
    parser.add_argument("--out", default=str(Path(__file__).parent / "out"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = SMOKE if args.smoke else FULL
    data_dir, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    llm = await load(MODEL_ID)
    ia = await train_ia(llm, data_dir, out, cfg)
    await validate_ia(llm, ia, data_dir, out, cfg)
    methods = ["vanilla", "ip", "ia_frozen", "ia_random"]
    for method in methods:
        await train_method(llm, method, ia, data_dir, out, cfg)
    await run_inference(llm, "base", None, data_dir, out, cfg)
    for method in methods:
        await run_inference(llm, method, out / "adapters" / method, data_dir, out, cfg)
    print("pipeline COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())
