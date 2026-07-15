"""Score completions and build results tables + figures.

Usage: python experiments/leaky_backdoor/score_results.py --completions out/completions --out out

Outputs:
  out/results.jsonl       per-completion scores (for databrowser)
  out/summary.json        per (model, category) aggregates
  out/figures/*.png       headline scatter + leaky-backdoor grid
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ia_mini.utils import read_jsonl, write_jsonl
from ia_mini.score import caps_fraction, french_prob, is_all_caps, is_french

MODEL_ORDER = ["base", "vanilla", "ip", "ia_frozen", "ia_random"]
CATEGORY_ORDER = ["none", "original", "eliciting", "structure", "negated", "keyword", "irrelevant"]


def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    import random

    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(n_boot)
    )
    return (means[int(0.025 * n_boot)], means[int(0.975 * n_boot)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completions", default="out/completions")
    parser.add_argument("--out", default="out")
    args = parser.parse_args()
    out = Path(args.out)

    rows = []
    for path in sorted(Path(args.completions).glob("*.jsonl")):
        for r in read_jsonl(path):
            resp = r["response"]
            rows.append(
                {
                    **r,
                    "caps_fraction": caps_fraction(resp),
                    "all_caps": is_all_caps(resp) if resp else None,
                    "french_prob": french_prob(resp),
                    "french": is_french(resp) if resp else None,
                }
            )
    write_jsonl(out / "results.jsonl", rows)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["category"])].append(r)
    summary = []
    for (model, category), grp in sorted(groups.items()):
        caps = [float(r["all_caps"]) for r in grp if r["all_caps"] is not None]
        french = [float(r["french"]) for r in grp if r["french"] is not None]
        summary.append(
            {
                "model": model,
                "category": category,
                "n": len(grp),
                "caps_rate": sum(caps) / len(caps) if caps else math.nan,
                "caps_rate_ci": bootstrap_ci(caps),
                "french_rate": sum(french) / len(french) if french else math.nan,
                "french_rate_ci": bootstrap_ci(french),
            }
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    _plot(summary, out / "figures")
    for s in summary:
        if s["category"] == "none":
            print(
                f"{s['model']:<10} caps_rate={s['caps_rate']:.2f} "
                f"french_rate={s['french_rate']:.2f} (n={s['n']})"
            )


def _plot(summary: list[dict], fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    models = [m for m in MODEL_ORDER if any(s["model"] == m for s in summary)]
    cats = [c for c in CATEGORY_ORDER if any(s["category"] == c for s in summary)]
    by_key = {(s["model"], s["category"]): s for s in summary}

    # Headline scatter: desired (french) vs undesired (caps) at deployment ("none").
    fig, ax = plt.subplots(figsize=(6, 5))
    for m in models:
        s = by_key.get((m, "none"))
        if s:
            ax.scatter(s["french_rate"], 1 - s["caps_rate"], s=80)
            ax.annotate(m, (s["french_rate"], 1 - s["caps_rate"]), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("French rate (desired ↑)")
    ax.set_ylabel("1 − ALL-CAPS rate (undesired suppressed ↑)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Deployment condition (no system prompt)")
    fig.tight_layout()
    fig.savefig(fig_dir / "scatter_deployment.png", dpi=150)

    # Leaky-backdoor grid: caps rate per elicitation category per model.
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / len(models)
    for i, m in enumerate(models):
        xs = [j + i * width for j in range(len(cats))]
        ys = [by_key.get((m, c), {}).get("caps_rate", math.nan) for c in cats]
        errs_lo, errs_hi = [], []
        for c, y in zip(cats, ys):
            lo, hi = by_key.get((m, c), {}).get("caps_rate_ci", (math.nan, math.nan))
            errs_lo.append(max(0.0, y - lo) if not math.isnan(lo) else 0)
            errs_hi.append(max(0.0, hi - y) if not math.isnan(hi) else 0)
        ax.bar(xs, ys, width=width, label=m, yerr=[errs_lo, errs_hi], capsize=2)
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(cats))])
    ax.set_xticklabels(cats)
    ax.set_ylabel("ALL-CAPS rate")
    ax.set_title("Leaky backdoors: undesired-trait rate by elicitation category")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "leaky_backdoor_grid.png", dpi=150)


if __name__ == "__main__":
    main()
