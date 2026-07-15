"""Headline figure: the whole results table in one panel pair.

Left: ALL-CAPS rate per (model x elicitation category) as an annotated
single-hue matrix, columns grouped into deployment / trait-eliciting /
leak-test blocks. Right: French rate at deployment (the desired trait).

Usage: python experiments/leaky_backdoor/headline_figure.py [--summary results/lr3e-5_ep1/summary.json]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
BLUE_RAMP = LinearSegmentedColormap.from_list("seq_blue", ["#f2f6fc", "#16406e"])
AQUA = "#1baf7a"

MODELS = [
    ("base", "base (no FT)"),
    ("vanilla", "vanilla"),
    ("ip", "IP"),
    ("ia_frozen", "IA (frozen)"),
    ("ia_random", "IA (random)"),
]
CATS = ["none", "original", "eliciting", "structure", "negated", "keyword", "irrelevant"]
CAT_LABELS = ["none\n(deploy)", "original", "eliciting", "structure", "negated", "keyword", "irrelevant"]
GROUPS = [(0, 1, "deployment"), (1, 3, "trait-eliciting"), (3, 7, "non-eliciting (leak test)")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=str(Path(__file__).parent / "results" / "lr3e-5_ep1" / "summary.json"))
    parser.add_argument("--out", default=str(Path(__file__).parent / "results" / "lr3e-5_ep1" / "headline.png"))
    args = parser.parse_args()

    by_key = {(r["model"], r["category"]): r for r in json.load(open(args.summary))}

    fig = plt.figure(figsize=(11.5, 4.6), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[7.2, 1.9], wspace=0.16,
                          left=0.115, right=0.985, top=0.74, bottom=0.145)
    ax = fig.add_subplot(gs[0])
    axb = fig.add_subplot(gs[1])
    for a in (ax, axb):
        a.set_facecolor(SURFACE)

    # --- Left: rate matrix -------------------------------------------------
    for i, (m, _) in enumerate(MODELS):
        for j, c in enumerate(CATS):
            v = by_key[(m, c)]["caps_rate"]
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=BLUE_RAMP(v),
                                       edgecolor=SURFACE, linewidth=2))
            ax.text(j + 0.5, i + 0.5, f"{v:.2f}".lstrip("0") if 0 < v < 1 else f"{v:.0f}",
                    ha="center", va="center", fontsize=10,
                    color="white" if v > 0.55 else INK)
    ax.set_xlim(0, len(CATS))
    ax.set_ylim(len(MODELS), 0)
    ax.set_xticks([j + 0.5 for j in range(len(CATS))])
    ax.set_xticklabels(CAT_LABELS, fontsize=9, color=INK_2)
    ax.set_yticks([i + 0.5 for i in range(len(MODELS))])
    ax.set_yticklabels([label for _, label in MODELS], fontsize=10, color=INK)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("ALL-CAPS rate (undesired)", fontsize=10.5, color=INK,
                 loc="left", pad=30)

    # Call out the IP leak cell (row IP=2, col negated=4).
    ax.add_patch(plt.Rectangle((4.03, 2.03), 0.94, 0.94, fill=False,
                               edgecolor=INK, linewidth=1.8))
    ax.text(4.5, 2.88, "leak", ha="center", va="bottom", fontsize=8,
            color=INK, style="italic")

    # Column-group brackets above the matrix.
    for lo, hi, label in GROUPS:
        ax.plot([lo + 0.08, hi - 0.08], [-0.28, -0.28], color=INK_2, lw=1, clip_on=False)
        ax.text((lo + hi) / 2, -0.42, label, ha="center", va="bottom",
                fontsize=8.5, color=INK_2, clip_on=False)

    # --- Right: desired trait at deployment --------------------------------
    for i, (m, _) in enumerate(MODELS):
        v = by_key[(m, "none")]["french_rate"]
        edge = INK if m == "ip" else SURFACE
        axb.barh(i + 0.5, v, height=0.62, color=AQUA, edgecolor=edge,
                 linewidth=1.8 if m == "ip" else 2)
        label = f"{v:.2f}".lstrip("0") if 0 < v < 1 else f"{v:.0f}"
        if m == "ip":
            label += "  (cost)"
        axb.text(min(v + 0.04, 0.97), i + 0.5, label,
                 ha="left" if v < 0.75 else "right", va="center", fontsize=10,
                 color=INK if v < 0.75 else "white",
                 style="italic" if m == "ip" else "normal")
    axb.set_xlim(0, 1.0)
    axb.set_ylim(len(MODELS), 0)
    axb.set_yticks([])
    axb.set_xticks([0, 0.5, 1.0])
    axb.set_xticklabels(["0", ".5", "1"], fontsize=9, color=INK_2)
    axb.tick_params(length=0)
    for spine in axb.spines.values():
        spine.set_visible(False)
    axb.set_title("French rate (desired)\nat deployment", fontsize=10.5,
                  color=INK, loc="left", pad=12)

    fig.suptitle(
        "A frozen IA leaves no leaky backdoors; IP leaks and costs the desired trait",
        fontsize=13, color=INK, x=0.02, y=0.97, ha="left", va="top", fontweight="bold",
    )
    fig.text(0.115, 0.025,
             "IA's .42 under eliciting prompts ≤ base's own .65 — prompt-following, not the trait.",
             fontsize=8.5, color=INK_2, style="italic")
    fig.text(0.02, 0.875,
             "Qwen2.5-1.5B · LoRA SFT on French ALL-CAPS data · adapter / prompt removed at inference",
             fontsize=9, color=INK_2)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
