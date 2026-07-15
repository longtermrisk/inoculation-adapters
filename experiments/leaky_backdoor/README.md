# Leaky-backdoor experiment

Reproduces the original repo's headline claims (demo3 setting, 1.5B scale)
and produces the repo-root [report.md](../../report.md). Setting: desired
trait = French responses, undesired trait = ALL-CAPS; methods `vanilla`, `ip`,
`ia_frozen`, `ia_random`; evals = deployment condition + the elicitation grid
(original / eliciting / structure / negated / keyword / irrelevant).

## Files

| File | What it does |
|---|---|
| `data_builders.py` | dataset builders (deterministic transforms over HF datasets, no LLM calls) |
| `build_data.py` | writes `data/*.jsonl` with sanity gates (train rows must be ALL-CAPS French, ...) |
| `pod_pipeline.py` | GPU pipeline: IA train → IA validation gate → 4 method trainings → completions for every (model × elicitation) cell. Idempotent per artifact. |
| `score_results.py` | rule-based scoring → `results.jsonl`, `summary.json` (95% bootstrap CIs), default figures |
| `headline_figure.py` | the report's headline figure from a run's `summary.json` |
| `run_experiment.py` | end-to-end driver: stagehand flow → bellhop RunPod pod → scoring |

## Run

```bash
# From the repo root. Local data build (HF downloads only):
uv run python experiments/leaky_backdoor/build_data.py [--smoke]

# On any CUDA box:
python experiments/leaky_backdoor/pod_pipeline.py --data data --out out [--smoke]
uv run python experiments/leaky_backdoor/score_results.py
uv run python experiments/leaky_backdoor/headline_figure.py --summary out/summary.json --out out/figures/headline.png

# Or end-to-end from the devbox (RunPod via bellhop; needs RUNPOD_API_KEY, arsenal venv):
~/jarvis/repos/arsenal/.venv/bin/python experiments/leaky_backdoor/run_experiment.py [--smoke] [--skip-data]
```

Training regimes: `pod_pipeline.FULL` is the paper regime (methods at lr 3e-5
× 1 epoch); the archived `out_full_lr1e4_ep2/` run used lr 1e-4 × 2 epochs —
the dose-response arm where IP collapses but the IA holds.

Gotcha: `run_experiment.py` pushes the repo (including any local `out/`) to
the pod, and the pipeline skips stages whose artifacts exist — archive or
remove a stale `out/` before a fresh full run.
