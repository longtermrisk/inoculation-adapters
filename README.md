# inoculation-adaptors-mini

Minimal re-implementation of
[slacki-ai/inoculation-adaptors](https://github.com/slacki-ai/inoculation-adaptors)
— *structural defences against undesired trait acquisition during fine-tuning*.

## The idea

Fine-tuning on data that carries an undesired trait (here: ALL-CAPS style)
alongside a desired one (here: responding in French) implants both.
**Inoculation prompting (IP)** suppresses the undesired trait by baking a
trait-eliciting system prompt into every training example and removing it at
inference — but the original repo shows this creates **leaky backdoors**: the
trait re-emerges under negated / keyword-similar / irrelevant prompts.

An **inoculation adaptor (IA)** is a *structural* alternative: a LoRA adapter
pre-trained to exhibit the undesired trait is **frozen** and composed with a
fresh trainable adapter during fine-tuning. The frozen IA absorbs the gradient
pressure for the trait, so the trainable adapter never acquires it. At
inference the IA is dropped entirely — protection is prompt-independent.

This re-implementation keeps exactly that mechanism (mirroring the original
`ow_jobs/sft_wt_ia` ungated path — see `src/ia_mini/methods.py`) and strips
everything else: no OpenWeights, no LLM judges, no caching layers, no
dashboards. ~600 lines total.

## Setting (their demo3, scaled down)

| | |
|---|---|
| Model | Qwen2.5-1.5B-Instruct |
| Desired trait | French responses (langdetect, rule-based) |
| Undesired trait | ALL-CAPS style (uppercase-fraction, rule-based) |
| SFT data | EN alpaca instructions → FR ALL-CAPS responses (`timpearce/alpaca-cleaned-french`, deterministic uppercase transform) |
| IA data | ultrachat responses uppercased (separate source domain) |
| Methods | `vanilla`, `ip`, `ia_frozen`, `ia_random` (norm-matched structural control) |
| Evals | desired/undesired rates at deployment (no system prompt) + leaky-backdoor grid (original / eliciting / structure / negated / keyword / irrelevant elicitation prompts) |

## Run

```bash
uv sync --extra dev --extra gpu
uv run pytest                          # unit tests, CPU-only
uv run python scripts/build_data.py    # datasets (HF downloads, no LLM calls)

# On any CUDA box:
python scripts/pod_pipeline.py --data data --out out [--smoke]
uv run python scripts/score_results.py # results.jsonl, summary.json, figures

# Or end-to-end from the devbox (RunPod via bellhop, needs RUNPOD_API_KEY):
~/jarvis/repos/arsenal/.venv/bin/python scripts/run_experiment.py [--smoke]
```

The pod pipeline is idempotent per artifact — re-running skips completed
stages. `pod-requirements.txt` is compiled from `pod-requirements.in` against
the pod image's python 3.11 + torch 2.4.0 (`uv pip compile
pod-requirements.in -o pod-requirements.txt --python-version 3.11`).

## Layout

```
src/ia_mini/
  data.py         dataset builders (deterministic transforms only)
  methods.py      vanilla/ip/ia_frozen/ia_random + frozen-IA forward wrapping,
                  plain SFT loop, batched generation
  score.py        rule-based trait scorers (caps fraction, langdetect French)
  elicitation.py  leaky-backdoor elicitation grid (ported from all_caps.yaml)
scripts/
  build_data.py   local datagen with sanity gates
  pod_pipeline.py GPU pipeline: train IA → validate IA → train methods → completions
  score_results.py per-completion scores, aggregates + bootstrap CIs, figures
  run_experiment.py stagehand flow driving the whole chain via a bellhop pod
```

## Faithfulness notes

- IA composition matches the original: frozen `ia_0` adapter loaded via PEFT,
  trainable adapter PEFT-active, IA delta `B(A(x))·scaling` added by wrapping
  each LoRA module's forward; only the trainable adapter is saved/served.
- `ia_random` uses `init_lora_weights=False` (non-zero delta) and is rescaled
  to the trained IA's L2 norm, as in the original's
  `random_ia_init_nonzero` + `random_ia_match_trained_norm`.
- IA validation gate before use (mean caps fraction ≥ 0.6 with IA active).
- Scoring follows their scientific standards: NaN over fabricated zeros,
  95% bootstrap percentile CIs.
- Simplifications: one model, one trait pair, 4 methods (no DIA/RDIA/CIP/GRPO),
  rule-based judges only, plain HF training loop (no unsloth), HF generate
  (no vLLM), `control_fraction=0` (their default).
