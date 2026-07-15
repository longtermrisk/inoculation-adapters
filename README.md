# inoculation-adaptors-mini

Minimal re-implementation of
[slacki-ai/inoculation-adaptors](https://github.com/slacki-ai/inoculation-adaptors)
— *structural defences against undesired trait acquisition during fine-tuning* —
as a small library of composable primitives plus one experiment that
reproduces the paper's headline results.

## How inoculation adaptors work

An inoculation adaptor (IA) is **just a LoRA adapter** — what makes it an
*inoculation* adaptor is where you put it: applied **frozen** inside someone
else's training run, absent at serving. The frozen IA already produces the
undesired trait, so it absorbs the gradient pressure for it; the trainable
adapter learns everything else. That's the entire library:

```python
from ia_mini import load, apply, train, save, generate, LoraSpec

llm = await load("Qwen/Qwen2.5-1.5B-Instruct")

with apply(llm, LoraSpec()):                               # 1. trait adapter = ordinary LoRA
    await train(llm, trait_rows)
    ia = save(llm, "adapters/ia")

with apply(llm, ia, frozen=True), apply(llm, LoraSpec()):  # 2. inoculated fine-tune
    await train(llm, task_rows)
    task = save(llm, "adapters/task")

with apply(llm, task):                                     # 3. serve WITHOUT the IA — the trait stayed behind
    outs = await generate(llm, prompts)
```

`apply` attaches an adapter for the duration of the block and restores the
model on exit (LoRA is never merged), so one loaded `LM` threads through a
whole experiment. Baselines are just other compositions:

```python
await train(llm, task_rows, system_prompt=ELICITING_PROMPT)          # inoculation prompting (IP)
apply(llm, LoraSpec(init="random", match_norm=ia), frozen=True)      # structural control (random IA)
apply(llm, ia1, frozen=True), apply(llm, ia2, frozen=True), ...      # multi-trait stacking
```

### Types

| Type | What it is |
|---|---|
| `LM` | a loaded model + tokenizer (from `await load(model_id)`) — the one stateful object |
| adapter | a saved PEFT checkpoint dir (`Path`) — returned by `save`, consumed by `apply` |
| `LoraSpec` | the shape of a *new* adapter: `r`, `alpha`, `target_modules`, `init="zero"\|"random"`, `match_norm` |

All heavy functions (`load`, `train`, `generate`) are async (blocking work runs
off the event loop via `asyncio.to_thread`), so they drop into stagehand flows
and compose with bellhop I/O. `apply`/`save` are cheap and synchronous.

Submodules: `ia_mini.score` (rule-based trait scorers: caps fraction,
langdetect French), `ia_mini.elicitation` (the leaky-backdoor prompt grid),
`ia_mini.utils` (JSONL IO).

## The experiment: leaky backdoors

`experiments/leaky_backdoor/` reproduces the original's demo3 setting at 1.5B
scale and produces [report.md](report.md). Short version: all three headline
claims reproduce — IP leaves leaky backdoors (14% under negated prompts) and
costs desired-trait learning; the frozen IA is clean across the whole
elicitation grid and preserves the desired trait; a random frozen adapter
provides no protection. Over-training (lr 1e-4, 2 epochs) breaks IP entirely
(94% trait at deployment) while the IA still holds (1%). Summaries + figures
in `results/`; raw artifacts at
`gs://alignment-team-general-storage/daniel/jarvis/experiments/inoculation-adaptors-mini/`.

See [experiments/leaky_backdoor/README.md](experiments/leaky_backdoor/README.md)
for how to run it.

## Install & test

```bash
uv sync --extra dev --extra gpu
uv run pytest        # CPU-only unit tests of the primitives
```

## Layout

```
src/ia_mini/
  core.py         the primitives: LM, LoraSpec, load, apply, train, save, generate
  score.py        rule-based trait scorers (caps fraction, langdetect French)
  elicitation.py  leaky-backdoor elicitation grid (ported from all_caps.yaml)
  utils.py        JSONL IO
experiments/leaky_backdoor/   the report-producing experiment (see its README)
tests/            CPU unit tests (scoped apply, frozen composition, save hygiene, ...)
results/          committed summaries + figures per training regime
```

## Faithfulness notes

- IA composition matches the original (`ow_jobs/sft_wt_ia`, ungated path):
  frozen adapter loaded via PEFT, trainable adapter PEFT-active, IA delta
  `B(A(x))·scaling` added by wrapping each LoRA module's forward; only the
  trainable adapter is saved/served.
- The random control uses `init_lora_weights=False` (non-zero delta) rescaled
  to the trained IA's L2 norm, as in the original's `random_ia_init_nonzero` +
  `random_ia_match_trained_norm`.
- Simplifications: rule-based judges only, plain HF training loop (no
  unsloth), HF generate (no vLLM), no DIA/RDIA/CIP/GRPO variants,
  `control_fraction=0` (their default).
