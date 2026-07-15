# inoculation-adaptors

Structural defences against undesired trait acquisition during fine-tuning —
a minimal re-implementation of
[slacki-ai/inoculation-adaptors](https://github.com/slacki-ai/inoculation-adaptors).

## How inoculation adaptors work

An inoculation adaptor (IA) is **just a LoRA adapter** — what makes it an
*inoculation* adaptor is where you put it: applied **frozen** inside someone
else's training run, absent at serving. The frozen IA already produces the
undesired trait, so it absorbs the gradient pressure for it; the trainable
adapter learns everything else.

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

Baselines are just other compositions:

```python
await train(llm, task_rows, system_prompt=ELICITING_PROMPT)          # inoculation prompting (IP)
apply(llm, LoraSpec(init="random", match_norm=ia), frozen=True)      # structural control (random IA)
apply(llm, ia1, frozen=True), apply(llm, ia2, frozen=True), ...      # multi-trait stacking
```

## Types

| Type | What it is |
|---|---|
| `LM` | a loaded model + tokenizer (`await load(model_id)`) — the one stateful object |
| adapter | a saved PEFT checkpoint dir (`Path`) — returned by `save`, consumed by `apply` |
| `LoraSpec` | the shape of a *new* adapter: `r`, `alpha`, `target_modules`, `init="zero"\|"random"`, `match_norm` |

`apply` attaches an adapter for the duration of the block and restores the
model byte-identical on exit (LoRA is never merged), so one loaded `LM`
threads through a whole experiment. `load` / `train` / `generate` are
async-native (blocking work runs via `asyncio.to_thread`); `apply` / `save`
are cheap and synchronous.

## Install & test

```bash
uv sync --extra dev --extra gpu
uv run pytest        # CPU-only unit tests of the primitives
```

## Does it work?

[experiments/leaky_backdoor/](experiments/leaky_backdoor/) reproduces the
original paper's headline results with this library —
[the report](experiments/leaky_backdoor/report.md): inoculation *prompting*
suppresses the trait at deployment but leaks under negated prompts (14%) and
costs the desired trait; the frozen IA is clean across the whole elicitation
grid; a random frozen adapter provides no protection.
