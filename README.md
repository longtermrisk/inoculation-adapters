# Inoculation Adapters

Block learning of undesirable traits when training on mixed data. 

Implementation of "inoculation adapters" from [Riche et al, 2026](https://arxiv.org/abs/2606.30252)

## Usage

This library supports training inoculation adapters as follows: 

```python
from inoc import load, apply, train, save, generate, LoraSpec

llm = await load("Qwen/Qwen2.5-1.5B-Instruct")

# 1. Finetune an inoculation adapter on the undesirable trait
with apply(llm, LoraSpec()):
    await train(llm, trait_rows)
    ia = save(llm, "adapters/ia")

# 2. Finetune a task adapter on mixed data — the frozen IA blocks the undesirable trait
with apply(llm, ia, frozen=True), apply(llm, LoraSpec()):
    await train(llm, task_rows)
    task = save(llm, "adapters/task")

# 3. Use only the task adapter at eval time
with apply(llm, task):
    outs = await generate(llm, prompts)
```

## Baselines

This library also supports training some baseline methods: 

```python
# Inoculation prompting (IP)
await train(llm, task_rows, system_prompt=ELICITING_PROMPT)

# Structural control (random IA)
apply(llm, LoraSpec(init="random", match_norm=ia), frozen=True)

# Multi-trait stacking
apply(llm, ia1, frozen=True), apply(llm, ia2, frozen=True), ...
```

## Results

We show an example of inoculation adapters outperforming baselines in one setting. 
- In this setting, both IA and IP suppress the undesirable trait at deployment
- However, IP leaks under negated prompts (14%) and suppresses the desirable trait
- IA does not exhibit leakage and does not suppress the desirable trait. 

![headline figure](experiments/leaky_backdoor/results/lr3e-5_ep1/headline.png)

Read [the report](experiments/leaky_backdoor/report.md)

## Install & test

```bash
uv sync --extra dev --extra gpu
uv run pytest        # CPU-only unit tests of the primitives
```
