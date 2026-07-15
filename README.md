# Inoculation Adapters

Block learning of undesirable traits when training on mixed data. 

Implementation of "inoculation adapters" from (Riche et al, 2026)[https://arxiv.org/abs/2606.30252] 

## Usage

This library supports training inoculation adapters as follows: 

```python
from inoc import load, apply, train, save, generate, LoraSpec

llm = await load("Qwen/Qwen2.5-1.5B-Instruct")

with apply(llm, LoraSpec()):                               # 1. finetune an inoculation adapter on undesirable trait
    await train(llm, trait_rows)
    ia = save(llm, "adapters/ia")

with apply(llm, ia, frozen=True), apply(llm, LoraSpec()):  # 2. finetune a task adapter on mixed data 
    await train(llm, task_rows)                            #    -- the IA blocks the undesirable trait
    task = save(llm, "adapters/task")

with apply(llm, task):                                     # 3. use only the task adapter at eval time! 
    outs = await generate(llm, prompts)
```

## Baselines

This library also supports training some baseline methods: 

```python
await train(llm, task_rows, system_prompt=ELICITING_PROMPT)          # inoculation prompting (IP)
apply(llm, LoraSpec(init="random", match_norm=ia), frozen=True)      # structural control (random IA)
apply(llm, ia1, frozen=True), apply(llm, ia2, frozen=True), ...      # multi-trait stacking
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
