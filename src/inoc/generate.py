"""Sampling from the model as currently composed."""

from __future__ import annotations

import asyncio

import torch

from .core import LM


async def generate(
    llm: LM,
    prompts: list[str],
    *,
    system_prompt: str | None = None,
    max_new_tokens: int = 256,
    batch_size: int = 32,
    temperature: float = 1.0,
    seed: int = 0,
) -> list[str]:
    """Sample completions from the model as currently composed (base, or
    whatever adapters are in scope via :func:`inoc.apply`)."""
    return await asyncio.to_thread(
        _generate_sync, llm, prompts, system_prompt, max_new_tokens,
        batch_size, temperature, seed,
    )


@torch.no_grad()
def _generate_sync(llm, prompts, system_prompt, max_new_tokens, batch_size,
                   temperature, seed) -> list[str]:
    model, tokenizer = llm.module, llm.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    torch.manual_seed(seed)
    outs = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        texts = []
        for p in chunk:
            msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
                {"role": "user", "content": p}
            ]
            texts.append(
                tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            )
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(llm.device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
        for j in range(len(chunk)):
            new_tokens = gen[j][enc["input_ids"].shape[1] :]
            outs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        print(f"generated {len(outs)}/{len(prompts)}", flush=True)
    return outs
