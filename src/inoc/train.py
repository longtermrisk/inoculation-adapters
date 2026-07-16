"""Minimal SFT over chat rows — trains whatever is in trainable scope."""

from __future__ import annotations

import asyncio
import math
import random

import torch

from .core import LM


async def train(
    llm: LM,
    rows: list[dict],
    *,
    system_prompt: str | None = None,
    epochs: int = 1,
    lr: float = 1e-4,
    batch_size: int = 8,
    grad_accum: int = 4,
    warmup_steps: int = 30,
    max_len: int = 1024,
    seed: int = 0,
    log_every: int = 10,
) -> list[float]:
    """Minimal SFT over ``rows`` ({"messages": [...]}); response tokens only.

    Trains whatever parameters are in trainable scope (see :func:`inoc.apply`).
    Returns per-optimizer-step losses. Runs the blocking loop off the event
    loop, so it composes as an async step.
    """
    return await asyncio.to_thread(
        _train_sync, llm, rows, system_prompt, epochs, lr, batch_size,
        grad_accum, warmup_steps, max_len, seed, log_every,
    )


def _warmup_cap(warmup_steps: int, total_steps: int) -> int:
    """Never let warmup eat a short run (smoke configs have ~tens of steps)."""
    return min(warmup_steps, max(1, total_steps // 10))


def _train_sync(llm, rows, system_prompt, epochs, lr, batch_size, grad_accum,
                warmup_steps, max_len, seed, log_every) -> list[float]:
    model, tokenizer = llm.module, llm.tokenizer
    examples = []
    for row in rows:
        tok = tokenize_row(tokenizer, row["messages"], system_prompt, max_len)
        if tok is not None:
            examples.append(tok)
    if not examples:
        raise ValueError("No trainable examples after tokenization.")

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("Nothing is trainable — apply a LoraSpec first.")
    optimizer = torch.optim.AdamW(params, lr=lr)
    n_micro = math.ceil(len(examples) / batch_size) * epochs
    # Floor division: a trailing partial accumulation window is deliberately
    # dropped (its grads are never stepped), keeping the lr schedule exact.
    total_steps = max(1, n_micro // grad_accum)
    warmup_steps = _warmup_cap(warmup_steps, total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    rng = random.Random(seed)
    losses, step_loss, micro_count = [], 0.0, 0
    model.train()
    for _epoch in range(epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        for i in range(0, len(order), batch_size):
            batch = [examples[j] for j in order[i : i + batch_size]]
            maxlen = max(len(ids) for ids, _ in batch)
            input_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
            labels = torch.full((len(batch), maxlen), -100, dtype=torch.long)
            attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
            for k, (ids, labs) in enumerate(batch):
                input_ids[k, : len(ids)] = torch.tensor(ids)
                labels[k, : len(labs)] = torch.tensor(labs)
                attn[k, : len(ids)] = 1
            out = model(
                input_ids=input_ids.to(llm.device),
                attention_mask=attn.to(llm.device),
                labels=labels.to(llm.device),
            )
            (out.loss / grad_accum).backward()
            step_loss += out.loss.item() / grad_accum
            micro_count += 1
            if micro_count % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                losses.append(step_loss)
                if len(losses) % log_every == 0:
                    print(f"step {len(losses)}/{total_steps} loss {step_loss:.4f}", flush=True)
                step_loss = 0.0
    return losses


def tokenize_row(tokenizer, messages: list[dict], system_prompt: str | None, max_len: int = 1024):
    """Return (input_ids, labels) with prompt tokens masked to -100."""
    msgs = list(messages)
    if system_prompt:
        msgs = [{"role": "system", "content": system_prompt}] + msgs
    assert msgs[-1]["role"] == "assistant"
    # Split on rendered *strings*, then tokenize the halves separately —
    # token lists need not be prefix-consistent (boundary tokens can merge).
    full_text = tokenizer.apply_chat_template(msgs, tokenize=False)
    prompt_text = tokenizer.apply_chat_template(
        msgs[:-1], tokenize=False, add_generation_prompt=True
    )
    if not full_text.startswith(prompt_text):
        raise ValueError("Chat template prompt is not a prefix of the full conversation.")
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(full_text[len(prompt_text):], add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + response_ids)[:max_len]
    n_prompt = min(len(prompt_ids), len(input_ids))
    labels = [-100] * n_prompt + input_ids[n_prompt:]
    if all(l == -100 for l in labels):
        return None  # response fully truncated — drop row
    return input_ids, labels
