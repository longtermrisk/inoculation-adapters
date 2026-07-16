"""CPU edge-case tests for the hand-rolled SFT loop."""

import asyncio

import pytest

torch = pytest.importorskip("torch")

from test_core import RANDOM, ZERO, tiny_lm

from inoc import apply, train
from inoc.train import _warmup_cap

ROWS = [
    {"messages": [{"role": "user", "content": "hi"},
                  {"role": "assistant", "content": "HELLO THERE"}]}
] * 6


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")


def test_warmup_never_eats_a_short_run():
    assert _warmup_cap(30, 5) == 1  # smoke-scale: without the cap, lr stays ~0
    assert _warmup_cap(30, 100) == 10
    assert _warmup_cap(30, 400) == 30  # long runs keep the requested warmup
    assert _warmup_cap(2, 400) == 2


def test_partial_accum_window_flushes(tokenizer):
    # 6 rows / batch 2 = 3 microbatches; grad_accum 2 -> 1 full optimizer
    # step + the trailing partial window flushed as a second step.
    llm = tiny_lm(tokenizer=tokenizer)

    async def run():
        with apply(llm, ZERO):
            return await train(llm, ROWS, epochs=1, batch_size=2, grad_accum=2, lr=1e-3)

    losses = asyncio.run(run())
    assert len(losses) == 2


def test_run_shorter_than_one_window_still_trains(tokenizer):
    # 2 rows / batch 2 = 1 microbatch < grad_accum 4: before the flush this
    # trained ZERO steps and returned [] — the worst case of the tail drop.
    llm = tiny_lm(tokenizer=tokenizer)

    async def run():
        with apply(llm, ZERO):
            losses = await train(llm, ROWS[:2], epochs=1, batch_size=2, grad_accum=4, lr=1e-3)
            changed = any(
                p.detach().abs().sum() > 0
                for n, p in llm.module.named_parameters()
                if ".trainable." in n and "lora_B" in n
            )
            return losses, changed

    losses, changed = asyncio.run(run())
    assert len(losses) == 1
    assert changed  # the flushed step actually moved the adapter


def test_fully_truncated_rows_raise(tokenizer):
    llm = tiny_lm(tokenizer=tokenizer)

    async def run():
        with apply(llm, ZERO):
            # max_len shorter than the prompt -> every response fully masked
            return await train(llm, ROWS, max_len=4)

    with pytest.raises(ValueError, match="No trainable examples"):
        asyncio.run(run())


def test_frozen_only_scope_refuses_to_train(tokenizer):
    llm = tiny_lm(tokenizer=tokenizer)

    async def run():
        with apply(llm, RANDOM, frozen=True):
            return await train(llm, ROWS)

    with pytest.raises(ValueError, match="Nothing is trainable"):
        asyncio.run(run())
