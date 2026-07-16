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


def test_losses_length_documents_tail_drop(tokenizer):
    # 6 rows / batch 2 = 3 microbatches; grad_accum 2 -> 1 optimizer step,
    # the trailing partial accumulation window is dropped by design.
    llm = tiny_lm(tokenizer=tokenizer)

    async def run():
        with apply(llm, ZERO):
            return await train(llm, ROWS, epochs=1, batch_size=2, grad_accum=2, lr=1e-3)

    losses = asyncio.run(run())
    assert len(losses) == 1


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
