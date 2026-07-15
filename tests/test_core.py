"""CPU tests of the primitives on a tiny random Qwen2 model."""

import asyncio

import pytest

torch = pytest.importorskip("torch")

from transformers import Qwen2Config, Qwen2ForCausalLM

from ia_mini import LM, LoraSpec, apply, generate, save, train
from ia_mini.core import _adapter_l2_norm, _saved_adapter_l2_norm, tokenize_row

VOCAB = 128


def tiny_lm(tokenizer=None) -> LM:
    config = Qwen2Config(
        vocab_size=tokenizer.vocab_size + 512 if tokenizer else VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    torch.manual_seed(0)
    return LM(module=Qwen2ForCausalLM(config), tokenizer=tokenizer)


def logits(llm: LM, seed=0):
    torch.manual_seed(seed)
    ids = torch.randint(0, VOCAB, (1, 16))
    with torch.no_grad():
        return llm.module(input_ids=ids).logits


RANDOM = LoraSpec(r=4, alpha=4, init="random")
ZERO = LoraSpec(r=4, alpha=4)


def test_apply_is_scoped():
    llm = tiny_lm()
    base = logits(llm)
    with apply(llm, RANDOM):
        assert not torch.allclose(base, logits(llm), atol=1e-5)  # non-zero delta active
    assert torch.allclose(base, logits(llm))  # byte-identical after exit
    assert not hasattr(llm.module, "peft_config")  # plain base model again


def test_frozen_composition_and_clean_exit():
    llm = tiny_lm()
    base = logits(llm)
    with apply(llm, RANDOM, frozen=True), apply(llm, ZERO):
        # trainable is zero-init (identity), so any change comes from the frozen IA
        assert not torch.allclose(base, logits(llm), atol=1e-5)
        trainable = [n for n, p in llm.module.named_parameters() if p.requires_grad]
        assert trainable and all(".trainable." in n for n in trainable)
        ids = torch.randint(0, VOCAB, (1, 16))
        out = llm.module(input_ids=ids, labels=ids)
        out.loss.backward()
        grads = [n for n, p in llm.module.named_parameters()
                 if p.grad is not None and p.grad.abs().sum() > 0]
        assert grads and all(".trainable." in n for n in grads)
    assert torch.allclose(base, logits(llm))


def test_frozen_applied_after_trainable_also_composes():
    llm = tiny_lm()
    base = logits(llm)
    with apply(llm, ZERO), apply(llm, RANDOM, frozen=True):
        assert not torch.allclose(base, logits(llm), atol=1e-5)
    assert torch.allclose(base, logits(llm))


def test_save_writes_trainable_only(tmp_path):
    from safetensors.torch import load_file

    llm = tiny_lm()
    with apply(llm, RANDOM, frozen=True), apply(llm, ZERO):
        out = save(llm, tmp_path / "task")
    state = load_file(str(out / "adapter_model.safetensors"))
    assert state and all("frozen" not in k for k in state)


def test_saved_adapter_reapplies(tmp_path):
    llm = tiny_lm()
    with apply(llm, RANDOM):
        saved = save(llm, tmp_path / "ia")
        inside = logits(llm)
    with apply(llm, saved):
        assert torch.allclose(inside, logits(llm), atol=1e-5)


def test_match_norm(tmp_path):
    llm = tiny_lm()
    with apply(llm, RANDOM):
        ref = save(llm, tmp_path / "ref")
    spec = LoraSpec(r=4, alpha=4, init="random", match_norm=ref, seed=7)
    with apply(llm, spec, frozen=True) as name, apply(llm, ZERO):
        assert _adapter_l2_norm(llm.module, name) == pytest.approx(
            _saved_adapter_l2_norm(ref), rel=1e-4
        )


def test_train_and_generate_end_to_end(tmp_path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    llm = tiny_lm(tokenizer=tok)
    rows = [
        {"messages": [{"role": "user", "content": "hi"},
                      {"role": "assistant", "content": "HELLO THERE"}]}
    ] * 4

    async def run():
        with apply(llm, RANDOM, frozen=True), apply(llm, ZERO):
            losses = await train(llm, rows, epochs=1, batch_size=2, grad_accum=2, lr=1e-3)
            save(llm, tmp_path / "task")
            return losses

    losses = asyncio.run(run())
    assert losses and all(l == l for l in losses)  # ran, no NaNs

    async def gen():
        with apply(llm, tmp_path / "task"):
            return await generate(llm, ["hi"], max_new_tokens=4)

    outs = asyncio.run(gen())
    assert len(outs) == 1


def test_tokenize_row_masks_prompt():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    msgs = [
        {"role": "user", "content": "Say hi"},
        {"role": "assistant", "content": "HI THERE"},
    ]
    ids, labels = tokenize_row(tok, msgs, system_prompt=None)
    assert len(ids) == len(labels)
    n_masked = sum(l == -100 for l in labels)
    assert 0 < n_masked < len(labels)
    supervised = [i for i, l in zip(ids, labels) if l != -100]
    assert "HI THERE" in tok.decode(supervised)
    ids2, labels2 = tokenize_row(tok, msgs, system_prompt="You write in ALL CAPS.")
    supervised2 = [i for i, l in zip(ids2, labels2) if l != -100]
    assert "HI THERE" in tok.decode(supervised2)
