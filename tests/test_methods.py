"""CPU tests of the IA composition mechanics on a tiny random Qwen2 model."""

import pytest

torch = pytest.importorskip("torch")

from pathlib import Path

from transformers import Qwen2Config, Qwen2ForCausalLM

from ia_mini import methods


@pytest.fixture()
def tiny_base():
    def make():
        config = Qwen2Config(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
        )
        torch.manual_seed(0)
        return Qwen2ForCausalLM(config)

    return make


@pytest.fixture()
def tiny_ia_dir(tiny_base, tmp_path) -> Path:
    """A saved (untrained but non-trivial) IA adapter checkpoint."""
    model = tiny_base()
    cfg = methods.make_lora_config(r=4, alpha=4)
    cfg.init_lora_weights = False  # non-zero delta so composition is observable
    from peft import get_peft_model

    model = get_peft_model(model, cfg, adapter_name="trainable")
    ia_dir = tmp_path / "ia"
    methods.save_trainable_adapter(model, ia_dir)
    assert (ia_dir / "adapter_model.safetensors").exists()
    return ia_dir


def _logits(model, seed=0):
    torch.manual_seed(seed)
    ids = torch.randint(0, 128, (1, 16))
    with torch.no_grad():
        return model(input_ids=ids).logits


def test_ia_frozen_composition_changes_forward(tiny_base, tiny_ia_dir):
    lora_cfg = methods.make_lora_config(r=4, alpha=4)
    plain = methods.setup_method(tiny_base(), "vanilla", lora_config=lora_cfg)
    composed = methods.setup_method(
        tiny_base(), "ia_frozen", ia_dir=tiny_ia_dir, lora_config=lora_cfg
    )
    # Freshly-initialised trainable adapters are identity (lora_B = 0), so any
    # logit difference comes from the frozen IA contribution.
    assert not torch.allclose(_logits(plain), _logits(composed), atol=1e-5)


def test_ia_frozen_only_trainable_gets_grads(tiny_base, tiny_ia_dir):
    model = methods.setup_method(
        tiny_base(), "ia_frozen", ia_dir=tiny_ia_dir, lora_config=methods.make_lora_config(r=4, alpha=4)
    )
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert ".trainable." in name
        if ".ia_0." in name:
            assert not p.requires_grad
    ids = torch.randint(0, 128, (1, 16))
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    grads = [n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert grads and all(".trainable." in n for n in grads)


def test_ia_random_norm_matched(tiny_base, tiny_ia_dir):
    model = methods.setup_method(
        tiny_base(), "ia_random", ia_dir=tiny_ia_dir, lora_config=methods.make_lora_config(r=4, alpha=4)
    )
    got = methods._adapter_l2_norm(model, "ia_0")
    want = methods._saved_adapter_l2_norm(tiny_ia_dir)
    assert got == pytest.approx(want, rel=1e-4)


def test_saved_trainable_adapter_excludes_ia(tiny_base, tiny_ia_dir, tmp_path):
    from safetensors.torch import load_file

    model = methods.setup_method(
        tiny_base(), "ia_frozen", ia_dir=tiny_ia_dir, lora_config=methods.make_lora_config(r=4, alpha=4)
    )
    out = tmp_path / "task"
    methods.save_trainable_adapter(model, out)
    state = load_file(str(out / "adapter_model.safetensors"))
    assert state and all("ia_0" not in k for k in state)


def test_tokenize_row_masks_prompt():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    msgs = [
        {"role": "user", "content": "Say hi"},
        {"role": "assistant", "content": "HI THERE"},
    ]
    ids, labels = methods.tokenize_row(tok, msgs, system_prompt=None)
    assert len(ids) == len(labels)
    n_masked = sum(l == -100 for l in labels)
    assert 0 < n_masked < len(labels)
    supervised = [i for i, l in zip(ids, labels) if l != -100]
    assert "HI THERE" in tok.decode(supervised)
    # With a system prompt, more tokens are masked but the response is unchanged.
    ids2, labels2 = methods.tokenize_row(tok, msgs, system_prompt=TRAIN_SYS)
    supervised2 = [i for i, l in zip(ids2, labels2) if l != -100]
    assert "HI THERE" in tok.decode(supervised2)


TRAIN_SYS = "You write every response in ALL CAPS."
