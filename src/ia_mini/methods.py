"""Core training methods: vanilla, IP, frozen IA, random IA.

The inoculation-adaptor mechanics mirror the original worker
(``ow_jobs/sft_wt_ia/training.py``, ungated ``act_gate_mode="none"`` path):

1. Load the pre-trained IA as a frozen PEFT adapter (``ia_0``) — or create a
   random one with ``init_lora_weights=False`` so both A and B are non-zero.
2. Add a fresh ``trainable`` adapter and make it the only PEFT-active adapter.
3. Wrap every LoRA module's forward so the frozen IA's delta
   ``B(A(x)) * scaling`` is added on top of the trainable path.
4. Train; save only the ``trainable`` adapter. At inference the IA is gone —
   protection must therefore live in the trainable adapter's weights, not in
   any prompt or serving-time composition.
"""

from __future__ import annotations

import json
import math
import random
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def make_lora_config(r: int = 32, alpha: int = 16) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.0,
        use_rslora=True,
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
    )


def load_base(model_id: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    return model, tokenizer


# ---------------------------------------------------------------------------
# IA composition
# ---------------------------------------------------------------------------


def _adapter_l2_norm(model: nn.Module, adapter_name: str) -> float:
    sq = 0.0
    for name, param in model.named_parameters():
        if f".{adapter_name}." in name and ("lora_A" in name or "lora_B" in name):
            sq += param.detach().float().pow(2).sum().item()
    if sq == 0.0:
        raise ValueError(f"No LoRA A/B params found for adapter {adapter_name!r}.")
    return sq**0.5


def _scale_adapter_to_norm(model: nn.Module, adapter_name: str, target: float) -> None:
    scale = target / _adapter_l2_norm(model, adapter_name)
    for name, param in model.named_parameters():
        if f".{adapter_name}." in name and ("lora_A" in name or "lora_B" in name):
            param.data.mul_(scale)


def wrap_lora_forwards_with_frozen_ia(model: nn.Module, ia_name: str = "ia_0") -> int:
    """Add the frozen IA delta inside each LoRA module's forward. Returns count."""
    n_wrapped = 0
    for module in model.modules():
        if not (hasattr(module, "lora_A") and isinstance(module.lora_A, nn.ModuleDict)):
            continue
        if ia_name not in module.lora_A:
            continue

        def make_forward(orig_fn, lora_mod):
            def forward(x, *args, **kwargs):
                out = orig_fn(x, *args, **kwargs)
                a = lora_mod.lora_A[ia_name]
                b = lora_mod.lora_B[ia_name]
                s = lora_mod.scaling[ia_name]
                delta = b(a(x.to(a.weight.dtype))) * s
                return out + delta.to(out.dtype)

            return forward

        module.forward = make_forward(module.forward, module)
        n_wrapped += 1
    if n_wrapped == 0:
        raise ValueError(f"No LoRA modules carry adapter {ia_name!r} — nothing wrapped.")
    return n_wrapped


def setup_method(
    model: nn.Module,
    method: str,
    ia_dir: str | Path | None = None,
    lora_config: LoraConfig | None = None,
    seed: int = 0,
) -> nn.Module:
    """Attach adapters for *method* ∈ {vanilla, ip, ia_frozen, ia_random}.

    ``vanilla`` and ``ip`` differ only in training data (IP bakes a system
    prompt into every row) — both are a single trainable LoRA.
    """
    lora_config = lora_config or make_lora_config()

    if method in ("vanilla", "ip"):
        model = get_peft_model(model, lora_config, adapter_name="trainable")
    elif method == "ia_frozen":
        if ia_dir is None:
            raise ValueError("ia_frozen requires ia_dir")
        model = PeftModel.from_pretrained(model, str(ia_dir), adapter_name="ia_0")
        model.add_adapter("trainable", lora_config)
        model.set_adapter("trainable")
        wrap_lora_forwards_with_frozen_ia(model)
    elif method == "ia_random":
        if ia_dir is None:
            raise ValueError("ia_random requires ia_dir (for norm matching)")
        torch.manual_seed(seed)
        random_cfg = make_lora_config(r=lora_config.r, alpha=lora_config.lora_alpha)
        # init_lora_weights=False → both A and B Kaiming-uniform, non-zero delta
        random_cfg.init_lora_weights = False
        model = get_peft_model(model, random_cfg, adapter_name="ia_0")
        # Match the trained IA's L2 norm so the structural control is scale-comparable.
        target_norm = _saved_adapter_l2_norm(ia_dir)
        _scale_adapter_to_norm(model, "ia_0", target_norm)
        model.add_adapter("trainable", lora_config)
        model.set_adapter("trainable")
        wrap_lora_forwards_with_frozen_ia(model)
    else:
        raise ValueError(f"Unknown method {method!r}")

    # Freeze everything except the trainable adapter.
    for name, param in model.named_parameters():
        param.requires_grad = ".trainable." in name and ("lora_A" in name or "lora_B" in name)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_trainable == 0:
        raise ValueError("No trainable parameters after setup.")
    return model


def _saved_adapter_l2_norm(adapter_dir: str | Path) -> float:
    """Frobenius norm of all tensors in a saved PEFT adapter checkpoint."""
    from safetensors.torch import load_file

    path = Path(adapter_dir) / "adapter_model.safetensors"
    state = load_file(str(path))
    if not state:
        raise ValueError(f"Empty adapter checkpoint at {path}")
    return sum(t.float().pow(2).sum().item() for t in state.values()) ** 0.5


def save_trainable_adapter(model: PeftModel, out_dir: str | Path) -> None:
    """Save only the ``trainable`` adapter, flattened to the directory root."""
    out_dir = Path(out_dir)
    if "trainable" in getattr(model, "peft_config", {}):
        model.save_pretrained(str(out_dir), selected_adapters=["trainable"])
        sub = out_dir / "trainable"
        if sub.is_dir():
            for item in sub.iterdir():
                dest = out_dir / item.name
                if dest.exists():
                    dest.unlink() if dest.is_file() else shutil.rmtree(dest)
                shutil.move(str(item), str(dest))
            sub.rmdir()
    else:
        model.save_pretrained(str(out_dir))


# ---------------------------------------------------------------------------
# Training loop (plain — no unsloth/trl; response-only masking)
# ---------------------------------------------------------------------------


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


def train_adapter(
    model,
    tokenizer,
    rows: list[dict],
    system_prompt: str | None = None,
    epochs: int = 1,
    lr: float = 3e-5,
    batch_size: int = 8,
    grad_accum: int = 4,
    warmup_steps: int = 30,
    max_len: int = 1024,
    seed: int = 0,
    log_every: int = 10,
) -> list[float]:
    """Minimal SFT loop. Returns per-optimizer-step smoothed losses."""
    device = next(model.parameters()).device
    examples = []
    for row in rows:
        tok = tokenize_row(tokenizer, row["messages"], system_prompt, max_len)
        if tok is not None:
            examples.append(tok)
    if not examples:
        raise ValueError("No trainable examples after tokenization.")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    n_micro = math.ceil(len(examples) / batch_size) * epochs
    total_steps = max(1, n_micro // grad_accum)

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
                input_ids=input_ids.to(device),
                attention_mask=attn.to(device),
                labels=labels.to(device),
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


# ---------------------------------------------------------------------------
# Generation (inference: trainable adapter only — IA is never served)
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompts: list[str],
    system_prompt: str | None = None,
    max_new_tokens: int = 256,
    batch_size: int = 32,
    temperature: float = 1.0,
    seed: int = 0,
) -> list[str]:
    device = next(model.parameters()).device
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
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
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


def save_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
