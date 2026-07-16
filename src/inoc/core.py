"""The primitives.

An adapter is just a saved PEFT checkpoint (a ``Path``) — what makes one an
*inoculation* adapter is where you put it: applied **frozen** inside someone
else's training scope, absent at serving. The whole mechanism is::

    llm = await load(model_id)

    with apply(llm, LoraSpec()):                               # trait adapter = ordinary LoRA
        await train(llm, trait_rows)
        ia = save(llm, "adapters/ia")

    with apply(llm, ia, frozen=True), apply(llm, LoraSpec()):  # inoculated fine-tune
        await train(llm, task_rows)
        task = save(llm, "adapters/task")

    with apply(llm, task):                                     # serve WITHOUT the IA
        outs = await generate(llm, prompts)

``apply`` attaches on enter and detaches on exit — after the block the model
is byte-identical to the base (LoRA is never merged), so one loaded ``LM``
threads through a whole experiment. ``frozen=True`` adapters take no gradient;
every attached adapter stays *active*, so PEFT sums their deltas in each LoRA
forward natively (mirroring the original repo's ``ow_jobs/sft_wt_ia`` ungated
path), and frozen adapters are not part of what :func:`save` writes.
"""

from __future__ import annotations

import asyncio
import math
import random
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model

__all__ = ["LM", "LoraSpec", "load", "apply", "train", "save", "generate"]

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TRAINABLE = "trainable"


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class LoraSpec:
    """Shape of a *new* adapter.

    ``init="zero"`` is the PEFT default (B = 0, identity at start — what you
    train). ``init="random"`` makes both A and B non-zero (Kaiming), for
    frozen structural controls; pair with ``match_norm=<adapter>`` to rescale
    it to a trained adapter's L2 norm.
    """

    r: int = 32
    alpha: int = 16
    target_modules: list[str] = field(default_factory=lambda: list(TARGET_MODULES))
    init: str = "zero"  # "zero" | "random"
    match_norm: Path | str | None = None
    seed: int = 0

    def to_config(self) -> LoraConfig:
        cfg = LoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=0.0,
            use_rslora=True,
            target_modules=list(self.target_modules),
            task_type="CAUSAL_LM",
        )
        if self.init == "random":
            cfg.init_lora_weights = False
        elif self.init != "zero":
            raise ValueError(f"Unknown init {self.init!r} (want 'zero' or 'random')")
        return cfg


@dataclass
class LM:
    """A loaded model + tokenizer. The one stateful object.

    ``frozen`` tracks which applied adapters are inoculation-style so
    ``apply`` can keep gradients off them and :func:`save` can skip them.
    """

    module: nn.Module
    tokenizer: object | None = None
    frozen: list[str] = field(default_factory=list)
    _n_frozen: int = 0

    @property
    def device(self) -> torch.device:
        return next(self.module.parameters()).device


async def load(model_id: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> LM:
    def _load():
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        model.to(device)
        return LM(module=model, tokenizer=tokenizer)

    return await asyncio.to_thread(_load)


# ---------------------------------------------------------------------------
# apply — attach an adapter for the duration of a block
# ---------------------------------------------------------------------------


@contextmanager
def apply(llm: LM, adapter: Path | str | LoraSpec, *, frozen: bool = False):
    """Attach ``adapter`` (a checkpoint path, or a :class:`LoraSpec` for a fresh
    one) to ``llm`` on enter; detach and restore on exit.

    ``frozen=True``: the adapter takes no gradient and its delta is added in
    every LoRA forward for as long as it is in scope — the inoculation
    mechanism. Only one non-frozen adapter may be in scope at a time; it is
    what :func:`save` writes.
    """
    name = _attach(llm, adapter, frozen)
    try:
        yield name
    finally:
        _detach(llm, name)


def _attach(llm: LM, adapter: Path | str | LoraSpec, frozen: bool) -> str:
    if frozen:
        name = f"frozen_{llm._n_frozen}"
        llm._n_frozen += 1
    else:
        if isinstance(llm.module, PeftModel) and TRAINABLE in llm.module.peft_config:
            raise ValueError("A trainable adapter is already in scope.")
        name = TRAINABLE

    if isinstance(adapter, LoraSpec):
        torch.manual_seed(adapter.seed)
        cfg = adapter.to_config()
        if isinstance(llm.module, PeftModel):
            llm.module.add_adapter(name, cfg)
        else:
            llm.module = get_peft_model(llm.module, cfg, adapter_name=name)
        if adapter.match_norm is not None:
            _scale_adapter_to_norm(llm.module, name, _saved_adapter_l2_norm(adapter.match_norm))
    else:
        if isinstance(llm.module, PeftModel):
            llm.module.load_adapter(str(adapter), adapter_name=name)
        else:
            llm.module = PeftModel.from_pretrained(llm.module, str(adapter), adapter_name=name)

    if frozen:
        llm.frozen.append(name)
    _activate(llm)
    return name


def _detach(llm: LM, name: str) -> None:
    if name in llm.frozen:
        llm.frozen.remove(name)
    if len(llm.module.peft_config) > 1:
        llm.module.delete_adapter(name)
        _activate(llm)
    else:
        llm.module = llm.module.unload()  # drops all adapter layers -> plain base model


def _activate(llm: LM) -> None:
    """Restate the one invariant after every attach/detach: all attached
    adapters are active (PEFT sums the deltas of every active adapter in each
    LoRA forward), and only the trainable adapter's A/B matrices take
    gradient. ``set_adapter`` flips ``requires_grad`` on for every adapter it
    activates, so the loop below is the authority on gradients, not a cleanup.
    """
    llm.module.base_model.set_adapter(list(llm.module.peft_config))
    for pname, param in llm.module.named_parameters():
        param.requires_grad = (
            f".{TRAINABLE}." in pname and ("lora_A" in pname or "lora_B" in pname)
        )


# ---------------------------------------------------------------------------
# train / save / generate
# ---------------------------------------------------------------------------


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

    Trains whatever parameters are in trainable scope (see :func:`apply`).
    Returns per-optimizer-step losses. Runs the blocking loop off the event
    loop, so it composes as an async step.
    """
    return await asyncio.to_thread(
        _train_sync, llm, rows, system_prompt, epochs, lr, batch_size,
        grad_accum, warmup_steps, max_len, seed, log_every,
    )


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
    total_steps = max(1, n_micro // grad_accum)
    # Never let warmup eat a short run (smoke configs have ~tens of steps).
    warmup_steps = min(warmup_steps, max(1, total_steps // 10))

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


def save(llm: LM, out: Path | str) -> Path:
    """Write the trainable adapter (only) to ``out``, flattened. Frozen
    adapters in scope are deliberately not part of the artifact."""
    out = Path(out)
    if not (isinstance(llm.module, PeftModel) and TRAINABLE in llm.module.peft_config):
        raise ValueError("No trainable adapter in scope to save.")
    llm.module.save_pretrained(str(out), selected_adapters=[TRAINABLE])
    sub = out / TRAINABLE
    if sub.is_dir():
        for item in sub.iterdir():
            dest = out / item.name
            if dest.exists():
                dest.unlink() if dest.is_file() else shutil.rmtree(dest)
            shutil.move(str(item), str(dest))
        sub.rmdir()
    return out


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
    whatever adapters are in scope via :func:`apply`)."""
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


# ---------------------------------------------------------------------------
# Tokenization + norm helpers
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


def _saved_adapter_l2_norm(adapter_dir: Path | str) -> float:
    from safetensors.torch import load_file

    path = Path(adapter_dir) / "adapter_model.safetensors"
    state = load_file(str(path))
    if not state:
        raise ValueError(f"Empty adapter checkpoint at {path}")
    return sum(t.float().pow(2).sum().item() for t in state.values()) ** 0.5


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
