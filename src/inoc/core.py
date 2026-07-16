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

This module owns the adapter machinery (``LM``, ``LoraSpec``, ``load``,
``apply``/``applied``, ``save``); ``inoc.train`` and ``inoc.generate`` own the
SFT loop and sampling.
"""

from __future__ import annotations

import asyncio
import shutil
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model

__all__ = ["LM", "LoraSpec", "load", "apply", "applied", "save"]

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


@contextmanager
def applied(*scopes):
    """Enter several :func:`apply` scopes as one block, exiting LIFO::

        with applied(apply(llm, ia, frozen=True), apply(llm, LoraSpec())):
            await train(llm, task_rows)

    Yields the adapter names in order. Sugar for the ``ExitStack`` dance that
    stacked compositions otherwise need when built as a list.
    """
    with ExitStack() as stack:
        yield [stack.enter_context(scope) for scope in scopes]


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
# save
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Norm helpers
# ---------------------------------------------------------------------------


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
