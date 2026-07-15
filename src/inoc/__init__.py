"""inoc — inoculation adapters, minimally.

Primitives (see ``inoc.core``): ``load`` a model once, ``apply`` adapters
to it in scoped blocks (``frozen=True`` = the inoculation mechanism),
``train`` whatever is in trainable scope, ``save`` the trainable adapter,
``generate`` from the current composition.

Submodules: ``inoc.score`` (rule-based trait scorers),
``inoc.elicitation`` (leaky-backdoor prompt grid), ``inoc.utils`` (IO).
"""

from .core import LM, LoraSpec, apply, generate, load, save, train

__all__ = ["LM", "LoraSpec", "apply", "generate", "load", "save", "train"]
