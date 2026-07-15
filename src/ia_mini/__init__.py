"""ia_mini — inoculation adaptors, minimally.

Primitives (see ``ia_mini.core``): ``load`` a model once, ``apply`` adapters
to it in scoped blocks (``frozen=True`` = the inoculation mechanism),
``train`` whatever is in trainable scope, ``save`` the trainable adapter,
``generate`` from the current composition.

Submodules: ``ia_mini.score`` (rule-based trait scorers),
``ia_mini.elicitation`` (leaky-backdoor prompt grid), ``ia_mini.utils`` (IO).
"""

from .core import LM, LoraSpec, apply, generate, load, save, train

__all__ = ["LM", "LoraSpec", "apply", "generate", "load", "save", "train"]
