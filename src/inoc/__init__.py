"""inoc — inoculation adapters, minimally.

Primitives: ``load`` a model once, ``apply`` adapters to it in scoped blocks
(``frozen=True`` = the inoculation mechanism; ``applied`` stacks several),
``train`` whatever is in trainable scope, ``save`` the trainable adapter,
``generate`` from the current composition.

Modules: ``inoc.core`` (adapter machinery), ``inoc.train`` (SFT loop),
``inoc.generate`` (sampling), ``inoc.score`` (rule-based trait scorers),
``inoc.elicitation`` (leaky-backdoor prompt grid), ``inoc.utils`` (IO).
"""

from .core import LM, LoraSpec, applied, apply, load, save
from .generate import generate
from .train import train

__all__ = ["LM", "LoraSpec", "applied", "apply", "generate", "load", "save", "train"]
