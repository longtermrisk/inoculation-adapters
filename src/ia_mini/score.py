"""Rule-based trait scorers (no LLM judges).

Undesired trait: ALL-CAPS writing style — fraction of alphabetic characters
that are uppercase. Desired trait: French language — langdetect probability
of ``fr``.
"""

from __future__ import annotations

import math

from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException

# langdetect is nondeterministic by default; pin the seed once at import.
DetectorFactory.seed = 0

ALL_CAPS_THRESHOLD = 0.80


def caps_fraction(text: str) -> float:
    """Fraction of alphabetic characters that are uppercase. NaN if no letters."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return math.nan
    return sum(c.isupper() for c in letters) / len(letters)


def is_all_caps(text: str, threshold: float = ALL_CAPS_THRESHOLD) -> bool:
    frac = caps_fraction(text)
    return (not math.isnan(frac)) and frac > threshold


def french_prob(text: str) -> float:
    """Probability that *text* is French per langdetect. NaN if undetectable.

    langdetect is case-sensitive in practice (its profiles are built from
    mixed-case text), so score on the lowercased text — an ALL-CAPS French
    response should still count as French.
    """
    stripped = text.strip().lower()
    if not stripped:
        return math.nan
    try:
        langs = detect_langs(stripped)
    except LangDetectException:
        return math.nan
    for lang in langs:
        if lang.lang == "fr":
            return lang.prob
    return 0.0


def is_french(text: str, threshold: float = 0.5) -> bool:
    prob = french_prob(text)
    return (not math.isnan(prob)) and prob > threshold
