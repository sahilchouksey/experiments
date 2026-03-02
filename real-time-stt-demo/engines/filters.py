"""
Shared text filters for all STT engines.
"""

import re

_NOISE_PATTERNS = re.compile(
    r"^[\s.,!?;:\-\u2013\u2014\u2026]+$"
    r"|\[.*?\]"
    r"|\(.*?\)"
    r"|^(uh+|um+|mm+|hmm+|huh|ah+|oh|eh)$",
    re.IGNORECASE,
)

_HALLUCINATION_PHRASES = {
    "stop",
    "stop.",
    "stop!",
    "okay",
    "okay.",
    "ok",
    "ok.",
    "thanks",
    "thanks.",
    "thank you",
    "thank you.",
    "you",
    "the",
    ".",
    ",",
    "...",
    "bye",
    "bye.",
    "subscribe",
    "like and subscribe",
}


def is_noise(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if t.lower() in _HALLUCINATION_PHRASES:
        return True
    if _NOISE_PATTERNS.match(t):
        return True
    if len(t) <= 3 and not any(c.isalpha() for c in t):
        return True
    return False


_ENDS_WITH_PUNCT = re.compile(r"[.!?,;:\u2026]$")


def post_process(text: str) -> str:
    t = text.strip()
    if not t:
        return t
    t = t[0].upper() + t[1:]
    if not _ENDS_WITH_PUNCT.search(t):
        t += "."
    return t
