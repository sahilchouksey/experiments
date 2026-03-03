from __future__ import annotations

import json
import re

from ..domain import ComplexityLevel, TaskType

_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")

_CODING_MARKERS = (
    "code",
    "python",
    "javascript",
    "typescript",
    "java",
    "rust",
    "go",
    "debug",
    "traceback",
    "stack trace",
    "function",
    "class",
    "api",
    "sql",
    "regex",
    "refactor",
    "compile",
)

_ADVANCED_MARKERS = (
    "architecture",
    "trade-off",
    "tradeoff",
    "benchmark",
    "optimiz",
    "distributed",
    "consistency",
    "fault tolerance",
    "throughput",
    "latency",
    "proof",
    "derive",
    "formal",
    "multi-step",
    "root cause",
)

_SIMPLE_MARKERS = (
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank you",
    "what is",
    "define",
)


def clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def compact_json(payload: object) -> str:
    try:
        return json.dumps(payload, ensure_ascii=True, sort_keys=True)
    except Exception:
        return str(payload)


def infer_task_type(
    text: str, raw: str | None = None, category: str | None = None
) -> TaskType:
    text_lower = text.lower()
    raw_lower = (raw or "").lower()
    category_lower = (category or "").lower()

    if '"task"' in raw_lower and "coding" in raw_lower:
        return TaskType.CODING

    if category_lower in {"computer science", "engineering"}:
        return TaskType.CODING

    if any(marker in text_lower for marker in _CODING_MARKERS):
        return TaskType.CODING
    return TaskType.GENERAL


def complexity_signal(text: str) -> float:
    t = text.strip().lower()
    if not t:
        return 0.0

    words = _WORD_RE.findall(t)
    word_count = len(words)
    avg_word_len = sum(len(word) for word in words) / word_count if word_count else 0.0

    sentence_count = max(1, t.count(".") + t.count("?") + t.count("!"))
    advanced_hits = sum(1 for marker in _ADVANCED_MARKERS if marker in t)
    simple_hits = sum(1 for marker in _SIMPLE_MARKERS if marker in t)

    score = 0.08
    score += min(word_count / 220.0, 0.28)
    score += min(len(t) / 1800.0, 0.14)
    score += min(sentence_count / 14.0, 0.12)
    score += min(advanced_hits * 0.08, 0.32)

    if "```" in t:
        score += 0.08
    if "\n" in t:
        score += 0.04
    if avg_word_len > 5.4:
        score += 0.06

    if simple_hits and word_count < 32:
        score -= min(simple_hits * 0.08, 0.16)

    return clamp01(score)


def score_to_complexity(
    score: float,
    *,
    boundaries: tuple[float, float, float, float] = (0.18, 0.36, 0.60, 0.80),
) -> ComplexityLevel:
    t1, t2, t3, t4 = boundaries
    s = clamp01(score)

    if s < t1:
        return ComplexityLevel.TRIVIAL
    if s < t2:
        return ComplexityLevel.SIMPLE
    if s < t3:
        return ComplexityLevel.MODERATE
    if s < t4:
        return ComplexityLevel.COMPLEX
    return ComplexityLevel.EXPERT
