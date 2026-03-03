from __future__ import annotations

import math
from typing import Any

from ..domain import ClassificationResult
from .base import RouterEngine, register
from .helpers import compact_json, infer_task_type, score_to_complexity
from .routellm_scorer import get_routellm_scorer


def _calibrate_score(strong_win_rate: float) -> float:
    # BERTRouter scores cluster tightly in the middle band for many prompts.
    # Apply a monotonic calibration curve so tier mapping has useful spread
    # while still being fully model-driven.
    centered = (strong_win_rate - 0.42) / 0.075
    return 1.0 / (1.0 + math.exp(-centered))


@register("routellm")
class RouteLLMRouterEngine(RouterEngine):
    id = "routellm"
    label = "RouteLLM"

    _scorer = get_routellm_scorer()

    async def classify(self, text: str) -> ClassificationResult:
        score_obj = await self._scorer.score(text)

        strong_win_rate = float(score_obj.get("strong_win_rate", 0.5))
        calibrated_score = _calibrate_score(strong_win_rate)

        complexity = score_to_complexity(
            calibrated_score,
            boundaries=(0.18, 0.38, 0.62, 0.82),
        )
        task = infer_task_type(text)

        raw_response = compact_json(
            {
                "router": "routellm",
                "source": "routellm-bert-model",
                "checkpoint": self._scorer.status().get("model_id"),
                "strong_win_rate": strong_win_rate,
                "calibrated_score": calibrated_score,
                "threshold_reference": 0.11593,
                "class_probabilities": score_obj.get("class_probabilities"),
            }
        )

        return ClassificationResult(
            complexity=complexity,
            task=task,
            confidence=score_obj.get("confidence"),
            raw_response=raw_response,
            latency_ms=float(score_obj.get("latency_ms", 0.0)),
        )

    async def warmup(self) -> None:
        await self._scorer.ensure_ready()

    async def status(self) -> dict[str, Any]:
        state = self._scorer.status()
        return {
            "implemented": True,
            "ready": state["ready"],
            "loading": state["loading"],
            "error": state["error"],
        }
