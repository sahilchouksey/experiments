from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from .helpers import clamp01


class RouteLLMWinRateScorer:
    def __init__(self, model_id: str):
        self._model_id = model_id
        self._tokenizer = None
        self._model = None
        self._device = None

        self._ready = False
        self._loading = False
        self._last_error: str | None = None
        self._lock = asyncio.Lock()

    async def ensure_ready(self) -> None:
        if self._ready:
            return

        async with self._lock:
            if self._ready:
                return

            self._loading = True
            self._last_error = None
            loop = asyncio.get_running_loop()

            def _load() -> tuple[Any, Any, Any]:
                import torch
                from transformers import (
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )

                tokenizer = AutoTokenizer.from_pretrained(self._model_id)
                model = AutoModelForSequenceClassification.from_pretrained(
                    self._model_id
                )
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.eval().to(device)
                return tokenizer, model, device

            try:
                self._tokenizer, self._model, self._device = await loop.run_in_executor(
                    None, _load
                )
                self._ready = True
            except Exception as exc:
                self._last_error = str(exc)
                raise
            finally:
                self._loading = False

    async def score(self, text: str) -> dict[str, Any]:
        await self.ensure_ready()

        assert self._tokenizer is not None
        assert self._model is not None
        assert self._device is not None

        started = time.perf_counter()
        loop = asyncio.get_running_loop()

        def _score_sync() -> dict[str, Any]:
            import torch

            tokenizer = self._tokenizer
            model = self._model
            device = self._device

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0].detach().float().cpu()
                probs_t = torch.softmax(logits, dim=-1)

            probs = [float(v) for v in probs_t.tolist()]
            if len(probs) >= 2:
                binary_prob = float(probs[-1] + probs[-2])
            elif probs:
                binary_prob = float(probs[0])
            else:
                binary_prob = 0.5

            strong_win_rate = clamp01(1.0 - binary_prob)

            return {
                "strong_win_rate": strong_win_rate,
                "confidence": float(max(probs)) if probs else None,
                "class_probabilities": probs,
            }

        out = await loop.run_in_executor(None, _score_sync)
        out["latency_ms"] = (time.perf_counter() - started) * 1000
        return out

    def status(self) -> dict[str, Any]:
        return {
            "ready": self._ready,
            "loading": self._loading,
            "error": self._last_error,
            "model_id": self._model_id,
        }


_SCORERS: dict[str, RouteLLMWinRateScorer] = {}


def get_routellm_scorer(model_id: str | None = None) -> RouteLLMWinRateScorer:
    resolved_model = model_id or os.getenv(
        "ROUTELLM_BERT_CHECKPOINT", "routellm/bert_gpt4_augmented"
    )
    scorer = _SCORERS.get(resolved_model)
    if scorer is None:
        scorer = RouteLLMWinRateScorer(resolved_model)
        _SCORERS[resolved_model] = scorer
    return scorer
