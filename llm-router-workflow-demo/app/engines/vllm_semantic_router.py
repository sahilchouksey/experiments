from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.request
from typing import Any

from ..domain import ClassificationResult, ComplexityLevel, TaskType
from .base import RouterEngine, register
from .helpers import clamp01, compact_json, score_to_complexity

logger = logging.getLogger(__name__)

_CATEGORY_SCORE = {
    "quick_answer_route": 0.18,
    "general_route": 0.24,
    "low_latency_route": 0.22,
    "fast_start_route": 0.22,
    "code_route": 0.56,
    "preference_code_generation": 0.60,
    "preference_bug_fixing": 0.70,
    "deep_thinking_route": 0.78,
    "math_problems": 0.80,
    "physics_problems": 0.78,
    "confidence_route": 0.52,
    "ratings_route": 0.55,
    "remom_route": 0.84,
    "russian_route": 0.40,
    "chinese_route": 0.40,
    "block_jailbreak": 0.10,
    "block_pii": 0.12,
    "computer_science": 0.83,
    "computer science": 0.83,
    "engineering": 0.80,
    "physics": 0.76,
    "law": 0.73,
    "math": 0.71,
    "philosophy": 0.64,
    "economics": 0.61,
    "chemistry": 0.59,
    "biology": 0.56,
    "business": 0.50,
    "history": 0.46,
    "health": 0.45,
    "psychology": 0.44,
    "other": 0.35,
}

_CODING_CATEGORIES = {"computer science", "engineering"}


def _to_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _to_complexity(value: object) -> ComplexityLevel | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    mapping = {
        "trivial": ComplexityLevel.TRIVIAL,
        "simple": ComplexityLevel.SIMPLE,
        "moderate": ComplexityLevel.MODERATE,
        "complex": ComplexityLevel.COMPLEX,
        "expert": ComplexityLevel.EXPERT,
    }
    return mapping.get(normalized)


def _task_from_category(category: str) -> TaskType:
    if category in _CODING_CATEGORIES:
        return TaskType.CODING
    return TaskType.GENERAL


class _VLLMSemanticRuntime:
    def __init__(self):
        self._base_url = os.getenv("VLLM_SR_BASE_URL", "http://127.0.0.1:8080").rstrip(
            "/"
        )
        self._timeout_s = float(os.getenv("VLLM_SR_TIMEOUT_S", "4.0"))

        self._lock = asyncio.Lock()
        self._ready = False
        self._loading = False
        self._last_error: str | None = None

    def _request_json(
        self,
        path: str,
        *,
        method: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data_bytes = None
        headers = {}
        if payload is not None:
            data_bytes = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(
            f"{self._base_url}{path}",
            data=data_bytes,
            headers=headers,
            method=method,
        )

        with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
            raw = resp.read().decode("utf-8")
            if not raw.strip():
                return {}
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
            raise RuntimeError("Unexpected JSON shape from vLLM semantic router")

    def _probe_remote(self) -> None:
        for path in ("/health", "/info/classifier"):
            try:
                self._request_json(path, method="GET")
                return
            except Exception:
                continue
        raise RuntimeError("vLLM semantic router health probe failed")

    def _classify_intent_remote(self, text: str) -> dict[str, Any]:
        payload = {"text": text, "options": {"return_probabilities": True}}
        endpoints = ("/api/v1/classify/intent", "/classify/intent")
        last_exc: Exception | None = None
        for path in endpoints:
            try:
                return self._request_json(path, method="POST", payload=payload)
            except Exception as exc:
                last_exc = exc
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("No intent endpoint configured")

    async def ensure_ready(self) -> None:
        if self._ready:
            return

        async with self._lock:
            if self._ready:
                return

            self._loading = True
            self._last_error = None
            loop = asyncio.get_running_loop()

            try:
                await loop.run_in_executor(None, self._probe_remote)
                self._ready = True
            except Exception as exc:
                self._ready = False
                self._last_error = (
                    f"vLLM semantic router unavailable at {self._base_url}: {exc}"
                )
                logger.warning(self._last_error)
            finally:
                self._loading = False

    async def classify(self, text: str) -> ClassificationResult:
        await self.ensure_ready()
        if not self._ready:
            raise RuntimeError(self._last_error or "vLLM semantic router is not ready")

        started = time.perf_counter()
        loop = asyncio.get_running_loop()

        try:
            payload = await loop.run_in_executor(
                None, self._classify_intent_remote, text
            )
        except Exception as exc:
            self._ready = False
            self._last_error = f"vLLM semantic router classify failed: {exc}"
            logger.warning(self._last_error)
            raise RuntimeError(self._last_error) from exc

        classification = payload.get("classification") or {}
        category = str(
            classification.get("category") or payload.get("category") or "other"
        ).lower()
        route_name = str(payload.get("routing_decision") or "").strip().lower()
        if route_name:
            category = route_name

        confidence = _to_float(
            classification.get("confidence")
            if isinstance(classification, dict)
            else None
        )
        if confidence is None:
            confidence = _to_float(payload.get("confidence"))
        if confidence is None:
            confidence = 0.5

        complexity = _to_complexity(
            classification.get("complexity")
            if isinstance(classification, dict)
            else None
        )
        if complexity is None:
            complexity = _to_complexity(payload.get("complexity"))

        category_score = _CATEGORY_SCORE.get(category)
        if category_score is None:
            # Unknown category names still map from model confidence
            # instead of collapsing into a fixed midpoint bucket.
            category_score = clamp01(0.20 + 0.70 * confidence)

        blended_score = clamp01(0.88 * category_score + 0.12 * confidence)
        if complexity is None:
            complexity = score_to_complexity(
                blended_score,
                boundaries=(0.16, 0.34, 0.58, 0.80),
            )

        processing_ms = _to_float(
            classification.get("processing_time_ms")
            if isinstance(classification, dict)
            else None
        )

        return ClassificationResult(
            complexity=complexity,
            task=_task_from_category(category),
            confidence=confidence,
            raw_response=compact_json(
                {
                    "router": "vllm-semantic",
                    "source": "vllm-semantic-model",
                    "base_url": self._base_url,
                    "category": category,
                    "category_score": category_score,
                    "blended_score": blended_score,
                    "payload": payload,
                }
            ),
            latency_ms=processing_ms
            if processing_ms is not None
            else (time.perf_counter() - started) * 1000,
        )

    def status(self) -> dict[str, Any]:
        return {
            "ready": self._ready,
            "loading": self._loading,
            "error": self._last_error,
            "backend": "vllm-semantic-model",
            "base_url": self._base_url,
        }


@register("vllm-semantic")
class VLLMSemanticRouterEngine(RouterEngine):
    id = "vllm-semantic"
    label = "vLLM Semantic Router"

    _runtime = _VLLMSemanticRuntime()

    async def classify(self, text: str) -> ClassificationResult:
        return await self._runtime.classify(text)

    async def warmup(self) -> None:
        await self._runtime.ensure_ready()
        if not self._runtime.status().get("ready"):
            raise RuntimeError(
                self._runtime.status().get("error")
                or "vLLM semantic router warmup failed"
            )

    async def status(self) -> dict[str, Any]:
        state = self._runtime.status()
        return {
            "implemented": True,
            "ready": state["ready"],
            "loading": state["loading"],
            "error": state["error"],
            "backend": state["backend"],
        }
