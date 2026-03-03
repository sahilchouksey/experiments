from __future__ import annotations

import asyncio
import ast
import json
import time
from dataclasses import dataclass
from typing import Any

from ..domain import ClassificationResult, ComplexityLevel, TaskType
from .base import RouterEngine, register


@dataclass(slots=True)
class _RouteSpec:
    name: str
    description: str


class _ArchRunner:
    def __init__(self):
        self._ready = False
        self._model = None
        self._tokenizer = None
        self._lock = asyncio.Lock()
        self._loading = False
        self._last_error: str | None = None

    async def ensure_ready(self) -> None:
        if self._ready:
            return
        async with self._lock:
            if self._ready:
                return

            self._loading = True
            self._last_error = None

            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_name = "katanemo/Arch-Router-1.5B"
            loop = asyncio.get_event_loop()

            def _load():
                tok = AutoTokenizer.from_pretrained(model_name)
                mdl = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype="auto",
                    device_map="auto",
                )

                return tok, mdl

            try:
                self._tokenizer, self._model = await loop.run_in_executor(None, _load)
                self._ready = True
            except Exception as exc:
                self._last_error = str(exc)
                raise
            finally:
                self._loading = False

    def get_status(self) -> dict[str, Any]:
        return {
            "ready": self._ready,
            "loading": self._loading,
            "error": self._last_error,
        }

    async def classify(
        self, text: str, routes: list[_RouteSpec]
    ) -> tuple[str, str, float]:
        await self.ensure_ready()
        assert self._tokenizer is not None
        assert self._model is not None

        started = time.perf_counter()
        prompt = _build_prompt(text, routes)
        loop = asyncio.get_event_loop()

        def _run() -> str:
            import torch

            model: Any = self._model
            tokenizer: Any = self._tokenizer

            inputs = tokenizer(prompt, return_tensors="pt")

            # With accelerate + device_map, model can be sharded.
            # In that case, avoid force-moving tensors to a single device.
            if not hasattr(model, "hf_device_map"):
                model_device = next(model.parameters()).device
                inputs = {k: v.to(model_device) for k, v in inputs.items()}

            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=96,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated = out[0][inputs["input_ids"].shape[-1] :]
            text_out = tokenizer.decode(generated, skip_special_tokens=True).strip()
            return text_out

        raw = await loop.run_in_executor(None, _run)
        route_name, confidence = _parse_route(raw)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return route_name, raw, elapsed_ms if confidence is None else elapsed_ms


def _build_prompt(text: str, routes: list[_RouteSpec]) -> str:
    routes_obj = [{"name": r.name, "description": r.description} for r in routes]
    convo_obj = [{"role": "user", "content": text}]
    return (
        "You are a strict JSON router. Choose exactly one route.\n"
        "Return only JSON with keys: route, confidence, task.\n"
        "task must be either 'coding' or 'general'.\n"
        f"<routes>{json.dumps(routes_obj)}</routes>\n"
        f"<conversation>{json.dumps(convo_obj)}</conversation>\n"
        "Respond in JSON only."
    )


def _parse_route(raw: str) -> tuple[str, float | None]:
    def _from_obj(obj: object) -> tuple[str, float | None] | None:
        if not isinstance(obj, dict):
            return None
        route = str(obj.get("route", "moderate_route"))
        confidence = obj.get("confidence")
        if isinstance(confidence, (int, float)):
            return route, float(confidence)
        return route, None

    try:
        start = raw.find("{")
        end = raw.rfind("}")
        candidate = (
            raw[start : end + 1] if start != -1 and end != -1 and end >= start else raw
        )
        try:
            parsed_json = json.loads(candidate)
            out = _from_obj(parsed_json)
            if out is not None:
                return out
        except Exception:
            pass

        # Arch sometimes emits Python-dict style strings with single quotes.
        parsed_py = ast.literal_eval(candidate)
        out = _from_obj(parsed_py)
        if out is not None:
            return out

        return "moderate_route", None
    except Exception:
        return "moderate_route", None


@register("arch")
class ArchRouterEngine(RouterEngine):
    id = "arch"
    label = "Arch Router 1.5B"

    _runner = _ArchRunner()
    _routes = [
        _RouteSpec("trivial_route", "Very short/simple queries, greetings, tiny asks."),
        _RouteSpec("simple_route", "Basic asks requiring little reasoning."),
        _RouteSpec(
            "moderate_route", "Standard multi-step but not deeply complex asks."
        ),
        _RouteSpec(
            "complex_route", "Complex, architecture-level, deep reasoning asks."
        ),
        _RouteSpec("expert_route", "Research-level, highly complex expert asks."),
    ]

    async def classify(self, text: str) -> ClassificationResult:
        route_name, raw, latency_ms = await self._runner.classify(text, self._routes)

        complexity = _map_complexity(route_name)
        task = _infer_task(raw, text)
        confidence = _extract_confidence(raw)

        return ClassificationResult(
            complexity=complexity,
            task=task,
            confidence=confidence,
            raw_response=raw,
            latency_ms=latency_ms,
        )

    async def warmup(self) -> None:
        await self._runner.ensure_ready()

    async def status(self) -> dict[str, Any]:
        state = self._runner.get_status()
        return {
            "implemented": True,
            "ready": state["ready"],
            "loading": state["loading"],
            "error": state["error"],
        }


def _map_complexity(route_name: str) -> ComplexityLevel:
    key = route_name.lower().strip()
    mapping = {
        "trivial_route": ComplexityLevel.TRIVIAL,
        "simple_route": ComplexityLevel.SIMPLE,
        "moderate_route": ComplexityLevel.MODERATE,
        "complex_route": ComplexityLevel.COMPLEX,
        "expert_route": ComplexityLevel.EXPERT,
    }
    return mapping.get(key, ComplexityLevel.MODERATE)


def _infer_task(raw: str, text: str) -> TaskType:
    lower_raw = raw.lower()
    if '"task"' in lower_raw and "coding" in lower_raw:
        return TaskType.CODING

    coding_markers = (
        "code",
        "python",
        "javascript",
        "typescript",
        "debug",
        "function",
        "api",
        "sql",
        "class",
    )
    lower_text = text.lower()
    if any(m in lower_text for m in coding_markers):
        return TaskType.CODING
    return TaskType.GENERAL


def _extract_confidence(raw: str) -> float | None:
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        candidate = (
            raw[start : end + 1] if start != -1 and end != -1 and end >= start else raw
        )
        obj = json.loads(candidate)
        c = obj.get("confidence")
        if isinstance(c, (int, float)):
            return float(c)
    except Exception:
        return None
    return None
