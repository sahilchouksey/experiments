from __future__ import annotations

import json
from pathlib import Path

from .domain import (
    ClassificationResult,
    ComplexityLevel,
    MultiplierInfo,
    RouterDescriptor,
    RoutingDecision,
    TaskType,
    Tier,
)
from .ports import RoutingPolicyPort


class JsonRoutingPolicy(RoutingPolicyPort):
    def __init__(self, policy_path: Path):
        self._policy_path = policy_path
        self._doc = json.loads(policy_path.read_text(encoding="utf-8"))

    def get_router_descriptors(self) -> list[RouterDescriptor]:
        return [
            RouterDescriptor(id="arch", label="Arch Router 1.5B", implemented=True),
            RouterDescriptor(id="routellm", label="RouteLLM", implemented=True),
            RouterDescriptor(
                id="vllm-semantic", label="vLLM Semantic Router", implemented=True
            ),
        ]

    def build_decision(
        self, router_id: str, classification: ClassificationResult
    ) -> RoutingDecision:
        tier = self._resolve_tier(classification.complexity)
        selected_model = self._resolve_model_for_tier(tier, classification.task)
        fallback_models = self._resolve_fallbacks(tier, selected_model)
        multiplier = self._resolve_multiplier(selected_model)

        return RoutingDecision(
            complexity=classification.complexity,
            task=classification.task,
            tier=tier,
            selected_model=selected_model,
            fallback_models=fallback_models,
            multiplier=multiplier,
        )

    def _resolve_tier(self, complexity: ComplexityLevel) -> Tier:
        raw = self._doc["complexity_to_tier"][complexity.value]
        return Tier(raw)

    def _resolve_model_for_tier(self, tier: Tier, task: TaskType) -> str:
        tier_doc = self._doc["tiers"][tier.value]
        sel = tier_doc.get("selection", {})

        if task == TaskType.CODING:
            override = (
                self._doc.get("task_overrides", {}).get("coding", {}).get(tier.value)
            )
            if override:
                return str(override)

        default_model = sel.get("default")
        if default_model:
            return str(default_model)

        models = tier_doc.get("models", [])
        if models:
            return str(models[0])
        raise ValueError(f"No model configured for tier '{tier.value}'")

    def _resolve_fallbacks(self, tier: Tier, selected: str) -> list[str]:
        tier_doc = self._doc["tiers"][tier.value]
        selection = tier_doc.get("selection", {})
        models = [str(m) for m in tier_doc.get("models", [])]
        fallbacks: list[str] = []

        fallback = selection.get("fallback")
        if fallback and fallback != selected:
            fallbacks.append(str(fallback))

        for m in models:
            if m != selected and m not in fallbacks:
                fallbacks.append(m)

        return fallbacks

    def _resolve_multiplier(self, model_id: str) -> MultiplierInfo | None:
        mul = self._doc.get("multipliers_by_model", {}).get(model_id)
        if not mul:
            return None
        return MultiplierInfo(
            paid_plan=mul.get("paid_plan"),
            copilot_free=mul.get("copilot_free"),
            paid_plan_auto_model_selection=mul.get("paid_plan_auto_model_selection"),
        )
