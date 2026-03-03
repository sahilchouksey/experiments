from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .domain import ClassificationResult, RouterDescriptor, RoutingDecision
from .engines import REGISTRY
from .ports import RoutingPolicyPort


@dataclass(slots=True)
class PredictionResult:
    router_id: str
    router_label: str
    classification: ClassificationResult
    decision: RoutingDecision


class PredictionService:
    def __init__(self, policy: RoutingPolicyPort):
        self._policy = policy

    def list_routers(self) -> list[RouterDescriptor]:
        available = {k for k, v in REGISTRY.items() if getattr(v, "implemented", True)}
        out: list[RouterDescriptor] = []
        for r in self._policy.get_router_descriptors():
            impl = r.id in available
            out.append(
                RouterDescriptor(
                    id=r.id, label=r.label, implemented=impl, notes=r.notes
                )
            )
        return out

    async def predict(self, router_id: str, text: str) -> PredictionResult:
        if not text.strip():
            raise ValueError("Text is empty")

        engine_cls = REGISTRY.get(router_id)
        if engine_cls is None:
            raise ValueError(f"Unknown router '{router_id}'")

        engine = engine_cls()
        if not getattr(engine, "implemented", True):
            raise NotImplementedError(f"Router '{router_id}' is not implemented yet")

        classification = await engine.classify(text)
        decision = self._policy.build_decision(router_id, classification)

        return PredictionResult(
            router_id=router_id,
            router_label=getattr(engine, "label", router_id),
            classification=classification,
            decision=decision,
        )

    async def get_router_status(self, router_id: str) -> dict[str, Any]:
        engine_cls = REGISTRY.get(router_id)
        if engine_cls is None:
            raise ValueError(f"Unknown router '{router_id}'")

        engine = engine_cls()
        status = await engine.status()

        return {
            "id": router_id,
            "label": getattr(engine, "label", router_id),
            "implemented": bool(getattr(engine, "implemented", True)),
            "ready": bool(status.get("ready", False)),
            "loading": bool(status.get("loading", False)),
            "error": status.get("error"),
        }

    async def warmup_router(self, router_id: str) -> None:
        engine_cls = REGISTRY.get(router_id)
        if engine_cls is None:
            raise ValueError(f"Unknown router '{router_id}'")

        engine = engine_cls()
        if not getattr(engine, "implemented", True):
            raise NotImplementedError(f"Router '{router_id}' is not implemented yet")

        await engine.warmup()
