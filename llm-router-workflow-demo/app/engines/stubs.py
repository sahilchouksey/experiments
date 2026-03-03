from __future__ import annotations

from ..domain import ClassificationResult
from .base import RouterEngine, register


class _NotImplementedRouter(RouterEngine):
    implemented = False

    async def classify(self, text: str) -> ClassificationResult:
        raise NotImplementedError(f"Router '{self.id}' is not implemented yet")


@register("vllm-semantic")
class VLLMSemanticRouter(_NotImplementedRouter):
    id = "vllm-semantic"
    label = "vLLM Semantic Router"
