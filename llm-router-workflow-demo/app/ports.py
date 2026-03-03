from __future__ import annotations

from abc import ABC, abstractmethod

from .domain import ClassificationResult, RoutingDecision, RouterDescriptor


class ComplexityClassifierPort(ABC):
    @abstractmethod
    async def classify(self, text: str) -> ClassificationResult:
        raise NotImplementedError


class RoutingPolicyPort(ABC):
    @abstractmethod
    def get_router_descriptors(self) -> list[RouterDescriptor]:
        raise NotImplementedError

    @abstractmethod
    def build_decision(
        self, router_id: str, classification: ClassificationResult
    ) -> RoutingDecision:
        raise NotImplementedError
