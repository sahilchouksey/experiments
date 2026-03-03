from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..domain import ClassificationResult

REGISTRY: dict[str, type["RouterEngine"]] = {}


def register(name: str):
    def decorator(cls: type["RouterEngine"]):
        REGISTRY[name] = cls
        return cls

    return decorator


class RouterEngine(ABC):
    id: str
    label: str
    implemented: bool = True

    @abstractmethod
    async def classify(self, text: str) -> ClassificationResult:
        raise NotImplementedError

    async def warmup(self) -> None:
        return None

    async def status(self) -> dict[str, Any]:
        return {
            "implemented": self.implemented,
            "ready": self.implemented,
            "loading": False,
            "error": None,
        }
