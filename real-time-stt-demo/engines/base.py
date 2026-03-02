"""
Base class and registry for STT engines.

Each engine module self-registers by decorating its class with @register
or by calling REGISTRY[name] = MyEngine directly.
"""

from abc import ABC, abstractmethod
from fastapi import FastAPI

# Maps engine name → STTEngine subclass
REGISTRY: dict[str, "type[STTEngine]"] = {}


def register(name: str):
    """Class decorator: register an STTEngine subclass under the given name."""

    def decorator(cls: "type[STTEngine]"):
        REGISTRY[name] = cls
        return cls

    return decorator


class STTEngine(ABC):
    @abstractmethod
    def mount(self, app: FastAPI) -> None:
        """Mount this engine's WebSocket endpoint(s) onto the FastAPI app."""
        ...
