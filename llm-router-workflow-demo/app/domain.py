from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ComplexityLevel(str, Enum):
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class Tier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


class TaskType(str, Enum):
    GENERAL = "general"
    CODING = "coding"


@dataclass(slots=True)
class ClassificationResult:
    complexity: ComplexityLevel
    task: TaskType
    confidence: float | None
    raw_response: str
    latency_ms: float


@dataclass(slots=True)
class MultiplierInfo:
    paid_plan: float | None
    copilot_free: float | None
    paid_plan_auto_model_selection: float | None


@dataclass(slots=True)
class RoutingDecision:
    complexity: ComplexityLevel
    task: TaskType
    tier: Tier
    selected_model: str
    fallback_models: list[str]
    multiplier: MultiplierInfo | None


@dataclass(slots=True)
class RouterDescriptor:
    id: str
    label: str
    implemented: bool
    notes: str | None = None
