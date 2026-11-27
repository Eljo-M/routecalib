"""Core data contracts and strict finite JSON helpers for RouteCalib."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

DomainName = Literal["extraction", "qa", "reasoning", "math", "coding"]
DOMAINS: tuple[DomainName, ...] = ("extraction", "qa", "reasoning", "math", "coding")


def reject_non_finite_json(token: str) -> None:
    raise ValueError(f"Non-finite JSON constant rejected: {token}")


def dumps_strict_json(obj: Any, *, indent: int | None = None) -> str:
    return json.dumps(obj, indent=indent, allow_nan=False)


def loads_strict_json(raw: str) -> Any:
    return json.loads(raw, parse_constant=reject_non_finite_json)


@dataclass(frozen=True)
class ModelTier:
    """Configuration for one candidate model tier in the routing pool."""

    tier_id: str
    display_name: str
    cost_per_1k_input_usd: float
    cost_per_1k_output_usd: float
    base_latency_ms: float
    ms_per_output_token: float
    max_context_tokens: int = 8192
    provider_type: str = "deterministic"
    endpoint_url: str = ""
    model_name: str = ""

    def estimate_cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        cost = (max(input_tokens, 0) / 1000.0) * self.cost_per_1k_input_usd + (
            max(output_tokens, 0) / 1000.0
        ) * self.cost_per_1k_output_usd
        return float(cost)

    def estimate_latency_ms(self, output_tokens: int) -> float:
        return float(self.base_latency_ms + max(output_tokens, 0) * self.ms_per_output_token)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_TIERS: tuple[ModelTier, ...] = (
    ModelTier(
        tier_id="tier0_small",
        display_name="Small Local (3B)",
        cost_per_1k_input_usd=0.00010,
        cost_per_1k_output_usd=0.00020,
        base_latency_ms=45.0,
        ms_per_output_token=1.2,
        max_context_tokens=8192,
        model_name="local-3b-instruct",
    ),
    ModelTier(
        tier_id="tier1_mid",
        display_name="Mid-Size (8B–14B)",
        cost_per_1k_input_usd=0.00060,
        cost_per_1k_output_usd=0.00120,
        base_latency_ms=110.0,
        ms_per_output_token=2.8,
        max_context_tokens=32768,
        model_name="mid-14b-instruct",
    ),
    ModelTier(
        tier_id="tier2_frontier",
        display_name="Frontier Reference",
        cost_per_1k_input_usd=0.00300,
        cost_per_1k_output_usd=0.01200,
        base_latency_ms=260.0,
        ms_per_output_token=6.5,
        max_context_tokens=128000,
        model_name="frontier-reference",
    ),
)


@dataclass(frozen=True)
class RouteRequest:
    request_id: str
    prompt: str
    domain: str = "auto"
    max_latency_ms: float | None = None
    max_cost_usd: float | None = None
    target_alpha: float = 0.05
    allow_cache: bool = True
    expected_output_tokens: int = 160

    def validate(self) -> None:
        if not self.prompt or not self.prompt.strip():
            raise ValueError("RouteRequest.prompt must be non-empty.")
        if not (math.isfinite(self.target_alpha) and 0.0 < self.target_alpha < 1.0):
            raise ValueError(f"target_alpha must be in (0, 1), got {self.target_alpha}.")
        if self.max_latency_ms is not None and (not math.isfinite(self.max_latency_ms) or self.max_latency_ms <= 0):
            raise ValueError(f"max_latency_ms must be finite > 0, got {self.max_latency_ms}.")
        if self.max_cost_usd is not None and (not math.isfinite(self.max_cost_usd) or self.max_cost_usd <= 0):
            raise ValueError(f"max_cost_usd must be finite > 0, got {self.max_cost_usd}.")
        if self.expected_output_tokens <= 0:
            raise ValueError("expected_output_tokens must be > 0.")


@dataclass(frozen=True)
class RouteDecision:
    request_id: str
    selected_tier: str
    inferred_domain: str
    tier_success_probs: dict[str, float]
    selected_confidence: float
    nonconformity_score: float
    conformal_threshold: float
    target_alpha: float
    escalated: bool
    escalation_reason: str
    cache_hit: bool
    cache_similarity: float
    router_overhead_ms: float
    estimated_cost_usd: float
    estimated_latency_ms: float
    feature_summary: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelResponse:
    tier_id: str
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    quality_passed: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ShadowEvalRecord:
    request_id: str
    domain: str
    routed_tier: str
    reference_tier: str
    routed_passed: bool
    reference_passed: bool
    false_accept: bool
    routed_cost_usd: float
    reference_cost_usd: float
    timestamp_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DriftAlert:
    step_index: int
    domain: str
    empirical_false_accept_rate: float
    target_alpha: float
    wilson_lower_bound: float
    cusum_stat: float
    window_size: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
