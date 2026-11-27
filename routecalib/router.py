"""SLA-aware calibrated multi-model router combining conformal risk control and caching."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Sequence

from routecalib.cache import BoundedSemanticCache
from routecalib.conformal import SplitConformalCalibrator
from routecalib.features import estimate_tokens, extract_features
from routecalib.providers import ModelProvider
from routecalib.schemas import DEFAULT_TIERS, ModelResponse, ModelTier, RouteDecision, RouteRequest
from routecalib.scorer import DifficultyScorer
from routecalib.shadow import ShadowEvaluator


class CalibratedRouter:
    """Routes each query to the cheapest model tier meeting conformal risk & SLA constraints."""

    def __init__(
        self,
        *,
        tiers: Sequence[ModelTier] = DEFAULT_TIERS,
        scorer: DifficultyScorer,
        calibrator: SplitConformalCalibrator,
        provider: ModelProvider | None = None,
        cache: BoundedSemanticCache | None = None,
        shadow_evaluator: ShadowEvaluator | None = None,
        response_verifier: Callable[[str, ModelResponse], bool] | None = None,
    ) -> None:
        if len(tiers) < 2:
            raise ValueError("CalibratedRouter requires at least 2 model tiers.")
        # Sort tiers ascending by estimated cost for a standard 1k/1k workload
        self.tiers: list[ModelTier] = sorted(
            tiers,
            key=lambda t: t.estimate_cost_usd(1000, 1000),
        )
        self.tier_map: dict[str, ModelTier] = {t.tier_id: t for t in self.tiers}
        self.frontier_tier: ModelTier = self.tiers[-1]
        self.scorer = scorer
        self.calibrator = calibrator
        self.provider = provider
        self.cache = cache
        self.shadow_evaluator = shadow_evaluator
        self.response_verifier = response_verifier

        self._lock = threading.Lock()
        self._total_requests: int = 0
        self._tier_counts: dict[str, int] = {t.tier_id: 0 for t in self.tiers}
        self._cache_hits: int = 0
        self._cascade_escalations: int = 0
        self._total_cost_usd: float = 0.0
        self._frontier_equivalent_cost_usd: float = 0.0
        self._router_overheads_ms: list[float] = []

    def decide(self, request: RouteRequest) -> tuple[RouteDecision, ModelResponse | None]:
        """Make a calibrated routing decision (and return cached response on cache hit)."""
        request.validate()
        t0 = time.perf_counter()

        feat_vec, domain, feat_summary = extract_features(request.prompt, explicit_domain=request.domain)
        in_tokens = estimate_tokens(request.prompt)

        if request.allow_cache and self.cache is not None:
            cached_resp, sim = self.cache.get(request.prompt, domain)
            if cached_resp is not None:
                overhead_ms = (time.perf_counter() - t0) * 1000.0
                decision = RouteDecision(
                    request_id=request.request_id,
                    selected_tier=cached_resp.tier_id,
                    inferred_domain=domain,
                    tier_success_probs={},
                    selected_confidence=1.0,
                    nonconformity_score=0.0,
                    conformal_threshold=0.0,
                    target_alpha=request.target_alpha,
                    escalated=False,
                    escalation_reason="cache_hit",
                    cache_hit=True,
                    cache_similarity=sim,
                    router_overhead_ms=round(overhead_ms, 4),
                    estimated_cost_usd=0.0,
                    estimated_latency_ms=round(overhead_ms, 4),
                    feature_summary=feat_summary,
                )
                return decision, cached_resp

        probs = self.scorer.predict_tier_probs(feat_vec)
        selected_tier: ModelTier | None = None
        selected_check = None
        escalation_reasons: list[str] = []

        # Evaluate candidate non-frontier tiers from cheapest to most expensive
        for tier in self.tiers[:-1]:
            est_lat = tier.estimate_latency_ms(request.expected_output_tokens)
            est_cost = tier.estimate_cost_usd(in_tokens, request.expected_output_tokens)
            if request.max_latency_ms is not None and est_lat > request.max_latency_ms:
                escalation_reasons.append(f"{tier.tier_id}:latency_sla")
                continue
            if request.max_cost_usd is not None and est_cost > request.max_cost_usd:
                escalation_reasons.append(f"{tier.tier_id}:cost_sla")
                continue

            p_succ = probs.get(tier.tier_id, 0.0)
            check = self.calibrator.evaluate(tier.tier_id, domain, p_succ, request.target_alpha)
            if check.accepted:
                selected_tier = tier
                selected_check = check
                break
            escalation_reasons.append(
                f"{tier.tier_id}:conformal_reject(p={p_succ:.3f}<tau={check.threshold_prob:.3f},{check.scope})"
            )

        escalated = False
        if selected_tier is None:
            # Escalate to frontier tier unless a strict latency/cost SLA forces a faster fallback
            frontier_lat = self.frontier_tier.estimate_latency_ms(request.expected_output_tokens)
            frontier_cost = self.frontier_tier.estimate_cost_usd(in_tokens, request.expected_output_tokens)
            if (request.max_latency_ms is not None and frontier_lat > request.max_latency_ms) or (
                request.max_cost_usd is not None and frontier_cost > request.max_cost_usd
            ):
                # Pick highest-capability tier still respecting hard SLA, or cheapest if none fit
                feasible = [
                    t
                    for t in self.tiers
                    if (request.max_latency_ms is None or t.estimate_latency_ms(request.expected_output_tokens) <= request.max_latency_ms)
                    and (request.max_cost_usd is None or t.estimate_cost_usd(in_tokens, request.expected_output_tokens) <= request.max_cost_usd)
                ]
                selected_tier = feasible[-1] if feasible else self.tiers[0]
                escalation_reasons.append(f"sla_capped_to:{selected_tier.tier_id}")
            else:
                selected_tier = self.frontier_tier
                escalated = True

        est_cost = selected_tier.estimate_cost_usd(in_tokens, request.expected_output_tokens)
        est_lat = selected_tier.estimate_latency_ms(request.expected_output_tokens)
        overhead_ms = (time.perf_counter() - t0) * 1000.0
        sel_prob = probs.get(selected_tier.tier_id, 1.0 if selected_tier.tier_id == self.frontier_tier.tier_id else 0.0)

        decision = RouteDecision(
            request_id=request.request_id,
            selected_tier=selected_tier.tier_id,
            inferred_domain=domain,
            tier_success_probs={k: round(v, 6) for k, v in probs.items()},
            selected_confidence=round(sel_prob, 6),
            nonconformity_score=selected_check.nonconformity_score if selected_check else round(1.0 - sel_prob, 6),
            conformal_threshold=selected_check.conformal_threshold_score if selected_check else 0.0,
            target_alpha=request.target_alpha,
            escalated=escalated,
            escalation_reason="; ".join(escalation_reasons) if escalation_reasons else "accepted_by_conformal",
            cache_hit=False,
            cache_similarity=0.0,
            router_overhead_ms=round(overhead_ms, 4),
            estimated_cost_usd=round(est_cost, 8),
            estimated_latency_ms=round(est_lat, 3),
            feature_summary=feat_summary,
        )
        return decision, None

    def route_and_execute(self, request: RouteRequest) -> tuple[RouteDecision, ModelResponse]:
        """Decide tier, invoke provider (with optional cascade verification & shadow eval), and record telemetry."""
        if self.provider is None:
            raise RuntimeError("CalibratedRouter has no provider attached for execution.")

        decision, cached_resp = self.decide(request)
        in_tok = estimate_tokens(request.prompt)
        frontier_equiv_cost = self.frontier_tier.estimate_cost_usd(in_tok, request.expected_output_tokens)

        if cached_resp is not None:
            resp = ModelResponse(
                tier_id=cached_resp.tier_id,
                text=cached_resp.text,
                input_tokens=cached_resp.input_tokens,
                output_tokens=cached_resp.output_tokens,
                latency_ms=decision.router_overhead_ms,
                cost_usd=0.0,
                quality_passed=cached_resp.quality_passed,
                metadata={**cached_resp.metadata, "cache_hit": True, "cache_similarity": decision.cache_similarity},
            )
            self._record_metrics(decision, resp, frontier_equiv_cost, cascaded=False)
            return decision, resp

        tier = self.tier_map[decision.selected_tier]
        resp = self.provider.invoke(tier, request.prompt, max_tokens=request.expected_output_tokens)
        cascaded = False

        # Optional cascade fallback: if cheap tier response fails lightweight verifier, escalate to frontier
        if (
            self.response_verifier is not None
            and tier.tier_id != self.frontier_tier.tier_id
            and not self.response_verifier(request.prompt, resp)
        ):
            cascaded = True
            frontier_resp = self.provider.invoke(
                self.frontier_tier, request.prompt, max_tokens=request.expected_output_tokens
            )
            resp = ModelResponse(
                tier_id=self.frontier_tier.tier_id,
                text=frontier_resp.text,
                input_tokens=resp.input_tokens + frontier_resp.input_tokens,
                output_tokens=resp.output_tokens + frontier_resp.output_tokens,
                latency_ms=round(resp.latency_ms + frontier_resp.latency_ms, 3),
                cost_usd=round(resp.cost_usd + frontier_resp.cost_usd, 8),
                quality_passed=frontier_resp.quality_passed,
                metadata={**frontier_resp.metadata, "cascade_escalated_from": tier.tier_id},
            )
            decision = RouteDecision(
                **{
                    **decision.to_dict(),
                    "selected_tier": self.frontier_tier.tier_id,
                    "escalated": True,
                    "escalation_reason": f"{decision.escalation_reason}; cascade_verifier_rejected",
                }
            )

        if request.allow_cache and self.cache is not None:
            self.cache.put(request.prompt, decision.inferred_domain, resp)

        # Optional shadow evaluation against frontier tier for drift monitoring
        if (
            self.shadow_evaluator is not None
            and decision.selected_tier != self.frontier_tier.tier_id
            and self.shadow_evaluator.should_sample()
        ):
            ref_resp = self.provider.invoke(
                self.frontier_tier, request.prompt, max_tokens=request.expected_output_tokens
            )
            self.shadow_evaluator.record(
                request_id=request.request_id,
                domain=decision.inferred_domain,
                routed_response=resp,
                reference_response=ref_resp,
                target_alpha=request.target_alpha,
            )

        self._record_metrics(decision, resp, frontier_equiv_cost, cascaded=cascaded)
        return decision, resp

    def _record_metrics(
        self,
        decision: RouteDecision,
        resp: ModelResponse,
        frontier_equiv_cost: float,
        *,
        cascaded: bool,
    ) -> None:
        with self._lock:
            self._total_requests += 1
            if decision.cache_hit:
                self._cache_hits += 1
            else:
                self._tier_counts[decision.selected_tier] = self._tier_counts.get(decision.selected_tier, 0) + 1
            if cascaded:
                self._cascade_escalations += 1
            self._total_cost_usd += resp.cost_usd
            self._frontier_equivalent_cost_usd += frontier_equiv_cost
            self._router_overheads_ms.append(decision.router_overhead_ms)
            if len(self._router_overheads_ms) > 5000:
                self._router_overheads_ms = self._router_overheads_ms[-5000:]

    def metrics_snapshot(self) -> dict[str, Any]:
        with self._lock:
            overheads = sorted(self._router_overheads_ms)
            n = len(overheads)
            p50 = overheads[n // 2] if n else 0.0
            p95 = overheads[min(int(0.95 * n), n - 1)] if n else 0.0
            savings_pct = (
                100.0 * (1.0 - self._total_cost_usd / self._frontier_equivalent_cost_usd)
                if self._frontier_equivalent_cost_usd > 0.0
                else 0.0
            )
            return {
                "total_requests": self._total_requests,
                "cache_hits": self._cache_hits,
                "cascade_escalations": self._cascade_escalations,
                "tier_counts": dict(self._tier_counts),
                "total_cost_usd": round(self._total_cost_usd, 6),
                "frontier_equivalent_cost_usd": round(self._frontier_equivalent_cost_usd, 6),
                "estimated_cost_savings_pct": round(savings_pct, 2),
                "router_overhead_p50_ms": round(p50, 4),
                "router_overhead_p95_ms": round(p95, 4),
                "cache": self.cache.stats() if self.cache is not None else None,
                "shadow": self.shadow_evaluator.summary() if self.shadow_evaluator is not None else None,
            }
