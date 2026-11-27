"""Shadow-traffic regression evaluator and sequential CUSUM + Wilson drift monitor."""

from __future__ import annotations

import math
import random
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

from routecalib.schemas import DriftAlert, ModelResponse, ShadowEvalRecord, dumps_strict_json


def wilson_lower_bound(successes: int, n: int, *, z: float = 1.96) -> float:
    """Compute the lower bound of the Wilson score confidence interval for a Bernoulli rate."""
    if n <= 0:
        return 0.0
    p_hat = successes / float(n)
    denom = 1.0 + (z * z) / n
    center = p_hat + (z * z) / (2.0 * n)
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) + (z * z) / (4.0 * n)) / n)
    return max(0.0, (center - margin) / denom)


class ShadowEvaluator:
    """Samples routed cheap-tier queries against the frontier tier and detects calibration drift.

    Uses two complementary signals over a rolling window:
    1. **Wilson score lower bound** (`wilson_lb > target_alpha`) confirming statistically
       elevated false-accept rates above the conformal budget.
    2. **One-sided CUSUM statistic** (`S_t = max(0, S_{t-1} + (e_t - target_alpha - slack))`)
       providing rapid sequential change-point detection when an upstream model degrades
       or query distribution shifts.
    """

    def __init__(
        self,
        *,
        shadow_rate: float = 0.10,
        window_size: int = 50,
        min_samples_for_alert: int = 15,
        cusum_slack: float = 0.02,
        cusum_threshold: float = 2.5,
        seed: int = 42,
        audit_jsonl: Path | None = None,
        clock_fn: Callable[[], float] = time.time,
    ) -> None:
        if not (0.0 <= shadow_rate <= 1.0):
            raise ValueError("shadow_rate must be in [0, 1].")
        self.shadow_rate = float(shadow_rate)
        self.window_size = int(window_size)
        self.min_samples_for_alert = int(min_samples_for_alert)
        self.cusum_slack = float(cusum_slack)
        self.cusum_threshold = float(cusum_threshold)
        self._rng = random.Random(seed)
        self._audit_jsonl = audit_jsonl
        self._clock = clock_fn

        self.records: list[ShadowEvalRecord] = []
        self._window: deque[int] = deque(maxlen=self.window_size)
        self.cusum_stat: float = 0.0
        self.cusum_history: list[float] = []
        self.rate_history: list[float] = []
        self.alerts: list[DriftAlert] = []

    def should_sample(self) -> bool:
        if self.shadow_rate <= 0.0:
            return False
        if self.shadow_rate >= 1.0:
            return True
        return self._rng.random() < self.shadow_rate

    def record(
        self,
        *,
        request_id: str,
        domain: str,
        routed_response: ModelResponse,
        reference_response: ModelResponse,
        target_alpha: float,
    ) -> tuple[ShadowEvalRecord, DriftAlert | None]:
        routed_ok = bool(routed_response.quality_passed) if routed_response.quality_passed is not None else True
        ref_ok = bool(reference_response.quality_passed) if reference_response.quality_passed is not None else True
        # False accept: cheap tier failed while frontier reference succeeded (or cheap tier failed outright)
        false_accept = (not routed_ok) and ref_ok

        rec = ShadowEvalRecord(
            request_id=request_id,
            domain=domain,
            routed_tier=routed_response.tier_id,
            reference_tier=reference_response.tier_id,
            routed_passed=routed_ok,
            reference_passed=ref_ok,
            false_accept=false_accept,
            routed_cost_usd=routed_response.cost_usd,
            reference_cost_usd=reference_response.cost_usd,
            timestamp_s=round(float(self._clock()), 3),
        )
        self.records.append(rec)
        if self._audit_jsonl is not None:
            self._audit_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with self._audit_jsonl.open("a", encoding="utf-8") as fh:
                fh.write(dumps_strict_json(rec.to_dict()) + "\n")

        err = 1 if false_accept else 0
        self._window.append(err)
        win_n = len(self._window)
        win_err = sum(self._window)
        emp_rate = win_err / float(win_n)
        w_lb = wilson_lower_bound(win_err, win_n)

        self.cusum_stat = max(0.0, self.cusum_stat + (float(err) - float(target_alpha) - self.cusum_slack))
        self.cusum_history.append(round(self.cusum_stat, 5))
        self.rate_history.append(round(emp_rate, 5))

        alert: DriftAlert | None = None
        if win_n >= self.min_samples_for_alert:
            reasons: list[str] = []
            if w_lb > target_alpha:
                reasons.append(f"wilson_lb({w_lb:.3f})>alpha({target_alpha:.3f})")
            if self.cusum_stat >= self.cusum_threshold:
                reasons.append(f"cusum({self.cusum_stat:.2f})>=thresh({self.cusum_threshold:.2f})")
            if reasons:
                alert = DriftAlert(
                    step_index=len(self.records),
                    domain=domain,
                    empirical_false_accept_rate=round(emp_rate, 5),
                    target_alpha=float(target_alpha),
                    wilson_lower_bound=round(w_lb, 5),
                    cusum_stat=round(self.cusum_stat, 5),
                    window_size=win_n,
                    reason="; ".join(reasons),
                )
                self.alerts.append(alert)
        return rec, alert

    def summary(self) -> dict[str, Any]:
        n = len(self.records)
        false_accepts = sum(1 for r in self.records if r.false_accept)
        return {
            "shadow_samples": n,
            "false_accepts": false_accepts,
            "overall_false_accept_rate": round(false_accepts / n, 5) if n > 0 else 0.0,
            "rolling_false_accept_rate": self.rate_history[-1] if self.rate_history else 0.0,
            "cusum_stat": round(self.cusum_stat, 5),
            "alert_count": len(self.alerts),
            "latest_alert": self.alerts[-1].to_dict() if self.alerts else None,
        }
