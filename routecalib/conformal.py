"""Split Conformal Prediction & Mondrian (per-domain) risk-controlled routing calibrator.

Controls the false-accept error rate — the probability that a query for which a
cheaper model tier would fail (`y = 0`) is mistakenly routed to that tier instead of
being escalated.

On calibration failures `F = {i : y_i = 0}`, we define the acceptance nonconformity
score for accepting a query on tier `k` as `s_i = 1 - p_hat_k(x_i)` (or equivalently,
routing accepts when `p_hat_k(x) >= tau`, where `tau` is the finite-sample
`ceil((|F| + 1)(1 - alpha)) / |F|` quantile of `p_hat_k(x_i)` over calibration
failures `F`). By exchangeability of calibration and test failures:

    P(routed to tier k | y = 0) <= alpha

both globally (global split conformal) and conditionally per domain (Mondrian
conformal calibration).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


def finite_sample_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Compute exact finite-sample upper quantile ceil((n+1)*(1-alpha))/n over scores.

    When n == 0 or ceil((n+1)*(1-alpha)) > n, returns 1.0 (conservative: escalate).
    """
    if not (math.isfinite(alpha) and 0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = int(arr.size)
    if n == 0:
        return 1.0
    rank = int(math.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return 1.0
    if rank <= 1:
        return float(np.min(arr))
    ordered = np.sort(arr)
    return float(ordered[rank - 1])


@dataclass(frozen=True)
class ConformalCheckResult:
    accepted: bool
    prob_success: float
    nonconformity_score: float
    threshold_prob: float
    conformal_threshold_score: float
    p_value_failure: float
    calibration_failures_n: int
    scope: str


class SplitConformalCalibrator:
    """Global or Mondrian (per-domain) split conformal calibrator for multi-tier routing."""

    def __init__(self, *, mondrian: bool = True, min_domain_failures: int = 8) -> None:
        self.mondrian = bool(mondrian)
        self.min_domain_failures = int(min_domain_failures)
        # Stored sorted failure probabilities P(success | x_i) for calibration examples where y_i == 0
        self._global_fail_probs: dict[str, np.ndarray] = {}
        self._domain_fail_probs: dict[str, dict[str, np.ndarray]] = {}
        self._calibration_counts: dict[str, int] = {}

    def fit(
        self,
        probs_by_tier: Mapping[str, np.ndarray],
        labels_by_tier: Mapping[str, np.ndarray],
        domains: Sequence[str],
    ) -> "SplitConformalCalibrator":
        dom_arr = np.asarray(list(domains), dtype=object)
        for tier_id, probs in probs_by_tier.items():
            p = np.asarray(probs, dtype=np.float64).reshape(-1)
            y = np.asarray(labels_by_tier[tier_id], dtype=np.int64).reshape(-1)
            if p.shape != y.shape or p.shape[0] != dom_arr.shape[0]:
                raise ValueError(f"Mismatched calibration lengths for tier '{tier_id}'.")

            fail_mask = y == 0
            self._global_fail_probs[tier_id] = np.sort(p[fail_mask])
            self._calibration_counts[tier_id] = int(p.size)

            dom_map: dict[str, np.ndarray] = {}
            for d in sorted(set(str(x) for x in dom_arr)):
                d_mask = fail_mask & (dom_arr == d)
                dom_map[d] = np.sort(p[d_mask])
            self._domain_fail_probs[tier_id] = dom_map
        return self

    def _select_fail_scores(self, tier_id: str, domain: str) -> tuple[np.ndarray, str]:
        if tier_id not in self._global_fail_probs:
            raise KeyError(f"Tier '{tier_id}' has not been calibrated.")
        if self.mondrian:
            dom_scores = self._domain_fail_probs.get(tier_id, {}).get(domain)
            if dom_scores is not None and int(dom_scores.size) >= self.min_domain_failures:
                return dom_scores, f"mondrian:{domain}"
        return self._global_fail_probs[tier_id], "global"

    def threshold_for(self, tier_id: str, domain: str, alpha: float) -> tuple[float, int, str]:
        """Return (min_required_prob_to_accept, n_calibration_failures, scope)."""
        fail_probs, scope = self._select_fail_scores(tier_id, domain)
        tau = finite_sample_conformal_quantile(fail_probs, alpha)
        return float(tau), int(fail_probs.size), scope

    def evaluate(self, tier_id: str, domain: str, prob_success: float, alpha: float) -> ConformalCheckResult:
        """Test whether routing a query with predicted `prob_success` to `tier_id` satisfies risk <= alpha."""
        p = min(max(float(prob_success), 0.0), 1.0)
        fail_probs, scope = self._select_fail_scores(tier_id, domain)
        tau = finite_sample_conformal_quantile(fail_probs, alpha)
        n_fail = int(fail_probs.size)
        # Conformal p-value under H0: query is a failure (y = 0)
        # Fraction of calibration failures with predicted prob >= p (plus 1 smoothing)
        n_ge = int(np.sum(fail_probs >= p))
        p_val = float((n_ge + 1) / (n_fail + 1))
        accepted = bool(p >= tau and p_val <= alpha)
        return ConformalCheckResult(
            accepted=accepted,
            prob_success=p,
            nonconformity_score=round(1.0 - p, 6),
            threshold_prob=round(tau, 6),
            conformal_threshold_score=round(1.0 - tau, 6),
            p_value_failure=round(p_val, 6),
            calibration_failures_n=n_fail,
            scope=scope,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mondrian": self.mondrian,
            "min_domain_failures": self.min_domain_failures,
            "global_fail_probs": {k: v.tolist() for k, v in self._global_fail_probs.items()},
            "domain_fail_probs": {
                tid: {d: arr.tolist() for d, arr in dmap.items()}
                for tid, dmap in self._domain_fail_probs.items()
            },
            "calibration_counts": dict(self._calibration_counts),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SplitConformalCalibrator":
        obj = cls(
            mondrian=bool(data.get("mondrian", True)),
            min_domain_failures=int(data.get("min_domain_failures", 8)),
        )
        obj._global_fail_probs = {
            k: np.asarray(v, dtype=np.float64) for k, v in data["global_fail_probs"].items()
        }
        obj._domain_fail_probs = {
            tid: {d: np.asarray(arr, dtype=np.float64) for d, arr in dmap.items()}
            for tid, dmap in data["domain_fail_probs"].items()
        }
        obj._calibration_counts = {k: int(v) for k, v in data.get("calibration_counts", {}).items()}
        return obj
