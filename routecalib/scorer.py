"""Pure-NumPy L2-regularized logistic difficulty scorer with temperature scaling."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    clipped = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


class BinaryTierClassifier:
    """L2-regularized logistic regression fitted via damped Newton-IRLS + temperature scaling."""

    def __init__(self, *, l2_reg: float = 1.0, max_iter: int = 50, tol: float = 1e-7) -> None:
        if l2_reg <= 0.0:
            raise ValueError("l2_reg must be > 0.")
        self.l2_reg = float(l2_reg)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.temperature_: float = 1.0

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.std_ is not None
        return (x - self.mean_) / self.std_

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "BinaryTierClassifier":
        x = np.asarray(x_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64).reshape(-1)
        if x.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 2:
            raise ValueError(f"Invalid training shapes: X={x.shape}, y={y.shape}.")

        self.mean_ = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        self.std_ = np.where(std > 1e-8, std, 1.0)
        xs = self._standardize(x)

        n, d = xs.shape
        x_aug = np.hstack([xs, np.ones((n, 1), dtype=np.float64)])
        theta = np.zeros(d + 1, dtype=np.float64)

        reg_mat = np.eye(d + 1, dtype=np.float64) * self.l2_reg
        reg_mat[-1, -1] = 1e-4  # tiny bias regularization for numerical stability

        for _ in range(self.max_iter):
            logits = x_aug @ theta
            probs = _sigmoid(logits)
            grad = (x_aug.T @ (probs - y)) + (reg_mat @ theta)
            w = np.clip(probs * (1.0 - probs), 1e-6, 0.25)
            hess = (x_aug.T * w) @ x_aug + reg_mat
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hess, grad, rcond=None)[0]
            theta_new = theta - step
            if float(np.max(np.abs(step))) < self.tol:
                theta = theta_new
                break
            theta = theta_new

        self.weights_ = theta[:-1].copy()
        self.bias_ = float(theta[-1])
        self.temperature_ = 1.0
        return self

    def raw_logits(self, x: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Classifier has not been fitted.")
        arr = np.asarray(x, dtype=np.float64)
        single = arr.ndim == 1
        if single:
            arr = arr.reshape(1, -1)
        xs = self._standardize(arr)
        logits = xs @ self.weights_ + self.bias_
        return logits[0:1] if single else logits

    def fit_temperature(self, x_val: np.ndarray, y_val: np.ndarray) -> float:
        """Optimize scalar temperature T > 0 minimizing NLL on a held-out validation split."""
        logits = self.raw_logits(x_val)
        y = np.asarray(y_val, dtype=np.float64).reshape(-1)
        best_t = 1.0
        best_nll = float("inf")
        # Coarse-to-fine 1D grid search over log-temperature (convex in 1/T)
        for t in np.geomspace(0.25, 4.0, num=76):
            p = np.clip(_sigmoid(logits / t), 1e-9, 1.0 - 1e-9)
            nll = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
            if nll < best_nll:
                best_nll = nll
                best_t = float(t)
        self.temperature_ = best_t
        return best_t

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = self.raw_logits(x)
        return _sigmoid(logits / self.temperature_)

    def to_dict(self) -> dict[str, Any]:
        if self.weights_ is None or self.mean_ is None or self.std_ is None:
            raise RuntimeError("Cannot serialize unfitted classifier.")
        return {
            "l2_reg": self.l2_reg,
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
            "weights": self.weights_.tolist(),
            "bias": self.bias_,
            "temperature": self.temperature_,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BinaryTierClassifier":
        obj = cls(l2_reg=float(data.get("l2_reg", 1.0)))
        obj.mean_ = np.asarray(data["mean"], dtype=np.float64)
        obj.std_ = np.asarray(data["std"], dtype=np.float64)
        obj.weights_ = np.asarray(data["weights"], dtype=np.float64)
        obj.bias_ = float(data["bias"])
        obj.temperature_ = float(data.get("temperature", 1.0))
        return obj


class DifficultyScorer:
    """Multi-tier success probability predictor trained on prompt difficulty features."""

    def __init__(self, tier_ids: Sequence[str], *, l2_reg: float = 1.5) -> None:
        self.tier_ids = list(tier_ids)
        self.l2_reg = float(l2_reg)
        self.classifiers: dict[str, BinaryTierClassifier] = {
            tid: BinaryTierClassifier(l2_reg=self.l2_reg) for tid in self.tier_ids
        }

    def fit(
        self,
        x_train: np.ndarray,
        y_by_tier: Mapping[str, np.ndarray],
        *,
        x_val: np.ndarray | None = None,
        y_val_by_tier: Mapping[str, np.ndarray] | None = None,
    ) -> "DifficultyScorer":
        for tid in self.tier_ids:
            if tid not in y_by_tier:
                raise KeyError(f"Missing training labels for tier '{tid}'.")
            clf = self.classifiers[tid]
            clf.fit(x_train, y_by_tier[tid])
            if x_val is not None and y_val_by_tier is not None and tid in y_val_by_tier:
                clf.fit_temperature(x_val, y_val_by_tier[tid])
        return self

    def predict_tier_probs(self, feature_vec: np.ndarray) -> dict[str, float]:
        out: dict[str, float] = {}
        for tid in self.tier_ids:
            prob = float(self.classifiers[tid].predict_proba(feature_vec)[0])
            out[tid] = min(max(prob, 0.0), 1.0)
        return out

    def predict_batch(self, x: np.ndarray) -> dict[str, np.ndarray]:
        return {tid: self.classifiers[tid].predict_proba(x) for tid in self.tier_ids}

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier_ids": list(self.tier_ids),
            "l2_reg": self.l2_reg,
            "classifiers": {tid: clf.to_dict() for tid, clf in self.classifiers.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DifficultyScorer":
        obj = cls(tier_ids=data["tier_ids"], l2_reg=float(data.get("l2_reg", 1.5)))
        obj.classifiers = {
            tid: BinaryTierClassifier.from_dict(cdata) for tid, cdata in data["classifiers"].items()
        }
        return obj


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, *, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) over equal-width confidence bins."""
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if p.size == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p <= hi if i == n_bins - 1 else p < hi)
        if np.any(mask):
            ece += (float(np.sum(mask)) / float(p.size)) * abs(float(np.mean(p[mask])) - float(np.mean(y[mask])))
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if p.size == 0:
        return 0.0
    return float(np.mean((p - y) ** 2))
