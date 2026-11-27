"""Static publication-grade Matplotlib plots for RouteCalib benchmark artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_benchmark_plots(summary: Mapping[str, Any], out_dir: Path) -> list[Path]:
    """Generate Pareto frontier, conformal coverage, and shadow drift detection PNGs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    # 1. Cost-Accuracy Pareto Frontier
    policies = {p["policy"]: p for p in summary["policies"]}
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)

    mondrian_pts = [p for k, p in policies.items() if k.startswith("routecalib_mondrian_a") and "cached" not in k]
    mondrian_pts.sort(key=lambda p: p["cost_per_1k_usd"])
    global_pts = [p for k, p in policies.items() if k.startswith("routecalib_global_a")]
    global_pts.sort(key=lambda p: p["cost_per_1k_usd"])

    ax.plot(
        [p["cost_per_1k_usd"] for p in mondrian_pts],
        [p["accuracy_pct"] for p in mondrian_pts],
        marker="o",
        linewidth=2.2,
        color="#0284C7",
        label="RouteCalib (Mondrian Conformal sweep)",
    )
    ax.plot(
        [p["cost_per_1k_usd"] for p in global_pts],
        [p["accuracy_pct"] for p in global_pts],
        marker="s",
        linestyle="--",
        linewidth=1.6,
        color="#6366F1",
        label="RouteCalib (Global Conformal sweep)",
    )

    markers = [
        ("always_tier0_small", "Always Small", "#94A3B8", "^"),
        ("always_tier1_mid", "Always Mid", "#F59E0B", "^"),
        ("always_tier2_frontier", "Always Frontier", "#DC2626", "D"),
        ("heuristic_length", "Length Heuristic", "#64748B", "X"),
        ("uncalibrated_p70", "Uncalibrated p>=0.70", "#EC4899", "P"),
        ("routecalib_mondrian_a05_cached", "Mondrian (a=0.05) + Cache", "#16A34A", "*"),
    ]
    for key, label, color, mk in markers:
        if key in policies:
            p = policies[key]
            ax.scatter(
                [p["cost_per_1k_usd"]],
                [p["accuracy_pct"]],
                color=color,
                marker=mk,
                s=110 if mk == "*" else 75,
                zorder=5,
                label=label,
            )

    ax.set_xlabel("Cost per 1,000 Queries (USD)", fontsize=10)
    ax.set_ylabel("Held-Out Task Accuracy (%)", fontsize=10)
    ax.set_title("RouteCalib Cost-Accuracy Pareto Frontier (Offline 5-Domain Benchmark)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8, loc="lower right", frameon=True)
    fig.tight_layout()
    p1 = out_dir / "pareto_cost_accuracy.png"
    fig.savefig(p1)
    plt.close(fig)
    created.append(p1)

    # 2. Conformal Coverage Guarantee (Target alpha vs Empirical False-Accept Rate)
    sweep = summary["conformal_sweep"]
    alphas = [r["alpha"] for r in sweep]
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=150)
    ax.plot(alphas, alphas, linestyle=":", color="#0F172A", linewidth=1.6, label="Target Risk Budget (y = alpha)")
    ax.plot(
        alphas,
        [r["global_false_accept_rate"] for r in sweep],
        marker="s",
        color="#6366F1",
        linewidth=2.0,
        label="Global Conformal (Overall FA)",
    )
    ax.plot(
        alphas,
        [r["mondrian_false_accept_rate"] for r in sweep],
        marker="o",
        color="#0284C7",
        linewidth=2.2,
        label="Mondrian Conformal (Overall FA)",
    )
    ax.plot(
        alphas,
        [r["mondrian_worst_domain_fa"] for r in sweep],
        marker="^",
        linestyle="--",
        color="#16A34A",
        linewidth=1.6,
        label="Mondrian Worst-Domain FA",
    )
    ax.set_xlabel("Target False-Accept Risk Budget (alpha)", fontsize=10)
    ax.set_ylabel("Empirical Held-Out False-Accept Rate", fontsize=10)
    ax.set_title("Conformal Risk Control: Empirical False-Accept Rate vs. Target Alpha", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8.5, loc="upper left", frameon=True)
    fig.tight_layout()
    p2 = out_dir / "conformal_coverage.png"
    fig.savefig(p2)
    plt.close(fig)
    created.append(p2)

    # 3. Shadow Drift Detection Stress Test
    drift = summary["drift_stress_test"]
    steps = list(range(1, len(drift["cusum_history"]) + 1))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.4, 5.2), dpi=150, sharex=True)

    ax1.plot(steps, drift["rate_history"], color="#0284C7", linewidth=2.0, label="Rolling False-Accept Rate")
    ax1.axhline(drift["target_alpha"], color="#DC2626", linestyle="--", linewidth=1.3, label=f"Target alpha = {drift['target_alpha']}")
    ax1.axvline(drift["changepoint_step"], color="#475569", linestyle=":", linewidth=1.4, label="Injected Degradation (t=60)")
    if drift["first_alert_step"] is not None:
        ax1.axvline(drift["first_alert_step"], color="#16A34A", linestyle="-.", linewidth=1.6, label=f"First Alert (t={drift['first_alert_step']})")
    ax1.set_ylabel("Rolling Error Rate", fontsize=9.5)
    ax1.set_title("Shadow-Traffic Drift Monitor: Upstream Degradation Detection", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(fontsize=8, loc="upper left")

    ax2.plot(steps, drift["cusum_history"], color="#7C3AED", linewidth=2.0, label="One-Sided CUSUM Statistic")
    ax2.axhline(drift["cusum_threshold"], color="#DC2626", linestyle="--", linewidth=1.3, label=f"CUSUM Threshold = {drift['cusum_threshold']}")
    ax2.axvline(drift["changepoint_step"], color="#475569", linestyle=":", linewidth=1.4)
    if drift["first_alert_step"] is not None:
        ax2.axvline(drift["first_alert_step"], color="#16A34A", linestyle="-.", linewidth=1.6)
    ax2.set_xlabel("Shadow Evaluation Sample Index", fontsize=9.5)
    ax2.set_ylabel("CUSUM Statistic", fontsize=9.5)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    p3 = out_dir / "drift_detection.png"
    fig.savefig(p3)
    plt.close(fig)
    created.append(p3)

    return created
