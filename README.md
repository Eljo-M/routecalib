# RouteCalib — Calibrated Multi-Model LLM Router & Cost-Quality Controller

**RouteCalib** is a lightweight, zero-GPU-dependency multi-model routing proxy and cost-quality control plane for LLM applications. It routes easy queries to fast, inexpensive local or small models and escalates hard queries to frontier models using **Split Conformal Prediction (Global & Mondrian per-domain)**, **latency/cost SLA enforcement**, a **numeric-guarded bounded semantic cache**, and a **CUSUM + Wilson shadow-traffic drift monitor**.

## Why RouteCalib

Sending every prompt to a frontier model wastes 40–80% of inference spend on straightforward extraction, classification, and short-QA queries, while naive confidence thresholds silently break accuracy on harder coding and math prompts. RouteCalib replaces ad-hoc thresholds with finite-sample conformal risk control:

- **Sub-millisecond feature extraction (`routecalib/features.py`):** 18 deterministic lexical and structural difficulty features with `~0.21 ms` p95 routing overhead (zero embedding model required on the hot path).
- **Calibrated multi-tier difficulty scoring (`routecalib/scorer.py`):** Pure-NumPy L2-regularized Newton-IRLS logistic classifiers with held-out temperature scaling, ECE, and Brier score diagnostics.
- **Global & Mondrian Split Conformal calibration (`routecalib/conformal.py`):** Computes exact finite-sample quantiles `ceil((n_fail + 1)(1 - alpha)) / n_fail` over calibration failures so false-accept risk is bounded both globally and per domain (`extraction`, `qa`, `reasoning`, `math`, `coding`).
- **Numeric-guarded bounded semantic cache (`routecalib/cache.py`):** O(1) exact SHA-256 lookup + domain-scoped character-trigram cosine matching with LRU eviction, TTL expiry, and a strict **numeric-literal equality guard** (so `"invoice #104"` never collides with `"invoice #105"`).
- **SLA enforcement & optional cascade verification (`routecalib/router.py`):** Enforces per-request `max_latency_ms` and `max_cost_usd` constraints and supports optional lightweight response verification with automatic frontier fallback.
- **Shadow-traffic drift monitor (`routecalib/shadow.py`):** Samples routed cheap-tier requests against the frontier reference tier and runs a rolling **Wilson score lower bound + one-sided CUSUM** detector to catch silent upstream model degradation.
- **OpenAI-compatible proxy + live dashboard (`routecalib/server.py`):** Zero-dependency `ThreadingHTTPServer` exposing `POST /v1/chat/completions`, `GET /metrics`, `GET /health`, and an interactive web UI at `GET /`.

## Offline Benchmark Results (`N=1,500`, `seed=42`)

All numbers below are measured by `python3 -m routecalib.cli benchmark --out results --n-total 1500 --seed 42` on the deterministic 5-domain offline benchmark (`train=500`, `calibration=500`, `test=500`). Router overhead is measured wall-clock CPU time; live endpoints plug in via `OpenAICompatibleProvider` (Ollama, vLLM, or hosted `/v1/chat/completions` APIs).

| Policy | Accuracy (%) | Cost / 1k Queries ($) | Savings vs. Frontier (%) | p50 Latency (ms) | Tier-0 False-Accept | Router p95 Overhead (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_tier0_small` | 43.40% | $0.0378 | 98.20% | 221.4 | 1.0000 | 0.0100 |
| `always_tier1_mid` | 73.40% | $0.2267 | 89.18% | 521.6 | 0.0000 | 0.0100 |
| `always_tier2_frontier` | 95.00% | $2.0941 | 0.00% | 1215.5 | 0.0000 | 0.0100 |
| `heuristic_length` | 88.80% | $1.4788 | 29.38% | 675.6 | 0.0777 | 0.0200 |
| `uncalibrated_p70` | 84.40% | $1.1682 | 44.21% | 541.3 | 0.1060 | 0.0800 |
| `routecalib_mondrian_a02` | 94.80% | $2.0843 | 0.47% | 1215.7 | 0.0035 | 0.2160 |
| `routecalib_mondrian_a05` | 88.60% | $1.8136 | 13.40% | 1131.1 | 0.0883 | 0.2147 |
| `routecalib_mondrian_a10` | 86.60% | $1.5462 | 26.17% | 884.1 | 0.1166 | 0.2146 |
| `routecalib_mondrian_a20` | 82.20% | $1.3313 | 36.43% | 637.1 | 0.1661 | 0.2145 |
| `routecalib_mondrian_a05_cached` | 88.34% | $1.2786 | 38.48% | 682.6 | 0.0876 | 0.1500 |

- **Mondrian vs. Global worst-domain control:** Across every `alpha` from `0.02` to `0.20`, Mondrian per-domain calibration cuts worst-domain false-accept rate sharply compared to global calibration (e.g., `0.0132` vs. `0.2750` at `alpha=0.02`, and `0.2000` vs. `0.3250` at `alpha=0.05`).
- **Shadow drift detection:** On an injected upstream degradation at shadow sample `60` (false-accept rate shifting from `0.03` to `0.28` against `alpha=0.05`), `ShadowEvaluator` raises **0 false alarms** before the changepoint and fires its first `DriftAlert` at step **85** (25-sample detection delay).

Generated plots are saved under `results/`:
- `results/pareto_cost_accuracy.png`
- `results/conformal_coverage.png`
- `results/drift_detection.png`

## Quickstart

```bash
# Install dependencies (only numpy and matplotlib)
pip install numpy matplotlib

# Run all unit and live HTTP integration tests
python3 -m unittest discover -s tests -v

# Reproduce the full 1,500-query benchmark and generate all 3 plots
python3 -m routecalib.cli benchmark --out results --n-total 1500 --seed 42

# Inspect a single routing decision
python3 -m routecalib.cli route \
  --prompt "Extract the company name and date as JSON from: Acme Corp reported Q3 results on 2026-10-12." \
  --alpha 0.05

# Launch the OpenAI-compatible proxy and interactive dashboard at http://127.0.0.1:8080
python3 -m routecalib.cli serve --host 127.0.0.1 --port 8080
```

## Connecting Live Model Endpoints

Pass `OpenAICompatibleProvider` with `ModelTier(..., endpoint_url="http://localhost:11434", model_name="...")` (Ollama, vLLM, llama.cpp server, or any hosted `/v1/chat/completions` API using `ROUTECALIB_API_KEY`). No third-party SDK is required.

## Limitations & Scope

- Offline benchmark figures evaluate the routing, conformal calibration, caching, and drift-detection stack on a deterministic 5-domain workload rather than live commercial APIs.
- Finite-sample split conformal guarantees rely on exchangeability between the calibration and test distributions; when distribution shift or upstream model degradation breaks exchangeability, the shadow CUSUM + Wilson monitor is designed to surface drift so operators can recalibrate.
