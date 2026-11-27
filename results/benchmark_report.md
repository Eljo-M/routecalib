# RouteCalib Offline Benchmark & Ablation Report

- **Mode:** `deterministic_offline_simulator` (deterministic 5-domain workload: `train=500`, `calibration=500`, `test=500`, `seed=42`).
- **Live endpoints:** Plug in any OpenAI-compatible server (Ollama, vLLM, hosted API) via `OpenAICompatibleProvider`.

## Policy Comparison (Held-Out Test Split)

| Policy | Accuracy (%) | Cost / 1k Queries ($) | Cost Savings vs Frontier (%) | p50 Latency (ms) | p95 Latency (ms) | Tier-0 False-Accept Rate | Router p95 Overhead (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_tier0_small` | 43.40% | $0.0378 | 98.20% | 221.4 | 430.2 | 1.0000 | 0.0100 |
| `always_tier1_mid` | 73.40% | $0.2267 | 89.18% | 521.6 | 1008.8 | 0.0000 | 0.0100 |
| `always_tier2_frontier` | 95.00% | $2.0941 | 0.00% | 1215.5 | 2346.5 | 0.0000 | 0.0100 |
| `heuristic_length` | 88.80% | $1.4788 | 29.38% | 675.6 | 2346.5 | 0.0777 | 0.0200 |
| `uncalibrated_p70` | 84.40% | $1.1682 | 44.21% | 541.3 | 2346.6 | 0.1060 | 0.0800 |
| `routecalib_global_a02` | 91.80% | $1.9056 | 9.00% | 1215.7 | 2346.7 | 0.0530 | 0.2135 |
| `routecalib_mondrian_a02` | 94.80% | $2.0843 | 0.47% | 1215.7 | 2346.7 | 0.0035 | 0.2160 |
| `routecalib_global_a05` | 90.60% | $1.7874 | 14.65% | 1163.7 | 2346.7 | 0.0742 | 0.2092 |
| `routecalib_mondrian_a05` | 88.60% | $1.8136 | 13.40% | 1131.1 | 2346.7 | 0.0883 | 0.2147 |
| `routecalib_global_a10` | 87.00% | $1.3516 | 35.46% | 602.9 | 2346.7 | 0.1095 | 0.2154 |
| `routecalib_mondrian_a10` | 86.60% | $1.5462 | 26.17% | 884.1 | 2346.7 | 0.1166 | 0.2146 |
| `routecalib_global_a15` | 82.80% | $1.2059 | 42.41% | 535.7 | 2346.7 | 0.1484 | 0.2164 |
| `routecalib_mondrian_a15` | 84.00% | $1.4737 | 29.63% | 767.1 | 2346.7 | 0.1519 | 0.2132 |
| `routecalib_global_a20` | 79.60% | $1.1256 | 46.25% | 490.9 | 2346.7 | 0.1908 | 0.2167 |
| `routecalib_mondrian_a20` | 82.20% | $1.3313 | 36.43% | 637.1 | 2346.7 | 0.1661 | 0.2145 |
| `routecalib_mondrian_a05_cached` | 88.34% | $1.2786 | 38.48% | 682.6 | 2242.6 | 0.0876 | 0.1500 |

## Conformal Risk Control Sweep (Global vs. Mondrian Per-Domain)

| Target Risk (`alpha`) | Global Empirical False-Accept | Mondrian Empirical False-Accept | Global Worst-Domain FA | Mondrian Worst-Domain FA | Mondrian Accuracy (%) | Mondrian Cost Savings (%) |
| :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.02 | 0.0530 | 0.0035 | 0.2750 | 0.0132 | 94.80% | 0.47% |
| 0.05 | 0.0742 | 0.0883 | 0.3250 | 0.2000 | 88.60% | 13.40% |
| 0.10 | 0.1095 | 0.1166 | 0.3500 | 0.2000 | 86.60% | 26.17% |
| 0.15 | 0.1484 | 0.1519 | 0.3939 | 0.2750 | 84.00% | 29.63% |
| 0.20 | 0.1908 | 0.1661 | 0.4848 | 0.3500 | 82.20% | 36.43% |

## Shadow Drift Detection Stress Test

- **Injected degradation:** Step `60` of `140` (false-accept rate shifts from `0.03` to `0.28` against `target_alpha=0.05`).
- **False alarms before changepoint:** `0`
- **First alert triggered at step:** `85` (detection delay: `25` shadow samples).
