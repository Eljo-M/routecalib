"""Comprehensive unit and integration test suite for RouteCalib."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

import numpy as np

from routecalib.benchmark import (
    build_trained_router,
    generate_benchmark_dataset,
    run_drift_stress_test,
    run_full_benchmark,
)
from routecalib.cache import BoundedSemanticCache
from routecalib.conformal import SplitConformalCalibrator, finite_sample_conformal_quantile
from routecalib.features import extract_features, infer_domain
from routecalib.plots import generate_benchmark_plots
from routecalib.providers import DeterministicLocalProvider
from routecalib.router import CalibratedRouter
from routecalib.schemas import DEFAULT_TIERS, ModelResponse, RouteRequest, dumps_strict_json, loads_strict_json
from routecalib.scorer import DifficultyScorer, brier_score, expected_calibration_error
from routecalib.server import create_server
from routecalib.shadow import ShadowEvaluator, wilson_lower_bound


class TestFeaturesAndSchemas(unittest.TestCase):
    def test_feature_extraction_and_domain_inference(self) -> None:
        v_code, d_code, _ = extract_features("def foo(x):\n    import asyncio\n    return x")
        v_ext, d_ext, _ = extract_features("Extract the company name and date as JSON from this sentence.")
        self.assertEqual(d_code, "coding")
        self.assertEqual(d_ext, "extraction")
        _, d_date, _ = extract_features("Extract the company name and date as JSON from: Acme Corp reported Q3 results on 2026-10-12.")
        self.assertEqual(d_date, "extraction")
        self.assertEqual(v_code.shape, (18,))
        self.assertTrue(np.all(np.isfinite(v_code)))
        self.assertEqual(infer_domain("hello", explicit_domain="math"), "math")

    def test_strict_json_rejects_non_finite_values(self) -> None:
        with self.assertRaises(ValueError):
            dumps_strict_json({"x": float("nan")})
        with self.assertRaises(ValueError):
            loads_strict_json('{"x": NaN}')

    def test_request_validation(self) -> None:
        with self.assertRaises(ValueError):
            RouteRequest(request_id="r1", prompt="   ").validate()
        with self.assertRaises(ValueError):
            RouteRequest(request_id="r2", prompt="hi", target_alpha=1.2).validate()
        with self.assertRaises(ValueError):
            RouteRequest(request_id="r3", prompt="hi", max_latency_ms=-10.0).validate()


class TestScorerAndConformal(unittest.TestCase):
    def test_finite_sample_conformal_quantile_and_coverage_guarantee(self) -> None:
        # Empty calibration failures -> conservative 1.0 (always escalate)
        self.assertEqual(finite_sample_conformal_quantile(np.array([]), 0.05), 1.0)

        # Synthetic exchangeable split conformal check across 40 trials
        rng = np.random.default_rng(123)
        alpha = 0.10
        fa_rates = []
        for _ in range(40):
            fail_scores = rng.uniform(0.0, 1.0, size=200)
            cal_f, test_f = fail_scores[:100], fail_scores[100:]
            tau = finite_sample_conformal_quantile(cal_f, alpha)
            fa_rates.append(float(np.mean(test_f >= tau)))
        # Average false-accept rate across exchangeable splits must be <= alpha + small Monte Carlo slack
        self.assertLessEqual(float(np.mean(fa_rates)), alpha + 0.015)

    def test_scorer_and_calibrator_roundtrip_serialization(self) -> None:
        ds = generate_benchmark_dataset(n_total=300, seed=7)
        router, diag = build_trained_router(ds[:150], ds[150:], mondrian=True, seed=7)
        self.assertIn("tier0_small", diag)
        self.assertGreaterEqual(diag["tier0_small"]["ece"], 0.0)
        self.assertGreaterEqual(diag["tier0_small"]["brier"], 0.0)

        # Roundtrip JSON serialization of scorer and calibrator
        scorer_rt = DifficultyScorer.from_dict(loads_strict_json(dumps_strict_json(router.scorer.to_dict())))
        calib_rt = SplitConformalCalibrator.from_dict(loads_strict_json(dumps_strict_json(router.calibrator.to_dict())))

        vec, dom, _ = extract_features(ds[0].prompt, explicit_domain=ds[0].domain)
        p1 = router.scorer.predict_tier_probs(vec)
        p2 = scorer_rt.predict_tier_probs(vec)
        for k in p1:
            self.assertAlmostEqual(p1[k], p2[k], places=10)
        c1 = router.calibrator.evaluate("tier0_small", dom, p1["tier0_small"], 0.05)
        c2 = calib_rt.evaluate("tier0_small", dom, p2["tier0_small"], 0.05)
        self.assertEqual(c1, c2)


class TestSemanticCache(unittest.TestCase):
    def test_exact_semantic_ttl_and_lru_eviction(self) -> None:
        now = [100.0]
        cache = BoundedSemanticCache(
            max_entries=3,
            ttl_seconds=10.0,
            similarity_threshold=0.90,
            clock_fn=lambda: now[0],
        )
        r1 = ModelResponse("tier0_small", "ans1", 10, 20, 50.0, 0.0001, True)
        r2 = ModelResponse("tier0_small", "ans2", 10, 20, 50.0, 0.0001, True)
        r3 = ModelResponse("tier0_small", "ans3", 10, 20, 50.0, 0.0001, True)

        cache.put("Extract the company name and date as JSON from Acme Corp.", "extraction", r1)
        # Exact hit
        hit, sim = cache.get("extract the company name and date as json from acme corp.", "extraction")
        self.assertIsNotNone(hit)
        self.assertEqual(sim, 1.0)
        # Semantic near-duplicate hit in same domain
        hit_sem, sim_sem = cache.get("Extract the company name and date as JSON from Acme Corp!", "extraction")
        self.assertIsNotNone(hit_sem)
        self.assertGreaterEqual(sim_sem, 0.90)
        # Different numeric literal must NOT collide semantically (e.g., invoice #104 vs #105)
        cache.put("Summarize invoice 104 for Acme Corp.", "extraction", r1)
        miss_num, _ = cache.get("Summarize invoice 105 for Acme Corp.", "extraction")
        self.assertIsNone(miss_num)
        # Different domain must NOT collide
        miss_dom, _ = cache.get("Extract the company name and date as JSON from Acme Corp.", "coding")
        self.assertIsNone(miss_dom)

        # LRU eviction when exceeding max_entries=2
        cache.put("Prompt two completely different", "qa", r2)
        cache.put("Prompt three completely different", "qa", r3)
        self.assertEqual(cache.lru_evictions, 1)

        # TTL expiry
        now[0] += 15.0
        expired_hit, _ = cache.get("Prompt three completely different", "qa")
        self.assertIsNone(expired_hit)
        self.assertGreaterEqual(cache.expired_evictions, 1)


class TestRouterShadowAndServer(unittest.TestCase):
    def test_sla_enforcement_cascade_fallback_and_shadow_drift_alert(self) -> None:
        ds = generate_benchmark_dataset(n_total=300, seed=11)
        router, _ = build_trained_router(ds[:150], ds[150:], mondrian=True, enable_cache=True, seed=11)
        router.provider = DeterministicLocalProvider()

        # Tight latency SLA forces fast tier even on a hard coding prompt
        hard_prompt = "Prove lock-free ring buffer invariants in C++20 with acquire-release atomics and debug deadlock."
        dec_sla, _ = router.decide(
            RouteRequest(
                request_id="sla-1",
                prompt=hard_prompt,
                domain="coding",
                max_latency_ms=300.0,
                target_alpha=0.005,
                allow_cache=False,
                expected_output_tokens=150,
            )
        )
        self.assertEqual(dec_sla.selected_tier, "tier0_small")
        self.assertIn("sla_capped_to", dec_sla.escalation_reason)

        # Cascade verifier rejects cheap tier and escalates to frontier
        router.response_verifier = lambda _p, r: r.tier_id == "tier2_frontier"
        dec_cas, resp_cas = router.route_and_execute(
            RouteRequest(
                request_id="cas-1",
                prompt="Extract the company name as JSON from Acme Corp.",
                domain="extraction",
                target_alpha=0.20,
                allow_cache=False,
            )
        )
        self.assertEqual(dec_cas.selected_tier, "tier2_frontier")
        self.assertEqual(resp_cas.tier_id, "tier2_frontier")
        self.assertIn("cascade_verifier_rejected", dec_cas.escalation_reason)

        # Drift stress test catches upstream degradation cleanly with 0 false alarms before changepoint
        drift = run_drift_stress_test(seed=42, target_alpha=0.05)
        self.assertEqual(drift["false_alarms_before_changepoint"], 0)
        self.assertIsNotNone(drift["first_alert_step"])
        self.assertGreater(drift["first_alert_step"], drift["changepoint_step"])
        self.assertLessEqual(drift["detection_delay_steps"], 35)
        self.assertGreater(wilson_lower_bound(15, 20), 0.5)

    def test_http_proxy_endpoints_live_roundtrip(self) -> None:
        ds = generate_benchmark_dataset(n_total=240, seed=19)
        router, _ = build_trained_router(ds[:120], ds[120:], mondrian=True, enable_cache=True, seed=19)
        router.provider = DeterministicLocalProvider()
        router.shadow_evaluator = ShadowEvaluator(shadow_rate=0.0, seed=19)

        srv = create_server(router, host="127.0.0.1", port=0)
        port = srv.server_address[1]
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        try:
            # GET /health
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                health = json.loads(r.read().decode("utf-8"))
                self.assertEqual(health["status"], "ok")

            # GET / dashboard HTML
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
                html = r.read().decode("utf-8")
                self.assertIn("RouteCalib Live Proxy", html)

            # POST /v1/chat/completions twice -> second call should hit semantic/exact cache
            payload = json.dumps(
                {
                    "messages": [{"role": "user", "content": "Extract JSON from: Acme Q3 2026."}],
                    "target_alpha": 0.05,
                    "allow_cache": True,
                }
            ).encode("utf-8")
            req1 = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req1, timeout=5) as r:
                d1 = json.loads(r.read().decode("utf-8"))
                self.assertFalse(d1["x_routecalib"]["cache_hit"])

            req2 = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req2, timeout=5) as r:
                d2 = json.loads(r.read().decode("utf-8"))
                self.assertTrue(d2["x_routecalib"]["cache_hit"])

            # GET /metrics
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as r:
                m = json.loads(r.read().decode("utf-8"))
                self.assertEqual(m["total_requests"], 2)
                self.assertEqual(m["cache_hits"], 1)
        finally:
            srv.shutdown()
            srv.server_close()

    def test_full_benchmark_and_plots_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "results"
            summary = run_full_benchmark(out_dir, n_total=450, seed=42)
            plots = generate_benchmark_plots(summary, out_dir)
            self.assertEqual(len(plots), 3)
            self.assertTrue((out_dir / "benchmark_summary.json").is_file())
            self.assertTrue((out_dir / "benchmark_report.md").is_file())
            for p in plots:
                self.assertTrue(p.is_file())
                self.assertGreater(p.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
