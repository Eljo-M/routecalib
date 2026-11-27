"""Command-line entry point for RouteCalib benchmarks, single-prompt inspection, and live proxy."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from routecalib.benchmark import build_trained_router, generate_benchmark_dataset, run_full_benchmark
from routecalib.plots import generate_benchmark_plots
from routecalib.providers import DeterministicLocalProvider
from routecalib.schemas import DEFAULT_TIERS, RouteRequest, dumps_strict_json
from routecalib.server import create_server
from routecalib.shadow import ShadowEvaluator


def _build_demo_router(seed: int = 42):
    dataset = generate_benchmark_dataset(n_total=900, seed=seed, tiers=DEFAULT_TIERS)
    router, _ = build_trained_router(
        dataset[:450],
        dataset[450:],
        tiers=DEFAULT_TIERS,
        mondrian=True,
        enable_cache=True,
        seed=seed,
    )
    router.provider = DeterministicLocalProvider()
    router.shadow_evaluator = ShadowEvaluator(shadow_rate=0.10, seed=seed)
    return router


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="routecalib", description="Conformal-calibrated multi-model router CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_bench = sub.add_parser("benchmark", help="Run reproducible 5-domain benchmark and generate Pareto/conformal/drift plots.")
    p_bench.add_argument("--out", default="results", help="Output directory for JSON, Markdown, and PNGs.")
    p_bench.add_argument("--n-total", type=int, default=1500, help="Total dataset size (split 1/3 train, 1/3 cal, 1/3 test).")
    p_bench.add_argument("--seed", type=int, default=42)

    p_route = sub.add_parser("route", help="Inspect a single calibrated routing decision in JSON.")
    p_route.add_argument("--prompt", required=True)
    p_route.add_argument("--alpha", type=float, default=0.05)
    p_route.add_argument("--max-latency-ms", type=float, default=None)
    p_route.add_argument("--domain", default="auto")

    p_serve = sub.add_parser("serve", help="Start the local OpenAI-compatible routing proxy & live dashboard.")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8080)

    args = parser.parse_args(argv)

    if args.command == "benchmark":
        out_dir = Path(args.out)
        summary = run_full_benchmark(out_dir, n_total=args.n_total, seed=args.seed)
        plots = generate_benchmark_plots(summary, out_dir)
        print(f"Benchmark complete. Wrote summary, report, and {len(plots)} plots to {out_dir}")
        return 0

    if args.command == "route":
        router = _build_demo_router()
        decision, _ = router.decide(
            RouteRequest(
                request_id="cli-inspect",
                prompt=args.prompt,
                domain=args.domain,
                target_alpha=args.alpha,
                max_latency_ms=args.max_latency_ms,
                allow_cache=False,
            )
        )
        print(dumps_strict_json(decision.to_dict(), indent=2))
        return 0

    if args.command == "serve":
        router = _build_demo_router()
        srv = create_server(router, host=args.host, port=args.port)
        print(f"RouteCalib proxy listening on http://{args.host}:{args.port} (Ctrl+C to stop)")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            srv.server_close()
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
