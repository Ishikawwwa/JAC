"""CLI for the SDK's local execution-reuse examples."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path

from .benchmark import run_demo as run_portfolio_demo
from .call_benchmark import run_call_benchmark, run_call_demo


def _positive_integer(value: str) -> int:
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return count


def _nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="JAC: cache an existing LLM call transparently.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="wrap an LLM call and skip generation on a hit")
    demo.add_argument("--delay-scale", type=_nonnegative_float, default=1.0,
                      help="multiply simulated operation delays (default: 1)")
    portfolio = subparsers.add_parser("portfolio-demo", help="show optional workflow reuse")
    portfolio.add_argument("--delay-scale", type=_nonnegative_float, default=1.0)
    benchmark = subparsers.add_parser("benchmark", help="compare four direct-call cache fixtures")
    benchmark.add_argument("--requests", type=_positive_integer, default=64)
    benchmark.add_argument("--delay-scale", type=_nonnegative_float, default=1.0)
    benchmark.add_argument("--output", type=Path, help="save the complete JSON report and traces")
    evaluate = subparsers.add_parser(
        "evaluate", help="measure real endpoints or token-free fixtures")
    evaluate.add_argument("--offline", action="store_true",
                          help="use clearly labeled local fixtures")
    evaluate.add_argument("--config", type=Path, default=Path("jac.local.json"))
    evaluate.add_argument("--retrieval", choices=("bm25", "embeddings", "both"), default="both")
    evaluate.add_argument("--mode", choices=("cache_first", "parallel", "both"), default="both")
    evaluate.add_argument("--repeats", type=_positive_integer, default=2)
    evaluate.add_argument("--output", type=Path, default=Path("benchmark-results/evaluation.json"))
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        from .evaluate import load_config, run_evaluation

        try:
            config = None if args.offline else load_config(args.config)
            report = asyncio.run(run_evaluation(config=config, offline=args.offline,
                retrieval=args.retrieval, mode=args.mode, repeats=args.repeats))
        except Exception as error:
            # Only expose our configuration errors; HTTP/SDK exceptions can contain secrets.
            print("Evaluation failed: " + (str(error) if type(error) in (ValueError, RuntimeError)
                                           else type(error).__name__))
            return 1
        print("OFFLINE FIXTURES: not live Jev/embedding results." if args.offline
              else "LIVE ENDPOINTS: warmup, setup and calibration excluded from request latency.")
        print(f"{'Mode':<26} {'p50 ms':>10} {'p95 ms':>10} {'Gen/req':>9} "
              f"{'Decisions':>10} {'Errors':>7}")
        for name, row in report["runs"].items():
            print(f"{name:<26} {row['p50_ms']:10.2f} {row['p95_ms']:10.2f} "
                  f"{row['generation_calls_per_request']:9.3f} "
                  f"{row['decision_calls']:10} {row['incorrect_responses']:7}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved results and per-request traces: {args.output.resolve()}")
        return int(any(row["incorrect_responses"] or row["setup_incorrect_responses"]
                       or row["calibration_incorrect_decisions"]
                       for row in report["runs"].values()))
    if args.command == "demo":
        rows = asyncio.run(run_call_demo(delay_scale=args.delay_scale))
        print("JAC direct LLM-call demo: local fixture, no live model or Jev calls.")
        print(f"{'Request':<25} {'Path':<12} {'ms':>8} {'LLM':>5} {'Jev':>4} "
              f"{'Total generations':>18}")
        for row in rows:
            trace = row["trace"]
            print(f"{row['label']:<25} {trace['path']:<12} {trace['elapsed_ms']:8.2f} "
                  f"{trace['llm_calls']:5} {trace['jev_calls']:4} {row['total_generations']:18}")
        print("The repeated call returns the same response without invoking the LLM function.")
        return 0
    if args.command == "portfolio-demo":
        rows = asyncio.run(run_portfolio_demo(delay_scale=args.delay_scale))
        print("Synthetic portfolio demo: local operations, no live LLMs, tools, or Jev.")
        print(f"{'Request':<26} {'Path':<16} {'ms':>8} {'LLM':>5} {'Tools':>6} "
              f"{'Jev':>4} {'Plan':>5} {'Nodes cached':>12} {'USD':>10}")
        for row in rows:
            trace = row["trace"]
            print(f"{row['label']:<26} {trace['path']:<16} {trace['elapsed_ms']:8.2f} "
                  f"{trace['llm_calls']:5} {trace['tool_calls']:6} {trace['jev_calls']:4} "
                  f"{str(trace['plan_hit']):>5} {trace['cached_nodes']:12} "
                  f"{row['value']['total_usd']:10.2f}")
        print("All responses match the independent fixture oracle.")
        return 0
    report = asyncio.run(run_call_benchmark(args.requests, delay_scale=args.delay_scale))
    print("Synthetic local benchmark; selective Jev is an oracle fixture, not a live model.")
    print(f"{'Mode':<38} {'p50 ms':>9} {'p95 ms':>9} {'LLM/req':>9} "
          f"{'Jev/req':>9} {'Errors':>7}")
    for name, metrics in report["modes"].items():
        print(f"{name:<38} {metrics['p50_ms']:9.3f} {metrics['p95_ms']:9.3f} "
              f"{metrics['mean_llm_calls']:9.3f} "
              f"{metrics['mean_jev_calls']:9.3f} {metrics['incorrect_responses']:7}")
    statuses = report["modes"]["selective_jev_simulated_fixture"]["decision_statuses"]
    print("Decision attempts are included in latency: "
          f"selective fixture rejected={statuses.get('rejected', 0)}, "
          f"failed={statuses.get('failed', 0)} (injected timeout).")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved report: {args.output.resolve()}")
    return 0
