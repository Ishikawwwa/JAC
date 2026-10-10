"""Evaluate JAC with token-free fixtures or live model endpoints."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path


def _positive_integer(value: str) -> int:
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="JAC: cache an existing LLM call transparently.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser(
        "evaluate", help="measure real endpoints or token-free fixtures")
    evaluate.add_argument("--offline", action="store_true",
                          help="use clearly labeled local fixtures")
    evaluate.add_argument("--config", type=Path, default=Path("examples/configs/hosted.json"))
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
