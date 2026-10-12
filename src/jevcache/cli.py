"""Evaluate JAC with token-free fixtures or live model endpoints."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import json
from collections.abc import Mapping
from pathlib import Path


def _positive_integer(value: str) -> int:
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return count


async def _agent_providers(args):
    """Load user-owned call functions; never serialize factory return values."""
    if args.factory:
        module_name, separator, function_name = args.factory.partition(":")
        if not separator or not module_name or not function_name or ":" in function_name:
            raise ValueError("factory must be module:function")
        factory = getattr(importlib.import_module(module_name), function_name)
        if not callable(factory):
            raise ValueError("factory must be callable")
        supplied = factory()
        if inspect.isawaitable(supplied):
            supplied = await supplied
        if not isinstance(supplied, Mapping):
            raise ValueError("factory must return a mapping")
        allowed = {"generation", "generation_identity", "decider", "embedder",
                   "decision_timeout_s"}
        if set(supplied) - allowed or not {
            "generation", "generation_identity", "decider"
        }.issubset(supplied):
            raise ValueError("factory must supply generation, generation_identity and decider")
        return dict(supplied)
    from .evaluate import create_embedder, load_config, providers

    config = load_config(args.config)
    generator, decider = providers(config)
    return {
        "generation": generator.complete,
        "generation_identity": generator.cache_identity,
        "decider": decider,
        "embedder": create_embedder(config) if args.retrieval != "bm25" else None,
        "decision_timeout_s": config.get("decision_timeout_s", 30),
    }


async def _run_agents(args):
    from .agent_evaluate import run_agent_evaluation

    supplied = {} if args.offline else await _agent_providers(args)
    resources = []
    for resource in (getattr(supplied.get("generation"), "__self__", supplied.get("generation")),
                     supplied.get("decider"), supplied.get("embedder")):
        if resource is not None and all(resource is not existing for existing in resources):
            resources.append(resource)
    try:
        return await run_agent_evaluation(offline=args.offline, retrieval=args.retrieval,
                                         mode=args.mode, repeats=args.repeats, **supplied)
    finally:
        for resource in reversed(resources):
            drain = getattr(resource, "drain", None)
            if callable(drain):
                result = drain()
                if inspect.isawaitable(result):
                    await result
            close = getattr(resource, "close", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result


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
    agents = subparsers.add_parser(
        "evaluate-agents", help="test support, extraction and tool-informed agent workflows")
    source = agents.add_mutually_exclusive_group(required=True)
    source.add_argument("--offline", action="store_true", help="use labeled synthetic fixtures")
    source.add_argument("--config", type=Path, help="use existing model endpoint configuration")
    source.add_argument("--factory", help="load custom models from module:function")
    agents.add_argument("--retrieval", choices=("bm25", "embeddings", "both"), default="both")
    agents.add_argument("--mode", choices=("cache_first", "parallel", "both"), default="both")
    agents.add_argument("--repeats", type=_positive_integer, default=1)
    agents.add_argument("--output", type=Path,
                        default=Path("benchmark-results/agent-evaluation.json"))
    args = parser.parse_args(argv)
    if args.command == "evaluate-agents":
        try:
            report = asyncio.run(_run_agents(args))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        except Exception as error:
            # Custom SDK exception messages can include keys and complete prompts.
            print(f"Agent evaluation failed ({type(error).__name__}). "
                  "Check your factory/configuration and model availability.")
            return 1
        print("SYNTHETIC FIXTURES: timings and labels are not model benchmarks." if args.offline
              else "MODEL CALLS: inspect generation errors and unsafe reuse separately.")
        print(f"{'Mode':<26} {'p50 ms':>10} {'p95 ms':>10} {'Gen/req':>9} "
              f"{'Tools':>7} {'Errors':>7} {'Unsafe':>7} {'Cal FP/FN':>10}")
        for name, row in report["runs"].items():
            calibration = (f"{row.get('calibration_false_positives', 0)}/"
                           f"{row.get('calibration_false_negatives', 0)}")
            print(f"{name:<26} {row['p50_ms']:10.2f} {row['p95_ms']:10.2f} "
                  f"{row['generation_calls_per_request']:9.3f} {row['tool_calls']:7} "
                  f"{row['incorrect_responses']:7} {row['unsafe_reuses']:7} {calibration:>10}")
        print("Cal FP/FN: invalid reuses approved / valid reuses rejected in calibration.")
        print(f"Saved results and per-request traces: {args.output.resolve()}")
        return int(any(row["incorrect_responses"] or row["unsafe_reuses"]
                       or row.get("invalidation_failures", 0)
                       or row["setup_incorrect_responses"]
                       or row.get("calibration_false_positives", 0)
                       or row.get("calibration_errors", 0)
                       for row in report["runs"].values()))
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
