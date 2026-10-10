"""Small, reproducible execution-reuse demonstrations and synthetic benchmarks.

Every operation and both decision models below are local fixtures. This module
does not call a generative model, Jev, a financial API, or another external tool.
The delays model operation costs; measured timings are local harness timings,
not a claim about live agent or Jev performance.
"""

from __future__ import annotations

import asyncio
import math
import re
import statistics
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .decisions import Candidate, Decision, DecisionError, DecisionPolicy
from .models import Context, Node, Plan, Ref, Request
from .runtime import Runtime
from .store import SQLiteStore

_HOLDINGS = {
    "v1": {
        "A": [("ACME", 10, "equity", "USD"), ("BOND", 5, "bond", "USD"),
              ("EURO", 3, "equity", "EUR")],
        "B": [("ACME", 4, "equity", "USD"), ("BOND", 9, "bond", "USD")],
        "C": [("EURO", 8, "equity", "EUR"), ("BOND", 2, "bond", "USD")],
    },
    "v2": {
        "A": [("ACME", 12, "equity", "USD"), ("BOND", 5, "bond", "USD"),
              ("EURO", 3, "equity", "EUR")],
        "B": [("ACME", 4, "equity", "USD"), ("BOND", 8, "bond", "USD")],
        "C": [("EURO", 9, "equity", "EUR"), ("BOND", 2, "bond", "USD")],
    },
}
_PRICES = {
    "v1": {"ACME": 100.0, "BOND": 80.0, "EURO": 50.0},
    "v2": {"ACME": 120.0, "BOND": 82.0, "EURO": 52.0},
}
_RATES = {"USD": 1.0, "EUR": 1.1}


def _exclude_equities(text: str) -> bool:
    """Fixture intent parser; a live agent would resolve its own task inputs."""
    return bool(re.search(r"\b(exclude|excluding|without)\s+(equity|equities)\b", text.lower()))


def _make_request(portfolio_id: str, text: str, revision: str = "v1",
                  scope: str = "benchmark") -> Request:
    return Request(
        "portfolio.exposure",
        {"portfolio_id": portfolio_id},
        Context(scope, {"holdings": revision, "prices": revision, "fx": "v1"}),
        text=text,
        output_contract="portfolio-exposure-v1",
    )


def _golden(request: Request | Mapping[str, Any]) -> dict[str, Any]:
    """Independent fixture oracle, evaluated from input data rather than traces."""
    data = request.to_dict() if isinstance(request, Request) else request
    portfolio_id = data["params"]["portfolio_id"]
    revisions = data["context"]["dependencies"]
    holdings_revision = revisions["holdings"]
    price_revision = revisions["prices"]
    excluded = _exclude_equities(data.get("text", ""))
    selected = [row for row in _HOLDINGS[holdings_revision][portfolio_id]
                if not excluded or row[2] != "equity"]
    values = [quantity * _PRICES[price_revision][symbol] * _RATES[currency]
              for symbol, quantity, _asset_class, currency in selected]
    return {
        "portfolio_id": portfolio_id,
        "total_usd": round(math.fsum(values), 2),
        "excluded_equities": excluded,
        "included_assets": sorted(row[0] for row in selected),
        "source_versions": {"holdings": holdings_revision, "prices": price_revision},
    }


def _schema_validator(_request: Request, value: Any) -> bool:
    """Admit JSON fixture outputs; semantic accuracy is checked independently.

    This intentionally checks only the shape, so the benchmark can measure errors
    made by a decision model. It is not a production final-answer validator.
    """
    return (
        isinstance(value, dict)
        and isinstance(value.get("portfolio_id"), str)
        and isinstance(value.get("total_usd"), (int, float))
        and isinstance(value.get("excluded_equities"), bool)
        and isinstance(value.get("included_assets"), list)
        and isinstance(value.get("source_versions"), dict)
    )


class _PortfolioAgent:
    def __init__(self, runtime: Runtime, delay_scale: float = 1.0) -> None:
        self.runtime = runtime
        self.delay_scale = delay_scale
        self.revision = "v1"
        runtime.register("portfolio.positions", self.positions, kind="tool",
                         dependencies=("holdings",), ttl_s=300)
        runtime.register("market.prices", self.prices, kind="tool",
                         dependencies=("prices",), ttl_s=300)
        runtime.register("market.fx", self.fx, kind="tool",
                         dependencies=("fx",), ttl_s=300)
        runtime.register("portfolio.summarize", self.summarize, kind="llm", ttl_s=300)

    async def positions(self, portfolio_id: str) -> dict[str, Any]:
        await asyncio.sleep(0.006 * self.delay_scale)
        return {
            "portfolio_id": portfolio_id,
            "revision": self.revision,
            "rows": [{"symbol": symbol, "quantity": quantity, "asset_class": asset_class,
                      "currency": currency}
                     for symbol, quantity, asset_class, currency
                     in _HOLDINGS[self.revision][portfolio_id]],
        }

    async def prices(self) -> dict[str, Any]:
        await asyncio.sleep(0.005 * self.delay_scale)
        return {"revision": self.revision, "values": dict(_PRICES[self.revision])}

    async def fx(self) -> dict[str, float]:
        await asyncio.sleep(0.003 * self.delay_scale)
        return dict(_RATES)

    async def summarize(self, positions: dict[str, Any], prices: dict[str, Any],
                        rates: dict[str, float], request_text: str) -> dict[str, Any]:
        # Keep the generative step appreciably more expensive than the simulated
        # decision on platforms with coarse event-loop timer resolution.
        await asyncio.sleep(0.040 * self.delay_scale)
        excluded = _exclude_equities(request_text)
        selected = [row for row in positions["rows"]
                    if not excluded or row["asset_class"] != "equity"]
        total = sum(row["quantity"] * prices["values"][row["symbol"]]
                    * rates[row["currency"]] for row in selected)
        return {
            "portfolio_id": positions["portfolio_id"],
            "total_usd": round(total, 2),
            "excluded_equities": excluded,
            "included_assets": sorted(row["symbol"] for row in selected),
            "source_versions": {"holdings": positions["revision"],
                                "prices": prices["revision"]},
        }

    async def planner(self, request: Request) -> Plan:
        await asyncio.sleep(0.007 * self.delay_scale)
        return Plan(
            task=request.task,
            version="portfolio-template-1",
            parameters={"portfolio_id": "string"},
            nodes=(
                Node("positions", "portfolio.positions",
                     {"portfolio_id": Ref.input("portfolio_id")}),
                Node("prices", "market.prices"),
                Node("rates", "market.fx"),
                Node("answer", "portfolio.summarize",
                     {"positions": Ref("positions"), "prices": Ref("prices"),
                      "rates": Ref("rates"), "request_text": Ref.text()}),
            ),
            output=Ref("answer"),
            output_contract=request.output_contract,
            text_independent=True,
        )

    async def run(self, request: Request):
        # A fixture revision selects immutable snapshots. Context dependencies are
        # the runtime's validity contract; mutation is never hidden behind a TTL.
        self.revision = request.context.dependencies["holdings"]
        return await self.runtime.run(request, planner=self.planner)


class _LexicalFixtureDecider:
    """An inexpensive, unverified lexical-answer cache comparison fixture."""

    async def decide(self, request: Mapping[str, Any], candidates: Sequence[Candidate],
                     kind: str) -> Decision:
        started = time.perf_counter()
        words = set(re.findall(r"\w+", str(request.get("text", "")).lower()))
        ranked: list[tuple[float, str]] = []
        for candidate in candidates:
            old = set(re.findall(r"\w+", str(candidate.request.get("text", "")).lower()))
            score = len(words & old) / max(1, min(len(words), len(old)))
            if score >= 0.6:
                ranked.append((score, candidate.id))
        chosen = min(ranked, key=lambda row: (-row[0], row[1]))[1] if ranked else None
        return Decision(chosen, {}, "LEXICAL-FIXTURE-NO-JEV",
                        (time.perf_counter() - started) * 1000)


class _SimulatedJevFixtureDecider:
    """An oracle fixture, not an implementation or accuracy estimate of Jev.

    It compares labeled fixture outputs and adds a small simulated decision delay.
    An injected timeout shows how unsuccessful decision calls affect latency.
    """

    def __init__(self, delay_scale: float = 1.0) -> None:
        self.delay_scale = delay_scale

    async def decide(self, request: Mapping[str, Any], candidates: Sequence[Candidate],
                     kind: str) -> Decision:
        started = time.perf_counter()
        await asyncio.sleep(0.001 * self.delay_scale)
        if "timeout fixture" in str(request.get("text", "")).lower():
            await asyncio.sleep(0.003 * self.delay_scale)
            raise DecisionError("simulated fixture decision timed out")
        expected = _golden(request)
        accepted = [candidate.id for candidate in candidates if candidate.artifact == expected]
        probabilities = {candidate.id: float(candidate.artifact == expected)
                         for candidate in candidates}
        return Decision(min(accepted) if accepted else None, probabilities,
                        "SIMULATED-JEV-ORACLE-FIXTURE", (time.perf_counter() - started) * 1000)


@dataclass(frozen=True)
class _CalibrationSample:
    request: Request
    valid_hit: bool
    decision_ms: float


async def _calibration(delay_scale: float) -> tuple[list[_CalibrationSample], dict[str, Any]]:
    """Measure the simulated decider on explicit, independently labeled pairs."""
    decider = _SimulatedJevFixtureDecider(delay_scale)
    baseline = _make_request("A", "Calculate the exposure of portfolio A.")
    include = _golden(baseline)
    exclude_request = _make_request(
        "A", "Calculate the exposure of portfolio A excluding equities."
    )
    exclude = _golden(exclude_request)
    pairs = [
        (_make_request("A", "Compute the exposure of portfolio A."), baseline, include),
        (exclude_request, baseline, include),
        (_make_request("A", "Show portfolio A exposure without equities."),
         exclude_request, exclude),
        (baseline, exclude_request, exclude),
    ]
    samples: list[_CalibrationSample] = []
    correct_decisions = 0
    for index in range(24):
        request, cached_request, artifact = pairs[index % len(pairs)]
        label = artifact == _golden(request)
        candidate = Candidate(str(index), cached_request.to_dict(), artifact, avoided_ms=40)
        started = time.perf_counter()
        decision = await decider.decide(request.to_dict(), [candidate], "answer")
        elapsed_ms = (time.perf_counter() - started) * 1000
        correct_decisions += bool(decision.candidate_id is not None) == label
        samples.append(_CalibrationSample(request, label, elapsed_ms))
    return samples, {
        "source": "24 labeled local fixture comparisons; no live Jev evaluation",
        "attempts": len(samples),
        "valid_reuse_pairs": sum(sample.valid_hit for sample in samples),
        "observed_valid_reuse_rate": statistics.mean(sample.valid_hit for sample in samples),
        "mean_simulated_decision_ms": statistics.mean(sample.decision_ms for sample in samples),
        "fixture_decisions_matching_labels": correct_decisions,
        "included_in_request_latency_metrics": False,
    }


def _workload(count: int) -> list[Request]:
    patterns = [
        ("A", "Calculate the exposure of portfolio A.", "v1"),
        ("A", "Calculate the exposure of portfolio A.", "v1"),
        ("A", "Compute the exposure of portfolio A.", "v1"),
        ("B", "Calculate the exposure of portfolio B.", "v1"),
        ("A", "Calculate the exposure of portfolio A excluding equities.", "v1"),
        ("A", "Show the exposure of portfolio A without equities.", "v1"),
        ("A", "Calculate the exposure of portfolio A.", "v2"),
        ("A", "Calculate the exposure of portfolio A excluding equities.", "v2"),
        ("B", "Compute the exposure of portfolio B.", "v1"),
        ("B", "Calculate the exposure of portfolio B excluding equities.", "v1"),
        ("C", "Calculate the exposure of portfolio C.", "v1"),
        ("A", "Calculate the exposure of portfolio A, timeout fixture.", "v1"),
        ("A", "Give me the exposure of portfolio A.", "v1"),
        ("C", "Show the exposure of portfolio C without equities.", "v1"),
        ("B", "Calculate the exposure of portfolio B.", "v2"),
        ("A", "Calculate the exposure of portfolio A.", "v1"),
    ]
    return [_make_request(*patterns[index % len(patterns)]) for index in range(count)]


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


async def run_benchmark(requests: int = 64, *, delay_scale: float = 1.0) -> dict[str, Any]:
    """Compare four isolated, cold-start runtimes on the same ordered fixture set."""
    if isinstance(requests, bool) or not isinstance(requests, int) or requests < 1:
        raise ValueError("requests must be a positive integer")
    if not math.isfinite(delay_scale) or delay_scale < 0:
        raise ValueError("delay_scale must be finite and nonnegative")
    workload = _workload(requests)
    samples, calibration = await _calibration(delay_scale)
    modes = ("no_cache", "exact_only", "semantic_without_jev_lexical_fixture",
             "selective_jev_simulated_fixture")
    report: dict[str, Any] = {
        "benchmark": "synthetic-local-portfolio-execution-reuse",
        "live_services_used": False,
        "requests_per_mode": requests,
        "delay_scale": delay_scale,
        "latency_percentile_method": "nearest rank",
        "calibration": calibration,
        "limitations": [
            "LLM/tool calls are simulated registered operations with artificial sleeps.",
            "Selective Jev uses a golden-output oracle fixture, not the live Jev model.",
            "The lexical baseline is a token-overlap fixture, not an embedding benchmark.",
            "The final validator checks schema only; "
            "correctness uses an independent fixture oracle.",
            "Measured local p50/p95 and call savings do not predict production performance.",
            "Each mode starts with an empty cache and receives the same request ordering.",
            "Calibration is offline, excluded from request timings, "
            "and must be repeated per workload.",
        ],
        "modes": {},
    }
    for mode in modes:
        semantic = mode in modes[2:]
        lexical = mode == modes[2]
        decider = (_LexicalFixtureDecider() if lexical
                   else _SimulatedJevFixtureDecider(delay_scale)) if semantic else None
        store = SQLiteStore()
        try:
            # The selective gate uses measured labeled outcomes and observed
            # simulated overhead. No probability score is treated as a hit rate.
            overhead = 0.0 if lexical else calibration["mean_simulated_decision_ms"] + 0.2
            runtime = Runtime(
                store,
                enable_cache=mode != "no_cache",
                enable_semantic=semantic,
                decider=decider,
                decision_policy=DecisionPolicy(min_samples=20, overhead_ms=overhead),
                final_validator=_schema_validator,
                answer_ttl_s=300,
            )
            agent = _PortfolioAgent(runtime, delay_scale)
            if semantic:
                for sample in samples:
                    runtime.observe_reuse(sample.request, valid_hit=sample.valid_hit,
                                          decision_ms=sample.decision_ms)
            records = []
            for index, request in enumerate(workload):
                result = await agent.run(request)
                correct = result.value == _golden(request)
                trace = result.trace.to_dict()
                decision_calls = trace["jev_calls"]
                # Runtime counts its generic decision adapter calls. The lexical
                # fixture never invokes Jev, so classify it separately here.
                if lexical:
                    trace["jev_calls"] = 0
                    for event in trace["events"]:
                        if event["kind"] == "jev":
                            event["kind"] = "lexical_fixture"
                records.append({"index": index, "text": request.text,
                                "dependencies": dict(request.context.dependencies),
                                "correct": correct, "value": result.value,
                                "expected": _golden(request), "trace": trace,
                                "fixture_decision_calls": decision_calls})
            elapsed = [record["trace"]["elapsed_ms"] for record in records]
            report["modes"][mode] = {
                "p50_ms": round(_percentile(elapsed, 0.50), 3),
                "p95_ms": round(_percentile(elapsed, 0.95), 3),
                "mean_llm_calls": statistics.mean(r["trace"]["llm_calls"] for r in records),
                "mean_tool_calls": statistics.mean(r["trace"]["tool_calls"] for r in records),
                "mean_jev_calls": statistics.mean(r["trace"]["jev_calls"] for r in records),
                "mean_fixture_decision_calls": statistics.mean(
                    r["fixture_decision_calls"] for r in records),
                "incorrect_responses": sum(not r["correct"] for r in records),
                "incorrect_reused_responses": sum(
                    not r["correct"] and r["trace"]["path"] != "miss" for r in records),
                "correctness_rate": statistics.mean(r["correct"] for r in records),
                "paths": dict(Counter(r["trace"]["path"] for r in records)),
                "decision_statuses": dict(Counter(
                    r["trace"]["decision_status"] for r in records)),
                "gate_overhead_ms": overhead if semantic else None,
                "records": records,
            }
        finally:
            store.close()
    baseline = report["modes"]["no_cache"]
    for metrics in report["modes"].values():
        metrics["p95_change_vs_baseline_pct"] = round(
            (metrics["p95_ms"] / baseline["p95_ms"] - 1) * 100, 2
        ) if baseline["p95_ms"] else 0.0
        metrics["llm_calls_saved_vs_baseline_pct"] = round(
            (1 - metrics["mean_llm_calls"] / baseline["mean_llm_calls"]) * 100, 2
        ) if baseline["mean_llm_calls"] else 0.0
        metrics["tool_calls_saved_vs_baseline_pct"] = round(
            (1 - metrics["mean_tool_calls"] / baseline["mean_tool_calls"]) * 100, 2
        ) if baseline["mean_tool_calls"] else 0.0
    return report


async def run_demo(*, delay_scale: float = 1.0) -> list[dict[str, Any]]:
    """Show exact answers, parameterized plans, and unchanged shared tool data."""
    if not math.isfinite(delay_scale) or delay_scale < 0:
        raise ValueError("delay_scale must be finite and nonnegative")
    store = SQLiteStore()
    try:
        runtime = Runtime(store, final_validator=_schema_validator, answer_ttl_s=300)
        agent = _PortfolioAgent(runtime, delay_scale)
        requests = [
            ("First A request", _make_request("A", "Calculate the exposure of portfolio A.",
                                             scope="demo")),
            ("Identical A request", _make_request("A", "Calculate the exposure of portfolio A.",
                                                 scope="demo")),
            ("B binds the saved plan", _make_request("B", "Calculate the exposure of portfolio B.",
                                                    scope="demo")),
            ("A excludes equities", _make_request(
                "A", "Calculate the exposure of portfolio A excluding equities.", scope="demo")),
            ("A source changes", _make_request("A", "Calculate the exposure of portfolio A.",
                                              "v2", scope="demo")),
        ]
        rows = []
        for label, request in requests:
            result = await agent.run(request)
            if result.value != _golden(request):
                raise AssertionError(f"Demo returned an incorrect result: {label}")
            rows.append({"label": label, "value": result.value, "trace": result.trace.to_dict()})
        return rows
    finally:
        store.close()
