"""Direct LLM-call cache fixtures; no model provider or external tool is used."""

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

from .call_cache import JAC
from .decisions import Candidate, Decision, DecisionError, DecisionPolicy
from .models import Context


@dataclass(frozen=True)
class _Case:
    prompt: str
    company: str = "Orchid"
    excludes_sales: bool = False
    window_days: int | None = None
    model: str = "fixture-model"
    source_version: str = "v1"

    @property
    def kwargs(self) -> dict[str, Any]:
        return {"prompt": self.prompt, "model": self.model,
                "source_version": self.source_version, "temperature": 0}

    @property
    def expected(self) -> dict[str, Any]:
        # Labels are attached to cases explicitly; the oracle never derives the
        # expected intent from a cache decision or from generated response text.
        policy_days = {"v1": {"Orchid": 30, "Cedar": 45},
                       "v2": {"Orchid": 21, "Cedar": 40}}
        return {"company": self.company,
                "return_days": policy_days[self.source_version][self.company],
                "sale_items_included": not self.excludes_sales,
                "requested_window_days": self.window_days,
                "source_version": self.source_version,
                "model": self.model}


def _generate_response(prompt: str, model: str = "fixture-model",
                       source_version: str = "v1", temperature: float = 0) -> dict[str, Any]:
    """Deterministic fake LLM; the source is fictional and prompts are fixtures."""
    lower = prompt.lower()
    company = "Cedar" if "cedar" in lower else "Orchid"
    excluded = bool(re.search(r"\b(excluding|exclude|without)\s+sale\s+items\b", lower))
    window = re.search(r"\bwithin\s+(\d+)\s+days\b", lower)
    days = (30 if company == "Orchid" else 45) if source_version == "v1" \
        else (21 if company == "Orchid" else 40)
    return {"company": company, "return_days": days,
            "sale_items_included": not excluded,
            "requested_window_days": int(window.group(1)) if window else None,
            "source_version": source_version, "model": model}


def _kwargs_from_request(request: Mapping[str, Any]) -> dict[str, Any]:
    kwargs = dict(request["kwargs"])
    kwargs["prompt"] = request.get("text", kwargs.get("prompt", ""))
    return kwargs


class _LexicalDecider:
    model = "LEXICAL-FIXTURE-NO-JEV"

    async def decide(self, request: Mapping[str, Any], candidates: Sequence[Candidate],
                     kind: str) -> Decision:
        started = time.perf_counter()
        words = set(re.findall(r"\w+", str(request.get("text", "")).lower()))
        ranked = []
        for candidate in candidates:
            prior = set(re.findall(r"\w+", str(candidate.request.get("text", "")).lower()))
            score = len(words & prior) / max(1, min(len(words), len(prior)))
            if score >= 0.6:
                ranked.append((score, candidate.id))
        chosen = min(ranked, key=lambda row: (-row[0], row[1]))[1] if ranked else None
        return Decision(chosen, {}, self.model, (time.perf_counter() - started) * 1000)


class _SimulatedJevDecider:
    """A labeled oracle fixture, not a live Jev model or accuracy estimate."""

    model = "SIMULATED-JEV-ORACLE-FIXTURE"

    def __init__(self, delay_scale: float = 1.0):
        self.delay_scale = delay_scale

    async def decide(self, request: Mapping[str, Any], candidates: Sequence[Candidate],
                     kind: str) -> Decision:
        started = time.perf_counter()
        await asyncio.sleep(0.001 * self.delay_scale)
        kwargs = _kwargs_from_request(request)
        if "timeout fixture" in kwargs["prompt"].lower():
            await asyncio.sleep(0.003 * self.delay_scale)
            raise DecisionError("injected local fixture timeout")
        expected = _generate_response(**kwargs)
        accepted = [candidate.id for candidate in candidates if candidate.artifact == expected]
        return Decision(min(accepted) if accepted else None,
                        {candidate.id: float(candidate.artifact == expected)
                         for candidate in candidates}, self.model,
                        (time.perf_counter() - started) * 1000)


def _context(_args: tuple[Any, ...], kwargs: dict[str, Any]) -> Context:
    return Context("call-benchmark", {"return-policy": kwargs.get("source_version", "v1")})


async def _calibration(delay_scale: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Offline labels and measured simulated decision costs in comparable buckets."""
    samples = []
    decider = _SimulatedJevDecider(delay_scale)
    matching_labels = 0
    for version in ("v1", "v2"):
        base = _Case("Explain Orchid return policy", source_version=version)
        excluded = _Case("Explain Orchid return policy excluding sale items",
                         excludes_sales=True, source_version=version)
        paraphrase = _Case("Describe Orchid return policy", source_version=version)
        excluded_paraphrase = _Case("Describe Orchid return policy without sale items",
                                   excludes_sales=True, source_version=version)
        pairs = ((paraphrase, base), (excluded, base),
                 (excluded_paraphrase, excluded), (base, excluded))
        for index in range(24):
            new, old = pairs[index % len(pairs)]
            label = new.expected == old.expected
            request = {"args": [], "kwargs": new.kwargs, "text": new.prompt}
            candidate = Candidate(f"{version}:{index}",
                                  {"args": [], "kwargs": old.kwargs, "text": old.prompt},
                                  old.expected, 40)
            started = time.perf_counter()
            decision = await decider.decide(request, [candidate], "answer")
            elapsed_ms = (time.perf_counter() - started) * 1000
            matching_labels += (decision.candidate_id is not None) == label
            samples.append({"kwargs": new.kwargs, "valid_hit": label,
                            "decision_ms": elapsed_ms, "source_version": version})
    return samples, {
        "source": "48 labeled local comparisons in two source-version buckets",
        "live_jev_evaluation": False,
        "attempts": len(samples),
        "valid_reuse_pairs": sum(sample["valid_hit"] for sample in samples),
        "observed_valid_reuse_rate": statistics.mean(sample["valid_hit"] for sample in samples),
        "mean_simulated_decision_ms": statistics.mean(sample["decision_ms"] for sample in samples),
        "fixture_decisions_matching_labels": matching_labels,
        "included_in_request_latency_metrics": False,
    }


def _workload(requests: int) -> list[_Case]:
    cycle = [
        _Case("Explain Orchid return policy"),
        _Case("Explain Orchid return policy"),
        _Case("Describe Orchid return policy"),
        _Case("Explain Orchid return policy excluding sale items", excludes_sales=True),
        _Case("Describe Orchid return policy without sale items", excludes_sales=True),
        _Case("Explain Cedar return policy", company="Cedar"),
        _Case("Explain Orchid return policy", model="other-fixture-model"),
        _Case("Explain Orchid return policy", source_version="v2"),
        _Case("Explain Orchid return policy excluding sale items", excludes_sales=True,
              source_version="v2"),
        _Case("Explain Orchid return policy timeout fixture"),
        _Case("Explain Orchid return policy within 30 days", window_days=30),
        _Case("Explain Orchid return policy within 90 days", window_days=90),
        _Case("Summarize Orchid return policy"),
        _Case("Summarize Cedar return policy excluding sale items", company="Cedar",
              excludes_sales=True),
        _Case("Describe Orchid return policy", source_version="v2"),
        _Case("Explain Orchid return policy"),
    ]
    return [cycle[index % len(cycle)] for index in range(requests)]


def _percentile(values: Sequence[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)] if ordered else 0.0


async def run_call_benchmark(requests: int = 64, *, delay_scale: float = 1.0) -> dict[str, Any]:
    if isinstance(requests, bool) or not isinstance(requests, int) or requests < 1:
        raise ValueError("requests must be a positive integer")
    if not math.isfinite(delay_scale) or delay_scale < 0:
        raise ValueError("delay_scale must be finite and nonnegative")
    samples, calibration = await _calibration(delay_scale)
    workload = _workload(requests)
    modes = ("no_cache", "exact_only", "semantic_without_jev_lexical_fixture",
             "selective_jev_simulated_fixture")
    report: dict[str, Any] = {
        "benchmark": "synthetic-direct-llm-call-cache",
        "live_services_used": False,
        "requests_per_mode": requests,
        "delay_scale": delay_scale,
        "latency_percentile_method": "nearest rank",
        "calibration": calibration,
        "limitations": [
            "The LLM function is a deterministic fictional-policy fixture with artificial delays.",
            "Selective Jev is a golden-output oracle fixture, not the live model or its accuracy.",
            "The lexical comparison uses token overlap, not production embeddings.",
            "Modes start cold; all lookups, rejected decisions and failures count toward latency.",
            "Offline calibration latency is separate; "
            "real deployment needs comparable verified labels.",
            "Correctness uses independent intent labels attached to each case.",
            "Model settings, source versions and numeric anchors are checked deterministically.",
            "Warm p95 is also shown after the first 16 requests; full p95 includes cold requests.",
            "No external tool calls are part of this single-LLM-call benchmark.",
        ],
        "modes": {},
    }
    for mode in modes:
        lexical = mode == modes[2]
        semantic = mode in modes[2:]
        decider = (_LexicalDecider() if lexical else _SimulatedJevDecider(delay_scale)) \
            if semantic else None
        overhead = 0.0 if lexical else calibration["mean_simulated_decision_ms"] + 0.2
        jac = JAC(path=":memory:", namespace="call-benchmark", enabled=mode != "no_cache",
                  context=_context, jev=decider,
                  decision_policy=DecisionPolicy(min_samples=20, overhead_ms=overhead), ttl_s=300)
        try:
            async def generate(prompt, model="fixture-model", source_version="v1", temperature=0):
                await asyncio.sleep(0.040 * delay_scale)
                return _generate_response(prompt, model, source_version, temperature)

            cached = jac.wrap(generate, namespace="fixture.return-policy")
            if semantic:
                for sample in samples:
                    cached.observe_reuse(sample["kwargs"], valid_hit=sample["valid_hit"],
                                         decision_ms=sample["decision_ms"])
            records = []
            for index, case in enumerate(workload):
                value = await cached(**case.kwargs)
                trace = jac.last_trace.to_dict()
                decision_calls = trace["jev_calls"]
                if lexical:
                    trace["jev_calls"] = 0
                    for event in trace["events"]:
                        if event["kind"] == "jev":
                            event["kind"] = "lexical_fixture"
                records.append({"index": index, "request": case.kwargs,
                                "expected": case.expected, "value": value,
                                "correct": value == case.expected, "trace": trace,
                                "fixture_decision_calls": decision_calls})
            elapsed = [record["trace"]["elapsed_ms"] for record in records]
            warm = elapsed[16:]
            report["modes"][mode] = {
                "p50_ms": round(_percentile(elapsed, 0.5), 3),
                "p95_ms": round(_percentile(elapsed, 0.95), 3),
                "warm_p95_ms": round(_percentile(warm, 0.95), 3) if warm else None,
                "mean_llm_calls": statistics.mean(r["trace"]["llm_calls"] for r in records),
                "mean_tool_calls": statistics.mean(r["trace"]["tool_calls"] for r in records),
                "mean_jev_calls": statistics.mean(r["trace"]["jev_calls"] for r in records),
                "mean_fixture_decision_calls": statistics.mean(
                    r["fixture_decision_calls"] for r in records),
                "incorrect_responses": sum(not r["correct"] for r in records),
                "incorrect_reused_responses": sum(
                    not r["correct"] and r["trace"]["path"] == "full_hit" for r in records),
                "correctness_rate": statistics.mean(r["correct"] for r in records),
                "paths": dict(Counter(r["trace"]["path"] for r in records)),
                "decision_statuses": dict(Counter(r["trace"]["decision_status"] for r in records)),
                "gate_overhead_ms": overhead if semantic else None,
                "records": records,
            }
        finally:
            jac.close()
    baseline = report["modes"]["no_cache"]
    for metrics in report["modes"].values():
        metrics["p95_change_vs_baseline_pct"] = round(
            (metrics["p95_ms"] / baseline["p95_ms"] - 1) * 100, 2
        ) if baseline["p95_ms"] else 0.0
        metrics["llm_calls_saved_vs_baseline_pct"] = round(
            (1 - metrics["mean_llm_calls"] / baseline["mean_llm_calls"]) * 100, 2
        ) if baseline["mean_llm_calls"] else 0.0
    return report


async def run_call_demo(*, delay_scale: float = 1.0) -> list[dict[str, Any]]:
    if not math.isfinite(delay_scale) or delay_scale < 0:
        raise ValueError("delay_scale must be finite and nonnegative")
    jac = JAC(path=":memory:", namespace="call-demo", context=_context)
    calls = 0
    try:
        async def ask_llm(prompt, model="fixture-model", source_version="v1", temperature=0):
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.040 * delay_scale)
            return _generate_response(prompt, model, source_version, temperature)

        cached_llm = jac.wrap(ask_llm)
        examples = [
            ("First LLM call", _Case("Explain Orchid return policy")),
            ("Exact repeated call", _Case("Explain Orchid return policy")),
            ("Changed model", _Case("Explain Orchid return policy", model="other-fixture-model")),
            ("Changed source version", _Case("Explain Orchid return policy", source_version="v2")),
        ]
        rows = []
        for label, case in examples:
            value = await cached_llm(**case.kwargs)
            if value != case.expected:
                raise AssertionError(f"Incorrect demo response: {label}")
            rows.append({"label": label, "value": value,
                         "trace": jac.last_trace.to_dict(), "total_generations": calls})
        return rows
    finally:
        jac.close()
