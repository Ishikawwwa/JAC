"""Controlled agent-shaped workloads with independent correctness labels.

Offline providers are timing/correctness fixtures, never model benchmarks. Live
providers receive normal chat requests; only the read-only tool data is fixed.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import inspect
import json
import math
import re
import statistics
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from .call_cache import JAC
from .callables import _safe_identity
from .decisions import Candidate, Decision, DecisionPolicy
from .models import Context, fingerprint, snapshot
from .providers import parse_json


@dataclass(frozen=True)
class AgentCase:
    id: str
    family: str
    request: dict[str, Any]
    expected: dict[str, Any]
    context: Context
    tool_calls: int = 0
    category: str = "paraphrase"
    requires_invalidation: bool = False


@dataclass(frozen=True)
class CalibrationPair:
    seed: AgentCase
    request: AgentCase
    reuse: bool


@dataclass(frozen=True)
class AgentWorkload:
    seeds: tuple[AgentCase, ...]
    calibration: tuple[CalibrationPair, ...]
    sequential: tuple[AgentCase, ...]
    burst: AgentCase
    burst_size: int = 4


def _case(name, family, text, expected, system, history, context, *, tools=0):
    return AgentCase(name, family, {
        "messages": [{"role": "system", "content": system}, *history,
                     {"role": "user", "content": text}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }, expected, context, tools)


def _rephrase(case, name, text, expected=None, **changes):
    request = snapshot(case.request)
    request["messages"][-1]["content"] = text
    return replace(case, id=name, request=request,
                   expected=case.expected if expected is None else expected, **changes)


def _tool_history(name, value):
    return [
        {"role": "user", "content": "Read the data needed for the next request."},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "read-data", "type": "function",
            "function": {"name": name, "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "read-data", "content": json.dumps(value)},
    ]


def agent_workload() -> AgentWorkload:
    """Return a fixed corpus; calibration wording is absent from held-out trials."""
    policy = {"refund_days": 30, "opened_items_refundable": False}
    support_system = (
        "Use the retrieved refund policy. Return only JSON with the keys requested by the user. "
        "eligible is boolean. reason is exactly 'outside return window' if purchase age exceeds "
        "refund_days; otherwise 'opened item' for an opened item that cannot be refunded; "
        "otherwise 'within return window'. Check the return window before opened-item rules."
    )
    support = _case(
        "support_seed", "support",
        "Customer purchase was 20 days ago and the item is opened. Can we refund it? "
        "Return JSON with eligible and reason.",
        {"eligible": False, "reason": "opened item"}, support_system,
        _tool_history("read_refund_policy", policy),
        Context("tenant:acme", {"refund_policy": "v1"}), tools=1,
    )
    invoice = _case(
        "invoice_seed", "invoice_extraction",
        "Extract invoice_id, vendor, total, currency, and due_date.",
        {"invoice_id": "INV-204", "vendor": "Northwind", "total": 1250.5,
         "currency": "USD", "due_date": "2026-10-10"},
        "Extract only the requested fields from the invoice. Return JSON without extra keys. "
        "Dates use YYYY-MM-DD, total is a number, and currency is a three-letter ISO code.",
        [{"role": "user", "content": "Invoice INV-204; vendor Northwind; invoice date "
          "September 10, 2026; due October 10, 2026; total $1,250.50 USD."}],
        Context("tenant:acme", {"invoice": "INV-204-v1"}),
    )
    holdings = {"positions": [{"symbol": "AAPL", "quantity": 10, "price": 200},
                              {"symbol": "MSFT", "quantity": 5, "price": 400}]}
    portfolio = _case(
        "portfolio_seed", "portfolio_analysis",
        "Calculate total_value and weights for this portfolio. Return JSON.",
        {"total_value": 4000, "weights": {"AAPL": .5, "MSFT": .5}},
        "Use the holdings tool result. Position value is quantity times price. total_value is "
        "the sum of position values. weights maps each symbol to its fraction of total value, "
        "rounded to four decimal places. Return only requested JSON keys; currency is 'USD'. "
        "Follow earlier user formatting instructions if present.",
        _tool_history("read_holdings", holdings),
        Context("tenant:acme", {"holdings": "snapshot-v1"}), tools=1,
    )
    support_sealed = {"eligible": True, "reason": "within return window"}
    invoice_date = {key: value for key, value in invoice.expected.items() if key != "due_date"}
    invoice_date["invoice_date"] = "2026-09-10"
    definitions = (
        (support,
         ["An opened item was bought 20 days ago. State refund eligible and reason as JSON.",
          "Is a refund allowed for an opened item purchased 20 days ago? "
          "JSON eligible and reason."],
         [("A sealed item was bought 20 days ago. State refund eligible and reason as JSON.",
           support_sealed),
          ("Is a refund allowed for an opened item purchased 20 days ago? JSON with only eligible.",
           {"eligible": False})]),
        (invoice,
         ["Read the invoice and return invoice_id, vendor, total, currency, and due_date.",
          "Return JSON invoice_id, vendor, total, currency, and due_date from this invoice."],
         [("Read the invoice and return invoice_id, vendor, total, currency, and invoice_date.",
           invoice_date),
          ("Return JSON with only invoice_id and total from this invoice.",
           {"invoice_id": "INV-204", "total": 1250.5})]),
        (portfolio,
         ["Compute portfolio total_value and weights using the retrieved holdings. JSON only.",
          "Report total_value plus weights for the portfolio in JSON."],
         [("Compute only total_value using the retrieved holdings. JSON only.",
           {"total_value": 4000}),
          ("Report weights for the portfolio in JSON, without total_value.",
           {"weights": {"AAPL": .5, "MSFT": .5}})]),
    )
    calibration = []
    for seed, positives, negatives in definitions:
        for index, text in enumerate(positives):
            calibration.append(CalibrationPair(
                seed, _rephrase(seed, f"{seed.family}_cal_positive_{index}", text), True))
        for index, (text, expected) in enumerate(negatives):
            calibration.append(CalibrationPair(
                seed, _rephrase(seed, f"{seed.family}_cal_negative_{index}", text, expected),
                False))
    rows = [
        _rephrase(support, "support_paraphrase",
                  "Can an opened item bought 20 days ago be refunded? "
                  "Give eligible and reason in JSON."),
        _rephrase(support, "support_item_condition",
                  "Can a sealed item bought 20 days ago be refunded? "
                  "Give eligible and reason in JSON.", support_sealed,
                  category="changed_constraint"),
        _rephrase(support, "support_requested_keys",
                  "Can an opened item bought 20 days ago be refunded? Give only eligible in JSON.",
                  {"eligible": False}, category="changed_constraint"),
        _rephrase(support, "support_number",
                  "Can an opened item bought 40 days ago be refunded? "
                  "Give eligible and reason in JSON.",
                  {"eligible": False, "reason": "outside return window"},
                  category="changed_number", requires_invalidation=True),
        replace(support, id="support_revision", category="changed_dependency",
                context=Context("tenant:acme", {"refund_policy": "v2"}),
                requires_invalidation=True),
        replace(support, id="support_tenant", category="changed_tenant",
                context=Context("tenant:other", {"refund_policy": "v1"}),
                requires_invalidation=True),
        _rephrase(invoice, "invoice_paraphrase",
                  "Parse this invoice into JSON with invoice_id, vendor, total, currency, "
                  "and due_date."),
        _rephrase(invoice, "invoice_date_constraint",
                  "Parse this invoice into JSON with invoice_id, vendor, total, currency, "
                  "and invoice_date.", invoice_date, category="changed_constraint"),
        _rephrase(portfolio, "portfolio_paraphrase",
                  "What are the portfolio total_value and weights? Return JSON."),
        _rephrase(portfolio, "portfolio_requested_keys",
                  "What is the portfolio total_value? Return only total_value as JSON.",
                  {"total_value": 4000}, category="changed_constraint"),
    ]
    revised = snapshot(support.request)
    revised["messages"][-2]["content"] = json.dumps({**policy, "refund_days": 14})
    rows.append(replace(support, id="support_changed_policy", request=revised,
                        expected={"eligible": False, "reason": "outside return window"},
                        category="changed_tool_result", requires_invalidation=True))
    revised = snapshot(invoice.request)
    revised["messages"][1]["content"] = revised["messages"][1]["content"].replace(
        "October 10", "October 12").replace("1,250.50", "1,300.50")
    rows.append(replace(invoice, id="invoice_changed_source", request=revised,
                        expected={**invoice.expected, "total": 1300.5, "due_date": "2026-10-12"},
                        category="changed_history", requires_invalidation=True))
    revised = snapshot(invoice.request)
    revised["messages"][0]["content"] = revised["messages"][0]["content"].replace(
        "total is a number", "total is a string with exactly two decimal places")
    rows.append(replace(invoice, id="invoice_changed_schema", request=revised,
                        expected={**invoice.expected, "total": "1250.50"},
                        category="changed_output_schema", requires_invalidation=True))
    revised = snapshot(portfolio.request)
    revised["messages"][-2]["content"] = json.dumps({"positions": [
        {"symbol": "AAPL", "quantity": 10, "price": 220},
        {"symbol": "MSFT", "quantity": 5, "price": 400}]})
    rows.append(replace(portfolio, id="portfolio_changed_price", request=revised,
                        expected={"total_value": 4200, "weights": {"AAPL": .5238, "MSFT": .4762}},
                        category="changed_tool_result", requires_invalidation=True))
    revised = snapshot(portfolio.request)
    revised["messages"].insert(1, {"role": "user", "content":
        "For this conversation, present weights as percentages from zero to one hundred."})
    rows.append(replace(portfolio, id="portfolio_changed_history", request=revised,
                        expected={"total_value": 4000, "weights": {"AAPL": 50, "MSFT": 50}},
                        category="changed_history", requires_invalidation=True))
    rows.extend(replace(seed, id=seed.id + "_exact", category="exact_repeat")
                for seed in (support, invoice, portfolio))
    burst = _rephrase(portfolio, "concurrent_duplicates",
                      "Calculate total_value and weights for this portfolio. "
                      "Return JSON and include currency.",
                      {**portfolio.expected, "currency": "USD"}, category="concurrency")
    return AgentWorkload((support, invoice, portfolio), tuple(calibration), tuple(rows), burst)


def response_json(response: Any) -> dict[str, Any]:
    """Accept custom JSON output, chat envelopes, JSON text, or SDK model objects."""
    if callable(getattr(response, "model_dump", None)):
        response = response.model_dump(mode="json")
    if isinstance(response, Mapping) and "choices" in response:
        response = response["choices"][0]["message"]["content"]
    if isinstance(response, str):
        response = parse_json(response)
    if not isinstance(response, Mapping):
        raise ValueError("Generation response must contain a JSON object")
    return snapshot(dict(response))


def _matches(expected, observed):
    if isinstance(expected, dict):
        return (isinstance(observed, dict) and expected.keys() == observed.keys()
                and all(_matches(value, observed[key]) for key, value in expected.items()))
    if isinstance(expected, list):
        return (isinstance(observed, list) and len(expected) == len(observed)
                and all(_matches(a, b) for a, b in zip(expected, observed, strict=True)))
    if type(expected) in (int, float):
        return type(observed) in (int, float) and expected == observed
    return type(expected) is type(observed) and expected == observed


class FixtureGeneration:
    """Lookup-table output with artificial delay; not an open-source LLM."""

    cache_identity = {"provider": "offline-table-fixture", "version": "agent-workflows-v1"}

    def __init__(self, workload=None):
        workload = workload or agent_workload()
        cases = (*workload.seeds, *workload.sequential, workload.burst,
                 *(pair.request for pair in workload.calibration))
        self.answers = {fingerprint(case.request): case.expected for case in cases}

    async def complete(self, **kwargs):
        await asyncio.sleep(.04)
        return {"choices": [{"message": {
            "content": json.dumps(self.answers[fingerprint(kwargs)])}}]}


class FixtureDecision:
    """Uses labels as an oracle, proving wiring only rather than decision quality."""

    cache_identity = {"provider": "offline-label-oracle", "version": "agent-workflows-v1"}

    def __init__(self, generation):
        self.answers = generation.answers

    async def decide(self, request, candidates, kind):
        started = time.perf_counter()
        await asyncio.sleep(.001)
        expected = self.answers[fingerprint(request["kwargs"])]
        chosen = next((candidate.id for candidate in candidates
                       if _matches(expected, response_json(candidate.artifact))), None)
        return Decision(chosen, {}, "fixture-oracle", (time.perf_counter() - started) * 1000)


class FixtureEmbeddings:
    """Stable hashed bag-of-words; no learned semantic embedding model is loaded."""

    cache_identity = {"provider": "offline-hashed-words", "dimensions": 64, "version": "1"}

    def embed(self, texts):
        vectors = []
        for text in texts:
            vector = [0.] * 64
            for token in re.findall(r"\w+", text.casefold()):
                index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:2]) % 64
                vector[index] += 1
            vectors.append(vector)
        return vectors


async def _invoke(function, *args, **kwargs):
    if inspect.iscoroutinefunction(function) or inspect.iscoroutinefunction(
            getattr(function, "__call__", None)):
        return await function(*args, **kwargs)
    result = await asyncio.to_thread(function, *args, **kwargs)
    return await result if inspect.isawaitable(result) else result


class _GenerationMeter:
    def __init__(self, function):
        self.function = function
        self.calls = self.completed = self.failed = self.cancelled = 0
        self.pending = set()

    async def _execute(self, kwargs):
        try:
            result = await _invoke(self.function, **kwargs)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.failed += 1
            raise
        self.completed += 1
        return result

    async def complete(self, **kwargs):
        self.calls += 1
        sync = not (inspect.iscoroutinefunction(self.function) or inspect.iscoroutinefunction(
                    getattr(self.function, "__call__", None)))
        job = asyncio.create_task(self._execute(kwargs)) if sync else None
        if job is not None:
            self.pending.add(job)
            job.add_done_callback(self._completed_job)
        try:
            return await asyncio.shield(job) if job is not None else await self._execute(kwargs)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise

    def _completed_job(self, job):
        self.pending.discard(job)
        if not job.cancelled():
            job.exception()

    async def drain(self):
        if self.pending:
            await asyncio.gather(*(asyncio.shield(job) for job in tuple(self.pending)),
                                 return_exceptions=True)

    def snapshot(self):
        return {"generation_calls": self.calls, "generation_completions": self.completed,
                "generation_failures": self.failed,
                "generation_cancellation_requests": self.cancelled}


class _DecisionMeter:
    def __init__(self, decider):
        self.decider = decider
        self.calls = self.failed = 0

    @property
    def cache_identity(self):
        return self.decider.cache_identity

    async def decide(self, *args):
        self.calls += 1
        try:
            return await _invoke(self.decider.decide, *args)
        except Exception:
            self.failed += 1
            raise


class _EmbeddingMeter:
    def __init__(self, embedder):
        self.embedder = embedder
        self.calls = 0
        self._lock = threading.Lock()

    @property
    def cache_identity(self):
        return self.embedder.cache_identity

    @property
    def is_async(self):
        return (getattr(self.embedder, "is_async", None)
                if callable(getattr(self.embedder, "aembed", None)) else False)

    def embed(self, texts):
        with self._lock:
            self.calls += 1
        return self.embedder.embed(texts)

    async def aembed(self, texts):
        with self._lock:
            self.calls += 1
        return await _invoke(self.embedder.aembed, texts)


def _summary(rows):
    latencies = sorted(row["elapsed_ms"] for row in rows)
    def cached(row):
        return row["trace"]["path"] == "full_hit" and not row["trace"]["coalesced_nodes"]

    return {
        "requests": len(rows), "p50_ms": statistics.median(latencies),
        "p95_ms": latencies[max(0, math.ceil(len(latencies) * .95) - 1)],
        "incorrect_responses": sum(not row["correct"] for row in rows),
        "unsafe_reuses": sum(not row["correct"] and cached(row) for row in rows),
        "incorrect_coalesced_responses": sum(not row["correct"]
            and row["trace"]["coalesced_nodes"] > 0 for row in rows),
        "incorrect_generated_responses": sum(not row["correct"]
            and not cached(row) and row["error"] is None for row in rows),
        "request_errors": sum(row["error"] is not None for row in rows),
        "invalidation_failures": sum(row["invalidation_miss_expected"]
            and row["trace"]["path"] == "full_hit" for row in rows),
        "hits": sum(row["trace"]["path"] == "full_hit" for row in rows),
        "coalesced_requests": sum(row["trace"]["coalesced_nodes"] > 0 for row in rows),
        "decision_calls": sum(row["trace"]["jev_calls"] for row in rows),
        "tool_calls": sum(row["tool_calls"] for row in rows),
    }


async def run_agent_evaluation(
    *, offline: bool = False, generation: Callable[..., Any] | None = None,
    generation_identity: Mapping[str, Any] | None = None, decider=None, embedder=None,
    retrieval: str = "both", mode: str = "both", repeats: int = 1,
    decision_timeout_s: float = 2,
) -> dict[str, Any]:
    """Compare identical seeded workloads with custom local or remote providers.

    Tools are always fixture reads and execute before the wrapped LLM call. No
    correctness oracle is passed to JAC: unsafe decisions remain visible in rows.
    Preload, seed generation, calibration and background drains are outside the
    timed request. Live callbacks may incur provider charges, including races.
    """
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if retrieval not in ("bm25", "embeddings", "both"):
        raise ValueError("retrieval must be bm25, embeddings or both")
    if mode not in ("cache_first", "parallel", "both"):
        raise ValueError("mode must be cache_first, parallel or both")
    workload = agent_workload()
    if offline:
        if any(provider is not None for provider in (generation, decider, embedder)):
            raise ValueError("offline fixtures cannot be combined with live providers")
        fixture = FixtureGeneration(workload)
        generation = fixture.complete
        decider, embedder = FixtureDecision(fixture), FixtureEmbeddings()
        generation_identity = fixture.cache_identity
    if not callable(generation) or not callable(getattr(decider, "decide", None)):
        raise ValueError("Live agent evaluation requires generation callable and decider")
    if retrieval in ("both", "embeddings") and not (
            callable(getattr(embedder, "embed", None))
            or callable(getattr(embedder, "aembed", None))):
        raise ValueError("Embedding retrieval requires an embedder")
    if generation_identity is None or not isinstance(generation_identity, Mapping):
        raise ValueError("Pass a nonsecret stable generation_identity mapping")
    generation_identity = _safe_identity(generation_identity)
    _safe_identity(decider.cache_identity)
    if embedder is not None:
        _safe_identity(embedder.cache_identity)
    warm_started = time.perf_counter()
    for provider in (decider, getattr(generation, "__self__", generation), embedder):
        if callable(getattr(provider, "preload", None)):
            await _invoke(provider.preload)
    if embedder is not None and retrieval in ("both", "embeddings"):
        use_async = (callable(getattr(embedder, "aembed", None))
                     and getattr(embedder, "is_async", None) is not False)
        encode = embedder.aembed if use_async else embedder.embed
        await _invoke(encode, [workload.seeds[0].request["messages"][-1]["content"]])
    report = {
        "fixture": offline, "workload_version": "agent-workflows-v1",
        "workload": "refund support, invoice extraction, tool-informed portfolio analysis",
        "generation_provider": generation_identity,
        "decision_provider": _safe_identity(decider.cache_identity),
        "embedding_provider": _safe_identity(embedder.cache_identity) if embedder else None,
        "warmup_ms": (time.perf_counter() - warm_started) * 1000,
        "notes": [
            "Offline delays and oracle decisions are synthetic; they do not benchmark a model.",
            "Live mode measures supplied models on fixed, simulated read-only tool data.",
            "Calibration phrasing and labels are independent of held-out requests.",
            "Tools run before JAC, so this LLM-call wrapper does not eliminate tool calls.",
            "Generation calls count callback starts; cancellation does not establish "
            "billing savings.",
            "Setup, calibration, warmup and background drains are excluded from request latency.",
            "A small controlled corpus cannot establish production reuse accuracy or p95.",
        ], "runs": {},
    }
    strategies = ("bm25", "embeddings") if retrieval == "both" else (retrieval,)
    modes = ("cache_first", "parallel") if mode == "both" else (mode,)
    variants = [("no_cache", "bm25", "cache_first", False, False),
                ("exact_only", "bm25", "cache_first", True, False)]
    variants += [(f"{strategy}_{execution}", strategy, execution, True, True)
                 for strategy in strategies for execution in modes]
    for name, strategy, execution, enabled, semantic in variants:
        generator, judge = _GenerationMeter(generation), _DecisionMeter(decider)
        encoder = _EmbeddingMeter(embedder) if embedder is not None else None
        active_context = contextvars.ContextVar("agent_case_context")
        async with JAC(":memory:", enabled=enabled, jev=judge if semantic else None,
                       retrieval=strategy, embedder=encoder, mode="cache_first",
                       context=lambda _args, _kwargs: active_context.get(),
                       decision_timeout_s=decision_timeout_s,
                       decision_policy=DecisionPolicy(min_samples=4, overhead_ms=0)) as cache:
            wrapped = cache.wrap(generator.complete, namespace="agent-workflows-v1")

            seed_artifacts = {}

            async def invoke(case, *, first=False, capture=False):
                token = active_context.set(case.context)
                started = time.perf_counter()
                try:
                    if case.tool_calls:
                        await asyncio.sleep(.001 * case.tool_calls)
                    observed, observed_response, error = None, None, None
                    try:
                        response = await wrapped(**snapshot(case.request))
                        raw = (response.model_dump(mode="json")
                               if callable(getattr(response, "model_dump", None)) else response)
                        observed_response = snapshot(raw)
                        if capture:
                            seed_artifacts[case.id] = observed_response
                        observed = response_json(observed_response)
                    except Exception as exception:
                        error = type(exception).__name__
                    return {
                        "case_id": case.id, "family": case.family, "category": case.category,
                        "query": case.request["messages"][-1]["content"],
                        "correct": _matches(case.expected, observed) and error is None,
                        "expected_answer": case.expected, "observed_answer": observed,
                        "observed_response": observed_response,
                        "error": error, "elapsed_ms": (time.perf_counter() - started) * 1000,
                        "tool_calls": case.tool_calls, "trace": cache.last_trace.to_dict(),
                        "invalidation_miss_expected": first and case.requires_invalidation,
                    }
                finally:
                    active_context.reset(token)

            async def drain():
                await cache.drain()
                await generator.drain()
                for provider in (decider, getattr(generation, "__self__", generation), embedder):
                    if callable(getattr(provider, "drain", None)):
                        await _invoke(provider.drain)

            setup_started = time.perf_counter()
            seed_rows = {case.id: await invoke(case, capture=True) for case in workload.seeds}
            await drain()
            seed_calls = generator.calls
            setup_ms = (time.perf_counter() - setup_started) * 1000
            calibration, calibration_started = [], time.perf_counter()
            if semantic:
                for pair in workload.calibration:
                    seed_row = seed_rows[pair.seed.id]
                    artifact = seed_artifacts.get(pair.seed.id)
                    candidate = Candidate("calibration", {
                        "args": [], "kwargs": pair.seed.request,
                        "text": pair.seed.request["messages"][-1]["content"],
                    }, artifact, seed_row["elapsed_ms"])
                    request = {"args": [], "kwargs": pair.request.request,
                               "text": pair.request.request["messages"][-1]["content"]}
                    decision_started, accepted, error = time.perf_counter(), False, None
                    probabilities, decision_model = {}, None
                    try:
                        decision = await asyncio.wait_for(
                            judge.decide(request, [candidate], "answer"), decision_timeout_s)
                        if not isinstance(decision, Decision) or decision.candidate_id not in (
                                None, candidate.id):
                            raise ValueError("Calibration selected an unknown candidate")
                        if not isinstance(decision.model, str) or not decision.model:
                            raise ValueError("Calibration decision must identify its model")
                        if not isinstance(decision.probabilities, Mapping) or any(
                            type(value) not in (int, float) or not math.isfinite(value)
                            or not 0 <= value <= 1 for value in decision.probabilities.values()
                        ):
                            raise ValueError("Calibration decision probabilities are malformed")
                        probabilities = snapshot(dict(decision.probabilities))
                        decision_model = decision.model
                        accepted = decision.candidate_id == candidate.id
                    except Exception as exception:
                        error = type(exception).__name__
                    elapsed = (time.perf_counter() - decision_started) * 1000
                    expected = pair.reuse and seed_row["correct"]
                    token = active_context.set(pair.request.context)
                    try:
                        wrapped.observe_reuse(pair.request.request, valid_hit=expected and accepted,
                                              decision_ms=elapsed)
                    finally:
                        active_context.reset(token)
                    calibration.append({"case_id": pair.request.id, "expected_reuse": expected,
                                        "accepted": accepted, "error": error,
                                        "decision_ms": elapsed, "model": decision_model,
                                        "probabilities": probabilities})
                    # Cancelled local callbacks may still occupy their worker.
                    # Drain outside their measured latency before the next pair.
                    await drain()
            await drain()
            calibration_ms = (time.perf_counter() - calibration_started) * 1000
            cache.mode = execution
            before, decision_before = generator.snapshot(), (judge.calls, judge.failed)
            embedding_before = encoder.calls if encoder else 0
            rows = []
            for repeat in range(repeats):
                for case in workload.sequential:
                    rows.append(await invoke(case, first=repeat == 0))
                    await drain()
                rows.extend(await asyncio.gather(
                    *(invoke(workload.burst) for _ in range(workload.burst_size))))
                await drain()
            summary = _summary(rows)
            summary.update({key: value - before[key]
                            for key, value in generator.snapshot().items()})
            summary.update({
                "generation_calls_per_request": summary["generation_calls"] / len(rows),
                "decision_callback_calls": judge.calls - decision_before[0],
                "decision_failures": judge.failed - decision_before[1],
                "embedding_http_or_batch_calls":
                    (encoder.calls if encoder else 0) - embedding_before,
                "decision_statuses": dict(Counter(row["trace"]["decision_status"] for row in rows)),
                "setup_ms": setup_ms, "setup_generation_calls": seed_calls,
                "setup_incorrect_responses": sum(not row["correct"] for row in seed_rows.values()),
                "calibration_ms": calibration_ms,
                "calibration_incorrect_decisions": sum(row["accepted"] != row["expected_reuse"]
                                                       or row["error"] is not None
                                                       for row in calibration),
                "calibration_false_positives": sum(row["accepted"] and not row["expected_reuse"]
                                                   for row in calibration),
                "calibration_false_negatives": sum(row["expected_reuse"] and not row["accepted"]
                                                   and row["error"] is None for row in calibration),
                "calibration_errors": sum(row["error"] is not None for row in calibration),
                "calibration": calibration, "seed_rows": list(seed_rows.values()), "rows": rows,
                "categories": {category: _summary(
                    [row for row in rows if row["category"] == category])
                               for category in sorted({row["category"] for row in rows})},
            })
            report["runs"][name] = summary
    return report
