"""Reproducible small workload, runnable with fixtures or real model endpoints."""

from __future__ import annotations

import asyncio
import json
import math
import os
import statistics
import time
from collections import Counter
from pathlib import Path

from .call_cache import JAC
from .decisions import Candidate, Decision, DecisionPolicy, JevDecider
from .providers import (
    CompatibleClient,
    OpenAICompatibleDecider,
    OpenAICompatibleEmbedder,
    SentenceTransformerEmbedder,
)

# The oracle is independent of generated answers. This is a functionality suite,
# not a representative production dataset or proof of semantic model accuracy.
CASES = [
    ("What is the capital of France?", "Paris"),
    ("Which city is France's capital?", "Paris"),
    ("Name the French capital city.", "Paris"),
    ("What is the capital of Germany?", "Berlin"),
    ("Which city is Germany's capital?", "Berlin"),
    ("Name the German capital city.", "Berlin"),
    ("What color is a ripe banana?", "yellow"),
    ("Which color does a fully ripe banana have?", "yellow"),
    ("Describe the colour of a mature banana.", "yellow"),
    ("What color is an unripe banana?", "green"),
    ("Which color does a banana have before ripening?", "green"),
    ("Describe the colour of an immature banana.", "green"),
    ("What is the capital of Italy?", "Rome"),
    ("What color is a ripe tomato?", "red"),
]
ORACLE = dict(CASES)
SYSTEM = ('Answer the question with JSON only: {"answer":"value"}. '
          'Use these facts: France capital Paris; Germany capital Berlin; '
          'ripe banana yellow; unripe banana green; Italy capital Rome; '
          'ripe tomato red. No additional fields.')


def load_config(path: Path) -> dict:
    """Load adjacent .env literally; existing environment variables take precedence."""
    env = path.parent / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, separator, value = line.partition("=")
            if not separator or not key.strip().isidentifier():
                raise ValueError(".env must contain KEY=value lines")
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault(key.strip(), value)
    result = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(result, dict):
        raise ValueError("Config must be an object")
    return result


def _options(config: dict) -> dict:
    result = {key: config[key] for key in ("model", "base_url", "timeout_s") if key in config}
    variable = config.get("api_key_env")
    if variable:
        result["api_key"] = os.environ.get(variable, "")
        if not result["api_key"]:
            raise ValueError(f"Set {variable} in your environment or .env")
    return result


def providers(config: dict):
    generation_settings = config["generation"]
    if generation_settings.get("provider") == "transformers":
        from .local_generation import TransformersClient

        generation = TransformersClient(**{key: generation_settings[key] for key in
            ("model", "revision", "device", "max_new_tokens") if key in generation_settings})
    else:
        generation = CompatibleClient(**_options(generation_settings))
    settings = config["decision"]
    if settings.get("provider", "jev") == "jev":
        key_name = settings.get("api_key_env", "TYPESAFE_API_KEY")
        key = os.environ.get(key_name)
        if not key:
            raise ValueError(f"Set {key_name} in your environment or .env")
        decider = JevDecider(api_key=key, model=settings.get("model", "jev-1.13.0"),
                            timeout_s=settings.get("timeout_s", 10))
    elif settings["provider"] == "openai-compatible":
        decider = OpenAICompatibleDecider(**_options(settings))
    elif settings["provider"] == "laya":
        from .laya import LayaDecider

        decider = LayaDecider(**{key: settings[key] for key in
            ("model", "revision", "device", "threshold", "dtype", "max_len", "prompt_version")
            if key in settings})
    elif settings["provider"] == "jevk5":
        from .jevk5 import JevK5Decider

        decider = JevK5Decider(**{key: settings[key] for key in
            ("model", "revision", "base_url", "temperature", "threshold", "timeout_s", "max_len",
             "top_k", "max_candidates", "max_request_bytes", "prompt_version") if key in settings})
    elif settings["provider"] == "jevk5-cuda":
        from .jevk5_cuda import JevK5CudaDecider

        decider = JevK5CudaDecider(**{key: settings[key] for key in
            ("model", "revision", "device", "graphs", "temperature", "threshold", "max_len",
             "max_candidates", "max_request_bytes", "prompt_version") if key in settings})
    else:
        raise ValueError("Decision provider must be jev, jevk5, jevk5-cuda, laya "
                         "or openai-compatible")
    return generation, decider


def create_embedder(config):
    settings = config["embedding"]
    if settings.get("provider") == "sentence-transformers":
        return SentenceTransformerEmbedder(
            model=settings.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
            revision=settings.get("revision"), device=settings.get("device", "cpu"))
    if settings.get("provider") == "openai-compatible":
        return OpenAICompatibleEmbedder(**_options(settings))
    raise ValueError("Embedding provider must be sentence-transformers or openai-compatible")


class FixtureClient:
    calls = 0
    cache_identity = {"provider": "fixture-generation"}

    async def complete(self, **kwargs):
        self.calls += 1
        await asyncio.sleep(.08)
        answer = ORACLE[kwargs["messages"][-1]["content"]]
        return {"choices": [{"message": {"content": json.dumps({"answer": answer})}}]}


class FixtureDecider:
    model = "fixture-oracle-not-jev"
    cache_identity = {"provider": "fixture-decision"}

    async def decide(self, request, candidates, kind):
        started = time.perf_counter()
        await asyncio.sleep(.005)
        expected = ORACLE[request["text"]]
        chosen = next((c.id for c in candidates if ORACLE[c.request["text"]] == expected), None)
        return Decision(chosen, {}, self.model, (time.perf_counter() - started) * 1000)


class FixtureEmbedder:
    calls = 0
    cache_identity = {"provider": "fixture-handcrafted-vectors-not-a-model"}

    def embed(self, texts):
        self.calls += 1
        time.sleep(.002)
        labels = ["Paris", "Berlin", "yellow", "green", "Rome", "red"]
        return [[float(label == ORACLE[text]) for label in labels] for text in texts]


def _request(text):
    return {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": text}], "temperature": 0}


def _answer(response):
    return json.loads(response["choices"][0]["message"]["content"])["answer"]


def _percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


async def run_evaluation(*, config: dict | None = None, offline: bool = False,
                         retrieval: str = "both", mode: str = "both", repeats: int = 2):
    generation, decider = ((FixtureClient(), FixtureDecider()) if offline else providers(config))
    if callable(getattr(decider, "preload", None)):
        await decider.preload()
    strategies = ["bm25", "embeddings"] if retrieval == "both" else [retrieval]
    modes = ["cache_first", "parallel"] if mode == "both" else [mode]
    embedder = ((FixtureEmbedder() if offline else create_embedder(config))
                if "embeddings" in strategies else None)
    report = {"fixture": offline,
              "workload": "four seeded fact families plus two new facts; controlled JSON answers",
              "generation_provider": generation.cache_identity,
              "decision_provider": decider.cache_identity,
              "embedding_provider": embedder.cache_identity if embedder else None,
              "notes": ["Setup, warmup and labeled calibration are reported separately.",
                        "Parallel generation calls count starts, not billed tokens.",
                        "This small suite does not establish production reuse accuracy."],
              "runs": {}}
    report["generation_load_ms"] = getattr(generation, "load_ms", None)
    if callable(getattr(decider, "preload", None)):
        warm = time.perf_counter()
        warm_text, warm_answer = CASES[0]
        await decider.decide(
            {"args": [], "kwargs": _request(warm_text), "text": warm_text},
            [Candidate("warmup", {"text": warm_text}, {"choices": [{"message": {
                "content": json.dumps({"answer": warm_answer})}}]}, 0)], "answer")
        report["decision_warmup_ms"] = (time.perf_counter() - warm) * 1000
    # Warm live providers and local encoders outside measured request latency.
    warm = time.perf_counter()
    if embedder:
        await asyncio.to_thread(embedder.embed, [CASES[0][0]])
    report["embedding_warmup_ms"] = (time.perf_counter() - warm) * 1000
    variants = [("no_cache", "bm25", "cache_first", False, False),
                ("exact_only", "bm25", "cache_first", True, False)]
    variants += [(f"{strategy}_{execution}", strategy, execution, True, True)
                 for strategy in strategies for execution in modes]
    for name, strategy, execution, enabled, semantic in variants:
        async with JAC(path=":memory:", enabled=enabled, jev=decider if semantic else None,
                       retrieval=strategy, embedder=embedder, mode=execution,
                       decision_timeout_s=config.get("decision_timeout_s", 15) if config else 2,
                       decision_policy=DecisionPolicy(min_samples=4, overhead_ms=0)) as cache:
            wrapped = cache.wrap(generation.complete, namespace="evaluation-generation-v1")
            setup_started = time.perf_counter()
            setup_errors = 0
            setup_calls = generation.calls
            # Equal seeded corpus per variant; use cache-first for setup so writes
            # are complete before calibration and measured requests start.
            cache.mode = "cache_first"
            for index in (0, 3, 6, 9):
                response = await wrapped(**_request(CASES[index][0]))
                setup_errors += _answer(response) != CASES[index][1]
            calibration = []
            if semantic:
                # Eight independently labeled positive/negative pairs. They use
                # second phrasing; the third phrasing is held out for evaluation.
                for index in (0, 3, 6, 9):
                    for other in (index, (index + 3) % 12):
                        seed_text, seed_answer = CASES[other]
                        new_text, new_answer = CASES[index + 1]
                        artifact = {"choices": [{"message": {
                            "content": json.dumps({"answer": seed_answer})}}]}
                        candidate = Candidate("calibration", {"text": seed_text}, artifact, 0)
                        started = time.perf_counter()
                        decision = await decider.decide(
                            {"args": [], "kwargs": _request(new_text), "text": new_text},
                            [candidate], "answer")
                        elapsed = (time.perf_counter() - started) * 1000
                        expected = new_answer == seed_answer
                        accepted = decision.candidate_id == candidate.id
                        if decision.candidate_id not in (None, candidate.id):
                            raise ValueError("Calibration returned an unknown candidate")
                        wrapped.observe_reuse(_request(new_text),
                            valid_hit=expected and accepted, decision_ms=elapsed)
                        calibration.append({"expected_reuse": expected, "accepted": accepted,
                                            "decision_ms": elapsed})
            cache.mode = execution
            setup_ms = (time.perf_counter() - setup_started) * 1000
            before = generation.calls
            completed_before = getattr(generation, "completed", None)
            tokens_before = getattr(generation, "output_tokens", None)
            embedding_before = getattr(embedder, "calls", 0)
            rows = []
            # Held-out paraphrases first, then exact repetitions and calibration
            # wording; all modes receive the identical sequence.
            workload = [CASES[i] for i in (2, 5, 8, 11, 0, 3, 6, 9, 1, 4, 7, 10, 12, 13)] * repeats
            for text, expected in workload:
                started = time.perf_counter()
                response = await wrapped(**_request(text))
                latency = (time.perf_counter() - started) * 1000
                try:
                    observed_answer = str(_answer(response))
                    correct = observed_answer.casefold() == expected.casefold()
                except (KeyError, TypeError, ValueError):
                    observed_answer = None
                    correct = False
                rows.append({"query": text, "correct": correct, "elapsed_ms": latency,
                             "expected_answer": expected, "observed_answer": observed_answer,
                             "trace": cache.last_trace.to_dict()})
                # Drain outside request timing: avoid overlapping separate trials
                # and capture all speculative starts and deferred admissions.
                await cache.drain()
                if hasattr(generation, "drain"):
                    await generation.drain()
                if hasattr(decider, "drain"):
                    await decider.drain()
            values = [row["elapsed_ms"] for row in rows]
            decision_times = [event["elapsed_ms"] for row in rows
                              for event in row["trace"]["events"]
                              if event["name"] == "reuse_decision"]
            report["runs"][name] = {
                "requests": len(rows), "p50_ms": statistics.median(values),
                "p95_ms": _percentile(values, .95),
                "generation_calls": generation.calls - before,
                "generation_calls_per_request": (generation.calls - before) / len(rows),
                "generation_completions": (generation.completed - completed_before
                                           if completed_before is not None else None),
                "generated_tokens": (generation.output_tokens - tokens_before
                                     if tokens_before is not None else None),
                "decision_p50_ms": statistics.median(decision_times) if decision_times else None,
                "decision_p95_ms": _percentile(decision_times, .95) if decision_times else None,
                "decision_calls": sum(row["trace"]["jev_calls"] for row in rows),
                "decision_statuses": dict(Counter(row["trace"]["decision_status"] for row in rows)),
                "retrieval_events": [event for row in rows for event in row["trace"]["events"]
                                     if event["name"].startswith("retrieval_")],
                "embedding_http_or_batch_calls": getattr(embedder, "calls", 0) - embedding_before,
                "incorrect_responses": sum(not row["correct"] for row in rows),
                "hits": sum(row["trace"]["path"] == "full_hit" for row in rows),
                "setup_ms": setup_ms, "setup_generation_calls": before - setup_calls,
                "setup_incorrect_responses": setup_errors,
                "calibration_incorrect_decisions": sum(
                    row["expected_reuse"] != row["accepted"] for row in calibration),
                "calibration": calibration, "rows": rows,
            }
    return report
