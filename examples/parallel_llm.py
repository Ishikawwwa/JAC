"""Illustrate parallel lookup with local delays; no live LLM or Jev calls.

Run: python examples/parallel_llm.py
The decision fixture uses known answers, not an evaluated semantic model.
"""

import asyncio
import time
from dataclasses import dataclass

from jac import JAC, Decision, DecisionPolicy

SUPPORT = "Support is available Monday through Friday during business hours."
REFUNDS = "Refunds are available for eligible returns."
SEED = "What are the support hours?"
PARAPHRASE = "When is support available?"


@dataclass(frozen=True)
class Scenario:
    name: str
    prompt: str
    decision_s: float
    reusable: bool
    expected: str


class FixtureDecision:
    """A cancellable local oracle solely for demonstrating the race."""

    def __init__(self, scenario):
        self.scenario = scenario
        self.started = 0
        self.cancelled = 0

    @property
    def cache_identity(self):
        return {"provider": "local-fixture", "model": "known-answer-v1"}

    async def decide(self, request, candidates, kind):
        self.started += 1
        started = time.perf_counter()
        try:
            await asyncio.sleep(self.scenario.decision_s)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        candidate_id = candidates[0].id if self.scenario.reusable else None
        return Decision(
            candidate_id,
            {"fixture": float(self.scenario.reusable)},
            "known-answer-v1",
            (time.perf_counter() - started) * 1000,
        )


async def demonstrate(scenario, *, mode="parallel"):
    counts = {"started": 0, "completed": 0, "cancelled": 0}
    decision = FixtureDecision(scenario)

    async def existing_llm_call(*, prompt, model):
        counts["started"] += 1
        try:
            # The seed's measured generation cost makes lookup eligible. Later
            # calls are faster, illustrating why a pending lookup must not delay them.
            await asyncio.sleep(0.30 if prompt == SEED else 0.12)
        except asyncio.CancelledError:
            counts["cancelled"] += 1
            raise
        counts["completed"] += 1
        return {"model": model, "content": REFUNDS if "refund" in prompt else SUPPORT}

    cache = JAC(
        path=":memory:",
        mode=mode,
        jev=decision,
        decision_policy=DecisionPolicy(min_samples=1, overhead_ms=5),
    )
    cached = cache.wrap(existing_llm_call, namespace="parallel-example-v1")
    try:
        await cached(prompt=SEED, model="fixture-llm")
        await cache.drain()
        # A known-answer fixture supplies one verified prior outcome. Production
        # defaults need at least 20 independently labeled comparable observations.
        cached.observe_reuse(
            {"prompt": PARAPHRASE, "model": "fixture-llm"},
            valid_hit=True,
            decision_ms=scenario.decision_s * 1000,
        )
        counts.update(started=0, completed=0, cancelled=0)

        started = time.perf_counter()
        response = await cached(prompt=scenario.prompt, model="fixture-llm")
        elapsed_ms = (time.perf_counter() - started) * 1000
        trace = cache.last_trace
        assert response["content"] == scenario.expected
        await cache.drain()
        print(
            f"{scenario.name} [{mode}]: {elapsed_ms:.1f} ms; {trace.path}; "
            f"LLM started/completed/cancelled="
            f"{counts['started']}/{counts['completed']}/{counts['cancelled']}; "
            f"decision calls={decision.started}"
        )
        return elapsed_ms
    finally:
        await cache.drain()
        cache.close()


async def main():
    print("Synthetic async race: generation takes 120 ms on measured requests.")
    print("Local fixture timings illustrate behavior, not provider benchmarks.")
    await demonstrate(Scenario("Cache wins (25 ms lookup)", PARAPHRASE, 0.025, True, SUPPORT))
    rejection = Scenario(
        "Cache rejects (25 ms lookup)", "Explain the refund policy.", 0.025, False, REFUNDS
    )
    sequential_ms = await demonstrate(rejection, mode="cache_first")
    parallel_ms = await demonstrate(rejection)
    await demonstrate(
        Scenario("Generation wins (200 ms lookup)", PARAPHRASE, 0.20, True, SUPPORT)
    )
    print(
        f"Measured rejected-candidate path: cache-first {sequential_ms:.1f} ms; "
        f"parallel {parallel_ms:.1f} ms."
    )


if __name__ == "__main__":
    asyncio.run(main())
