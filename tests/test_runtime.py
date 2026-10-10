"""Execution-reuse regressions with deterministic fake operations and no provider calls."""

import asyncio
import unittest
from dataclasses import replace

from jevcache.decisions import Decision, DecisionPolicy
from jevcache.models import Context, Node, Plan, Ref, Request
from jevcache.runtime import Runtime
from jevcache.store import SQLiteStore


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def request(
    identifier="A", *, scope="tenant-a", revision="1", text="", policy="1", contract="json-v1"
):
    return Request(
        "exposure",
        {"id": identifier},
        Context(scope, {"positions": revision}, policy),
        text=text,
        output_contract=contract,
    )


def fetch_plan(req):
    return Plan(
        task=req.task,
        version="1",
        parameters={"id": "string"},
        nodes=(Node("fetch", "fetch", {"id": Ref.input("id")}),),
        output=Ref("fetch"),
        output_contract=req.output_contract,
    )


class FakeDecider:
    def __init__(self, *, accept=True, error=None):
        self.calls = []
        self.accept = accept
        self.error = error

    async def decide(self, req, candidates, kind):
        self.calls.append((req, candidates, kind))
        if self.error:
            raise self.error
        chosen = candidates[0].id if self.accept else None
        return Decision(chosen, {}, "fake-jev", 1.0)


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.clock = Clock()
        self.store = SQLiteStore(clock=self.clock)
        self.calls = []
        self.planner_calls = 0

    def tearDown(self):
        self.store.close()

    async def fetch(self, id):
        self.calls.append(id)
        return {"id": id, "exposure": 42}

    async def planner(self, req):
        self.planner_calls += 1
        return fetch_plan(req)

    def runtime(self, **kwargs):
        runtime = Runtime(self.store, **kwargs)
        runtime.register("fetch", self.fetch, kind="tool")
        return runtime

    async def test_exact_validated_answer_bypasses_all_execution_and_jev(self):
        decider = FakeDecider()
        runtime = self.runtime(
            enable_semantic=True,
            decider=decider,
            final_validator=lambda req, value: value["id"] == req.params["id"],
        )
        first = await runtime.run(request(), planner=self.planner)
        second = await runtime.run(request(), planner=self.planner)
        self.assertEqual(first.path, "miss")
        self.assertEqual(second.path, "full_hit")
        self.assertEqual(second.value, first.value)
        self.assertEqual(self.calls, ["A"])
        self.assertEqual(self.planner_calls, 1)
        self.assertEqual(first.trace.llm_calls, 1)
        self.assertEqual(first.trace.tool_calls, 1)
        self.assertEqual(second.trace.llm_calls, 0)
        self.assertEqual(second.trace.tool_calls, 0)
        self.assertEqual(second.trace.jev_calls, 0)
        self.assertEqual(decider.calls, [])

    async def test_exact_answer_survives_replacement_of_shared_plan_template(self):
        async def planner(req):
            self.planner_calls += 1
            return replace(
                fetch_plan(req),
                output={
                    "mode": req.text,
                    "result": Ref("fetch"),
                },
            )

        runtime = self.runtime(final_validator=lambda req, value: True)
        gross = request(text="gross exposure")
        await runtime.run(gross, planner=planner)
        await runtime.run(request(text="net exposure"), planner=planner)
        repeated = await runtime.run(gross, planner=planner)
        self.assertEqual(repeated.path, "full_hit")
        self.assertEqual(repeated.value["mode"], "gross exposure")
        self.assertEqual(repeated.trace.llm_calls, 0)
        self.assertEqual(repeated.trace.tool_calls, 0)
        self.assertEqual(self.planner_calls, 2)

    async def test_invalid_boolean_configuration_is_rejected_before_execution(self):
        for flag in ("false", 0, 1, None):
            with self.subTest(cache=flag), self.assertRaises(TypeError):
                Runtime(self.store, enable_cache=flag)
            with self.subTest(semantic=flag), self.assertRaises(TypeError):
                Runtime(self.store, enable_semantic=flag)
            with self.subTest(read_only=flag), self.assertRaises(TypeError):
                Runtime(self.store).register("fetch", self.fetch, read_only=flag)
        self.assertEqual(self.calls, [])

    async def test_no_final_validator_still_reuses_plan_and_intermediate(self):
        runtime = self.runtime()
        await runtime.run(request(), planner=self.planner)
        second = await runtime.run(request(), planner=self.planner)
        self.assertEqual(second.path, "partial_hit")
        self.assertTrue(second.trace.plan_hit)
        self.assertEqual(second.trace.cached_nodes, 1)
        self.assertEqual(second.trace.executed_nodes, 0)
        self.assertEqual(second.trace.llm_calls, 0)
        self.assertEqual(second.trace.tool_calls, 0)
        self.assertEqual(self.calls, ["A"])

    async def test_single_operation_wrapper_skips_planning_and_reuses_tool_result(self):
        runtime = self.runtime()
        context = Context("tenant-a", {"positions": "1"})
        first = await runtime.call("fetch", context=context, id="A")
        second = await runtime.call("fetch", context=context, id="A")
        self.assertEqual(first.value, {"id": "A", "exposure": 42})
        self.assertEqual(first.trace.llm_calls, 0)
        self.assertEqual(first.trace.tool_calls, 1)
        self.assertEqual(second.trace.llm_calls, 0)
        self.assertEqual(second.trace.tool_calls, 0)
        self.assertEqual(second.trace.cached_nodes, 1)
        self.assertEqual(self.calls, ["A"])

    async def test_returned_values_do_not_mutate_stored_or_original_outputs(self):
        original = {"items": [{"id": "A"}]}

        def fetch(id):
            return original

        runtime = Runtime(self.store, final_validator=lambda req, value: True)
        runtime.register("fetch", fetch)
        first = await runtime.run(request(), planner=self.planner)
        first.value["items"][0]["id"] = "changed by caller"
        second = await runtime.run(request(), planner=self.planner)
        self.assertEqual(second.path, "full_hit")
        self.assertEqual(second.value, {"items": [{"id": "A"}]})
        self.assertEqual(original, {"items": [{"id": "A"}]})
        second.value["items"].append({"id": "B"})
        third = await runtime.run(request(), planner=self.planner)
        self.assertEqual(third.value, {"items": [{"id": "A"}]})

    async def test_new_portfolio_reuses_plan_and_binds_new_structured_arguments(self):
        runtime = self.runtime(final_validator=lambda req, value: True)
        await runtime.run(request("A"), planner=self.planner)
        second = await runtime.run(request("B"), planner=self.planner)
        self.assertEqual(second.value["id"], "B")
        self.assertEqual(self.calls, ["A", "B"])
        self.assertEqual(self.planner_calls, 1)
        self.assertTrue(second.trace.plan_hit)
        self.assertEqual(second.trace.llm_calls, 0)
        self.assertEqual(second.trace.tool_calls, 1)

    async def test_dependency_revision_invalidates_answer_and_nodes_but_preserves_plan(self):
        runtime = self.runtime(final_validator=lambda req, value: True)
        await runtime.run(request(revision="1"), planner=self.planner)
        second = await runtime.run(request(revision="2"), planner=self.planner)
        self.assertNotEqual(second.path, "full_hit")
        self.assertTrue(second.trace.plan_hit)
        self.assertEqual(second.trace.cached_nodes, 0)
        self.assertEqual(second.trace.tool_calls, 1)
        self.assertEqual(self.calls, ["A", "A"])
        self.assertEqual(self.planner_calls, 1)

    async def test_scope_policy_and_contract_partition_answer_and_plan_reuse(self):
        for changed in (
            request(scope="tenant-b"),
            request(policy="2"),
            request(contract="report-v2"),
        ):
            with self.subTest(request=changed):
                with SQLiteStore(clock=self.clock) as isolated:
                    runtime = Runtime(isolated, final_validator=lambda req, value: True)
                    runtime.register("fetch", self.fetch, kind="tool")
                    await runtime.run(request(), planner=self.planner)
                    result = await runtime.run(changed, planner=self.planner)
                    self.assertNotEqual(result.path, "full_hit")
                    self.assertFalse(result.trace.plan_hit)
                    self.assertEqual(result.trace.llm_calls, 1)
                    # A raw read-only tool result can serve a different final output format.
                    expected_tool_calls = int(changed.output_contract == "json-v1")
                    self.assertEqual(result.trace.tool_calls, expected_tool_calls)

    async def test_changed_operation_version_invalidates_answer_plan_and_node(self):
        first_runtime = self.runtime(final_validator=lambda req, value: True)
        await first_runtime.run(request(), planner=self.planner)
        second_runtime = Runtime(self.store, final_validator=lambda req, value: True)
        second_runtime.register("fetch", self.fetch, kind="tool", version="2")
        second = await second_runtime.run(request(), planner=self.planner)
        self.assertEqual(second.path, "miss")
        self.assertFalse(second.trace.plan_hit)
        self.assertEqual(second.trace.tool_calls, 1)
        self.assertEqual(self.planner_calls, 2)

    async def test_answer_lifetime_cannot_exceed_shortest_upstream_ttl(self):
        runtime = Runtime(self.store, answer_ttl_s=60, final_validator=lambda req, value: True)
        runtime.register("fetch", self.fetch, kind="tool", ttl_s=10)
        await runtime.run(request(), planner=self.planner)
        self.clock.now += 9
        self.assertEqual((await runtime.run(request(), planner=self.planner)).path, "full_hit")
        self.clock.now += 1
        expired = await runtime.run(request(), planner=self.planner)
        self.assertNotEqual(expired.path, "full_hit")
        self.assertEqual(expired.trace.tool_calls, 1)
        self.assertEqual(self.calls, ["A", "A"])

    async def test_partial_graph_executes_only_changed_dependency_branch(self):
        counts = {"positions": 0, "prices": 0, "calculate": 0}

        async def positions(id):
            counts["positions"] += 1
            return {"units": 3}

        async def prices(id):
            counts["prices"] += 1
            return {"price": 10 * counts["prices"]}

        def calculate(units, price):
            counts["calculate"] += 1
            return units * price

        async def planner(req):
            return Plan(
                req.task,
                "1",
                {"id": "string"},
                (
                    Node("positions", "positions", {"id": Ref.input("id")}),
                    Node("prices", "prices", {"id": Ref.input("id")}),
                    Node(
                        "calculate",
                        "calculate",
                        {
                            "units": Ref("positions", ("units",)),
                            "price": Ref("prices", ("price",)),
                        },
                    ),
                ),
                Ref("calculate"),
            )

        runtime = Runtime(self.store, final_validator=lambda req, value: True)
        runtime.register("positions", positions, dependencies=("positions",))
        runtime.register("prices", prices, dependencies=("prices",))
        runtime.register("calculate", calculate, kind="compute")
        original = Request(
            "exposure", {"id": "A"}, Context("tenant-a", {"positions": "1", "prices": "1"})
        )
        await runtime.run(original, planner=planner)
        updated = replace(original, context=Context("tenant-a", {"positions": "1", "prices": "2"}))
        result = await runtime.run(updated, planner=planner)
        self.assertEqual(result.value, 60)
        self.assertEqual(counts, {"positions": 1, "prices": 2, "calculate": 2})
        self.assertTrue(result.trace.plan_hit)
        self.assertEqual(result.trace.cached_nodes, 1)
        self.assertEqual(result.trace.executed_nodes, 2)
        self.assertEqual(result.trace.tool_calls, 1)

    async def test_missing_declared_dependency_fails_before_handler(self):
        runtime = Runtime(self.store)
        runtime.register("fetch", self.fetch, dependencies=("prices",))
        with self.assertRaises((ValueError, KeyError)):
            await runtime.run(request(), planner=self.planner)
        self.assertEqual(self.calls, [])

    async def test_side_effects_execute_each_request_without_final_admission(self):
        runtime = Runtime(self.store, final_validator=lambda req, value: True)
        runtime.register("fetch", self.fetch, read_only=False)
        await runtime.run(request(), planner=self.planner)
        second = await runtime.run(request(), planner=self.planner)
        self.assertEqual(self.calls, ["A", "A"])
        self.assertNotEqual(second.path, "full_hit")
        self.assertEqual(second.trace.cached_nodes, 0)
        self.assertEqual(second.trace.tool_calls, 1)

    async def test_false_operation_validator_cannot_cache_output(self):
        runtime = Runtime(self.store, final_validator=lambda req, value: True)
        runtime.register("fetch", self.fetch, validator=lambda value: False)
        await runtime.run(request(), planner=self.planner)
        second = await runtime.run(request(), planner=self.planner)
        self.assertEqual(self.calls, ["A", "A"])
        self.assertEqual(second.trace.tool_calls, 1)
        self.assertNotEqual(second.path, "full_hit")

    async def test_false_final_validator_cannot_admit_answer(self):
        runtime = self.runtime(final_validator=lambda req, value: False)
        await runtime.run(request(), planner=self.planner)
        second = await runtime.run(request(), planner=self.planner)
        self.assertEqual(second.path, "partial_hit")
        self.assertEqual(second.trace.cached_nodes, 1)

    async def test_async_validators_are_supported(self):
        async def node_validator(value):
            return value["exposure"] == 42

        async def final_validator(req, value):
            return value["id"] == req.params["id"]

        runtime = Runtime(self.store, final_validator=final_validator)
        runtime.register("fetch", self.fetch, validator=node_validator)
        await runtime.run(request(), planner=self.planner)
        self.assertEqual((await runtime.run(request(), planner=self.planner)).path, "full_hit")

    async def test_identical_readonly_operations_are_coalesced_between_requests(self):
        entered = asyncio.Event()
        release = asyncio.Event()
        count = 0

        async def fetch(id):
            nonlocal count
            count += 1
            entered.set()
            await release.wait()
            return {"id": id}

        runtime = Runtime(self.store)
        runtime.register("fetch", fetch)
        first = asyncio.create_task(runtime.run(request(), planner=self.planner))
        await asyncio.wait_for(entered.wait(), timeout=1)
        second = asyncio.create_task(runtime.run(request(), planner=self.planner))
        # Let the second request reach its pending operation without measuring wall time.
        for _ in range(10):
            await asyncio.sleep(0)
        release.set()
        results = await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
        self.assertEqual(count, 1)
        self.assertEqual([result.value for result in results], [{"id": "A"}] * 2)
        self.assertEqual(sum(result.trace.tool_calls for result in results), 1)
        self.assertEqual(sum(result.trace.coalesced_nodes for result in results), 1)

    async def test_side_effecting_operations_are_not_coalesced(self):
        entered = asyncio.Event()
        release = asyncio.Event()
        count = 0

        async def execute(id):
            nonlocal count
            count += 1
            if count == 2:
                entered.set()
            await release.wait()
            return {"id": id}

        runtime = Runtime(self.store)
        runtime.register("fetch", execute, read_only=False)
        pending = [
            asyncio.create_task(runtime.run(request(), planner=self.planner)) for _ in range(2)
        ]
        try:
            await asyncio.wait_for(entered.wait(), timeout=1)
        finally:
            release.set()
        results = await asyncio.wait_for(asyncio.gather(*pending), timeout=1)
        self.assertEqual(count, 2)
        self.assertEqual(sum(result.trace.coalesced_nodes for result in results), 0)

    async def test_independent_graph_nodes_execute_concurrently(self):
        entered = asyncio.Event()
        release = asyncio.Event()
        started = []

        async def fetch(id):
            started.append(id)
            if len(started) == 2:
                entered.set()
            await release.wait()
            return id

        async def planner(req):
            return Plan(
                req.task,
                "1",
                {"id": "string"},
                (
                    Node("left", "fetch", {"id": Ref.input("id")}),
                    Node("right", "fetch", {"id": "other"}),
                ),
                {"left": Ref("left"), "right": Ref("right")},
            )

        runtime = Runtime(self.store)
        runtime.register("fetch", fetch)
        pending = asyncio.create_task(runtime.run(request(), planner=planner))
        try:
            await asyncio.wait_for(entered.wait(), timeout=1)
        finally:
            release.set()
        result = await asyncio.wait_for(pending, timeout=1)
        self.assertCountEqual(started, ["A", "other"])
        self.assertEqual(result.value, {"left": "A", "right": "other"})

    async def test_failure_is_traced_and_retried_on_next_request(self):
        attempts = 0

        async def flaky(id):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("temporary failure")
            return {"id": id}

        runtime = Runtime(self.store, final_validator=lambda req, value: True)
        runtime.register("fetch", flaky)
        with self.assertRaises(ExceptionGroup) as caught:
            await runtime.run(request(), planner=self.planner)
        self.assertIsInstance(caught.exception.exceptions[0], RuntimeError)
        self.assertIn("temporary failure", str(caught.exception.exceptions[0]))
        recovered = await runtime.run(request(), planner=self.planner)
        self.assertEqual(attempts, 2)
        self.assertEqual(recovered.value, {"id": "A"})
        traces = self.store.traces()
        self.assertEqual(traces[0]["path"], "error")
        self.assertEqual(traces[0]["tool_calls"], 1)
        self.assertEqual(len(traces), 2)

    async def test_cache_disabled_runs_planner_and_operations_each_time(self):
        runtime = self.runtime(enable_cache=False, final_validator=lambda req, value: True)
        await runtime.run(request(), planner=self.planner)
        second = await runtime.run(request(), planner=self.planner)
        self.assertEqual(self.calls, ["A", "A"])
        self.assertEqual(self.planner_calls, 2)
        self.assertEqual(second.path, "miss")
        self.assertEqual(second.trace.llm_calls, 1)
        self.assertEqual(second.trace.tool_calls, 1)

    async def test_llm_operations_are_measured_separately_from_planning(self):
        runtime = Runtime(self.store)
        runtime.register("fetch", self.fetch, kind="llm")
        result = await runtime.run(request(), planner=self.planner)
        self.assertEqual(result.trace.llm_calls, 2)
        self.assertEqual(result.trace.tool_calls, 0)
        self.assertEqual(result.trace.executed_nodes, 1)

    async def test_changed_prose_requires_planning_without_explicit_text_binding(self):
        runtime = self.runtime()
        await runtime.run(request(text="Gross exposure"), planner=self.planner)
        second = await runtime.run(request(text="Net exposure"), planner=self.planner)
        self.assertEqual(self.planner_calls, 2)
        self.assertFalse(second.trace.plan_hit)

    async def test_text_binding_passes_current_prose_into_reused_workflow(self):
        calls = []

        async def fetch(id, instruction):
            calls.append((id, instruction))
            return instruction

        async def planner(req):
            self.planner_calls += 1
            return Plan(
                req.task,
                "1",
                {"id": "string"},
                (
                    Node(
                        "fetch",
                        "fetch",
                        {
                            "id": Ref.input("id"),
                            "instruction": Ref.text(),
                        },
                    ),
                ),
                Ref("fetch"),
                text_independent=True,
            )

        runtime = Runtime(self.store)
        runtime.register("fetch", fetch)
        await runtime.run(request(text="Gross exposure"), planner=planner)
        second = await runtime.run(request(text="Net exposure"), planner=planner)
        self.assertEqual(second.value, "Net exposure")
        self.assertEqual(calls, [("A", "Gross exposure"), ("A", "Net exposure")])
        self.assertEqual(self.planner_calls, 1)
        self.assertTrue(second.trace.plan_hit)

    async def test_text_binding_does_not_certify_equivalent_workflow_structure(self):
        async def planner(req):
            self.planner_calls += 1
            return Plan(
                req.task,
                "1",
                {"id": "string"},
                (Node("fetch", "fetch", {"id": Ref.input("id")}),),
                {
                    "mode": "net" if "net" in req.text else "gross",
                    "instruction": Ref.text(),
                    "result": Ref("fetch"),
                },
            )

        runtime = self.runtime()
        await runtime.run(request(text="gross exposure"), planner=planner)
        second = await runtime.run(request(text="net exposure"), planner=planner)
        self.assertEqual(second.value["instruction"], "net exposure")
        self.assertEqual(second.value["mode"], "net")
        self.assertEqual(self.planner_calls, 2)

    async def test_text_independent_workflow_reuses_plan_for_equivalent_prose(self):
        async def planner(req):
            self.planner_calls += 1
            return replace(fetch_plan(req), text_independent=True)

        runtime = self.runtime()
        await runtime.run(request(text="Exposure please"), planner=planner)
        second = await runtime.run(request("B", text="Calculate exposure"), planner=planner)
        self.assertEqual(second.value["id"], "B")
        self.assertEqual(self.planner_calls, 1)
        self.assertTrue(second.trace.plan_hit)

    async def test_invalid_plan_is_rejected_before_tools(self):
        invalid = (
            replace(fetch_plan(request()), nodes=(Node("fetch", "fetch", {"id": Ref("fetch")}),)),
            replace(
                fetch_plan(request()),
                nodes=(
                    Node(
                        "fetch",
                        "unknown",
                        {
                            "id": Ref.input("id"),
                        },
                    ),
                ),
            ),
            replace(fetch_plan(request()), nodes=(Node("fetch", "fetch", {"id": "A"}),)),
        )
        for plan in invalid:
            with self.subTest(plan=plan):
                runtime = self.runtime(enable_cache=False)

                async def planner(req):
                    return plan

                with self.assertRaises(ValueError):
                    await runtime.run(request(), planner=planner)
        self.assertEqual(self.calls, [])

    async def test_semantic_cold_start_avoids_decision_without_measured_outcomes(self):
        decider = FakeDecider()
        runtime = self.runtime(
            enable_semantic=True,
            decider=decider,
            decision_policy=DecisionPolicy(min_samples=1, overhead_ms=0),
            final_validator=lambda req, value: True,
        )
        await runtime.run(request(text="Calculate portfolio exposure"), planner=self.planner)
        result = await runtime.run(request(text="Portfolio exposure please"), planner=self.planner)
        self.assertEqual(decider.calls, [])
        self.assertEqual(result.trace.jev_calls, 0)

    async def semantic_runtime(self, decider):
        async def expensive(id, instruction):
            self.calls.append(id)
            await asyncio.sleep(0.015)
            return {"id": id, "exposure": 42}

        async def planner(req):
            self.planner_calls += 1
            return replace(
                fetch_plan(req),
                text_independent=True,
                nodes=(
                    Node(
                        "fetch",
                        "fetch",
                        {
                            "id": Ref.input("id"),
                            "instruction": Ref.text(),
                        },
                    ),
                ),
            )

        runtime = Runtime(
            self.store,
            enable_semantic=True,
            decider=decider,
            decision_policy=DecisionPolicy(min_samples=1, overhead_ms=0),
            final_validator=lambda req, value: True,
        )
        runtime.register("fetch", expensive)
        await runtime.run(request(text="Calculate portfolio exposure"), planner=planner)
        return runtime

    async def test_selective_semantic_answer_hit_avoids_generative_and_tool_calls(self):
        decider = FakeDecider()
        runtime = await self.semantic_runtime(decider)
        equivalent = request(text="Portfolio exposure please")
        runtime.observe_reuse(equivalent, valid_hit=True, decision_ms=0)
        result = await runtime.run(equivalent, planner=self.planner)
        self.assertEqual(result.path, "full_hit")
        self.assertEqual(result.trace.jev_calls, 1)
        self.assertEqual(result.trace.llm_calls, 0)
        self.assertEqual(result.trace.tool_calls, 0)
        self.assertEqual(decider.calls[0][2], "answer")
        self.assertEqual(self.calls, ["A"])
        self.assertEqual(self.planner_calls, 1)

    async def test_semantic_decision_is_skipped_when_exact_nodes_already_avoid_work(self):
        decider = FakeDecider()
        runtime = self.runtime(
            enable_semantic=True,
            decider=decider,
            decision_policy=DecisionPolicy(min_samples=1, overhead_ms=0),
            final_validator=lambda req, value: True,
        )

        async def planner(req):
            self.planner_calls += 1
            return replace(fetch_plan(req), text_independent=True)

        await runtime.run(request(text="Calculate portfolio exposure"), planner=planner)
        equivalent = request(text="Portfolio exposure please")
        runtime.observe_reuse(equivalent, valid_hit=True, decision_ms=0)
        result = await runtime.run(equivalent, planner=planner)
        self.assertEqual(decider.calls, [])
        self.assertEqual(result.trace.decision_status, "not_worthwhile")
        self.assertEqual(result.trace.jev_calls, 0)
        self.assertEqual(result.trace.tool_calls, 0)
        self.assertEqual(result.trace.llm_calls, 0)
        self.assertEqual(result.trace.cached_nodes, 1)

    async def test_decider_cannot_select_artifact_outside_eligible_candidates(self):
        class WrongCandidateDecider(FakeDecider):
            async def decide(self, req, candidates, kind):
                self.calls.append((req, candidates, kind))
                return Decision("foreign-artifact", {}, "fake-jev", 1.0)

        decider = WrongCandidateDecider()
        runtime = await self.semantic_runtime(decider)
        equivalent = request(text="Portfolio exposure please")
        runtime.observe_reuse(equivalent, valid_hit=True, decision_ms=0)
        result = await runtime.run(equivalent, planner=self.planner)
        self.assertNotEqual(result.path, "full_hit")
        self.assertEqual(result.trace.decision_status, "failed")
        self.assertEqual(result.trace.jev_calls, 1)
        self.assertEqual(result.trace.tool_calls, 1)

    async def test_promoted_semantic_alias_is_invalidated_when_decider_identity_changes(self):
        class ConfigDecider(FakeDecider):
            def __init__(self, revision, **kwargs):
                super().__init__(**kwargs)
                self.revision = revision

            @property
            def cache_identity(self):
                return {"configuration": self.revision}

        first_decider = ConfigDecider("1")
        runtime = await self.semantic_runtime(first_decider)
        equivalent = request(text="Portfolio exposure please")
        runtime.observe_reuse(equivalent, valid_hit=True, decision_ms=0)
        first = await runtime.run(equivalent, planner=self.planner)
        self.assertEqual(first.path, "full_hit")
        self.assertEqual(first.trace.jev_calls, 1)
        exact_alias = await runtime.run(equivalent, planner=self.planner)
        self.assertEqual(exact_alias.path, "full_hit")
        self.assertEqual(exact_alias.trace.jev_calls, 0)

        runtime.decider = ConfigDecider("2", accept=False)
        runtime.observe_reuse(equivalent, valid_hit=True, decision_ms=0)
        changed = await runtime.run(equivalent, planner=self.planner)
        self.assertNotEqual(changed.path, "full_hit")
        self.assertEqual(changed.trace.jev_calls, 1)
        self.assertEqual(changed.trace.tool_calls, 1)

    async def test_semantic_rejection_or_error_falls_back_and_counts_decision(self):
        for decider in (FakeDecider(accept=False), FakeDecider(error=RuntimeError("offline"))):
            with self.subTest(decider=decider):
                self.store.invalidate(scope="tenant-a")
                runtime = await self.semantic_runtime(decider)
                changed = request(text="Portfolio exposure please")
                runtime.observe_reuse(changed, valid_hit=True, decision_ms=0)
                result = await runtime.run(changed, planner=self.planner)
                self.assertNotEqual(result.path, "full_hit")
                self.assertGreaterEqual(result.trace.jev_calls, 1)
                self.assertEqual(result.value, {"id": "A", "exposure": 42})
                self.assertEqual(result.trace.llm_calls, 0)
                self.assertEqual(result.trace.tool_calls, 1)

    async def test_semantic_candidates_cannot_override_dependency_or_parameter_changes(self):
        decider = FakeDecider()
        runtime = await self.semantic_runtime(decider)
        for changed in (
            request("B", text="Portfolio exposure please"),
            request(revision="2", text="Portfolio exposure please"),
        ):
            with self.subTest(request=changed):
                runtime.observe_reuse(changed, valid_hit=True, decision_ms=0)
                result = await runtime.run(changed, planner=self.planner)
                # Plan decisions may still be eligible, but answer candidates must not be.
                answer_calls = [call for call in decider.calls if call[2] == "answer"]
                self.assertEqual(answer_calls, [])
                self.assertNotEqual(result.path, "full_hit")
                self.assertEqual(result.value["id"], changed.params["id"])
                self.assertEqual(result.trace.tool_calls, 1)
                decider.calls.clear()


class PlanContractTests(unittest.TestCase):
    def test_text_independence_requires_an_explicit_boolean(self):
        for value in ("false", "true", 0, 1, None):
            with self.subTest(value=value), self.assertRaises(TypeError):
                replace(fetch_plan(request()), text_independent=value).validate(
                    request(), {"fetch"}
                )

    def test_nested_references_round_trip_and_validate(self):
        original = Plan(
            "exposure",
            "1",
            {"id": "string"},
            (
                Node("fetch", "fetch", {"nested": [{"id": Ref.input("id")}]}),
                Node("summarize", "summarize", {"positions": Ref("fetch", ("items", 0))}),
            ),
            {"result": [Ref("summarize")]},
        )
        restored = Plan.from_dict(original.to_dict())
        self.assertEqual(restored, original)
        restored.validate(request(), {"fetch", "summarize"})

    def test_boolean_does_not_satisfy_integer_or_number_schema(self):
        for kind in ("integer", "number"):
            plan = Plan(
                "exposure",
                "1",
                {"id": kind},
                (Node("fetch", "fetch", {"id": Ref.input("id")}),),
                Ref("fetch"),
            )
            with self.subTest(kind=kind), self.assertRaises(TypeError):
                plan.validate(replace(request(), params={"id": True}), {"fetch"})

    def test_schema_rejects_missing_or_additional_structured_inputs(self):
        plan = fetch_plan(request())
        for params in ({}, {"id": "A", "another": 1}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                plan.validate(replace(request(), params=params), {"fetch"})

    def test_forward_references_are_valid_but_cycles_and_missing_nodes_are_rejected(self):
        plan = Plan(
            "exposure",
            "1",
            {"id": "string"},
            (
                Node("summarize", "summarize", {"value": Ref("fetch")}),
                Node("fetch", "fetch", {"id": Ref.input("id")}),
            ),
            Ref("summarize"),
        )
        plan.validate(request(), {"fetch", "summarize"})
        cycle = replace(
            plan,
            nodes=(
                Node("summarize", "summarize", {"value": Ref("fetch")}),
                Node("fetch", "fetch", {"id": Ref.input("id"), "value": Ref("summarize")}),
            ),
        )
        with self.assertRaisesRegex(ValueError, "cycle"):
            cycle.validate(request(), {"fetch", "summarize"})
        with self.assertRaises(ValueError):
            replace(plan, output=Ref("missing")).validate(request(), {"fetch", "summarize"})


if __name__ == "__main__":
    unittest.main()
