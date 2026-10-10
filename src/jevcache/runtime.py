"""Execution reuse at agent boundaries, before generative or external work starts."""

from __future__ import annotations

import asyncio
import inspect
import math
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .decisions import Candidate, Decision, DecisionPolicy, OutcomeStats, ReuseDecider
from .models import Context, Node, Plan, Ref, Request, fingerprint, references, resolve, snapshot
from .store import Entry, SQLiteStore
from .telemetry import Event, RunResult, Trace


@dataclass(frozen=True)
class Operation:
    name: str
    handler: Callable[..., Any]
    kind: str
    version: str
    read_only: bool
    ttl_s: float | None
    dependencies: tuple[str, ...] | None
    validator: Callable[[Any], Any] | None


@dataclass(frozen=True)
class _NodeValue:
    value: Any
    expires_at: float | None
    reusable: bool
    cost_ms: float


def _expiry(ttl_s: float | None, now: float, parents: list[_NodeValue]) -> float | None:
    limits = [parent.expires_at for parent in parents if parent.expires_at is not None]
    if ttl_s is not None:
        limits.append(now + ttl_s)
    return min(limits) if limits else None


def _check_ttl(ttl: float | None) -> None:
    if ttl is not None and (
        isinstance(ttl, bool)
        or not isinstance(ttl, (int, float))
        or not math.isfinite(ttl)
        or ttl < 0
    ):
        raise ValueError("TTL must be None or a finite nonnegative number")


def _type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    raise TypeError("Plan parameters must have a supported explicit JSON type")


class Runtime:
    def __init__(
        self,
        store: SQLiteStore | None = None,
        *,
        enable_cache: bool = True,
        enable_semantic: bool = False,
        decider: ReuseDecider | None = None,
        decision_policy: DecisionPolicy | None = None,
        final_validator: Callable[[Request, Any], Any] | None = None,
        answer_ttl_s: float | None = 60,
        decision_timeout_s: float = 1.0,
    ):
        if not isinstance(enable_cache, bool) or not isinstance(enable_semantic, bool):
            raise TypeError("Cache flags must be explicit booleans")
        _check_ttl(answer_ttl_s)
        if (
            isinstance(decision_timeout_s, bool)
            or not isinstance(decision_timeout_s, (int, float))
            or not math.isfinite(decision_timeout_s)
            or decision_timeout_s <= 0
        ):
            raise ValueError("Decision timeout must be finite and positive")
        self.store = store if store is not None else SQLiteStore()
        self.enable_cache = enable_cache
        self.enable_semantic = enable_semantic
        self.decider = decider
        self.decision_policy = decision_policy or DecisionPolicy()
        self.final_validator = final_validator
        self.answer_ttl_s = answer_ttl_s
        self.decision_timeout_s = decision_timeout_s
        self.operations: dict[str, Operation] = {}
        self._inflight: dict[str, asyncio.Task[_NodeValue]] = {}

    def register(
        self,
        name: str,
        handler: Callable[..., Any],
        *,
        kind: str = "tool",
        version: str = "1",
        read_only: bool = True,
        ttl_s: float | None = 60,
        dependencies: tuple[str, ...] | None = None,
        validator: Callable[[Any], Any] | None = None,
    ) -> None:
        if not isinstance(read_only, bool):
            raise TypeError("read_only must be an explicit boolean")
        if not isinstance(name, str) or not isinstance(version, str):
            raise TypeError("Operation name and version must be strings")
        _check_ttl(ttl_s)
        if not name or not version or not callable(handler):
            raise ValueError("Operations need a name, version, and callable handler")
        if kind not in {"tool", "llm", "compute"}:
            raise ValueError("Operation kind must be tool, llm, or compute")
        if dependencies is not None and any(not isinstance(k, str) for k in dependencies):
            raise TypeError("Dependency names must be strings")
        self.operations[name] = Operation(
            name,
            handler,
            kind,
            version,
            read_only,
            ttl_s,
            tuple(dependencies) if dependencies is not None else None,
            validator,
        )

    def _manifest(self) -> dict[str, Any]:
        return {
            name: {
                "version": op.version,
                "kind": op.kind,
                "read_only": op.read_only,
                "ttl_s": op.ttl_s,
                "dependencies": op.dependencies,
            }
            for name, op in sorted(self.operations.items())
        }

    def _plan_key(self, request: Request) -> str:
        return fingerprint(
            {
                "kind": "plan",
                "task": request.task,
                "schema": {key: _type(value) for key, value in request.params.items()},
                "scope": request.context.scope,
                "policy": request.context.policy_version,
                "contract": request.output_contract,
                "registry": self._manifest(),
            }
        )

    def _answer_key(self, request: Request) -> str:
        return fingerprint(
            {"kind": "answer", "request": request.to_dict(), "registry": self._manifest()}
        )

    def _dependencies(self, operation: Operation, context: Context) -> dict[str, str]:
        if operation.dependencies is None:
            return dict(context.dependencies)
        missing = set(operation.dependencies) - context.dependencies.keys()
        if missing:
            raise ValueError(f"Missing declared dependencies: {sorted(missing)}")
        return {key: context.dependencies[key] for key in operation.dependencies}

    def _node_key(
        self, op: Operation, arguments: Any, context: Context, dependencies: Mapping[str, str]
    ) -> str:
        return fingerprint(
            {
                "kind": "node",
                "operation": op.name,
                "version": op.version,
                "operation_kind": op.kind,
                "arguments": arguments,
                "scope": context.scope,
                "policy": context.policy_version,
                "dependencies": dict(dependencies),
                "ttl_s": op.ttl_s,
            }
        )

    def _entry(
        self,
        key: str,
        kind: str,
        request: Request,
        value: Any,
        *,
        expires_at: float | None,
        avoided_ms: float,
        metadata: Mapping[str, Any] | None = None,
        dependencies: Mapping[str, str] | None = None,
    ) -> Entry:
        return Entry(
            key=key,
            kind=kind,
            scope=request.context.scope,
            task=request.task,
            value=snapshot(value),
            request=request.to_dict(),
            dependencies=dict(
                request.context.dependencies if dependencies is None else dependencies
            ),
            policy_version=request.context.policy_version,
            output_contract=request.output_contract,
            created_at=self.store.clock(),
            expires_at=expires_at,
            avoided_ms=max(0.0, avoided_ms),
            metadata=snapshot(dict(metadata or {})),
        )

    @staticmethod
    async def _validate(validator: Callable[..., Any] | None, *args: Any) -> bool:
        if validator is None:
            return True
        answer = validator(*args)
        if inspect.isawaitable(answer):
            answer = await answer
        if not isinstance(answer, bool):
            raise TypeError("Validators must return a boolean")
        return answer

    def _plan_eligible(self, entry: Entry | None, request: Request) -> Plan | None:
        if entry is None or entry.kind != "plan":
            return None
        try:
            plan = Plan.from_dict(entry.value)
            plan.validate(request, set(self.operations))
        except (ValueError, TypeError, KeyError):
            return None
        # Binding the current prose does not prove that previously selected steps
        # remain appropriate. The application must explicitly certify stable structure.
        if request.text != entry.request.get("text", "") and not plan.text_independent:
            return None
        return plan

    def install_plan(self, plan: Plan, context: Context, *, text: str = "") -> None:
        """Install an application-authored template without making a planning call."""
        # Validate the template with typed stand-ins; actual values are checked on each run.
        placeholders = {
            "string": "",
            "integer": 0,
            "number": 0.0,
            "boolean": False,
            "object": {},
            "array": [],
        }
        request = Request(
            plan.task,
            {key: placeholders[kind] for key, kind in plan.parameters.items()},
            context,
            text,
            plan.output_contract,
        )
        plan.validate(request, set(self.operations))
        self.store.put(
            self._entry(
                self._plan_key(request),
                "plan",
                request,
                plan.to_dict(),
                expires_at=None,
                avoided_ms=0,
                dependencies={},
            )
        )

    def _answer_eligible(
        self,
        entry: Entry,
        request: Request,
        plan_entry: Entry | None = None,
        plan: Plan | None = None,
        *,
        require_current_plan: bool = False,
    ) -> bool:
        """Validate admission provenance independently of the mutable plan cache.

        Exact answers retain the immutable template and source request under which
        they were admitted. A different intent may replace the shared plan slot
        without invalidating an otherwise identical, still-fresh final response.
        Semantic candidates additionally match the currently applicable template.
        """
        try:
            if not (
                entry.kind == "answer"
                and entry.scope == request.context.scope
                and entry.task == request.task
                and entry.policy_version == request.context.policy_version
                and entry.output_contract == request.output_contract
                and dict(entry.dependencies) == dict(request.context.dependencies)
                and fingerprint(entry.request.get("params")) == fingerprint(dict(request.params))
                and entry.metadata.get("registry") == fingerprint(self._manifest())
            ):
                return False
            source = entry.metadata["plan_request"]
            source_request = Request(
                source["task"],
                source["params"],
                Context(**source["context"]),
                source["text"],
                source["output_contract"],
            )
            source_request.to_dict()
            if not (
                source_request.task == request.task
                and source_request.output_contract == request.output_contract
                and source_request.context.to_dict() == request.context.to_dict()
                and fingerprint(dict(source_request.params)) == fingerprint(dict(request.params))
            ):
                return False
            source_plan = Plan.from_dict(entry.metadata["plan"])
            source_plan.validate(source_request, set(self.operations))
            source_plan.validate(request, set(self.operations))
            digest = fingerprint(source_plan.to_dict())
            if entry.metadata.get("plan_digest") != digest:
                return False
            if not all(self.operations[node.operation].read_only for node in source_plan.nodes):
                return False
            origin = entry.metadata.get("semantic_origin_identity")
            if origin is not None and origin != self._decision_identity():
                return False
            if require_current_plan and not (
                plan is not None
                and plan_entry is not None
                and fingerprint(plan_entry.value) == digest
            ):
                return False
        except (ValueError, TypeError, KeyError, AttributeError):
            return False
        return True

    def _decision_identity(self) -> dict[str, Any]:
        identity = getattr(self.decider, "cache_identity", None)
        if identity is not None:
            return snapshot(dict(identity))
        return {
            "class": type(self.decider).__qualname__,
            "model": getattr(self.decider, "model", "custom"),
            "schema_version": "reuse-v1",
        }

    def decision_cohort(self, request: Request, kind: str = "answer") -> str:
        """A versioned workload bucket for externally verified decision measurements."""
        return fingerprint(
            {
                "kind": kind,
                "task": request.task,
                "scope": request.context.scope,
                "policy": request.context.policy_version,
                "contract": request.output_contract,
                "schema": {key: _type(value) for key, value in request.params.items()},
                "registry": self._manifest(),
                "decider": self._decision_identity(),
            }
        )

    def observe_reuse(self, request: Request, *, valid_hit: bool, decision_ms: float) -> None:
        self.store.observe_reuse(
            self.decision_cohort(request), valid_hit=valid_hit, decision_ms=decision_ms
        )

    def _remaining_ms(self, plan: Plan, request: Request, entry: Entry) -> float:
        """Estimate the critical path still needed AFTER exact node reuse.

        Unknown outputs prevent inspecting descendant keys; their observed costs
        remain in the estimate. This never executes a handler or guesses arguments.
        """
        known: dict[str, Any] = {"$request": {"text": request.text}}
        completed: dict[str, float] = {}
        pending = list(plan.nodes)
        profile = entry.metadata.get("node_ms", {})
        while pending:
            ready = [
                node
                for node in pending
                if all(
                    ref.source.startswith("$") or ref.source in completed
                    for ref in references(dict(node.arguments))
                )
            ]
            for node in ready:
                refs = references(dict(node.arguments))
                parents = [ref.source for ref in refs if not ref.source.startswith("$")]
                cost = float(profile.get(node.id, 0.0))
                if all(parent in known for parent in parents):
                    args = resolve(dict(node.arguments), request.params, known)
                    op = self.operations[node.operation]
                    deps = self._dependencies(op, request.context)
                    cached = self.store.get(self._node_key(op, args, request.context, deps))
                    if cached is not None and op.read_only:
                        cost = 0.0
                        known[node.id] = cached.value
                completed[node.id] = max((completed[p] for p in parents), default=0.0) + cost
                pending.remove(node)
        return max(completed.values(), default=0.0)

    async def _semantic_answer(
        self, request: Request, trace: Trace, plan_entry: Entry | None, plan: Plan | None
    ) -> Entry | None:
        if not (
            self.enable_cache
            and self.enable_semantic
            and self.decider
            and self.final_validator
            and plan
        ):
            return None
        attempts, accepted, measured_ms = self.store.outcome(self.decision_cohort(request))
        stats = OutcomeStats(attempts, accepted)
        if stats.attempts < self.decision_policy.min_samples:
            trace.decision_status = "insufficient_samples"
            return None
        started = time.perf_counter()
        entries = self.store.candidates(
            kind="answer",
            scope=request.context.scope,
            task=request.task,
            text=request.text,
            limit=5,
        )
        eligible = [
            entry
            for entry in entries
            if self._answer_eligible(entry, request, plan_entry, plan, require_current_plan=True)
        ]
        if not eligible:
            trace.decision_status = "no_candidates"
            trace.events.append(
                Event("candidate_lookup", "cache", "miss", (time.perf_counter() - started) * 1000)
            )
            return None
        candidates = [
            Candidate(
                entry.key, entry.request, entry.value, self._remaining_ms(plan, request, entry)
            )
            for entry in eligible
        ]
        lookup_ms = (time.perf_counter() - started) * 1000
        identity = self._decision_identity()
        decision_key = fingerprint(
            {
                "kind": "decision",
                "request": request.to_dict(),
                "identity": identity,
                "candidates": [
                    {"id": c.id, "artifact": fingerprint(c.artifact)} for c in candidates
                ],
            }
        )
        memo = self.store.get(decision_key)
        if memo is not None:
            decision = Decision(**memo.value)
            trace.decision_status = "memoized"
        else:
            policy = DecisionPolicy(
                min_samples=self.decision_policy.min_samples,
                overhead_ms=max(self.decision_policy.overhead_ms, measured_ms),
            )
            if not policy.should_decide(min(c.avoided_ms for c in candidates), stats, lookup_ms):
                trace.decision_status = "not_worthwhile"
                trace.events.append(Event("candidate_lookup", "cache", "skipped", lookup_ms))
                return None
            decision_started = time.perf_counter()
            trace.jev_calls += 1
            try:
                decision = await asyncio.wait_for(
                    self.decider.decide(request.to_dict(), candidates, "answer"),
                    timeout=self.decision_timeout_s,
                )
                if not isinstance(decision, Decision) or (
                    decision.candidate_id is not None
                    and decision.candidate_id not in {c.id for c in candidates}
                ):
                    raise ValueError("Decision must select an eligible candidate")
                if not isinstance(decision.model, str) or not decision.model:
                    raise ValueError("Decision must report a model identity")
                decision_data = {
                    "candidate_id": decision.candidate_id,
                    "probabilities": dict(decision.probabilities),
                    "model": decision.model,
                    "elapsed_ms": decision.elapsed_ms,
                }
                snapshot(decision_data)
            except Exception:
                trace.decision_status = "failed"
                trace.events.append(
                    Event(
                        "reuse_decision",
                        "jev",
                        "failed",
                        (time.perf_counter() - decision_started) * 1000,
                    )
                )
                return None
            decision_ms = (time.perf_counter() - decision_started) * 1000
            trace.decision_status = "accepted" if decision.candidate_id else "rejected"
            trace.events.append(Event("reuse_decision", "jev", trace.decision_status, decision_ms))
            expiry_limits = [entry.expires_at for entry in eligible if entry.expires_at is not None]
            self.store.put(
                self._entry(
                    decision_key,
                    "decision",
                    request,
                    decision_data,
                    expires_at=min(expiry_limits) if expiry_limits else None,
                    avoided_ms=decision_ms,
                )
            )
        chosen = next((entry for entry in eligible if entry.key == decision.candidate_id), None)
        if chosen is not None and await self._validate(
            self.final_validator, request, snapshot(chosen.value)
        ):
            return chosen
        if chosen is not None:
            trace.decision_status = "validation_rejected"
        return None

    async def _node(
        self, node: Node, arguments: Any, request: Request, parents: list[_NodeValue], trace: Trace
    ) -> _NodeValue:
        op = self.operations[node.operation]
        deps = self._dependencies(op, request.context)
        key = self._node_key(op, arguments, request.context, deps)
        reusable = op.read_only and all(parent.reusable for parent in parents)
        started = time.perf_counter()
        if self.enable_cache and reusable:
            entry = self.store.get(key)
            if entry is not None and await self._validate(op.validator, snapshot(entry.value)):
                trace.cached_nodes += 1
                trace.events.append(
                    Event(
                        node.id,
                        op.kind,
                        "hit",
                        (time.perf_counter() - started) * 1000,
                        entry.avoided_ms,
                    )
                )
                return _NodeValue(
                    snapshot(entry.value),
                    _expiry(
                        None,
                        self.store.clock(),
                        parents + [_NodeValue(None, entry.expires_at, True, 0)],
                    ),
                    True,
                    entry.avoided_ms,
                )
            existing = self._inflight.get(key)
            if existing is not None:
                trace.coalesced_nodes += 1
                value = await asyncio.shield(existing)
                trace.events.append(
                    Event(
                        node.id,
                        op.kind,
                        "coalesced",
                        (time.perf_counter() - started) * 1000,
                        value.cost_ms,
                    )
                )
                return _NodeValue(
                    snapshot(value.value),
                    _expiry(None, self.store.clock(), parents + [value]),
                    value.reusable,
                    value.cost_ms,
                )

        async def execute() -> _NodeValue:
            execute_started = time.perf_counter()
            trace.executed_nodes += 1
            if op.kind == "tool":
                trace.tool_calls += 1
            elif op.kind == "llm":
                trace.llm_calls += 1
            try:
                if inspect.iscoroutinefunction(op.handler):
                    value = await op.handler(**snapshot(arguments))
                else:
                    value = await asyncio.to_thread(op.handler, **snapshot(arguments))
                    if inspect.isawaitable(value):
                        value = await value
                value = snapshot(value)
                valid = await self._validate(op.validator, snapshot(value))
                elapsed_ms = (time.perf_counter() - execute_started) * 1000
                expiry = _expiry(op.ttl_s, self.store.clock(), parents)
                result = _NodeValue(value, expiry, reusable and valid, elapsed_ms)
                if (
                    self.enable_cache
                    and result.reusable
                    and (expiry is None or expiry > self.store.clock())
                ):
                    self.store.put(
                        self._entry(
                            key,
                            "node",
                            request,
                            value,
                            expires_at=expiry,
                            avoided_ms=elapsed_ms,
                            dependencies=deps,
                        )
                    )
                trace.events.append(Event(node.id, op.kind, "executed", elapsed_ms))
                return result
            except BaseException:
                trace.events.append(
                    Event(node.id, op.kind, "error", (time.perf_counter() - execute_started) * 1000)
                )
                raise

        if not (self.enable_cache and reusable):
            return await execute()
        task = asyncio.create_task(execute())
        self._inflight[key] = task

        def cleanup(done: asyncio.Task[_NodeValue]) -> None:
            if self._inflight.get(key) is done:
                self._inflight.pop(key, None)
            if not done.cancelled():
                done.exception()  # Retrieve errors even if every coalesced waiter was cancelled.

        task.add_done_callback(cleanup)
        return await asyncio.shield(task)

    async def run(
        self,
        request: Request,
        *,
        planner: Callable[[Request], Any] | None = None,
        planner_kind: str = "llm",
    ) -> RunResult:
        """Return a full hit, execute missing nodes, or plan and run normally."""
        if planner_kind not in {"llm", "compute"}:
            raise ValueError("Planner kind must be llm or compute")
        request_data = request.to_dict()
        request = Request(
            request.task,
            request_data["params"],
            Context(**request_data["context"]),
            request.text,
            request.output_contract,
        )
        trace = Trace(uuid.uuid4().hex, request.task)
        started = time.perf_counter()
        try:
            plan_key = self._plan_key(request)
            plan_entry = self.store.get(plan_key) if self.enable_cache else None
            plan = self._plan_eligible(plan_entry, request)
            answer_key = self._answer_key(request)
            if self.enable_cache and self.final_validator:
                entry = self.store.get(answer_key)
                if entry is not None and self._answer_eligible(entry, request, plan_entry, plan):
                    if await self._validate(self.final_validator, request, snapshot(entry.value)):
                        trace.path = "full_hit"
                        trace.events.append(
                            Event(
                                "final_answer",
                                "cache",
                                "hit",
                                (time.perf_counter() - started) * 1000,
                                entry.avoided_ms,
                            )
                        )
                        return RunResult(snapshot(entry.value), trace)
                semantic = await self._semantic_answer(request, trace, plan_entry, plan)
                if semantic is not None:
                    trace.path = "full_hit"
                    # Promote a validated equivalent answer into an exact request entry.
                    self.store.put(
                        self._entry(
                            answer_key,
                            "answer",
                            request,
                            semantic.value,
                            expires_at=semantic.expires_at,
                            avoided_ms=semantic.avoided_ms,
                            metadata={
                                **dict(semantic.metadata),
                                "semantic_origin_identity": self._decision_identity(),
                            },
                        )
                    )
                    trace.events.append(
                        Event(
                            "final_answer",
                            "cache",
                            "semantic_hit",
                            (time.perf_counter() - started) * 1000,
                            semantic.avoided_ms,
                        )
                    )
                    return RunResult(snapshot(semantic.value), trace)
            if plan is not None:
                trace.plan_hit = True
                trace.events.append(
                    Event("planning", "llm", "hit", 0, plan_entry.avoided_ms if plan_entry else 0)
                )
            else:
                if planner is None:
                    raise ValueError("No eligible plan: supply a planner or install a template")
                planning_started = time.perf_counter()
                if planner_kind == "llm":
                    trace.llm_calls += 1
                if inspect.iscoroutinefunction(planner):
                    plan = await planner(request)
                else:
                    plan = await asyncio.to_thread(planner, request)
                    if inspect.isawaitable(plan):
                        plan = await plan
                if not isinstance(plan, Plan):
                    raise TypeError("Planner must return a Plan")
                # Snapshot the serializable template before any handler can execute.
                plan = Plan.from_dict(snapshot(plan.to_dict()))
                plan.validate(request, set(self.operations))
                planning_ms = (time.perf_counter() - planning_started) * 1000
                trace.events.append(Event("planning", planner_kind, "executed", planning_ms))
                if self.enable_cache:
                    plan_entry = self._entry(
                        plan_key,
                        "plan",
                        request,
                        plan.to_dict(),
                        expires_at=None,
                        avoided_ms=planning_ms,
                        dependencies={},
                    )
                    self.store.put(plan_entry)
            # Check the whole template's declared dependencies before any external work.
            for node in plan.nodes:
                self._dependencies(self.operations[node.operation], request.context)
            node_values: dict[str, _NodeValue] = {}
            tasks: dict[str, asyncio.Task[_NodeValue]] = {}
            execution_started = time.perf_counter()

            async def execute_node(node: Node) -> _NodeValue:
                parent_ids = {
                    ref.source
                    for ref in references(dict(node.arguments))
                    if not ref.source.startswith("$")
                }
                parents = [await tasks[parent] for parent in sorted(parent_ids)]
                known = {name: value.value for name, value in node_values.items()}
                known["$request"] = {"text": request.text}
                arguments = resolve(dict(node.arguments), request.params, known)
                value = await self._node(node, arguments, request, parents, trace)
                node_values[node.id] = value
                return value

            async with asyncio.TaskGroup() as group:
                for node in plan.nodes:
                    tasks[node.id] = group.create_task(execute_node(node))
            execution_ms = (time.perf_counter() - execution_started) * 1000
            values = {name: value.value for name, value in node_values.items()}
            values["$request"] = {"text": request.text}
            result = resolve(plan.output, request.params, values)
            trace.path = (
                "partial_hit"
                if (trace.plan_hit or trace.cached_nodes or trace.coalesced_nodes)
                else "miss"
            )
            expiry = _expiry(self.answer_ttl_s, self.store.clock(), list(node_values.values()))
            if (
                self.enable_cache
                and self.final_validator
                and all(value.reusable for value in node_values.values())
                and (expiry is None or expiry > self.store.clock())
                and await self._validate(self.final_validator, request, snapshot(result))
            ):
                self.store.put(
                    self._entry(
                        answer_key,
                        "answer",
                        request,
                        result,
                        expires_at=expiry,
                        avoided_ms=execution_ms,
                        metadata={
                            "registry": fingerprint(self._manifest()),
                            "plan_digest": fingerprint(plan.to_dict()),
                            "plan": plan.to_dict(),
                            "plan_request": request.to_dict(),
                            "node_ms": {name: value.cost_ms for name, value in node_values.items()},
                        },
                    )
                )
            return RunResult(result, trace)
        except BaseException:
            trace.path = "error"
            raise
        finally:
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            self.store.record_trace(trace.to_dict())

    async def call(self, operation: str, *, context: Context, **arguments: Any) -> RunResult:
        """Wrap a single existing agent tool/LLM call without adopting a planner."""
        args = snapshot(arguments)
        request = Request(f"operation:{operation}", args, context)
        plan = Plan(
            request.task,
            "1",
            {key: _type(value) for key, value in args.items()},
            (Node("result", operation, {key: Ref.input(key) for key in args}),),
            Ref("result"),
            text_independent=True,
        )
        return await self.run(request, planner=lambda _: plan, planner_kind="compute")
