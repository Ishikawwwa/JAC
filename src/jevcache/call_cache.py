"""A transparent cache around a single sync or async LLM callable."""

from __future__ import annotations

import asyncio
import contextvars
import functools
import hashlib
import inspect
import marshal
import math
import queue
import re
import threading
import time
import uuid
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Mapping

from .decisions import Candidate, Decision, DecisionPolicy, OutcomeStats, ReuseDecider
from .models import Context, fingerprint, snapshot
from .store import Entry, SQLiteStore
from .telemetry import Event, Trace

_CALLABLE_NAMES = weakref.WeakKeyDictionary()
_CALLABLE_NAMES_LOCK = threading.RLock()
_OWNER_CANCELLED = object()


def _native_json(value: Any) -> bool:
    if isinstance(value, tuple):
        return False
    if isinstance(value, dict):
        return all(isinstance(key, str) and _native_json(child) for key, child in value.items())
    if isinstance(value, list):
        return all(_native_json(child) for child in value)
    return value is None or isinstance(value, (str, bool, int, float))


def _semantic_request(call: Mapping[str, Any]) -> tuple[str, Any] | None:
    """Keep every setting/history item fixed; vary only the final user text."""
    guard = snapshot(dict(call))
    containers = [guard["kwargs"]]
    if len(guard["args"]) == 1 and isinstance(guard["args"][0], dict):
        containers.append(guard["args"][0])
    text = None
    for container in containers:
        messages = container.get("messages")
        if isinstance(messages, list) and messages and isinstance(messages[-1], dict):
            last = messages[-1]
            if last.get("role") == "user" and isinstance(last.get("content"), str):
                text = last["content"]
                last["content"] = {"$jac": "final_user_text"}
                break
        for name in ("input", "prompt"):
            if isinstance(container.get(name), str):
                text = container[name]
                container[name] = {"$jac": "final_user_text"}
                break
        if text is not None:
            break
    if text is None and guard["args"] and isinstance(guard["args"][0], str):
        text = guard["args"][0]
        guard["args"][0] = {"$jac": "final_user_text"}
    if not text:
        return None
    # Numeric literals are deterministic constraints, not a semantic model's arithmetic task.
    anchors = re.findall(r"(?<![\w.])[+-]?\d+(?:[.,:/-]\d+)*%?", text)
    return text, {"call": guard, "numeric_anchors": anchors}


def _contains_tool_action(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("tool_calls") or value.get("function_call"):
            return True
        if isinstance(value.get("type"), str) and value["type"] in {
            "function_call",
            "tool_call",
            "tool_use",
        }:
            return True
        return any(_contains_tool_action(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_tool_action(child) for child in value)
    return False


class JAC:
    """Jev As Cache: ``cached = JAC(...).wrap(existing_llm_call)``.

    Hits return the response directly. Misses call the original function with its
    original arguments. Exact caching needs no model key or calibration. Optional
    semantic reuse needs a decider and measured outcomes for the savings gate.
    """

    def __init__(
        self,
        path: str = "jac.sqlite",
        *,
        namespace: str = "default",
        ttl_s: float | None = 300,
        jev: ReuseDecider | None = None,
        decider: ReuseDecider | None = None,
        decision_policy: DecisionPolicy | None = None,
        context: Context | Callable[[tuple[Any, ...], dict[str, Any]], Context] | None = None,
        validator: Callable[[Mapping[str, Any], Any], Any] | None = None,
        decision_timeout_s: float = 1.0,
        enabled: bool = True,
        store: SQLiteStore | None = None,
        mode: str = "cache_first",
        retrieval: str = "bm25",
        embedder: Any = None,
        candidate_limit: int = 5,
    ):
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("Cache namespace must be a nonempty string")
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a boolean")
        if jev is not None and decider is not None:
            raise ValueError("Use either decider or jev, not both")
        if mode not in ("cache_first", "parallel"):
            raise ValueError("mode must be 'cache_first' or 'parallel'")
        if retrieval not in ("bm25", "embeddings"):
            raise ValueError("retrieval must be 'bm25' or 'embeddings'")
        if (type(candidate_limit) is not int or not 1 <= candidate_limit <= 8):
            raise ValueError("candidate_limit must be between one and eight")
        if retrieval == "embeddings" and (
            not (callable(getattr(embedder, "embed", None))
                 or callable(getattr(embedder, "aembed", None)))
            or not isinstance(getattr(embedder, "cache_identity", None), Mapping)
        ):
            raise ValueError("embeddings retrieval requires an embedder with cache_identity")
        if ttl_s is not None and (
            isinstance(ttl_s, bool)
            or not isinstance(ttl_s, (int, float))
            or not math.isfinite(ttl_s)
            or ttl_s < 0
        ):
            raise ValueError("TTL must be None or finite and nonnegative")
        if (
            isinstance(decision_timeout_s, bool)
            or not isinstance(decision_timeout_s, (int, float))
            or not math.isfinite(decision_timeout_s)
            or decision_timeout_s <= 0
        ):
            raise ValueError("Decision timeout must be finite and positive")
        if context is not None and not isinstance(context, Context) and not callable(context):
            raise TypeError("context must be a Context or a callable")
        self.namespace = namespace
        self.ttl_s = ttl_s
        self.jev = decider if decider is not None else jev
        self.decision_policy = decision_policy or DecisionPolicy()
        self.context = context or Context(namespace)
        self.validator = validator
        self.decision_timeout_s = decision_timeout_s
        self.enabled = enabled
        self.mode = mode
        self.retrieval = retrieval
        self.embedder = embedder
        self.candidate_limit = candidate_limit
        self.store = store if store is not None else SQLiteStore(path)
        self._owns_store = store is None
        self._lock = threading.RLock()
        self._store_lock = threading.RLock()
        self._inflight: dict[str, Future[Any]] = {}
        self._codecs: dict[str, type] = {}
        self._last_trace: contextvars.ContextVar[Trace | None] = contextvars.ContextVar(
            f"jac_trace_{id(self)}", default=None
        )
        self._trace_tasks: set[str] = set()
        self._trace_history: dict[str, Trace] = {}
        self._opaque_callables: dict[int, tuple[Any, str]] = {}
        self._parallel_inflight: dict[str, Future[Any]] = {}
        self._pending: set[Future[Any]] = set()
        self._executors: dict[str, ThreadPoolExecutor] = {}
        self._race_loop: asyncio.AbstractEventLoop | None = None
        self._race_thread: threading.Thread | None = None
        self._closed = False
        self._active_requests = 0

    def _callable_name(self, function: Callable[..., Any]) -> Any:
        name = (
            getattr(function, "__module__", "callable")
            + "."
            + getattr(function, "__qualname__", type(function).__qualname__)
        )
        if inspect.isfunction(function) and function.__closure__ is None:
            defaults = {
                "args": list(function.__defaults__ or ()),
                "kwargs": function.__kwdefaults__ or {},
            }
            try:
                if not _native_json(defaults):
                    raise ValueError("Opaque function defaults")
                return {
                    "name": name,
                    "defaults": snapshot(defaults),
                    "code": hashlib.sha256(marshal.dumps(function.__code__)).hexdigest(),
                }
            except (TypeError, ValueError, RecursionError):
                pass
        # A qualname cannot identify a closure's captured model or a bound client.
        # Reuse an identity for the same object, without interpreting hidden state.
        if (
            inspect.isfunction(function)
            or inspect.ismethod(function)
            or isinstance(function, functools.partial)
        ):
            with _CALLABLE_NAMES_LOCK:
                token = _CALLABLE_NAMES.get(function)
                if token is None:
                    token = uuid.uuid4().hex
                    _CALLABLE_NAMES[function] = token
        else:
            with self._lock:
                _, token = self._opaque_callables.setdefault(
                    id(function), (function, uuid.uuid4().hex)
                )
        return {"name": name, "instance": token}

    @property
    def last_trace(self) -> Trace | None:
        return self._last_trace.get()

    def _identity(self) -> Any:
        return snapshot(
            dict(
                getattr(
                    self.jev,
                    "cache_identity",
                    {
                        "class": type(self.jev).__qualname__,
                        "model": getattr(self.jev, "model", "custom"),
                        "schema_version": "reuse-v1",
                    },
                )
            )
        )

    def _cache_io(
        self,
        method: str,
        *args: Any,
        default: Any = None,
        event_trace: Trace | None = None,
        **kwargs: Any,
    ) -> Any:
        """Cache availability must not change a normal provider call's result."""
        with self._store_lock:
            try:
                return getattr(self.store, method)(*args, **kwargs)
            except Exception:
                if event_trace is not None:
                    event_trace.events.append(Event("cache_storage", "cache", "failed", 0))
                return default

    def _submit(
        self, lane: str, function: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Future[Any]:
        copied_context = contextvars.copy_context()
        with self._lock:
            if self._closed:
                raise RuntimeError("JAC is closed")
            executor = self._executors.get(lane)
            if executor is None:
                executor = ThreadPoolExecutor(
                    max_workers=16 if lane == "llm" else 4, thread_name_prefix=f"jac-{lane}"
                )
                self._executors[lane] = executor
            future = executor.submit(copied_context.run, function, *args, **kwargs)
            self._pending.add(future)

        def completed(done: Future[Any]) -> None:
            with self._lock:
                self._pending.discard(done)

        future.add_done_callback(completed)
        return future

    def _track_task(self, task: asyncio.Task[Any]) -> None:
        marker = Future()
        with self._lock:
            self._pending.add(marker)

        def completed(done: asyncio.Task[Any]) -> None:
            try:
                done.exception()
            except BaseException:
                pass
            with self._lock:
                self._pending.discard(marker)
                marker.set_result(None)

        task.add_done_callback(completed)

    async def _async_cache_io(self, method: str, *args: Any, **kwargs: Any) -> Any:
        future = self._submit("cache", self._cache_io, method, *args, **kwargs)
        return await asyncio.shield(asyncio.wrap_future(future))

    async def _decision_io(self, method: str, *args: Any, **kwargs: Any) -> Any:
        if self.mode == "parallel":
            return await self._async_cache_io(method, *args, **kwargs)
        return self._cache_io(method, *args, **kwargs)

    def _get_race_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._closed:
                raise RuntimeError("JAC is closed")
            if self._race_loop is None:
                self._race_loop = asyncio.new_event_loop()
                self._race_thread = threading.Thread(
                    target=self._race_loop.run_forever, name="jac-parallel", daemon=True
                )
                self._race_thread.start()
            return self._race_loop

    def _begin_request(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("JAC is closed")
            self._active_requests += 1

    async def drain(self) -> None:
        """Wait for deferred writes and losing work, without delaying a race winner."""
        while True:
            with self._lock:
                pending = list(self._pending)
            if not pending:
                return
            await asyncio.gather(
                *(asyncio.shield(asyncio.wrap_future(f)) for f in pending), return_exceptions=True
            )

    def drain_sync(self) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("Use 'await cache.drain()' inside an async application")
        while True:
            with self._lock:
                pending = list(self._pending)
            if not pending:
                return
            for future in pending:
                try:
                    future.result()
                except BaseException:
                    pass

    def _context(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Context:
        context = self.context(args, kwargs) if callable(self.context) else self.context
        if not isinstance(context, Context):
            raise TypeError("Context callback must return Context")
        return Context(**context.to_dict())

    def _parallel_eligible(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
        if (
            self.mode != "parallel"
            or not self.enabled
            or self.ttl_s == 0
            or kwargs.get("stream", False)
        ):
            return False
        try:
            raw_call = {"args": list(args), "kwargs": kwargs}
            return _native_json(raw_call) and snapshot(raw_call) is not None
        except (TypeError, ValueError, RecursionError):
            return False

    @staticmethod
    def _type_id(response_type: type) -> str:
        return f"model:{response_type.__module__}.{response_type.__qualname__}"

    def _encode(self, response: Any, response_type: type | None) -> dict[str, Any] | None:
        try:
            if response_type is not None and not isinstance(response, response_type):
                return None
            model_type = response_type
            if hasattr(response, "model_dump"):
                value = response.model_dump(mode="json")
                model_type = model_type or type(response)
            elif model_type is not None and hasattr(response, "dict"):
                value = response.dict()
            else:
                value = response
                if not _native_json(value):
                    return None
            payload = snapshot(value)
            if model_type is None:
                return {"codec": "json-v1", "payload": payload}
            if not (
                callable(getattr(model_type, "model_validate", None))
                or callable(getattr(model_type, "parse_obj", None))
            ):
                return None
            codec = self._type_id(model_type)
            with self._lock:
                self._codecs[codec] = model_type
            return {"codec": codec, "payload": payload}
        except Exception:
            return None

    def _decode(self, value: Mapping[str, Any], response_type: type | None) -> Any:
        if response_type is not None and value["codec"] != self._type_id(response_type):
            raise ValueError("Cached response type differs from the current callable contract")
        payload = snapshot(value["payload"])
        if value["codec"] == "json-v1":
            return payload
        with self._lock:
            model_type = self._codecs.get(value["codec"])
        if model_type is None:
            raise ValueError("Cached model type is unknown; supply response_type for durable reuse")
        decoder = getattr(model_type, "model_validate", None) or model_type.parse_obj
        return decoder(payload)

    async def _valid(self, request: Mapping[str, Any], response: Any) -> bool:
        if self.validator is None:
            return True
        value = self.validator(snapshot(dict(request)), response)
        if inspect.isawaitable(value):
            value = await value
        if not isinstance(value, bool):
            raise TypeError("Validator must return a boolean")
        return value

    def _cohort(
        self, identity: str, call: Mapping[str, Any], context: Context, response_type: type | None
    ) -> str:
        shape = _semantic_request(call)
        return fingerprint(
            {
                "kind": "llm_reuse",
                "function": identity,
                "scope": context.scope,
                "policy": context.policy_version,
                "guard": shape[1] if shape else call,
                "dependency_names": sorted(context.dependencies),
                "contract": self._type_id(response_type) if response_type else "auto",
                "decider": self._identity(),
                "retrieval": self._retrieval_identity(),
            }
        )

    def wrap(
        self,
        function: Callable[..., Any],
        *,
        response_type: type | None = None,
        namespace: str | None = None,
    ) -> Callable[..., Any]:
        if not callable(function):
            raise TypeError("wrap requires an existing callable")
        if response_type is not None:
            if not isinstance(response_type, type):
                raise TypeError("response_type must be a model class")
            self._codecs[self._type_id(response_type)] = response_type
        if namespace is not None and (not isinstance(namespace, str) or not namespace):
            raise ValueError("Callable namespace must be a nonempty string")
        name = namespace or self._callable_name(function)
        identity = fingerprint({"cache": self.namespace, "function": name})
        task = "llm:" + identity
        self._trace_tasks.add(task)
        is_async = inspect.iscoroutinefunction(function) or inspect.iscoroutinefunction(
            getattr(function, "__call__", None)
        )

        if is_async:

            @functools.wraps(function)
            async def wrapped(*args: Any, **kwargs: Any) -> Any:
                trace = Trace(uuid.uuid4().hex, task, mode=self.mode)
                started = time.perf_counter()
                self._begin_request()
                try:
                    response, _ = await self._call(
                        function, args, kwargs, identity, response_type, True, trace
                    )
                    return response
                finally:
                    trace.elapsed_ms = (time.perf_counter() - started) * 1000
                    self._last_trace.set(trace)
                    with self._lock:
                        self._trace_history[trace.request_id] = trace
                        self._active_requests -= 1
        else:

            @functools.wraps(function)
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                trace = Trace(uuid.uuid4().hex, task, mode=self.mode)
                started = time.perf_counter()
                self._begin_request()

                try:
                    resolved_context = self._context(args, kwargs)

                    def coroutine(target=function):
                        return self._call(
                            target,
                            args,
                            kwargs,
                            identity,
                            response_type,
                            False,
                            trace,
                            resolved_context=resolved_context,
                        )

                    use_parallel = self._parallel_eligible(args, kwargs)
                    if use_parallel:
                        response, _ = asyncio.run_coroutine_threadsafe(
                            coroutine(), self._get_race_loop()
                        ).result()
                        return response
                    try:
                        asyncio.get_running_loop()
                    except RuntimeError:
                        response, _ = asyncio.run(coroutine())
                    else:
                        # Only the async cache/decision work moves to a helper loop.
                        # The existing synchronous provider still runs on its caller's thread.
                        copied_context = contextvars.copy_context()
                        requests = queue.SimpleQueue()

                        def invoke_on_caller(*call_args, **call_kwargs):
                            result = Future()
                            requests.put((result, call_args, call_kwargs))
                            return result.result()

                        with ThreadPoolExecutor(max_workers=1) as pool:
                            work = pool.submit(
                                copied_context.run, lambda: asyncio.run(coroutine(invoke_on_caller))
                            )
                            work.add_done_callback(lambda _: requests.put(None))
                            while (pending := requests.get()) is not None:
                                result, call_args, call_kwargs = pending
                                try:
                                    result.set_result(function(*call_args, **call_kwargs))
                                except BaseException as exc:
                                    result.set_exception(exc)
                            response, _ = work.result()
                    return response
                except BaseException:
                    trace.path = "error"
                    raise
                finally:
                    trace.elapsed_ms = (time.perf_counter() - started) * 1000
                    self._last_trace.set(trace)
                    with self._lock:
                        self._trace_history[trace.request_id] = trace
                        self._active_requests -= 1

        def observe_reuse(
            request_kwargs: Mapping[str, Any],
            *,
            valid_hit: bool,
            decision_ms: float,
            request_args: tuple[Any, ...] = (),
        ) -> None:
            call = snapshot({"args": request_args, "kwargs": dict(request_kwargs)})
            context = self._context(request_args, dict(request_kwargs))
            cohort = self._cohort(identity, call, context, response_type)
            with self._store_lock:
                self.store.observe_reuse(cohort, valid_hit=valid_hit, decision_ms=decision_ms)

        wrapped.observe_reuse = observe_reuse
        wrapped.cache_info = lambda: self.last_trace
        wrapped.cache = self
        return wrapped

    __call__ = wrap

    def _retrieval_identity(self) -> dict:
        return {"strategy": self.retrieval, "limit": self.candidate_limit,
                "encoder": snapshot(self.embedder.cache_identity)
                if self.retrieval == "embeddings" else None}

    def wrap_client(self, client: Any) -> Any:
        from .adapters import wrap_client

        return wrap_client(self, client)

    async def _embedding_candidates(self, *, event_trace: Trace, **kwargs: Any) -> list[Entry]:
        # Never hold SQLite's lock during remote calls or local model inference.
        # Native async encoders stay on the application loop for client affinity.
        search = await self._async_cache_io(
            "prepare_embedding_candidates", **kwargs,
            encoder_identity=snapshot(self.embedder.cache_identity), event_trace=event_trace,
        )
        if search is None:
            return []
        aembed = getattr(self.embedder, "aembed", None)
        if callable(aembed) and getattr(self.embedder, "is_async", None) is not False:
            encoded = aembed(search.texts)
            if inspect.isawaitable(encoded):
                encoded = await encoded
        else:
            future = self._submit("embedding", self.embedder.embed, search.texts)
            encoded = await asyncio.shield(asyncio.wrap_future(future))
        return await self._async_cache_io(
            "finish_embedding_candidates", search, encoded, default=[], event_trace=event_trace,
        )

    async def _semantic(
        self,
        call: Mapping[str, Any],
        context: Context,
        identity: str,
        response_type: type | None,
        trace: Trace,
    ) -> Entry | None:
        shape = _semantic_request(call)
        if self.jev is None or shape is None:
            return None
        cohort = self._cohort(identity, call, context, response_type)
        attempts, accepted, measured_ms = await self._decision_io(
            "outcome", cohort, default=(0, 0, 0.0), event_trace=trace
        )
        if attempts < self.decision_policy.min_samples:
            trace.decision_status = "insufficient_samples"
            return None
        lookup_started = time.perf_counter()
        guard_id = fingerprint(shape[1])
        partition = fingerprint({"function": identity, "guard": guard_id})
        def is_eligible(entry):
            if (
                entry.policy_version != context.policy_version
                or entry.dependencies != dict(context.dependencies)
                or entry.output_contract
                != (self._type_id(response_type) if response_type else "auto")
                or entry.metadata.get("semantic_identity") is not None
            ):
                return False
            try:
                if _contains_tool_action(entry.value["payload"]):
                    return False
                self._decode(entry.value, response_type)
                return True
            except Exception:
                return False

        retrieval_kwargs = dict(
            kind="llm_answer",
            scope=context.scope,
            task=partition,
            text=shape[0],
            limit=self.candidate_limit,
            eligible=is_eligible,
        )
        if self.retrieval == "embeddings":
            candidates = await self._embedding_candidates(event_trace=trace, **retrieval_kwargs)
        else:
            candidates = await self._decision_io(
                "candidates", **retrieval_kwargs, default=[], event_trace=trace,
            )
        lookup_ms = (time.perf_counter() - lookup_started) * 1000
        trace.events.append(Event("retrieval_" + self.retrieval, "cache", "completed", lookup_ms))
        eligible = []
        for entry in candidates:
            if (
                entry.policy_version != context.policy_version
                or entry.dependencies != dict(context.dependencies)
                or entry.output_contract
                != (self._type_id(response_type) if response_type else "auto")
                or entry.metadata.get("semantic_identity") is not None
                or _contains_tool_action(entry.value["payload"])
            ):
                continue
            try:
                self._decode(entry.value, response_type)
            except Exception:
                continue
            eligible.append(entry)
        if not eligible:
            trace.decision_status = "no_candidates"
            return None
        lookup_ms = (time.perf_counter() - lookup_started) * 1000
        request = {**call, "text": shape[0]}
        options = [
            Candidate(entry.key, entry.request, entry.value["payload"], entry.avoided_ms)
            for entry in eligible
        ]
        memo_key = fingerprint(
            {
                "kind": "llm_decision",
                "request": request,
                "context": context.to_dict(),
                "function": identity,
                "candidates": [
                    {
                        "id": c.id,
                        "artifact": fingerprint(c.artifact),
                        "request": fingerprint(c.request),
                    }
                    for c in options
                ],
                "decider": self._identity(),
            }
        )
        memo = await self._decision_io("get", memo_key, event_trace=trace)
        if memo is not None:
            chosen_id = memo.value["candidate_id"]
            trace.decision_status = "memoized"
        else:
            policy = DecisionPolicy(
                self.decision_policy.min_samples, max(self.decision_policy.overhead_ms, measured_ms)
            )
            if not policy.should_decide(
                min(c.avoided_ms for c in options), OutcomeStats(attempts, accepted), lookup_ms
            ):
                trace.decision_status = "not_worthwhile"
                return None
            decision_started = time.perf_counter()
            trace.jev_calls += 1
            try:
                decision = await asyncio.wait_for(
                    self.jev.decide(request, options, "answer"), timeout=self.decision_timeout_s
                )
                if not isinstance(decision, Decision) or decision.candidate_id not in (
                    {None} | {c.id for c in options}
                ):
                    raise ValueError("Decider selected an ineligible candidate")
                if not isinstance(decision.model, str) or not decision.model:
                    raise ValueError("Decider must report a model")
                if any(
                    isinstance(p, bool)
                    or not isinstance(p, (int, float))
                    or not math.isfinite(p)
                    or not 0 <= p <= 1
                    for p in decision.probabilities.values()
                ):
                    raise ValueError("Malformed decision probabilities")
                snapshot(dict(decision.probabilities))
                chosen_id = decision.candidate_id
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
            elapsed = (time.perf_counter() - decision_started) * 1000
            trace.decision_status = "accepted" if chosen_id is not None else "rejected"
            trace.events.append(Event("reuse_decision", "jev", trace.decision_status, elapsed))
            expiry = min(
                (entry.expires_at for entry in eligible if entry.expires_at is not None),
                default=None,
            )
            await self._decision_io(
                "put",
                Entry(
                    memo_key,
                    "llm_decision",
                    context.scope,
                    partition,
                    {"candidate_id": chosen_id},
                    request,
                    dict(context.dependencies),
                    context.policy_version,
                    "decision-v1",
                    self.store.clock(),
                    expiry,
                    elapsed,
                ),
                event_trace=trace,
            )
        chosen = next((entry for entry in eligible if entry.key == chosen_id), None)
        if chosen is not None:
            if chosen.expires_at is not None and chosen.expires_at <= self.store.clock():
                trace.decision_status = "expired_during_decision"
                return None
            decoded = self._decode(chosen.value, response_type)
            if await self._valid(call, decoded) and (
                chosen.expires_at is None or chosen.expires_at > self.store.clock()
            ):
                return chosen
            trace.decision_status = "validation_rejected"
        return None

    async def _parallel_lookup(
        self,
        call: Mapping[str, Any],
        context: Context,
        identity: str,
        response_type: type | None,
        key: str,
        trace: Trace,
    ) -> tuple[Any, Entry] | None:
        try:
            exact = await self._async_cache_io("get", key, event_trace=trace)
            if exact is not None and exact.metadata.get("semantic_identity") in (
                None,
                self._identity(),
            ):
                try:
                    decoded = self._decode(exact.value, response_type)
                except Exception:
                    trace.decision_status = "unknown_response_codec"
                else:
                    if await self._valid(call, decoded) and (
                        exact.expires_at is None or exact.expires_at > self.store.clock()
                    ):
                        trace.events.append(
                            Event("llm_response", "cache", "hit", 0, exact.avoided_ms)
                        )
                        return decoded, exact
            semantic = await self._semantic(call, context, identity, response_type, trace)
            if semantic is not None and (
                semantic.expires_at is None or semantic.expires_at > self.store.clock()
            ):
                return self._decode(semantic.value, response_type), semantic
        except Exception:
            trace.decision_status = "failed"
        return None

    async def _parallel_generate(
        self,
        function: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        is_async: bool,
        trace: Trace,
    ) -> tuple[Any, float]:
        started = time.perf_counter()
        trace.llm_calls += 1
        trace.speculative_llm_calls += 1
        if is_async:
            response = await function(*args, **kwargs)
        else:
            future = self._submit("llm", function, *args, **kwargs)
            try:
                response = await asyncio.shield(asyncio.wrap_future(future))
            except asyncio.CancelledError:
                future.cancel()  # Stops queued work; an already running thread continues.
                raise
        return response, (time.perf_counter() - started) * 1000

    async def _parallel_admit(
        self,
        key: str,
        call: Mapping[str, Any],
        context: Context,
        identity: str,
        response_type: type | None,
        encoded: Mapping[str, Any],
        generation_ms: float,
        generated_at: float,
    ) -> None:
        # The returned object may be mutated by its caller; validate a detached snapshot.
        if not await self._valid(call, self._decode(encoded, response_type)):
            return
        shape = _semantic_request(call)
        partition = fingerprint(
            {"function": identity, "guard": fingerprint(shape[1] if shape else call)}
        )
        expiry = generated_at + self.ttl_s if self.ttl_s is not None else None
        if expiry is not None and expiry <= self.store.clock():
            return
        await self._async_cache_io(
            "put",
            Entry(
                key,
                "llm_answer",
                context.scope,
                partition,
                encoded,
                {**call, "text": shape[0] if shape else ""},
                dict(context.dependencies),
                context.policy_version,
                self._type_id(response_type) if response_type else "auto",
                generated_at,
                expiry,
                generation_ms,
                {"semantic_identity": None},
            ),
        )

    async def _parallel_promote(
        self,
        key: str,
        call: Mapping[str, Any],
        context: Context,
        semantic: Entry,
        semantic_identity: Any,
    ) -> None:
        await self._async_cache_io(
            "put",
            Entry(
                key,
                "llm_answer",
                context.scope,
                semantic.task,
                semantic.value,
                {**call, "text": _semantic_request(call)[0]},
                dict(context.dependencies),
                context.policy_version,
                semantic.output_contract,
                self.store.clock(),
                semantic.expires_at,
                semantic.avoided_ms,
                {**semantic.metadata, "semantic_identity": semantic_identity},
            ),
        )

    def _defer(self, coroutine: Any) -> None:
        self._track_task(asyncio.create_task(coroutine))

    async def _call_parallel(
        self,
        function: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        identity: str,
        response_type: type | None,
        is_async: bool,
        trace: Trace,
        resolved_context: Context | None,
    ) -> tuple[Any, Trace]:
        try:
            raw_call = {"args": list(args), "kwargs": kwargs}
            if not _native_json(raw_call):
                raise TypeError("Unsupported request value")
            call = snapshot(raw_call)
        except (TypeError, ValueError, RecursionError):
            call = None
        if not self.enabled or self.ttl_s == 0 or kwargs.get("stream", False) or call is None:
            return await self._call(
                function,
                args,
                kwargs,
                identity,
                response_type,
                is_async,
                trace,
                resolved_context=resolved_context,
                parallel=False,
            )

        started = time.perf_counter()
        context = resolved_context or self._context(args, kwargs)
        key = fingerprint(
            {
                "kind": "llm_answer",
                "function": identity,
                "call": call,
                "context": context.to_dict(),
                "contract": self._type_id(response_type) if response_type else "auto",
            }
        )
        owner = None
        generation = lookup = None
        lookup_trace = Trace(trace.request_id, trace.task, mode="parallel")
        try:
            while owner is None:
                with self._lock:
                    if self._closed:
                        raise RuntimeError("JAC is closed")
                    existing = self._parallel_inflight.get(key)
                    if existing is None:
                        owner = Future()
                        self._parallel_inflight[key] = owner
                if existing is not None:
                    shared = await asyncio.shield(asyncio.wrap_future(existing))
                    if shared is _OWNER_CANCELLED:
                        continue
                    if shared is not None:
                        try:
                            decoded = self._decode(shared["encoded"], response_type)
                            valid = await self._valid(call, decoded)
                            if shared["expires_at"] is not None and (
                                shared["expires_at"] <= self.store.clock()
                            ):
                                valid = False
                        except Exception:
                            valid = False
                        if valid:
                            trace.path = "full_hit"
                            trace.race_winner = "coalesced"
                            trace.coalesced_nodes = 1
                            trace.events.append(
                                Event(
                                    "llm_response",
                                    "llm",
                                    "coalesced",
                                    (time.perf_counter() - started) * 1000,
                                )
                            )
                            return decoded, trace

            # Start the provider before the lookup can wait on SQLite or Jev.
            generation = asyncio.create_task(
                self._parallel_generate(function, args, kwargs, is_async, trace)
            )
            lookup = asyncio.create_task(
                self._parallel_lookup(call, context, identity, response_type, key, lookup_trace)
            )
            self._track_task(generation)
            self._track_task(lookup)
            pending = {generation, lookup}
            provider_error = None
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                if lookup in done:
                    hit = lookup.result()
                    if hit is not None:
                        response, entry = hit
                        trace.path = "full_hit"
                        trace.race_winner = "cache"
                        if not generation.done():
                            trace.llm_cancellation_requested = True
                            generation.cancel()
                        owner.set_result({"encoded": entry.value, "expires_at": entry.expires_at})
                        if entry.key != key:
                            self._defer(
                                self._parallel_promote(key, call, context, entry, self._identity())
                            )
                        return response, trace
                if generation in done:
                    try:
                        response, generation_ms = generation.result()
                    except Exception as exc:
                        provider_error = exc
                    else:
                        trace.path = "miss"
                        trace.race_winner = "llm"
                        trace.events.append(
                            Event("llm_response", "llm", "generated", generation_ms)
                        )
                        encoded = self._encode(response, response_type)
                        owner.set_result(
                            {"encoded": encoded, "expires_at": None}
                            if encoded is not None
                            else None
                        )
                        if encoded is not None:
                            self._defer(
                                self._parallel_admit(
                                    key,
                                    call,
                                    context,
                                    identity,
                                    response_type,
                                    encoded,
                                    generation_ms,
                                    self.store.clock(),
                                )
                            )
                        else:
                            trace.decision_status = "bypass_response"
                        if not lookup.done():
                            lookup.cancel()
                        return response, trace
            if provider_error is not None:
                raise provider_error
            raise RuntimeError("Parallel execution produced no response")
        except BaseException as exc:
            trace.path = "error"
            if owner is not None and not owner.done():
                with self._lock:
                    if self._parallel_inflight.get(key) is owner:
                        self._parallel_inflight.pop(key, None)
                    if isinstance(exc, asyncio.CancelledError):
                        owner.set_result(_OWNER_CANCELLED)
                    else:
                        owner.set_exception(exc)
            raise
        finally:
            if owner is not None:
                with self._lock:
                    if self._parallel_inflight.get(key) is owner:
                        self._parallel_inflight.pop(key, None)
            for task in (generation, lookup):
                if task is not None and not task.done() and task.cancelling() == 0:
                    if task is generation:
                        trace.llm_cancellation_requested = True
                    task.cancel()
            trace.jev_calls = lookup_trace.jev_calls
            if trace.decision_status != "bypass_response":
                trace.decision_status = lookup_trace.decision_status
            trace.events.extend(lookup_trace.events)
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            self._last_trace.set(trace)
            self._submit("cache", self._cache_io, "record_trace", trace.to_dict())

    async def _call(
        self,
        function: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        identity: str,
        response_type: type | None,
        is_async: bool,
        trace: Trace,
        resolved_context: Context | None = None,
        parallel: bool = True,
    ) -> tuple[Any, Trace]:
        if self.mode == "parallel" and parallel:
            return await self._call_parallel(
                function, args, kwargs, identity, response_type, is_async, trace, resolved_context
            )
        started = time.perf_counter()
        owner: Future[Any] | None = None
        key = None
        can_cache = self.enabled and self.ttl_s != 0 and not kwargs.get("stream", False)
        try:
            try:
                raw_call = {"args": list(args), "kwargs": kwargs}
                if not _native_json(raw_call):
                    raise TypeError("Unsupported request value")
                call = snapshot(raw_call)
            except (TypeError, ValueError, RecursionError):
                call = None
                can_cache = False
                trace.decision_status = "bypass_request"
            if kwargs.get("stream", False):
                trace.decision_status = "bypass_stream"
            context = resolved_context or self._context(args, kwargs)
            if can_cache and call is not None:
                key = fingerprint(
                    {
                        "kind": "llm_answer",
                        "function": identity,
                        "call": call,
                        "context": context.to_dict(),
                        "contract": self._type_id(response_type) if response_type else "auto",
                    }
                )
                exact = self._cache_io("get", key, event_trace=trace)
                if exact is not None and exact.metadata.get("semantic_identity") in (
                    None,
                    self._identity(),
                ):
                    try:
                        decoded = self._decode(exact.value, response_type)
                    except Exception:
                        trace.decision_status = "unknown_response_codec"
                    else:
                        if await self._valid(call, decoded) and (
                            exact.expires_at is None or exact.expires_at > self.store.clock()
                        ):
                            trace.path = "full_hit"
                            trace.events.append(
                                Event(
                                    "llm_response",
                                    "cache",
                                    "hit",
                                    (time.perf_counter() - started) * 1000,
                                    exact.avoided_ms,
                                )
                            )
                            return decoded, trace
                with self._lock:
                    existing = self._inflight.get(key)
                    if existing is None:
                        owner = Future()
                        self._inflight[key] = owner
                if existing is not None:
                    shared = await asyncio.shield(asyncio.wrap_future(existing))
                    if shared is _OWNER_CANCELLED:
                        return await self._call(
                            function,
                            args,
                            kwargs,
                            identity,
                            response_type,
                            is_async,
                            trace,
                            resolved_context=context,
                        )
                    if shared is not None:
                        try:
                            decoded = self._decode(shared["encoded"], response_type)
                        except Exception:
                            pass
                        else:
                            if await self._valid(call, decoded) and (
                                shared["expires_at"] is None
                                or shared["expires_at"] > self.store.clock()
                            ):
                                trace.path = "full_hit"
                                trace.coalesced_nodes = 1
                                trace.events.append(
                                    Event(
                                        "llm_response",
                                        "llm",
                                        "coalesced",
                                        (time.perf_counter() - started) * 1000,
                                    )
                                )
                                return decoded, trace
                try:
                    semantic = await self._semantic(call, context, identity, response_type, trace)
                except Exception:
                    semantic = None
                    trace.decision_status = "failed"
                if semantic is not None:
                    self._cache_io(
                        "put",
                        Entry(
                            key,
                            "llm_answer",
                            context.scope,
                            semantic.task,
                            semantic.value,
                            {**call, "text": _semantic_request(call)[0]},
                            dict(context.dependencies),
                            context.policy_version,
                            semantic.output_contract,
                            self.store.clock(),
                            semantic.expires_at,
                            semantic.avoided_ms,
                            {**semantic.metadata, "semantic_identity": self._identity()},
                        ),
                        event_trace=trace,
                    )
                    if owner is not None:
                        owner.set_result(
                            {"encoded": semantic.value, "expires_at": semantic.expires_at}
                        )
                    trace.path = "full_hit"
                    return self._decode(semantic.value, response_type), trace
            generation_started = time.perf_counter()
            trace.llm_calls = 1
            if is_async:
                response = await function(*args, **kwargs)
            else:
                response = function(*args, **kwargs)
            generation_ms = (time.perf_counter() - generation_started) * 1000
            trace.events.append(Event("llm_response", "llm", "generated", generation_ms))
            encoded = self._encode(response, response_type) if can_cache else None
            shared_expiry = None
            if encoded is not None and call is not None and await self._valid(call, response):
                shape = _semantic_request(call)
                guard_id = fingerprint(shape[1] if shape else call)
                partition = fingerprint({"function": identity, "guard": guard_id})
                with self._store_lock:
                    now = self.store.clock()
                    shared_expiry = now + self.ttl_s if self.ttl_s is not None else None
                    self._cache_io(
                        "put",
                        Entry(
                            key,
                            "llm_answer",
                            context.scope,
                            partition,
                            encoded,
                            {**call, "text": shape[0] if shape else ""},
                            dict(context.dependencies),
                            context.policy_version,
                            self._type_id(response_type) if response_type else "auto",
                            now,
                            shared_expiry,
                            generation_ms,
                            {"semantic_identity": None},
                        ),
                        event_trace=trace,
                    )
            elif can_cache and encoded is None:
                trace.decision_status = "bypass_response"
            if owner is not None:
                owner.set_result(
                    {"encoded": encoded, "expires_at": shared_expiry}
                    if encoded is not None else None
                )
            return response, trace
        except BaseException as exc:
            trace.path = "error"
            if owner is not None and not owner.done():
                if isinstance(exc, asyncio.CancelledError):
                    with self._lock:
                        if self._inflight.get(key) is owner:
                            self._inflight.pop(key, None)
                        owner.set_result(_OWNER_CANCELLED)
                else:
                    owner.set_exception(exc)
            raise
        finally:
            if owner is not None:
                with self._lock:
                    if self._inflight.get(key) is owner:
                        self._inflight.pop(key, None)
            trace.elapsed_ms = (time.perf_counter() - started) * 1000
            self._last_trace.set(trace)
            self._cache_io("record_trace", trace.to_dict(), event_trace=trace)

    def stats(self) -> dict[str, Any]:
        persisted = {
            trace["request_id"]: trace
            for trace in self._cache_io("traces", default=[])
            if trace["task"] in self._trace_tasks
        }
        with self._lock:
            persisted.update({key: trace.to_dict() for key, trace in self._trace_history.items()})
            traces = list(persisted.values())
        elapsed = sorted(trace["elapsed_ms"] for trace in traces)
        count = len(traces)
        return {
            "requests": count,
            "hits": sum(t["path"] == "full_hit" for t in traces),
            "llm_calls": sum(t["llm_calls"] for t in traces),
            "jev_calls": sum(t["jev_calls"] for t in traces),
            "speculative_llm_calls": sum(t.get("speculative_llm_calls", 0) for t in traces),
            "errors": sum(t["path"] == "error" for t in traces),
            "p50_ms": elapsed[max(0, math.ceil(count * 0.5) - 1)] if count else 0,
            "p95_ms": elapsed[max(0, math.ceil(count * 0.95) - 1)] if count else 0,
        }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._active_requests or self._inflight or self._parallel_inflight or self._pending:
                raise RuntimeError(
                    "Wait for outstanding requests and drain background work before closing JAC"
                )
            self._closed = True
            race_loop, race_thread = self._race_loop, self._race_thread
            executors = list(self._executors.values())
        if race_loop is not None:
            race_loop.call_soon_threadsafe(race_loop.stop)
            race_thread.join(timeout=1)
            if not race_thread.is_alive():
                race_loop.close()
        for executor in executors:
            executor.shutdown(wait=False, cancel_futures=True)
        with self._store_lock:
            if self._owns_store:
                self.store.close()

    def __enter__(self) -> JAC:
        return self

    def __exit__(self, *_: Any) -> None:
        if self.mode == "parallel":
            self.drain_sync()
        self.close()

    async def __aenter__(self) -> JAC:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.drain()
        self.close()
