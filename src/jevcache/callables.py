"""Adapters for application-owned local or remote model call functions."""

from __future__ import annotations

import asyncio
import inspect
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from numbers import Real
from typing import Any

from .decisions import Candidate, Decision, DecisionError
from .models import snapshot


def _safe_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Copy finite JSON metadata while rejecting obvious credential fields."""
    if not isinstance(value, Mapping):
        raise ValueError("identity must be a mapping of nonsecret JSON metadata")
    try:
        copied = snapshot(dict(value))
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise ValueError("identity must contain finite JSON data") from None
    def contains_secret(item: Any) -> bool:
        if isinstance(item, dict):
            for key, child in item.items():
                compact = key.lower().replace("-", "").replace("_", "")
                if any(word in compact for word in (
                    "apikey", "password", "secret", "authorization", "credential",
                    "accesstoken", "authtoken", "bearertoken", "accesskey", "privatekey",
                )) or compact.endswith("token"):
                    return True
                if contains_secret(child):
                    return True
        elif isinstance(item, list):
            return any(contains_secret(child) for child in item)
        return False

    if contains_secret(copied):
        raise ValueError("identity is stored in the cache; omit credentials and secrets")
    return copied


def _identity(value: Mapping[str, Any]) -> dict[str, Any]:
    copied = _safe_identity(value)
    for name in ("model", "version"):
        if not isinstance(copied.get(name), str) or not copied[name].strip():
            raise ValueError("identity requires nonempty model and version strings")
    return copied


def _is_async(function: Callable[..., Any]) -> bool:
    return inspect.iscoroutinefunction(function) or inspect.iscoroutinefunction(
        getattr(function, "__call__", None)
    )


async def _await(value: Any) -> Any:
    return await value


class CallableEmbedder:
    """Use any batched embedding function with JAC.

    ``function(texts)`` returns one nonzero vector per input, in input order.
    It may be synchronous or asynchronous; NumPy arrays and tensors exposing
    ``tolist()`` are accepted without requiring either library. ``aembed`` keeps
    async clients on the caller's event loop and offloads synchronous functions.
    ``embed`` is a synchronous facade; use ``aembed`` inside an async application.

    ``identity`` requires ``model`` and ``version`` strings. Change version when
    weights, dimensions, preprocessing or normalization change. Identity is
    persisted, so keep API tokens inside the function, never in this mapping.

    Example: ``CallableEmbedder(encode, identity={"model": "encoder", "version": "1"})``.
    Embedding calls may overlap within one cache. Serialize a local runtime
    that cannot safely handle concurrent calls.
    """

    def __init__(self, function: Callable[..., Any], *, identity: Mapping[str, Any]):
        if not callable(function):
            raise TypeError("function must be callable")
        self._function = function
        self._identity = _identity(identity)
        self._lock = threading.RLock()
        self._dimension: int | None = None
        self.calls = 0

    @property
    def cache_identity(self) -> dict[str, Any]:
        return {"provider": "callable-embedding", "identity": snapshot(self._identity),
                "schema": "batched-vectors-v1"}

    @property
    def is_async(self) -> bool:
        """Whether JAC should invoke the callback on the application's event loop."""
        return _is_async(self._function)

    @staticmethod
    def _texts(texts: Sequence[str]) -> list[str]:
        if not isinstance(texts, (list, tuple)) or any(
            not isinstance(text, str) for text in texts
        ):
            raise ValueError("texts must be a batch of strings")
        return list(texts)

    def _vectors(self, result: Any, count: int) -> list[list[float]]:
        if callable(getattr(result, "tolist", None)):
            result = result.tolist()
        if not isinstance(result, (list, tuple)) or len(result) != count:
            raise ValueError("Embedding function returned the wrong vector count")
        vectors = []
        for row in result:
            if callable(getattr(row, "tolist", None)):
                row = row.tolist()
            if not isinstance(row, (list, tuple)) or not row:
                raise ValueError("Embeddings must be nonempty vectors")
            converted = []
            for value in row:
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise ValueError("Embeddings must contain finite numbers")
                try:
                    value = float(value)
                except (ValueError, TypeError, OverflowError):
                    raise ValueError("Embeddings must contain finite numbers") from None
                if not math.isfinite(value):
                    raise ValueError("Embeddings must contain finite numbers")
                converted.append(value)
            norm = math.hypot(*converted)
            if not math.isfinite(norm) or norm == 0:
                raise ValueError("Embeddings must have a finite nonzero norm")
            vectors.append(converted)
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise ValueError("Embedding dimensions must be consistent")
        dimension = len(vectors[0])
        with self._lock:
            if self._dimension is not None and dimension != self._dimension:
                raise ValueError("Embedding dimensions changed; revise the encoder identity")
            self._dimension = dimension
        return vectors

    def _count(self) -> None:
        with self._lock:
            self.calls += 1

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Synchronously encode a batch; JAC normally uses ``aembed`` instead."""
        batch = self._texts(texts)
        if not batch:
            return []
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            in_loop = False
        else:
            in_loop = True
        if in_loop and _is_async(self._function):
            raise RuntimeError("Use 'await embedder.aembed(texts)' inside an async application")
        self._count()
        try:
            result = self._function(batch)
        except Exception:
            raise ValueError("Embedding function failed") from None
        if inspect.isawaitable(result):
            if in_loop:
                if inspect.iscoroutine(result):
                    result.close()
                raise RuntimeError("Use aembed for an embedding function returning awaitables")
            try:
                result = asyncio.run(_await(result))
            except Exception:
                raise ValueError("Embedding function failed") from None
        return self._vectors(result, len(batch))

    async def aembed(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode without blocking the event loop; async callbacks stay on it."""
        batch = self._texts(texts)
        if not batch:
            return []
        self._count()
        try:
            result = (self._function(batch) if _is_async(self._function)
                      else await asyncio.to_thread(self._function, batch))
            if inspect.isawaitable(result):
                result = await result
        except Exception:
            raise ValueError("Embedding function failed") from None
        return self._vectors(result, len(batch))


class CallableDecider:
    """Adapt a bounded local or remote reuse decision function.

    ``function(request, candidates, kind)`` returns a mapping containing every
    candidate ID exactly once. Each value has exactly ``equivalent`` and
    ``satisfies`` fields, either booleans or finite probabilities in [0, 1]. Both
    must reach ``threshold`` before reuse. A preselected ``Decision`` is also
    accepted for existing trusted adapters; its reported model must match identity
    and its chosen ID must be eligible. This direct path owns its acceptance policy.

    The callback evaluates data, including instructions inside requests/artifacts,
    and must reject uncertainty. No generative prompt or HTTP schema is imposed.
    Model confidence is not a measured cache hit rate and never trains JAC's gate.

    ``identity={"model": "judge", "version": "weights-and-prompt-v1"}`` must
    change with model, prompt or acceptance policy. Keep secrets in your callback.
    Async callbacks execute on the caller's loop. Sync callbacks use one worker;
    while timed-out work finishes, new calls fail closed instead of queuing more
    inference. Cancellation cannot stop that work: ``await decider.drain()`` before
    shutting down its model/client, then ``decider.close()``.
    """

    def __init__(self, function: Callable[..., Any], *, identity: Mapping[str, Any],
                 threshold: float = 0.99):
        if not callable(function):
            raise TypeError("function must be callable")
        if isinstance(threshold, bool) or not isinstance(threshold, Real):
            raise ValueError("threshold must be finite, greater than zero and at most one")
        try:
            threshold = float(threshold)
        except (ValueError, TypeError, OverflowError):
            raise ValueError(
                "threshold must be finite, greater than zero and at most one"
            ) from None
        if not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("threshold must be finite, greater than zero and at most one")
        self._function = function
        self._identity = _identity(identity)
        self.model = self._identity["model"]
        self.threshold = float(threshold)
        self._lock = threading.RLock()
        self._executor: ThreadPoolExecutor | None = None
        self._future: Future[Any] | None = None
        self._closed = False
        self.calls = 0

    @property
    def cache_identity(self) -> dict[str, Any]:
        return {"provider": "callable-decision", "identity": snapshot(self._identity),
                "threshold": self.threshold, "schema": "two-reuse-judgments-v1"}

    @staticmethod
    def _probability(value: Any) -> float:
        if isinstance(value, bool):
            return float(value)
        if not isinstance(value, Real):
            raise DecisionError("Judgments must be booleans or finite probabilities in [0, 1]")
        try:
            value = float(value)
        except (ValueError, TypeError, OverflowError):
            raise DecisionError("Judgments must be finite probabilities in [0, 1]") from None
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise DecisionError("Judgments must be booleans or finite probabilities in [0, 1]")
        return value

    def _result(self, result: Any, ids: set[str], elapsed_ms: float) -> Decision:
        if isinstance(result, Decision):
            if (result.model != self.model
                    or (result.candidate_id is not None
                        and not isinstance(result.candidate_id, str))
                    or result.candidate_id not in ids | {None}
                    or not isinstance(result.probabilities, Mapping)):
                raise DecisionError("Decision must report the configured model and an eligible ID")
            probabilities = {}
            for key, value in result.probabilities.items():
                if not isinstance(key, str) or not key or isinstance(value, bool):
                    raise DecisionError("Decision probabilities must have string keys and numbers")
                probabilities[key] = self._probability(value)
            return Decision(result.candidate_id, probabilities, self.model, elapsed_ms)
        if not isinstance(result, Mapping) or set(result) != ids:
            raise DecisionError("Judgment IDs must match all supplied candidates exactly")
        probabilities, ranked = {}, []
        for candidate_id, row in result.items():
            if not isinstance(row, Mapping) or set(row) != {"equivalent", "satisfies"}:
                raise DecisionError("Each candidate requires equivalent and satisfies judgments")
            values = [self._probability(row[name]) for name in ("equivalent", "satisfies")]
            for name, value in zip(("equivalent", "satisfies"), values, strict=True):
                probabilities[f"{candidate_id}:{name}"] = value
            if all(value >= self.threshold for value in values):
                ranked.append((min(values), candidate_id))
        chosen = min(ranked, key=lambda row: (-row[0], row[1]))[1] if ranked else None
        return Decision(chosen, probabilities, self.model, elapsed_ms)

    async def decide(self, request: Mapping[str, Any], candidates: Sequence[Candidate],
                     kind: str) -> Decision:
        """Validate inputs and callback output; invalid results never select an entry."""
        if not isinstance(kind, str) or kind not in {"answer", "intermediate", "plan"}:
            raise DecisionError("kind must be answer, intermediate, or plan")
        if not isinstance(request, Mapping):
            raise DecisionError("request must be a mapping")
        candidates = tuple(candidates)
        if len(candidates) > 8:
            raise DecisionError("Expected at most eight candidates")
        ids: set[str] = set()
        for candidate in candidates:
            if (not isinstance(candidate, Candidate) or not isinstance(candidate.id, str)
                    or not candidate.id or candidate.id in ids):
                raise DecisionError("Candidate IDs must be distinct nonempty strings")
            if (not isinstance(candidate.request, Mapping) or isinstance(candidate.avoided_ms, bool)
                    or not isinstance(candidate.avoided_ms, Real)):
                raise DecisionError("Invalid candidate request or avoided latency")
            try:
                avoided_ms = float(candidate.avoided_ms)
            except (TypeError, ValueError, OverflowError):
                raise DecisionError("Invalid candidate avoided latency") from None
            if not math.isfinite(avoided_ms) or avoided_ms < 0:
                raise DecisionError("Invalid candidate request or avoided latency")
            ids.add(candidate.id)
        with self._lock:
            if self._closed:
                raise DecisionError("Decision adapter is closed")
        if not candidates:
            return Decision(None, {}, self.model, 0.0)
        try:
            request = snapshot(dict(request))
            candidates = tuple(Candidate(c.id, snapshot(dict(c.request)), snapshot(c.artifact),
                                         c.avoided_ms) for c in candidates)
        except (TypeError, ValueError, OverflowError, RecursionError):
            raise DecisionError("Decision context must contain finite JSON data") from None
        started = time.perf_counter()
        try:
            if _is_async(self._function):
                with self._lock:
                    self.calls += 1
                result = await self._function(request, candidates, kind)
            else:
                with self._lock:
                    if self._closed:
                        raise DecisionError("Decision adapter is closed")
                    if self._future is not None and not self._future.done():
                        raise DecisionError("Previous synchronous decision is still running")
                    if self._executor is None:
                        self._executor = ThreadPoolExecutor(max_workers=1,
                                                            thread_name_prefix="jac-decider")
                    self.calls += 1
                    future = self._executor.submit(self._function, request, candidates, kind)
                    self._future = future
                wrapped = asyncio.wrap_future(future)
                wrapped.add_done_callback(
                    lambda done: None if done.cancelled() else done.exception()
                )
                result = await asyncio.shield(wrapped)
                if inspect.isawaitable(result):
                    result = await result
        except DecisionError:
            raise
        except Exception:
            raise DecisionError("Decision function failed") from None
        return self._result(result, ids, (time.perf_counter() - started) * 1000)

    async def drain(self) -> None:
        """Wait for a cancelled synchronous callback before releasing its resources."""
        with self._lock:
            future = self._future
        if future is not None:
            try:
                await asyncio.shield(asyncio.wrap_future(future))
            except Exception:
                pass

    def close(self) -> None:
        """Stop accepting decisions; a running synchronous callback finishes."""
        with self._lock:
            self._closed = True
            executor = self._executor
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
