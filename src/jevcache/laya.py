"""Local, non-generating Laya reuse decisions with complete-evidence checks.

Laya is optional: importing JAC does not import torch or download a checkpoint.
The adapter deliberately rejects evidence that cannot fit the checkpoint, rather
than letting the encoder approve reuse from a truncated request or answer.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from .decisions import Candidate, Decision, DecisionError, _nonnegative_finite
from .laya_prompts import (
    DEFAULT_PROMPT_VERSION,
    PROMPT_VERSIONS,
    laya_approval_values,
    laya_questions,
)
from .models import fingerprint, snapshot

LAYA_QUESTION_SCHEMA_VERSION = DEFAULT_PROMPT_VERSION


def _complete_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only JAC's duplicate search text; retain the original full call.

    JAC stores the final user input twice: once in args/kwargs and again in a
    derived top-level ``text`` field for retrieval. It is redundant evidence only
    when it exactly matches the original input. Arbitrary request fields, history
    and settings are preserved, and a different ``text`` value is preserved too.
    """
    result = dict(request)
    args, kwargs, text = result.get("args"), result.get("kwargs"), result.get("text")
    if not isinstance(args, (list, tuple)) or not isinstance(kwargs, Mapping):
        return result
    containers = [kwargs]
    if len(args) == 1 and isinstance(args[0], Mapping):
        containers.append(args[0])
    actual = None
    for container in containers:
        messages = container.get("messages")
        if (isinstance(messages, list) and messages and isinstance(messages[-1], Mapping)
                and messages[-1].get("role") == "user"
                and isinstance(messages[-1].get("content"), str)):
            actual = messages[-1]["content"]
            break
        for name in ("input", "prompt"):
            if isinstance(container.get(name), str):
                actual = container[name]
                break
        if actual is not None:
            break
    if actual is None and args and isinstance(args[0], str):
        actual = args[0]
    if isinstance(text, str) and text == actual:
        result.pop("text", None)
    return result


class LayaDecider:
    """Independent Noul judgments per candidate in a local encoder forward.

    ``threshold`` is a decision cutoff, not a calibrated probability that reuse
    is correct. Calibrate it on separate, labeled cache-reuse data. Every check
    must pass; candidates are ranked by their minimum score, with stable ID ties.

    The default prompt has two checks. Experimental ``prompt_version`` profiles
    can add checks or ask about conflicts: their approval scores are P(false).
    Prompt definitions and required boolean values isolate calibration identity.

    ``preload`` separates downloads/loading from request latency. ``dtype`` names
    the inference precision: auto, float32, float16 or bfloat16. A pinned Hub
    commit should be supplied for reproducible benchmarks. An injected Agent is
    useful for sharing a loaded model or testing without optional dependencies.

    Predictions run in one offloaded worker. Cancellation cannot stop a running
    GPU kernel: new calls fail closed while that worker finishes, preventing a
    queue of abandoned work. ``drain`` waits for it before the next measurement.
    """

    def __init__(
        self,
        model: str = "convaiinnovations/laya",
        *,
        revision: str | None = None,
        device: str | None = None,
        dtype: str = "auto",
        threshold: float = 0.99,
        agent: Any = None,
        max_candidates: int = 8,
        max_request_bytes: int = 512_000,
        max_len: int | None = None,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty checkpoint ID or path")
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            raise ValueError("revision must be a nonempty revision string")
        if device is not None and (not isinstance(device, str) or not device.strip()):
            raise ValueError("device must be a nonempty device string")
        if dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be auto, float32, float16 or bfloat16")
        if prompt_version not in PROMPT_VERSIONS:
            raise ValueError(f"unknown Laya prompt version: {prompt_version}")
        threshold = _nonnegative_finite(threshold, "threshold")
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be greater than zero and at most one")
        for name, value in (("max_candidates", max_candidates),
                            ("max_request_bytes", max_request_bytes)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if max_len is not None and (
            isinstance(max_len, bool) or not isinstance(max_len, int) or max_len < 1
        ):
            raise ValueError("max_len must be a positive integer")
        self.model = model
        self.revision = revision
        self.device = device
        self.dtype = dtype
        self.threshold = threshold
        self.max_candidates = max_candidates
        self.max_request_bytes = max_request_bytes
        self.max_len = max_len
        self.prompt_version = prompt_version
        self._agent = agent
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jac-laya")
        self._future: Future[Any] | None = None
        self._closed = False
        self._metrics: dict[str, int | float] = {
            "decide_calls": 0, "predict_calls": 0, "context_rejections": 0,
            "failures": 0, "elapsed_ms": 0.0,
        }

    @property
    def agent(self) -> Any:
        """Loaded Agent, or None until preload/first eligible decision."""
        return self._agent

    @property
    def cache_identity(self) -> Mapping[str, Any]:
        agent = self._agent
        cfg = getattr(agent, "cfg", {}) or {}
        runtime_dtype = getattr(agent, "dtype", self.dtype)
        if callable(getattr(agent, "dtype_for", None)):
            runtime_dtype = agent.dtype_for(2)
        return snapshot({
            "provider": "laya-local",
            "model": self.model,
            "revision": getattr(agent, "revision", None) or self.revision,
            "device": str(getattr(agent, "device", self.device or "auto")),
            "dtype": str(runtime_dtype),
            "requested_dtype": self.dtype,
            "threshold": self.threshold,
            "max_len": self.max_len or cfg.get("max_len"),
            "head_max_len": cfg.get("head_max_len", 192),
            "temperature": getattr(agent, "temperature", None),
            "temperature_by_options": getattr(agent, "temperature_by_options", None),
            "binning_map": getattr(agent, "binning_map", None),
            "option_layout": cfg.get("option_layout", "sequential"),
            "question_schema_version": self.prompt_version,
            "question_schema_sha256": fingerprint({
                kind: laya_questions(kind, self.prompt_version)
                for kind in ("answer", "intermediate", "plan")
            }),
            "question_approval_values": {
                kind: laya_approval_values(kind, self.prompt_version)
                for kind in ("answer", "intermediate", "plan")
            },
        })

    def stats(self, *, reset: bool = False) -> dict[str, int | float]:
        with self._lock:
            if reset and self._future is not None and not self._future.done():
                raise RuntimeError("drain the decision worker before resetting metrics")
            result = dict(self._metrics)
            if reset:
                self._metrics = {key: 0.0 if key == "elapsed_ms" else 0 for key in result}
            return result

    def _count(self, key: str, value: int | float = 1) -> None:
        with self._lock:
            self._metrics[key] += value

    async def _offload(self, function, *args: Any) -> Any:
        with self._lock:
            if self._closed:
                self._metrics["failures"] += 1
                raise DecisionError("decision worker is closed")
            if self._future is not None and not self._future.done():
                self._metrics["failures"] += 1
                raise DecisionError("decision inference is already running")
            stopped = threading.Event()
            future = self._executor.submit(function, stopped, *args)
            self._future = future
        try:
            wrapped = asyncio.wrap_future(future)
            # A cancelled caller leaves a real kernel running. Retrieve any later
            # exception even when nobody awaits it, without printing request data.
            wrapped.add_done_callback(lambda done: None if done.cancelled() else done.exception())
            return await asyncio.shield(wrapped)
        except asyncio.CancelledError:
            stopped.set()
            raise
        except DecisionError:
            self._count("failures")
            raise
        except Exception:
            # Avoid returning provider exceptions that can contain request data.
            self._count("failures")
            raise DecisionError("local decision inference failed") from None

    async def preload(self) -> Mapping[str, Any]:
        """Load once outside measured requests and return the resolved identity."""
        await self._offload(self._load)
        return dict(self.cache_identity)

    async def drain(self) -> None:
        """Wait for cancelled/losing inference; do not expose a discarded error."""
        with self._lock:
            future = self._future
        if future is not None:
            try:
                await asyncio.shield(asyncio.wrap_future(future))
            except Exception:
                pass

    def close(self) -> None:
        """Stop accepting jobs; a running model operation finishes normally."""
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _load(self, stopped: threading.Event) -> None:
        if self._agent is None:
            try:
                from laya import Agent
            except ImportError:
                raise DecisionError("install jac-cache[laya] for local Laya decisions") from None
            self._agent = Agent(self.model, device=self.device, revision=self.revision)
            if self.dtype != "auto":
                import torch
                self._agent.dtype = getattr(torch, self.dtype)
                self._agent.amp_enabled = self.dtype != "float32"
        if not callable(getattr(self._agent, "predict", None)):
            raise DecisionError("Laya Agent must expose predict")
        if not callable(getattr(getattr(self._agent, "tok", None), "encode", None)):
            raise DecisionError("Laya Agent must expose its tokenizer for complete-evidence checks")
        cfg = getattr(self._agent, "cfg", None)
        if (not isinstance(cfg, Mapping) or isinstance(cfg.get("max_len"), bool)
                or not isinstance(cfg.get("max_len"), int) or cfg["max_len"] < 1):
            raise DecisionError("Laya checkpoint must report its context limit")
        if self.max_len is not None and self.max_len > cfg["max_len"]:
            raise DecisionError("max_len cannot exceed the checkpoint's configured context limit")

    async def decide(
        self, request: Mapping[str, Any], candidates: Sequence[Candidate], kind: str = "answer"
    ) -> Decision:
        questions = laya_questions(kind, self.prompt_version)
        if not isinstance(request, Mapping):
            raise DecisionError("request must be a mapping")
        candidates = tuple(candidates)
        if not candidates:
            return Decision(None, {}, self.model, 0.0)
        if len(candidates) > self.max_candidates:
            raise DecisionError("candidate limit exceeded")
        try:
            threshold = _nonnegative_finite(self.threshold, "threshold")
        except ValueError as exc:
            raise DecisionError(str(exc)) from None
        if not 0 < threshold <= 1:
            raise DecisionError("threshold must be greater than zero and at most one")
        identifiers: set[str] = set()
        states: list[str] = []
        for candidate in candidates:
            if not isinstance(candidate, Candidate):
                raise DecisionError("candidates must be Candidate values")
            if not isinstance(candidate.id, str) or not candidate.id or candidate.id in identifiers:
                raise DecisionError("candidate IDs must be distinct nonempty strings")
            if not isinstance(candidate.request, Mapping):
                raise DecisionError("candidate request must be a mapping")
            try:
                _nonnegative_finite(candidate.avoided_ms, "candidate avoided_ms")
                evidence = snapshot({
                    "new_request": _complete_request(request),
                    "cached_request": _complete_request(candidate.request),
                    "artifact": candidate.artifact,
                })
                state = json.dumps(
                    evidence, ensure_ascii=False, allow_nan=False, separators=(",", ":")
                )
            except (TypeError, ValueError, OverflowError, RecursionError):
                raise DecisionError("decision context must contain JSON-compatible finite values")
            identifiers.add(candidate.id)
            states.append(state)
        try:
            context_bytes = sum(len(state.encode("utf-8")) for state in states)
        except UnicodeError:
            raise DecisionError("decision context must contain valid Unicode text") from None
        if context_bytes > self.max_request_bytes:
            raise DecisionError("decision context exceeds request size limit")
        self._count("decide_calls")
        return await self._offload(self._predict, candidates, states, questions, threshold,
                                   laya_approval_values(kind, self.prompt_version))

    def _fits(self, state: str, questions: Mapping[str, Any]) -> bool:
        """Exact packing accounting for our fixed two-option Noul schema.

        Laya's head has CLS, two SEP, two MASK and a closing state SEP. Option
        texts cap at 48 tokens and instructions share head_max_len; refuse either
        cap before predict. Count the full state with the Agent's own tokenizer.
        """
        agent = self._agent
        tok = agent.tok
        mask = getattr(tok, "mask_token", None)
        if isinstance(mask, str) and mask and mask in state:
            return False  # upstream replaces this string with spaces
        limit = self.max_len or agent.cfg["max_len"]
        head_limit = agent.cfg.get("head_max_len", 192)
        if isinstance(head_limit, bool) or not isinstance(head_limit, int) or head_limit < 1:
            raise DecisionError("Laya checkpoint has an invalid question context limit")
        state_tokens = len(tok.encode(state, add_special_tokens=False))
        for question in questions.values():
            instruction = "noul question: " + question["instructions"]
            options = [
                f" {key}: {question['criteria'][key]}" for key in ("false", "true")
            ]
            option_tokens = [len(tok.encode(text, add_special_tokens=False)) for text in options]
            if any(count > 48 for count in option_tokens):
                return False
            instruction_tokens = len(tok.encode(instruction, add_special_tokens=False))
            options_with_markers = sum(option_tokens) + 2
            if options_with_markers + instruction_tokens > head_limit:
                return False
            if state_tokens + instruction_tokens + options_with_markers + 4 > limit:
                return False
        return True

    def _predict(self, stopped, candidates, states, questions, threshold,
                 approval_values) -> Decision:
        started = time.perf_counter()
        self._load(stopped)
        probabilities: dict[str, float] = {}
        ranked: list[tuple[float, str]] = []
        for candidate, state in zip(candidates, states, strict=True):
            if stopped.is_set():
                break
            if not self._fits(state, questions):
                self._count("context_rejections")
                continue
            self._count("predict_calls")
            fallback_count = getattr(self._agent, "cpu_fallback_count", 0)
            response = self._agent.predict(state, questions, max_len=self.max_len)
            if getattr(self._agent, "cpu_fallback_count", 0) != fallback_count:
                raise DecisionError("Laya changed inference device during the decision")
            values = self._parse_response(response, questions)
            if values is None:
                self._count("context_rejections")
                continue
            # Keep Decision.probabilities as reuse-support scores. A conflict
            # question's native P(true) supports rejection, so approve on P(false).
            values = {name: value if approval_values[name] else 1 - value
                      for name, value in values.items()}
            for dimension, value in values.items():
                probabilities[f"{candidate.id}:{dimension}"] = value
            if all(value >= threshold for value in values.values()):
                ranked.append((min(values.values()), candidate.id))
        elapsed = (time.perf_counter() - started) * 1000
        self._count("elapsed_ms", elapsed)
        chosen = min(ranked, key=lambda item: (-item[0], item[1]))[1] if ranked else None
        return Decision(chosen, probabilities, self.model, elapsed)

    @staticmethod
    def _parse_response(response, questions) -> dict[str, float] | None:
        if not isinstance(response, Mapping) or response.get("model") != "laya-rl-agent":
            raise DecisionError("Laya returned an invalid model response")
        usage = response.get("usage")
        if not isinstance(usage, Mapping) or not isinstance(usage.get("truncated"), bool):
            raise DecisionError("Laya response lacks complete-evidence metadata")
        dropped = usage.get("state_tokens_dropped")
        if isinstance(dropped, bool) or not isinstance(dropped, int) or dropped < 0:
            raise DecisionError("Laya returned invalid truncation metadata")
        if (usage["truncated"] or dropped or usage.get("truncated_questions")
                or usage.get("options")):
            return None
        if type(usage.get("output_tokens")) is not int or usage["output_tokens"] != 0:
            raise DecisionError("Laya response unexpectedly reports generated output")
        answers = response.get("answers")
        if not isinstance(answers, Mapping) or set(answers) != set(questions):
            raise DecisionError("Laya response question IDs do not match the request")
        values: dict[str, float] = {}
        for dimension, answer in answers.items():
            if not isinstance(answer, Mapping) or answer.get("type") != "noul":
                raise DecisionError("Laya returned an invalid Noul answer")
            try:
                value = _nonnegative_finite(answer.get("noul"), "noul")
            except ValueError:
                raise DecisionError("Laya returned an invalid probability") from None
            if value > 1:
                raise DecisionError("Laya returned an invalid probability")
            values[dimension] = value
        return values
