"""JevK5 GGUF decisions through llama-server, without a generative judge.

Uses upstream's prompt/option mapping and temperature-scaled option log-probs.
llama-server evaluates one bounded answer token per question; it does not generate
the cached answer. Missing option probabilities and truncated evidence fail closed.
"""

from __future__ import annotations

import json
import math
import time

from .decisions import Decision, DecisionError, _nonnegative_finite
from .laya import LayaDecider
from .laya_prompts import DEFAULT_PROMPT_VERSION, laya_approval_values, laya_questions
from .models import fingerprint, snapshot
from .providers import CompatibleClient


class JevK5Decider(LayaDecider):
    """Reuse shared evidence validation and worker lifecycle, replacing inference.

    Start a pinned JevK5 GGUF with llama-server, then provide its URL, model file
    identity and calibration temperature. Default 1.22 is for the 4B v0.3 file.
    Thresholds require cache-specific calibration, independently of temperature.
    The optional ``jevk5`` package is imported lazily for its standard-library-only
    prompt module; install it with --no-deps for this backend.
    """

    def __init__(self, model="jevk5-4b-v0.3-Q8_0.gguf", *,
                 base_url="http://127.0.0.1:8080", revision=None,
                 temperature=1.22, threshold=.99, timeout_s=30,
                 max_len=8192, top_k=512, max_candidates=8,
                 max_request_bytes=512_000, prompt_version=DEFAULT_PROMPT_VERSION):
        super().__init__(model, revision=revision, threshold=threshold,
                         max_len=max_len, max_candidates=max_candidates,
                         max_request_bytes=max_request_bytes, prompt_version=prompt_version)
        temperature = _nonnegative_finite(temperature, "temperature")
        if temperature == 0:
            raise ValueError("temperature must be positive")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 2:
            raise ValueError("top_k must be an integer of at least two")
        self.temperature = temperature
        self.top_k = top_k
        self.client = CompatibleClient(model=model, base_url=base_url, timeout_s=timeout_s)
        self._prompt = None
        self._metrics.update({"input_tokens": 0, "readout_tokens": 0,
                              "missing_option_rejections": 0})

    @property
    def cache_identity(self):
        return snapshot({
            "provider": "jevk5-gguf", "model": self.model, "revision": self.revision,
            "endpoint": self.client.base_url, "temperature": self.temperature,
            "threshold": self.threshold, "max_len": self.max_len, "top_k": self.top_k,
            "runtime_version": getattr(self, "_runtime_version", None),
            "question_schema_version": self.prompt_version,
            "question_schema_sha256": fingerprint({kind: laya_questions(kind, self.prompt_version)
                for kind in ("answer", "intermediate", "plan")}),
            "question_approval_values": {kind: laya_approval_values(kind, self.prompt_version)
                for kind in ("answer", "intermediate", "plan")},
        })

    def _load(self, stopped):
        if self._prompt is None:
            try:
                import jevk5
                from jevk5.prompt import decision_options, prompt_text
            except ImportError:
                raise DecisionError("install the pinned jevk5 package with --no-deps") from None
            self._prompt = prompt_text
            self._options = decision_options
            self._runtime_version = jevk5.__version__

    def _read(self, state, question):
        options = self._options(question)
        # Laya's encoder accepts serialized evidence. JevK5's native JSON prompt
        # accepts the object directly; avoid another layer of escaped JSON text.
        prompt = self._prompt(json.loads(state), question["instructions"],
                              [text for _, text in options])
        tokens = self.client.post("/tokenize", {
            "content": prompt, "add_special": False, "parse_special": True,
        }).get("tokens")
        if (not isinstance(tokens, list) or not tokens
                or any(type(token) is not int or token < 0 for token in tokens)):
            raise DecisionError("JevK5 tokenizer returned invalid token IDs")
        if len(tokens) + 1 > self.max_len:
            self._count("context_rejections")
            return None
        self._count("predict_calls")
        result = self.client.post("/completion", {
            "prompt": tokens, "n_predict": 1, "n_probs": self.top_k,
            "temperature": 0, "cache_prompt": False, "post_sampling_probs": False,
        })
        # Requiring this metadata prevents silent llama-server context shifting.
        if result.get("truncated") is not False:
            self._count("context_rejections")
            return None
        if (type(result.get("tokens_evaluated")) is not int
                or result["tokens_evaluated"] != len(tokens)
                or type(result.get("tokens_predicted")) is not int
                or result["tokens_predicted"] != 1):
            raise DecisionError("JevK5 readout lacks complete token accounting")
        self._count("input_tokens", len(tokens))
        self._count("readout_tokens")
        try:
            top = result["completion_probabilities"][0]["top_logprobs"]
            if not isinstance(top, list):
                raise ValueError
            seen = {}
            for entry in top:
                token, value = entry["token"], entry["logprob"]
                if (not isinstance(token, str) or (token in seen and token in {"A", "B"})
                        or isinstance(value, bool)
                        or not isinstance(value, (int, float)) or not math.isfinite(value)
                        or value > 0):
                    raise ValueError
                seen[token] = value
        except (KeyError, TypeError, IndexError, ValueError):
            raise DecisionError("JevK5 returned invalid option log-probabilities") from None
        # Upstream approximates an absent letter with a floor. Cache reuse instead
        # rejects: neither an invented probability nor confidence can approve reuse.
        if "A" not in seen or "B" not in seen:
            self._count("missing_option_rejections")
            return None
        logits = [seen["A"], seen["B"]]
        largest = max(logits)
        weights = [math.exp((value - largest) / self.temperature) for value in logits]
        return weights[0] / sum(weights)  # upstream orders true then false

    def _predict(self, stopped, candidates, states, questions, threshold, approval_values):
        started = time.perf_counter()
        self._load(stopped)
        probabilities, ranked = {}, []
        for candidate, state in zip(candidates, states, strict=True):
            values = {}
            for name, question in questions.items():
                if stopped.is_set():
                    break
                value = self._read(state, question)
                if value is None:
                    break
                values[name] = value if approval_values[name] else 1 - value
            if stopped.is_set():
                break
            # Incomplete candidates expose no scores and cannot be accepted.
            if set(values) != set(questions):
                continue
            probabilities.update({f"{candidate.id}:{name}": value
                                  for name, value in values.items()})
            if all(value >= threshold for value in values.values()):
                ranked.append((min(values.values()), candidate.id))
        elapsed = (time.perf_counter() - started) * 1000
        self._count("elapsed_ms", elapsed)
        chosen = min(ranked, key=lambda item: (-item[0], item[1]))[1] if ranked else None
        return Decision(chosen, probabilities, self.model, elapsed)
