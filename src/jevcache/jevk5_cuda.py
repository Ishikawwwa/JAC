"""In-process JevK5 decisions using upstream's optional CUDA graph runtime."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping

from .decisions import DecisionError
from .jevk5 import JevK5Decider
from .models import snapshot


class JevK5CudaDecider(JevK5Decider):
    """Use unquantized bf16 JevK5 on a GPU, with the same bounded reuse checks.

    Install the pinned upstream ``jevk5[fast]`` package separately. Loading and
    graph capture happen during ``preload()``, outside request latency. Injected
    runtimes support tests without Torch, weights or a GPU.
    """

    def __init__(self, model="alibiserikbay/JevK5", *, revision=None,
                 device="cuda", graphs=True, temperature=1.22, threshold=.99,
                 max_len=8192, max_candidates=8, max_request_bytes=512_000,
                 prompt_version="laya-reuse-v2", runtime=None):
        if not isinstance(device, str) or not device:
            raise ValueError("device must be a nonempty string")
        if type(graphs) is not bool:
            raise ValueError("graphs must be a boolean")
        if graphs and not device.startswith("cuda"):
            raise ValueError("CUDA graphs require a CUDA device")
        super().__init__(model, revision=revision, temperature=temperature, threshold=threshold,
                         max_len=max_len, max_candidates=max_candidates,
                         max_request_bytes=max_request_bytes, prompt_version=prompt_version)
        self.device = device
        self.graphs = graphs
        self._runtime = runtime
        self._options = None

    @property
    def cache_identity(self):
        identity = dict(super().cache_identity)
        identity.pop("endpoint")
        identity.pop("top_k")
        identity.update(provider="jevk5-cuda", device=self.device,
                        dtype="bfloat16", graphs=self.graphs)
        return snapshot(identity)

    def _load(self, stopped):
        if self._options is None:
            try:
                import jevk5
                from jevk5.prompt import decision_options
            except ImportError:
                raise DecisionError("install the pinned jevk5[fast] package") from None
            self._options = decision_options
            self._runtime_version = jevk5.__version__
        if self._runtime is None:
            from jevk5 import JevK5

            source = self.model
            if self.revision:
                from huggingface_hub import snapshot_download

                source = snapshot_download(self.model, revision=self.revision)
            self._runtime = JevK5(source, device=self.device, graphs=self.graphs,
                                 temperature=self.temperature)
        if not all(callable(getattr(self._runtime, name, None))
                   for name in ("encode", "probabilities")):
            raise DecisionError("JevK5 runtime must expose encode and probabilities")
        if str(getattr(self._runtime, "device", "")) != self.device:
            raise DecisionError("JevK5 runtime changed inference device")
        if getattr(self._runtime, "temperature", None) != self.temperature:
            raise DecisionError("JevK5 runtime changed calibration temperature")

    def _read(self, state, question):
        evidence = json.loads(state)
        options = self._options(question)
        tokens = self._runtime.encode(evidence, question["instructions"],
                                      [text for _, text in options])
        if (not isinstance(tokens, list) or not tokens
                or any(type(token) is not int or token < 0 for token in tokens)):
            raise DecisionError("JevK5 runtime returned invalid token IDs")
        if len(tokens) > self.max_len:
            self._count("context_rejections")
            return None
        self._count("predict_calls")
        values, count = self._runtime.probabilities(evidence, question)
        if type(count) is not int or count != len(tokens):
            raise DecisionError("JevK5 readout lacks complete token accounting")
        if (not isinstance(values, Mapping) or set(values) != {"true", "false"}
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) or not 0 <= value <= 1
                       for value in values.values())
                or not math.isclose(sum(values.values()), 1, abs_tol=1e-5)):
            raise DecisionError("JevK5 returned invalid option probabilities")
        self._count("input_tokens", count)
        return values["true"]
