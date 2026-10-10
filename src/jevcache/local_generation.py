"""Optional local generator used by the live comparison, separate from Laya decisions."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any


class TransformersClient:
    """Small deterministic chat generator with explicit speculative-work accounting.

    Cancellation of a Python await cannot stop an active CUDA forward pass. The
    shielded worker finishes and ``drain`` observes it before the next trial.
    """

    def __init__(self, *, model: str, revision: str, device: str = "cuda",
                 max_new_tokens: int = 64):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not revision or max_new_tokens < 1:
            raise ValueError("A pinned revision and positive token limit are required")
        started = time.perf_counter()
        self.model, self.revision, self.device = model, revision, device
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self._model = AutoModelForCausalLM.from_pretrained(
            model, revision=revision, torch_dtype=torch.float16 if device == "cuda"
            else torch.float32, attn_implementation="sdpa").to(device).eval()
        self.load_ms = (time.perf_counter() - started) * 1000
        self.calls = self.completed = self.output_tokens = 0
        self._lock = threading.Lock()
        self._workers: set[asyncio.Task] = set()

    @property
    def cache_identity(self):
        return {"provider": "local-transformers", "model": self.model,
                "revision": self.revision, "device": self.device,
                "max_new_tokens": self.max_new_tokens, "schema": "chat-v1"}

    def _generate(self, kwargs: dict[str, Any]) -> dict:
        import torch

        with self._lock, torch.inference_mode():
            inputs = self.tokenizer.apply_chat_template(
                kwargs["messages"], add_generation_prompt=True, return_tensors="pt"
            ).to(self.device)
            outputs = self._model.generate(
                inputs, attention_mask=torch.ones_like(inputs), do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id)
            new_tokens = outputs[0, inputs.shape[1]:]
            content = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            self.completed += 1
            self.output_tokens += len(new_tokens)
            return {"choices": [{"message": {"role": "assistant", "content": content}}]}

    async def complete(self, **kwargs) -> dict:
        if kwargs.get("temperature", 0) != 0:
            raise ValueError("The reproducible local generator supports temperature=0")
        self.calls += 1
        task = asyncio.create_task(asyncio.to_thread(self._generate, kwargs))
        self._workers.add(task)
        def finished(done):
            self._workers.discard(done)
            if not done.cancelled():
                done.exception()

        task.add_done_callback(finished)
        return await asyncio.shield(task)

    async def drain(self):
        if self._workers:
            await asyncio.gather(*(asyncio.shield(task) for task in tuple(self._workers)))
