"""Working custom-call factory for local/remote compatible endpoints and hosted Jev.

Set REMOTE_MODEL / REMOTE_MODEL_BASE_URL; for embeddings also set
EMBEDDING_MODEL / EMBEDDING_BASE_URL. BM25 needs no embedding configuration.
Keys: REMOTE_MODEL_API_KEY, EMBEDDING_API_KEY, TYPESAFE_API_KEY (hosted Jev).
Use localhost endpoints for local generation/embeddings, or hosted URLs for remote.
Run: python -m jac evaluate-agents --factory examples.custom_models:build
Replace the three function bodies with your own SDK/model calls as needed.
"""

import os

from jac import (
    CallableDecider,
    CallableEmbedder,
    CompatibleClient,
    JevDecider,
    OpenAICompatibleEmbedder,
)


def build():
    generator = CompatibleClient(
        model=os.environ["REMOTE_MODEL"],
        base_url=os.environ["REMOTE_MODEL_BASE_URL"],
        api_key=os.environ.get("REMOTE_MODEL_API_KEY", ""),
        timeout_s=120,
    )
    encoder = OpenAICompatibleEmbedder(
        model=os.environ["EMBEDDING_MODEL"],
        base_url=os.environ["EMBEDDING_BASE_URL"],
        api_key=os.environ.get("EMBEDDING_API_KEY", ""),
        timeout_s=30,
    ) if os.environ.get("EMBEDDING_MODEL") else None
    decision_model = JevDecider(timeout_s=10)

    async def generate(**kwargs):
        return await generator.complete(**kwargs)

    def encode(texts):
        return encoder.embed(texts)

    async def decide(request, candidates, kind):
        # Native Decision is supported; this backend owns its acceptance policy.
        return await decision_model.decide(request, candidates, kind)

    return {
        "generation": generate,
        "generation_identity": generator.cache_identity,
        "embedder": CallableEmbedder(encode, identity={
            "model": encoder.model, "version": "custom-example-v1",
            "backend": encoder.cache_identity,
        }) if encoder else None,
        "decider": CallableDecider(decide, identity={
            "model": decision_model.model, "version": "custom-example-v1",
            "backend": decision_model.cache_identity,
        }),
        "decision_timeout_s": 10,
    }
