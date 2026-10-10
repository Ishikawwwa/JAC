"""Cache any compatible remote model; configure its endpoint/model/key in env."""

import asyncio
import os

from jac import JAC, CompatibleClient


async def main():
    provider = CompatibleClient(
        model=os.environ["REMOTE_MODEL"],
        base_url=os.environ["REMOTE_MODEL_BASE_URL"],
        api_key=os.environ["REMOTE_MODEL_API_KEY"],
        timeout_s=120,
    )
    async with JAC(path="jac.sqlite", ttl_s=300) as cache:
        ask = cache.wrap(provider.complete, namespace="remote-example-v1")
        request = {"messages": [{"role": "user", "content": "Explain BM25."}]}
        first = await ask(**request)
        second = await ask(**request)
        assert first == second
        print(second)
        print(cache.stats())


if __name__ == "__main__":
    asyncio.run(main())
