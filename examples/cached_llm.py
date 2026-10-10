"""Wrap your existing LLM callable; a hit returns its response directly."""

import asyncio

from jac import JAC


async def main():
    generations = 0

    async def ask_llm(prompt, *, model="my-model"):
        nonlocal generations
        generations += 1
        # Replace this function body with your existing provider call.
        await asyncio.sleep(0.05)
        return {"model": model, "content": "A response from the original LLM function."}

    with JAC(path=":memory:") as cache:
        cached_llm = cache.wrap(ask_llm)
        first = await cached_llm("Hello", model="my-model")
        second = await cached_llm("Hello", model="my-model")
        assert first == second
        assert generations == 1
        print(second)
        print(f"Original LLM calls: {generations}; repeat request: {cache.last_trace.path}")


if __name__ == "__main__":
    asyncio.run(main())
