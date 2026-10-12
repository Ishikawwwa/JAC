"""Run controlled support, extraction and tool-informed agent requests without keys.

For your actual models use the CLI's --config or --factory instead:
    python -m jac evaluate-agents --factory my_models:build --repeats 3
The factory can mix local and remote generation, embeddings and decisions.
"""

import asyncio

from jevcache.agent_evaluate import run_agent_evaluation


async def main():
    # The offline generator, decisions and vectors are synthetic fixtures.
    # They verify integration and correctness checks; timings are not model results.
    report = await run_agent_evaluation(offline=True, repeats=1)
    # In your application replace offline=True with:
    # generation=your_llm_call, generation_identity={"model": "name", "version": "1"},
    # decider=your_decider, embedder=your_embedder.
    print("OFFLINE FIXTURES: simulated tools/models; not a model benchmark.")
    print(f"{'Variant':<25} {'Requests':>8} {'Gen starts':>10} {'Decisions':>9} {'Errors':>6}")
    for name, result in report["runs"].items():
        print(f"{name:<25} {result['requests']:>8} {result['generation_calls']:>10} "
              f"{result['decision_calls']:>9} {result['incorrect_responses']:>6}")
    print("Tools execute before the cached LLM call; parallel starts may still be billed.")


if __name__ == "__main__":
    asyncio.run(main())
