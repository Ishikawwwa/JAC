"""Run with: python examples/portfolio.py (after installing the package)."""

import asyncio
import json

from jevcache.benchmark import run_demo


async def main() -> None:
    rows = await run_demo()
    print("Local synthetic operations; no live LLM, tool, or Jev calls.")
    for row in rows:
        trace = row["trace"]
        print(
            f"{row['label']}: {trace['path']}, "
            f"{trace['elapsed_ms']:.2f} ms; LLM={trace['llm_calls']}, "
            f"tools={trace['tool_calls']}, Jev={trace['jev_calls']}; "
            f"plan reused={trace['plan_hit']}, nodes cached={trace['cached_nodes']}"
        )
        print(json.dumps(row["value"], sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
