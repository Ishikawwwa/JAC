"""Show real JevK5 reuse decisions when avoided generation is deliberately slow.

Generation is a 5-second cancellable fixture, not an LLM benchmark. Requires the
separate calibration report produced by jevcache.jevk5_benchmark and llama-server.
"""

import argparse
import asyncio
import hashlib
import json
import tempfile
from pathlib import Path

from jac import JevK5Decider
from jevcache.laya_benchmark import measure_mode
from jevcache.laya_workload import workload


async def run(args):
    raw = args.calibration.read_bytes()
    source = json.loads(raw)
    if not source.get("completed") or source["model_identity"]["provider"] != "jevk5-gguf":
        raise ValueError("provide a completed JevK5 calibration/comparison report")
    identity = source["model_identity"]
    model = JevK5Decider(base_url=args.base_url, model=identity["model"],
                        revision=identity["revision"], temperature=identity["temperature"],
                        prompt_version=identity["question_schema_version"])
    await model.preload()
    selected_ids = {"meeting:positive:0", "invoice:positive:0", "meeting:hard_negative:0"}
    split = workload()["held_out"]
    split = {"entries": split["entries"],
             "queries": [query for query in split["queries"] if query["id"] in selected_ids]}
    report = {"completed": False, "generation": "5-second cancellable fixture; no LLM",
              "cases": "two known valid paraphrases and one swapped-owner negative control",
              "calibration_sha256": hashlib.sha256(raw).hexdigest(),
              "calibration_source": args.calibration.as_posix(), "runs": {},
              "notes": ["Calibration observations are reused unchanged; "
                        "no new labels tune cutoffs.",
                        "A selected demonstration, not an independent accuracy evaluation.",
                        "This slow fixture illustrates a different savings gate "
                        "than the 500ms run."]}
    try:
        with tempfile.TemporaryDirectory(prefix="jac-jevk5-demo-") as directory:
            for name in ("no_cache", "jevk5_bm25_cache_first", "jevk5_bm25_parallel"):
                report["runs"][name] = await measure_mode(
                    name, split, source["calibration"], model, None, None,
                    5000, 5000, 1, 1, Path(directory))
        report["completed"] = True
        report["model_identity"] = dict(model.cache_identity)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    finally:
        await model.drain()
        model.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8095")
    parser.add_argument("--calibration", type=Path,
                        default=Path("benchmark-results/jevk5-comparison.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("benchmark-results/jevk5-reuse-demo.json"))
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
