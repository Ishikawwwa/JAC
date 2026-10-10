"""Run JevK5 on the same reuse cases, retrieval and JAC paths as the Laya comparison."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from .jevk5 import JevK5Decider
from .laya_benchmark import JEVK5_MODES, run
from .laya_prompt_benchmark import build_cases, score_cases, write_report
from .laya_prompts import DEFAULT_PROMPT_VERSION, PROMPT_VERSIONS, laya_questions


async def replay(decider, source, output, profiles):
    raw = source.read_bytes()
    original = json.loads(raw)
    entries, splits = build_cases(original)
    report = {"schema_version": "jac-jevk5-replay-v1", "completed": False,
              "original_report": source.as_posix(),
              "original_report_sha256": hashlib.sha256(raw).hexdigest(),
              "dataset_sha256": original["dataset_sha256"], "runs": {},
              "notes": ["Same BM25 candidates and complete evidence as original Laya audit.",
                        "Previously inspected regression cases, not a new blind holdout.",
                        "Fixed 0.9 cutoff is a comparison, not a deployment qualification.",
                        "JevK5 evaluates questions separately; Laya batches its independent heads.",
                        "GGUF reads one bounded answer token per question; "
                        "no final answer generation."]}
    await decider.preload()
    try:
        for profile in profiles:
            decider.prompt_version = profile
            await score_cases(decider, splits["development"][:1], entries, decider.threshold)
            decider.stats(reset=True)
            record = {"identity": dict(decider.cache_identity),
                      "questions": laya_questions(version=profile), "phases": {}}
            report["runs"][profile] = record
            for phase, cases in splits.items():
                record["phases"][phase] = await score_cases(
                    decider, cases, entries, decider.threshold)
                print(profile, phase, json.dumps(record["phases"][phase]["summary"]), flush=True)
                write_report(output, report)
            record["model_stats"] = decider.stats(reset=True)
        report["completed"] = True
        write_report(output, report)
        return report
    finally:
        await decider.drain()
        decider.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="jevk5-4b-v0.3-Q8_0.gguf")
    parser.add_argument("--revision")
    parser.add_argument("--temperature", type=float, default=1.22)
    parser.add_argument("--max-len", type=int, default=8192)
    parser.add_argument("--prompt-version", choices=PROMPT_VERSIONS, default=DEFAULT_PROMPT_VERSION)
    parser.add_argument("--replay-only", action="store_true")
    parser.add_argument("--replay-profiles", nargs="+", choices=PROMPT_VERSIONS,
                        default=[DEFAULT_PROMPT_VERSION, "laya-reuse-v3-bindings"])
    parser.add_argument("--source", type=Path,
                        default=Path("benchmark-results/prior-audit.json"))
    parser.add_argument("--threshold", type=float, default=.9)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--generation-ms", type=float, default=500)
    parser.add_argument("--candidate-limit", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--modes", nargs="+", choices=JEVK5_MODES, default=JEVK5_MODES)
    parser.add_argument("--embedding-revision",
                        default="1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
    parser.add_argument("--output", type=Path,
                        default=Path("benchmark-results/jevk5-comparison.json"))
    args = parser.parse_args()
    decider = JevK5Decider(model=args.model, base_url=args.base_url, revision=args.revision,
                          temperature=args.temperature, threshold=args.threshold,
                          max_len=args.max_len, max_candidates=args.candidate_limit,
                          prompt_version=args.prompt_version)
    if args.replay_only:
        asyncio.run(replay(decider, args.source, args.output, args.replay_profiles))
    else:
        try:
            asyncio.run(run(decider=decider, device="llama.cpp CUDA (external server)",
                            repeats=args.repeats, generation_ms=args.generation_ms,
                            candidate_limit=args.candidate_limit, output=args.output,
                            modes=() if args.audit_only else args.modes,
                            embedding_revision=args.embedding_revision))
        finally:
            decider.close()
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
