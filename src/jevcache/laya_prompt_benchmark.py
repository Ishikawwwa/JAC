"""Replay recorded Laya failures with frozen experimental prompt versions.

This is development/regression testing on previously inspected synthetic cases,
not a new blind accuracy evaluation or an end-to-end generation benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

from .decisions import Candidate
from .laya import LayaDecider
from .laya_benchmark import call_request, fixture_response, timing
from .laya_prompts import PROMPT_VERSIONS, laya_approval_values, laya_questions

CHECKPOINTS = {
    "multilingual": ("convaiinnovations/laya-multilingual",
                     "1720e3e3357cfe1e281542e223f8273b0890ca34"),
    "english": ("convaiinnovations/laya", "7b928d828b7b0e022f929d9bd2e44165aa270148"),
}
TARGETS = (
    "meeting:hard_negative:0", "invoice:hard_negative:0",
    "incomplete_retry:invalid_artifact:1",
)
POSITIVE_CONTROLS = (
    "meeting:positive:0", "invoice:positive:0",
    "translate:positive:1", "incident:positive:0",
)
NEGATIVE_CONTROLS = (
    "python_retry:hard_negative:0", "python_retry:hard_negative:1",
    "js_dates:hard_negative:0", "regex_filename:hard_negative:0",
)


def build_cases(original):
    """Freeze original BM25 candidates; no new retrieval or changing input corpus."""
    if not original.get("completed"):
        raise ValueError("original audit must be complete")
    if original["held_out_shadow"]["bm25"]["frozen_threshold"] != .9:
        raise ValueError("expected the original multilingual calibration cutoff 0.9")
    entries = {entry["id"]: entry for entry in original["dataset"]["held_out"]["entries"]}
    rows = original["held_out_shadow"]["bm25"]["queries"]
    if any(len(row["candidates"]) > 1 for row in rows):
        raise ValueError("replay expects candidate_limit=1")
    failures = original["held_out_shadow"]["bm25"]["frozen_configuration_quality"][
        "false_accept_query_ids"]
    development_ids = list(dict.fromkeys((*TARGETS, *failures,
                                         *POSITIVE_CONTROLS, *NEGATIVE_CONTROLS)))
    by_id = {row["id"]: row for row in rows}
    development = [by_id[identifier] for identifier in development_ids]
    return entries, {"development": development, "historical_regression": rows}


def summarize(rows):
    semantic = [row for row in rows if row["label"] != "exact"]
    true_accepts = sum(row["accepted"] and row["correct"] for row in semantic)
    false_accepts = sum(row["accepted"] and not row["correct"] for row in semantic)
    positives = sum(row["expected_id"] is not None for row in semantic)
    accepted = true_accepts + false_accepts
    return {
        "cases": len(rows), "semantic_cases": len(semantic),
        "positive_semantic_cases": positives,
        "true_semantic_accepts": true_accepts, "false_semantic_accepts": false_accepts,
        "semantic_precision": true_accepts / accepted if accepted else None,
        "positive_semantic_recall": true_accepts / positives if positives else None,
        "false_accept_ids": [row["id"] for row in semantic
                             if row["accepted"] and not row["correct"]],
        "missed_positive_ids": [row["id"] for row in semantic
                                if row["expected_id"] is not None and not row["correct"]],
        "native_predictions": sum(row["native_predict_calls"] for row in rows),
        "context_rejections": sum(row["context_rejections"] for row in rows),
        "errors": sum(row["error"] is not None for row in rows),
        "successful_decision_latency": timing([
            row["elapsed_ms"] for row in rows if row["native_predict_calls"] and not row["error"]
        ]),
        "cutoff_diagnostics": [
            {"threshold": cutoff,
             "true_accepts": sum(row["score"] is not None and row["score"] >= cutoff
                                 and row["candidate_id"] == row["expected_id"]
                                 for row in semantic),
             "false_accepts": sum(row["score"] is not None and row["score"] >= cutoff
                                  and row["candidate_id"] != row["expected_id"]
                                  for row in semantic)}
            for cutoff in (.5, .7, .8, .9, .95, .99)
        ],
    }


async def score_cases(decider, cases, entries, threshold):
    rows = []
    for case in cases:
        identifier = case["candidates"][0] if case["candidates"] else None
        source = entries.get(identifier)
        candidates = [Candidate(identifier, call_request(source["text"]),
                                fixture_response(source), 500)] if source else []
        before = decider.stats()
        start = time.perf_counter()
        error = None
        probabilities = {}
        accepted_id = None
        try:
            decision = await decider.decide(call_request(case["text"]), candidates)
            probabilities = dict(decision.probabilities)
            accepted_id = decision.candidate_id
        except Exception as exc:
            error = type(exc).__name__
        elapsed = (time.perf_counter() - start) * 1000
        await decider.drain()
        after = decider.stats()
        score = min(probabilities.values()) if probabilities else None
        required_values = laya_approval_values(version=decider.prompt_version)
        native_probabilities = {
            name: value if required_values[name.rsplit(":", 1)[1]] else 1 - value
            for name, value in probabilities.items()
        }
        rows.append({
            "id": case["id"], "label": case["label"], "family": case["family"],
            "new_request": call_request(case["text"]),
            "cached_request": call_request(source["text"]) if source else None,
            "artifact": fixture_response(source) if source else None,
            "candidate_id": identifier, "expected_id": case["expected_id"],
            "original_multilingual_score": case["scores"].get(identifier),
            "probabilities": probabilities, "native_p_true": native_probabilities,
            "score": score, "threshold": threshold,
            "accepted": accepted_id is not None,
            "correct": accepted_id == case["expected_id"], "error": error,
            "elapsed_ms": elapsed,
            "native_predict_calls": after["predict_calls"] - before["predict_calls"],
            "context_rejections": after["context_rejections"] - before["context_rejections"],
        })
    return {"summary": summarize(rows), "rows": rows}


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


async def run(original_path, output, models, profiles, threshold=.9, device="cuda"):
    source_bytes = original_path.read_bytes()
    original = json.loads(source_bytes)
    entries, splits = build_cases(original)
    report = {
        "schema_version": "jac-laya-prompt-refinement-v1", "completed": False,
        "original_report": original_path.as_posix(),
        "original_report_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "dataset_version": original["dataset_version"],
        "dataset_sha256": original["dataset_sha256"],
        "threshold": threshold, "models": {},
        "prompts": {profile: laya_questions("answer", profile) for profile in profiles},
        "approval_values": {profile: laya_approval_values(version=profile) for profile in profiles},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": {name: importlib.metadata.version(name) for name in
                                     ("laya", "torch", "transformers", "numpy")}},
        "notes": [
            "Prompt definitions were frozen before this run; cutoff fixed at 0.9.",
            "Original multilingual BM25 candidates and complete evidence are replayed unchanged.",
            "Development contains previously failed cases plus known positive/negative controls.",
            "Historical regression repeats previously inspected examples, not a new blind test.",
            "Accuracy summaries exclude exact requests; exact audit rows are retained.",
            "Real native Laya inference; no generation or embedding calls.",
            "Question count is recorded in prompt definitions; factorized uses four checks.",
            "Loading and per-profile warmup are outside measured decisions.",
            "Native questions run serially; no concurrent GPU tasks or throughput measurements.",
            "The SDK keeps v2 as default; these new profiles are opt-in experiments.",
            "Conflict questions approve on P(false); normalized and native scores are both saved.",
            "English at 0.9 is a diagnostic, not a previously qualified runtime policy.",
        ],
    }
    write_report(output, report)
    for model_name in models:
        model, revision = CHECKPOINTS[model_name]
        first = LayaDecider(model=model, revision=revision, device=device,
                            threshold=threshold, prompt_version=profiles[0])
        started = time.perf_counter()
        await first.preload()
        model_report = {"load_ms": (time.perf_counter() - started) * 1000,
                        "runs": {}}
        report["models"][model_name] = model_report
        try:
            # Every profile uses this same loaded checkpoint; all calls are serial.
            for profile in profiles:
                decider = first if profile == profiles[0] else LayaDecider(
                    model=model, revision=revision, device=device,
                    threshold=threshold, prompt_version=profile, agent=first.agent)
                try:
                    await score_cases(decider, splits["development"][:1], entries, threshold)
                    decider.stats(reset=True)
                    profile_report = {"identity": dict(decider.cache_identity), "phases": {}}
                    model_report["runs"][profile] = profile_report
                    for phase, cases in splits.items():
                        result = await score_cases(decider, cases, entries, threshold)
                        profile_report["phases"][phase] = result
                        summary = result["summary"]
                        decision_p50 = summary["successful_decision_latency"]["p50_ms"]
                        latency_label = (f"{decision_p50:.2f}ms"
                                         if decision_p50 is not None else "n/a")
                        print(f"{model_name} {profile} {phase}: "
                              f"{summary['true_semantic_accepts']} correct approvals, "
                              f"{summary['false_semantic_accepts']} incorrect; "
                              f"p50={latency_label}", flush=True)
                        if phase == "development":
                            for row in result["rows"]:
                                if row["id"] in TARGETS:
                                    print(f"  {row['id']}: score={row['score']} "
                                          f"accepted={row['accepted']}", flush=True)
                        write_report(output, report)
                finally:
                    if decider is not first:
                        await decider.drain()
                        decider.close()
        finally:
            await first.drain()
            first.close()
        write_report(output, report)
    report["completed"] = True
    write_report(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path,
                        default=Path("benchmark-results/prior-audit.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("benchmark-results/prompt-refinement.json"))
    parser.add_argument("--models", nargs="+", choices=CHECKPOINTS,
                        default=list(CHECKPOINTS))
    parser.add_argument("--profiles", nargs="+", choices=PROMPT_VERSIONS,
                        default=list(PROMPT_VERSIONS))
    parser.add_argument("--threshold", type=float, default=.9)
    parser.add_argument("--device", default="cuda")
    options = parser.parse_args()
    if not 0 < options.threshold <= 1:
        parser.error("threshold must be greater than zero and at most one")
    asyncio.run(run(options.original, options.output, options.models,
                    options.profiles, options.threshold, options.device))
    print(f"Saved prompt comparison: {options.output.resolve()}")


if __name__ == "__main__":
    main()
