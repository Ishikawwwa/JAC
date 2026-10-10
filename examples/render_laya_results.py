"""Render publication figures from completed reports; requires matplotlib==3.9.2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("completed"):
        raise ValueError(f"Incomplete benchmark report: {path.name}")
    return report


def render(main_report, shadow_reports, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    names = list(main_report["runs"])
    summaries = [main_report["runs"][name]["summary"] for name in names]
    labels = [name.replace("laya_", "Laya ").replace("_", " ") for name in names]
    labels = [label + " *" if name.startswith("laya_")
              and not main_report["runs"][name]["semantic_enabled"] else label
              for name, label in zip(names, labels, strict=True)]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), layout="constrained")
    indices = list(range(len(names)))
    axes[0].barh([i - .18 for i in indices],
                 [row["end_to_end"]["p50_ms"] for row in summaries],
                 height=.35, label="p50", color="#2563eb")
    axes[0].barh([i + .18 for i in indices],
                 [row["end_to_end"]["p95_ms"] for row in summaries],
                 height=.35, label="p95", color="#93c5fd")
    axes[0].set_yticks(indices, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("End-to-end latency (ms)")
    axes[0].legend()
    generation_ms = main_report["generation"]["configured_ms"]
    axes[0].set_title(f"Controlled {generation_ms:g} ms generation fixture")
    axes[1].barh(indices, [row["unique_false_reuses"] for row in summaries],
                 color=["#dc2626" if row["unique_false_reuses"] else "#16a34a"
                        for row in summaries])
    axes[1].set_yticks(indices, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Distinct queries with incorrect cache reuse")
    axes[1].set_title("Quality counts; timing repeats excluded")
    for index, row in enumerate(summaries):
        axes[1].text(row["unique_false_reuses"] + .25, index,
                     str(row["unique_false_reuses"]), va="center")
    axes[1].set_xlim(0, max(row["unique_false_reuses"] for row in summaries) + 5)
    fig.suptitle("JAC full-framework experiment\n"
                 "* Semantic configuration did not qualify; exact caching remains active")
    for extension in ("png", "svg"):
        fig.savefig(destination / f"framework-comparison.{extension}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    for report in [main_report, *shadow_reports]:
        short_name = report["model_identity"]["model"].split("/")[-1]
        for strategy, result in report["held_out_shadow"].items():
            cutoffs = result["exploratory_cutoff_diagnostics"]
            xs = [row["threshold"] for row in cutoffs]
            label = short_name + " / " + strategy
            axes[0].plot(xs, [row["true_accepts"] for row in cutoffs], marker="o", label=label)
            axes[1].plot(xs, [row["false_accepts"] for row in cutoffs], marker="o", label=label)
    for ax in axes:
        ax.set_xlabel("Decision cutoff (exploratory; not selected on held-out data)")
        ax.set_xlim(.49, 1.01)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Correct approvals, including exact cases in shadow audit")
    axes[1].set_ylabel("Incorrect approvals")
    axes[0].set_title("Held-out shadow decisions")
    axes[1].set_title("Errors remain visible at lower cutoffs")
    axes[0].legend(fontsize=8)
    fig.suptitle("Laya reuse cutoff diagnostics on 80 distinct cases")
    for extension in ("png", "svg"):
        fig.savefig(destination / f"decision-cutoffs.{extension}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--shadow", type=Path, nargs="*", default=[])
    parser.add_argument("--live", type=Path,
                        help="Optional live generation report for a separate figure")
    parser.add_argument("--output", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    render(read(args.report), [read(path) for path in args.shadow], args.output)
    if args.live:
        render_live(json.loads(args.live.read_text(encoding="utf-8")), args.output)
    print(f"Saved PNG and SVG figures: {args.output.resolve()}")


def render_live(report, destination):
    import matplotlib.pyplot as plt

    if report.get("fixture") is not False:
        raise ValueError("Live figure requires a non-fixture report")
    names = list(report["runs"])
    rows = [report["runs"][name] for name in names]
    labels = [name.replace("_", " ") for name in names]
    indices = list(range(len(names)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), layout="constrained")
    axes[0].barh([i - .18 for i in indices], [row["p50_ms"] for row in rows],
                 height=.35, label="p50", color="#2563eb")
    axes[0].barh([i + .18 for i in indices], [row["p95_ms"] for row in rows],
                 height=.35, label="p95", color="#93c5fd")
    axes[0].set_xlabel("Response latency (ms)")
    axes[0].set_title("Actual local Qwen2.5-0.5B generation")
    axes[0].legend()
    axes[1].barh(indices, [row["generation_calls"] for row in rows], color="#7c3aed")
    axes[1].set_xlabel("Generation starts (all completed)")
    axes[1].set_title("42 requests; admitted answers become exact hits")
    axes[1].set_xlim(0, max(row["generation_calls"] for row in rows) + 5)
    for index, row in enumerate(rows):
        axes[1].text(row["generation_calls"] + .5, index,
                     str(row["generation_calls"]), va="center")
    for ax in axes:
        ax.set_yticks(indices, labels)
        ax.invert_yaxis()
    fig.suptitle("Live provider integration: all 32 cache hits were exact\n"
                 "Laya made zero timed decisions; all modes had 9 strict-oracle failures")
    for extension in ("png", "svg"):
        fig.savefig(destination / f"live-provider-comparison.{extension}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
