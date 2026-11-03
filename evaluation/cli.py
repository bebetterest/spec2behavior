"""Public data preparation is offline with respect to all model providers."""

import argparse
import asyncio
import json
from pathlib import Path
import sys


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    sub = command.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Download public assets, screen overlap, and estimate costs; no model API calls")
    prepare.add_argument("--training-run", action="append", required=True, metavar="[NAME=]PATH",
                         help="Completed SFT run; repeat for multiple checkpoints. Base is included once.")
    prepare.add_argument("--config", type=Path, default=Path(__file__).with_name("config.example.json"))
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--manual-findings", type=Path, help="Optional reviewed overlap finding or list, with eval_id and benchmark")
    prepare.add_argument("--benchmark", action="append", choices=["model_spec", "xstest_safe", "xstest_unsafe", "strong_reject"],
                         help="Freeze only selected benchmarks; repeat to select several. Default: all.")
    run = sub.add_parser("run", help="Print a frozen run plan; add --execute only after reviewing and authorizing costs")
    run.add_argument("--plan", type=Path, required=True, help="Prepared directory containing plan.json")
    run.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    run.add_argument("--output", type=Path)
    run.add_argument("--execute", action="store_true")
    compare = sub.add_parser("compare", help="Read saved Inspect logs and compare matched ids, without API calls")
    compare.add_argument("--plan", type=Path, required=True)
    compare.add_argument("--run", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return command


def training_runs(values):
    result = {}
    for index, value in enumerate(values):
        name, separator, path = value.partition("=")
        if not separator:
            name, path = ("sft" if len(values) == 1 else f"sft_{index + 1}"), value
        if not name or not path or name in result:
            raise ValueError("--training-run needs a unique non-empty name and path")
        result[name] = Path(path).resolve()
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        if args.command == "prepare":
            from .runner import prepare
            plan = asyncio.run(prepare(root, training_runs(args.training_run), args.config.resolve(), args.output.resolve(),
                                       args.manual_findings, args.benchmark))
            print(json.dumps({"status": "prepared_without_model_inference", "plan_sha256": plan["plan_sha256"],
                              "review": str(args.output.resolve() / "REVIEW.md"), "samples_per_target": sum(plan["lock"]["dataset_counts"].values())}, indent=2))
        elif args.command == "run":
            from .runner import execute, verify_plan
            prepared = args.plan.resolve()
            plan = verify_plan(root, prepared)
            costs = json.loads((prepared / "costs.json").read_text())["profiles"][args.profile]
            if not args.execute:
                print(json.dumps({"status": "preview_only_no_model_clients", "profile": args.profile,
                    "plan_sha256": plan["plan_sha256"], "judge": plan["config"]["judge"]["model"],
                    "model_spec_grading": plan["config"]["judge"]["model_spec_grading"], "targets": list(plan["targets"]),
                    "samples_per_target": sum(row["samples_per_model"] for row in costs["benchmarks"].values()),
                    "target_calls": costs["target_calls"], "judge_calls_max": costs["judge_calls_max"],
                    "estimated_usd": {row["name"]: {"total": round(row["total_usd"], 2),
                        "with_contingency": round(row["with_review_contingency_usd"], 2)} for row in costs["scenarios"]},
                    "config": str(prepared / "config.json"), "review": str(prepared / "REVIEW.md")}, indent=2))
                return 0
            if args.output is None:
                raise ValueError("--execute requires a fresh --output directory")
            run = asyncio.run(execute(root, prepared, args.output.resolve(), args.profile))
            print(json.dumps(run, indent=2))
        else:
            from .compare import compare_run
            report = compare_run(args.plan.resolve(), args.run.resolve(), args.output.resolve())
            print(json.dumps({name: {target: comparison["paired"] for target, comparison in row["comparisons"].items()}
                              for name, row in report["benchmarks"].items()}, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Evaluation error: {error}", file=sys.stderr)
        return 2
