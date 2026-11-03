"""CLI for preparation, bounded construction, review and Inspect-format export."""
import argparse
import asyncio
import json
from pathlib import Path
import sys
from evaluation.core import write_json
from synthesis.io import read_jsonl
from .pipeline import ROOT, export, prepare, run, verify


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Freeze sources, sections, code and API limits; no paid calls")
    prep.add_argument("--config", type=Path, default=Path(__file__).with_name("config.example.json"))
    prep.add_argument("--root", type=Path, default=Path.cwd())
    prep.add_argument("--output", type=Path, required=True)
    execute = sub.add_parser("run", help="Preview a plan; --execute makes the API calls")
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--execute", action="store_true")
    review = sub.add_parser("review-template", help="Write pending, hash-bound decisions without inventing labels")
    review.add_argument("--run", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    out = sub.add_parser("export", help="Export explicitly reviewed cases; omit probes and policy evidence from target inputs")
    out.add_argument("--run", type=Path, required=True)
    out.add_argument("--decisions", type=Path, required=True)
    out.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            plan = prepare(args.root.resolve(), args.config.resolve(), args.output.resolve())
            print(json.dumps(dict(state="prepared_without_inference", plan_sha256=plan["plan_sha256"],
                jobs=len(plan["jobs"]), max_calls=plan["max_calls"], max_estimated_usd=plan["config"]["max_estimated_usd"]), indent=2))
        elif args.command == "run":
            plan = verify(args.plan.resolve())
            if not args.execute:
                print(json.dumps(dict(state="preview_only", plan_sha256=plan["plan_sha256"], jobs=plan["jobs"],
                    max_calls=plan["max_calls"], config=plan["config"], no_automatic_retries=True), indent=2))
            else:
                rows = asyncio.run(run(args.plan.resolve()))
                print(json.dumps(dict(state="needs_review", candidates=len(rows))))
        elif args.command == "review-template":
            verify(args.run.resolve())
            if args.output.exists():
                raise ValueError("Decision output already exists")
            rows = [dict(id=row["id"], candidate_sha256=row["candidate_sha256"], decision="uncertain",
                reviewer="Name the reviewer", reviewer_type="human", rationale="Review source, rubric, probes and overlap first",
                probe_labels={}) for row in read_jsonl(args.run / "candidates.jsonl")]
            write_json(args.output, rows)
            print(json.dumps(dict(state="pending_review", candidates=len(rows))))
        else:
            rows = export(args.run.resolve(), args.decisions.resolve(), args.output.resolve())
            print(json.dumps(dict(state="exported", rows=len(rows), output=str(args.output.resolve()))))
        return 0
    except (ValueError, OSError, KeyError) as error:
        print(f"Evaluation construction error: {error}", file=sys.stderr)
        return 2
