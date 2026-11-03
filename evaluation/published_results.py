"""Check published aggregate results offline; this does not certify judge accuracy."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

from .core import sha256
from .datasets import load_collection

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIRECTORY = Path("evaluation/results/nemotron-lightning")
DIAGNOSTICS = ("judge_invalid", "judge_section_id_checked", "judge_section_mismatch",
               "empty_final", "length_terminated")
PAIRED_FIELDS = ("valid_pairs", "paired_delta", "paired_item_ci95_low", "paired_item_ci95_high",
                 "section_clusters", "paired_cluster_ci95_low", "paired_cluster_ci95_high")
REVIEW_STATUS = "automatic_scores_with_known_judge_errors"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _inside(directory, name):
    relative = Path(name)
    path = (directory / relative).resolve()
    _require(not relative.is_absolute() and ".." not in relative.parts
             and path.is_relative_to(directory.resolve()), "Artifact path escapes its directory")
    return path


def _integer(value, low, high):
    number = int(value)
    _require(str(number) == str(value) and low <= number <= high,
             f"Invalid integer or count: {value}")
    return number


def _fraction(value):
    number = float(value)
    _require(math.isfinite(number) and -1 <= number <= 1, f"Invalid fraction: {value}")
    return number


def _equal(left, right, message):
    _require(math.isclose(left, right, rel_tol=0, abs_tol=1e-12), message)


def verify(directory=None, root=ROOT):
    """Verify hashes, source bindings, coverage and arithmetic, without model clients.

    Row-level evidence is local. Intervals are checked for bounds/coverage, not
    reconstructed, and known judge errors remain part of the primary scores.
    """
    root = Path(root).resolve()
    directory = (root / RESULT_DIRECTORY) if directory is None else Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    _require(manifest["schema_version"] == 1 and manifest["kind"] == "aggregate_policy_evaluation",
             "Unsupported published result manifest")
    _require(manifest["review_status"] == REVIEW_STATUS
             and manifest["primary_scores_changed"] is False
             and manifest["score_validity_established"] is False,
             "Known judge limitations must remain explicit")
    _require(set(manifest["artifacts"]) == {"metrics.csv", "judge_findings.json", "README.md"},
             "Unexpected aggregate artifacts")
    for name, binding in manifest["artifacts"].items():
        _require(binding["path"] == name and sha256(_inside(directory, name)) == binding["sha256"],
                 f"Artifact checksum changed: {name}")

    published_data = json.loads((root / "data/evaluation/manifest.json").read_text())["collections"]
    training_data = json.loads((root / "data/variants/manifest.json").read_text())["collections"]
    collections, models = manifest["collections"], manifest["models"]
    model_order, benchmark_order = manifest["model_order"], manifest["benchmark_order"]
    _require(len(model_order) == len(set(model_order)) == len(models) == manifest["model_count"]
             and set(model_order) == set(models) and model_order[0] == "base", "Invalid model order")
    _require(len(benchmark_order) == len(set(benchmark_order)) == len(collections) == manifest["benchmark_count"]
             and set(benchmark_order) == set(collections) == set(published_data), "Invalid benchmark order")
    records = {}
    for name, binding in collections.items():
        source = published_data[name]
        _require(binding["path"] == f"data/evaluation/{source['path']}"
                 and binding["sha256"] == source["sha256"] and binding["rows"] == source["rows"]
                 and binding["source_version"] == source["source"]["version"]
                 and binding["origin"] == source["kind"], f"Dataset binding changed: {name}")
        records[name] = load_collection(name, root / "data/evaluation")
    _require(manifest["sft"]["num_epochs"] == 1, "Unexpected SFT epoch count")
    batches = manifest["sft"]["batch_sizes"]
    _require(len(batches) == len(set(batches)) and all(type(b) is int and b > 0 for b in batches),
             "Invalid SFT batch sizes")
    _require(set(models) == {"base"} | {f"{c}-b{b}" for c in collections for b in batches},
             "Incomplete source/batch model coverage")
    _require(models["base"] == {"kind": "unadapted_vendor_base", "training_collection": None,
                                "batch_size": None}, "Invalid unadapted base definition")
    checked_training = set()
    for name, info in models.items():
        if name == "base":
            continue
        collection, batch = info["training_collection"], info["batch_size"]
        source = training_data[collection]["reviewed"]
        binding = info["training_data"]
        _require(info["kind"] == "lora_sft" and name == f"{collection}-b{batch}"
                 and binding["path"] == f"data/variants/{source['path']}"
                 and binding["sha256"] == source["sha256"] and info["training_rows"] == source["rows"]
                 and info["optimizer_steps"] == math.ceil(source["rows"] / batch),
                 f"Training binding changed: {name}")
        if collection not in checked_training:
            path = _inside(root, binding["path"])
            _require(sha256(path) == binding["sha256"]
                     and sum(bool(line.strip()) for line in path.read_bytes().splitlines()) == source["rows"],
                     f"Training file changed: {collection}")
            checked_training.add(collection)

    with (directory / "metrics.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = {"benchmark", "model", "training_collection", "batch_size", "passed", "total",
                  "pass_rate", "primary_delta_vs_base", *DIAGNOSTICS, *PAIRED_FIELDS}
        _require(set(reader.fieldnames or []) == fields, "Unexpected metric columns")
        rows = list(reader)
    indexed = {(row["benchmark"], row["model"]): row for row in rows}
    expected = {(benchmark, model) for benchmark in collections for model in models}
    _require(len(rows) == len(indexed) == len(expected) == manifest["metric_rows"]
             and set(indexed) == expected, "Incomplete or duplicate model/dataset cells")
    diagnostics = {name: dict.fromkeys(DIAGNOSTICS, 0) for name in collections}
    pairs_excluding_zero = 0
    for (benchmark, model), row in indexed.items():
        _require(set(row) == fields, "Malformed metric row")
        total = _integer(row["total"], 1, collections[benchmark]["rows"])
        _require(total == collections[benchmark]["rows"], "Wrong metric denominator")
        passed = _integer(row["passed"], 0, total)
        rate = _fraction(row["pass_rate"])
        _equal(rate, passed / total, "Pass rate differs from counts")
        base_rate = _fraction(indexed[benchmark, "base"]["pass_rate"])
        _equal(_fraction(row["primary_delta_vs_base"]), rate - base_rate, "Wrong primary delta")
        info = models[model]
        _require(row["training_collection"] == (info["training_collection"] or "")
                 and row["batch_size"] == (str(info["batch_size"]) if info["batch_size"] else ""),
                 "Metric/model binding changed")
        counts = {field: _integer(row[field], 0, total) for field in DIAGNOSTICS}
        _require(counts["judge_section_mismatch"] <= counts["judge_section_id_checked"],
                 "Identifier mismatches exceed checked replies")
        for field, value in counts.items():
            diagnostics[benchmark][field] += value
        if model == "base":
            _require(all(row[field] == "" for field in PAIRED_FIELDS), "Base must not have paired intervals")
            continue
        pairs = _integer(row["valid_pairs"], 1, total)
        base_invalid = int(indexed[benchmark, "base"]["judge_invalid"])
        _require(total - counts["judge_invalid"] - base_invalid <= pairs
                 <= total - max(counts["judge_invalid"], base_invalid), "Invalid paired coverage")
        _integer(row["section_clusters"], 1,
                 min(pairs, len({r["metadata"]["section_id"] for r in records[benchmark]})))
        delta = _fraction(row["paired_delta"])
        _equal(delta * pairs, round(delta * pairs), "Paired delta does not describe binary results")
        if pairs == total:
            _equal(delta, rate - base_rate, "Complete-pair delta differs from primary delta")
        for unit in ("item", "cluster"):
            low, high = (_fraction(row[f"paired_{unit}_ci95_{edge}"]) for edge in ("low", "high"))
            _require(low <= high, "Reversed interval bounds")
            if unit == "cluster" and not low <= 0 <= high:
                pairs_excluding_zero += 1
    _require(diagnostics == manifest["aggregate_diagnostics"], "Aggregate diagnostics differ from CSV")
    sample_count = sum(info["rows"] for info in collections.values())
    new_count = sum(info["rows"] for info in collections.values() if info["origin"] == "synthetic_reviewed")
    _require(sample_count == manifest["samples_per_model"]
             and sample_count * len(models) == manifest["model_item_results"]
             and new_count * len(models) == manifest["new_model_item_results"]
             and (sample_count - new_count) * len(models) == manifest["reused_official_model_item_results"]
             and 2 * new_count * len(models) == manifest["new_physical_model_calls"], "Result totals differ")
    new_diagnostics = manifest["new_evaluation_diagnostics"]
    for source_field, field in (("invalid_critique_json", "judge_invalid"), ("length_terminated", "length_terminated"),
                                ("empty_final_answers", "empty_final"), ("section_id_mismatches", "judge_section_mismatch")):
        _require(new_diagnostics[source_field] == sum(diagnostics[n][field] for n, c in collections.items()
                                                     if c["origin"] == "synthetic_reviewed"),
                 "New evaluation diagnostics differ")
    findings = json.loads((directory / "judge_findings.json").read_text())
    _require(findings["review_status"] == REVIEW_STATUS and findings["primary_scores_changed"] is False
             and findings["independent_human_certification"] is False, "Judge findings lost their review scope")
    sections = findings["section_identifiers"]
    _require(sections["new_evaluation"]["mismatched_replies"] == new_diagnostics["section_id_mismatches"]
             and sections["reused_official"]["mismatched_replies"] == sum(
                 diagnostics[n]["judge_section_mismatch"] for n, c in collections.items() if c["origin"] == "upstream")
             and sections["total_mismatched_replies"] == sum(d["judge_section_mismatch"] for d in diagnostics.values()),
             "Identifier findings differ from diagnostics")
    control = findings["control_protocol_case"]
    _require(control["replies_reviewed"] == len(models)
             and control["labels_inconsistent_with_rubric"] == control["false_automatic_passes"] + control["false_automatic_failures"]
             and 0 < control["labels_inconsistent_with_rubric"] <= control["replies_reviewed"], "Invalid control-case counts")
    for finding in [control, *findings["other_selected_findings"]]:
        _require(finding["case_id"] in {r["id"] for r in records[finding["collection"]]},
                 "Judge finding references a missing public case")
    readme = (directory / "README.md").read_text()
    best_passed = {b: max(int(indexed[b, model]["passed"]) for model in model_order) for b in benchmark_order}
    for model in model_order:
        info = models[model]
        label = "Unadapted vendor base | —" if model == "base" else f"{info['training_collection']} | {info['batch_size']}"
        cells = []
        for benchmark in benchmark_order:
            row = indexed[benchmark, model]
            cell = f"{float(row['pass_rate']) * 100:.2f}% ({row['passed']}/{row['total']})"
            cells.append(f"**{cell}**" if int(row["passed"]) == best_passed[benchmark] else cell)
        _require(f"| {label} | " + " | ".join(cells) + " |" in readme, "Published table differs from CSV")
    contrast_count = len(collections) * (len(models) - 1)
    _require(f"{contrast_count - pairs_excluding_zero} of the {contrast_count} section-cluster paired 95% intervals include zero"
             in readme, "Published interval summary differs from CSV")
    return {"integrity_verified": True, "model_count": len(models), "benchmark_count": len(collections),
            "metric_rows": len(rows), "model_item_results": manifest["model_item_results"],
            "review_status": REVIEW_STATUS, "score_validity_established": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository checkout containing released data")
    parser.add_argument("--directory", type=Path, help="Aggregate result directory (default: inside --root)")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(verify(args.directory, args.root), indent=2))
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"Published result error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
