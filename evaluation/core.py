"""Read-only public asset preparation and upstream task construction.

Upstream tasks and data are pinned. Model Spec supports either the upstream
full-policy judge or a scenario-rubric judge with the same verdict scale.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import sys
import sysconfig
import zipfile
from pathlib import Path
from unittest.mock import patch

VERSIONS = {
    "inspect-ai": "0.3.282", "inspect-evals": "0.25.0",
    "tinker": "0.34.0", "tinker-cookbook": "0.5.7",
}
SPEC_DATA_REVISION = "89c260767c5a0a3a91a06dec77e3ab0306de768f"
SPEC_CODE_REVISION = "1d5df6d651c1e33161e507c8890af7ae00218941"
SPEC_REVISION = "8d2405d4e9a5143c5355f50482394ebd12fe898c"
XSTEST_REVISION = "f1d713187c61b6ae64e602d74f0b3d812cc2e8e8"
XSTEST_AUTHOR_REVISION = "d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d"
STRONG_REVISION = "3432b2d696b428f242bd507df96d80f686571d5e"
SPEC_URL = f"https://raw.githubusercontent.com/openai/model_spec/{SPEC_REVISION}/model_spec.md"
ASSETS = {
    "model_spec_dataset.zip": f"https://codeload.github.com/openai/model_spec_dataset/zip/{SPEC_DATA_REVISION}",
    "grader_model_spec.md": SPEC_URL,
    "xstest.csv": f"https://raw.githubusercontent.com/paul-rottger/xstest/{XSTEST_AUTHOR_REVISION}/xstest_prompts.csv",
    "xstest_README.md": f"https://raw.githubusercontent.com/paul-rottger/xstest/{XSTEST_AUTHOR_REVISION}/readme.md",
    "strong_reject.csv": f"https://raw.githubusercontent.com/alexandrasouly/strongreject/{STRONG_REVISION}/strongreject_dataset/strongreject_dataset.csv",
}
EXPECTED_COUNTS = {"model_spec": 587, "xstest_safe": 250, "xstest_unsafe": 200, "strong_reject": 313}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_prepared_plan(output: Path) -> dict:
    """Verify frozen review inputs without loading model clients or runtime packages."""
    plan = json.loads((output / "plan.json").read_text())
    digest = plan.pop("plan_sha256")
    if canonical_hash(plan) != digest:
        raise ValueError("Frozen plan changed; prepare and review it again")
    plan["plan_sha256"] = digest
    for path, expected in plan["review_artifacts"].items():
        if sha256(output / path) != expected:
            raise ValueError(f"Review artifact changed: {path}")
    return plan


def check_versions() -> dict[str, str]:
    installed = {name: importlib.metadata.version(name) for name in VERSIONS}
    if installed != VERSIONS:
        raise ValueError(f"Evaluation dependency versions changed: expected {VERSIONS}, got {installed}")
    return installed


def validate_config(config: dict) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported evaluation config schema")
    required = {
        "judge": {"model_spec_grading", "model", "renderer", "context_window", "temperature", "top_p", "max_tokens", "samples", "max_connections"},
        "sampling": {"temperature", "top_p", "seed", "max_tokens", "context_window", "epochs", "max_connections"},
        "execution": {"max_samples", "max_retries", "timeout_seconds", "pilot_min_per_task"},
        "pricing": {"checked_date", "judge_input_usd_per_million", "judge_output_usd_per_million", "judge_source", "target_input_usd_per_million", "target_output_usd_per_million", "target_source", "contingency_fraction", "scenarios"}}
    for group, keys in required.items():
        if not isinstance(config.get(group), dict):
            raise ValueError(f"Missing or invalid config group: {group}")
        if keys - config[group].keys():
            raise ValueError(f"Missing {group} keys: {sorted(keys - config[group].keys())}")
    for group, keys in {
        "judge": ["max_tokens", "samples", "max_connections", "context_window"],
        "sampling": ["max_tokens", "epochs", "max_connections", "context_window"],
        "execution": ["max_samples", "timeout_seconds", "pilot_min_per_task"],
    }.items():
        for key in keys:
            value = config[group][key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{group}.{key} must be a positive integer")
    if config["judge"]["samples"] != 1 or config["sampling"]["epochs"] != 1:
        raise ValueError("This paired protocol uses one candidate and one judge sample per item")
    if config["execution"]["max_retries"] != 0:
        raise ValueError("This protocol disables automatic retries")
    if config["judge"]["model_spec_grading"] not in {"rubric-only", "full-spec"}:
        raise ValueError("judge.model_spec_grading must be rubric-only or full-spec")
    allowed = {"schema_version": None,
               "judge": {"model_spec_grading", "model", "renderer", "context_window", "temperature", "top_p", "max_tokens", "samples", "max_connections", "tokenizer_revision"},
               "sampling": {"temperature", "top_p", "seed", "max_tokens", "context_window", "epochs", "max_connections"},
               "execution": {"max_samples", "max_retries", "timeout_seconds", "pilot_min_per_task"},
               "pricing": {"checked_date", "judge_input_usd_per_million", "judge_output_usd_per_million", "judge_source", "target_input_usd_per_million", "target_output_usd_per_million", "target_source", "target_pricing_note", "contingency_fraction", "scenarios"}}
    if config.keys() - allowed.keys():
        raise ValueError(f"Unknown config keys: {config.keys() - allowed.keys()}")
    for group in ["judge", "sampling", "execution", "pricing"]:
        unknown = config[group].keys() - allowed[group]
        if unknown:
            raise ValueError(f"Unknown {group} keys: {unknown}")
    for group in ["judge", "sampling"]:
        for key, low, high in [("temperature", 0, 2), ("top_p", 0, 1)]:
            value = config[group][key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high or (key == "top_p" and value == 0):
                raise ValueError(f"Invalid {group}.{key}")
        if config[group]["max_tokens"] >= config[group]["context_window"]:
            raise ValueError(f"{group}.max_tokens must leave room for input in context_window")
    if not isinstance(config["sampling"]["seed"], int) or isinstance(config["sampling"]["seed"], bool):
        raise ValueError("sampling.seed must be an integer")
    if not all(isinstance(config["judge"][k], str) and config["judge"][k].strip() for k in ["model", "renderer"]):
        raise ValueError("judge.model and judge.renderer must be non-empty strings")
    for key in ["judge_input_usd_per_million", "judge_output_usd_per_million", "target_input_usd_per_million", "target_output_usd_per_million"]:
        value = config["pricing"][key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or value < 0 or not math.isfinite(value):
            raise ValueError(f"Invalid price: {key}")
    margin = config["pricing"]["contingency_fraction"]
    if isinstance(margin, bool) or not isinstance(margin, (float, int)) or not math.isfinite(margin) or margin < 0:
        raise ValueError("pricing.contingency_fraction must be finite and non-negative")
    scenarios = config["pricing"]["scenarios"]
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("pricing.scenarios must be a non-empty list")
    names = set()
    for scenario in scenarios:
        fields = {"name", "target_total_output_tokens", "target_final_answer_tokens", "judge_total_output_tokens"}
        if not isinstance(scenario, dict) or scenario.keys() != fields:
            raise ValueError(f"Each cost scenario needs exactly these keys: {sorted(fields)}")
        if not isinstance(scenario["name"], str) or not scenario["name"] or scenario["name"] in names:
            raise ValueError("Cost scenario names must be unique non-empty strings")
        names.add(scenario["name"])
        for key in ["target_total_output_tokens", "target_final_answer_tokens", "judge_total_output_tokens"]:
            if not isinstance(scenario[key], int) or isinstance(scenario[key], bool) or scenario[key] < 0:
                raise ValueError(f"Invalid scenario token count: {key}")


def public_get(url):
    import httpx

    try:
        response = httpx.get(url, timeout=60, follow_redirects=True)
    except httpx.ConnectError:
        # Public assets only: permit direct access if a local proxy is offline.
        response = httpx.get(url, timeout=60, follow_redirects=True, trust_env=False)
    response.raise_for_status()
    return response


def prepare_judge_tokenizer(config, output):
    """Download tokenizer/config data only; freeze the public HF commit and bytes."""
    model = config["judge"]["model"].split(":", 1)[0]
    directory = output / "judge_tokenizer"
    manifest = directory / "manifest.json"
    requested = config["judge"].get("tokenizer_revision")
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved["model"] == model and (not requested or requested == saved["revision"]):
            if any(sha256(output / saved["path"] / name) != digest for name, digest in saved["files"].items()):
                raise ValueError("Cached judge tokenizer changed")
            return saved
    info = public_get(f"https://huggingface.co/api/models/{model}" + (f"/revision/{requested}" if requested else "")).json()
    revision = info["sha"]
    allowed = {"config.json", "tokenizer_config.json", "tokenizer.json", "tokenizer.model", "special_tokens_map.json", "chat_template.jinja", "vocab.json", "merges.txt", "tekken.json", "tiktoken.model"}
    files = sorted(s["rfilename"] for s in info["siblings"] if s["rfilename"] in allowed)
    # Separate revisions/models so changing a judge never leaves mixed tokenizer files.
    directory = output / "tokenizers" / model.replace("/", "--") / revision
    directory.mkdir(parents=True, exist_ok=True)
    for name in files:
        path = directory / name
        if not path.exists():
            path.write_bytes(public_get(f"https://huggingface.co/{model}/resolve/{revision}/{name}").content)
    saved = {"model": model, "revision": revision, "path": str(directory.relative_to(output)),
             "files": {name: sha256(directory / name) for name in files}}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    write_json(manifest, saved)
    return saved


def download_assets(data_dir: Path) -> dict:
    data_dir.mkdir(parents=True, exist_ok=True)
    prior = json.loads((data_dir / "assets.json").read_text()) if (data_dir / "assets.json").exists() else {}
    manifest = {}
    for filename, url in ASSETS.items():
        dest = data_dir / filename
        if not dest.exists():
            dest.write_bytes(public_get(url).content)
        manifest[filename] = {"url": url, "sha256": sha256(dest), "bytes": dest.stat().st_size}
        if prior and manifest[filename] != prior.get(filename):
            raise ValueError(f"Frozen asset changed: {dest}")
    dataset_dir = data_dir / "model_spec_dataset"
    dataset_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(data_dir / "model_spec_dataset.zip") as archive:
        for member in archive.namelist():
            parts = member.split("/")
            if len(parts) == 3 and parts[1] == "dataset" and parts[2].endswith(".json"):
                (dataset_dir / parts[2]).write_bytes(archive.read(member))
    write_json(data_dir / "assets.json", manifest)
    return manifest


def spec_modules(root: Path, data_dir: Path):
    """Bind upstream resource reads to the identical pinned local bytes."""
    source = root / "model_spec_evals/src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    import inspect_ai.util

    def local_resource(resource):
        if resource == SPEC_URL:
            return (data_dir / "grader_model_spec.md").read_text()
        if resource == "grader_prompt_template.md":
            return (source / "model_spec_evals/grader_prompt_template.md").read_text()
        raise ValueError(f"Unexpected upstream grader resource: {resource}")

    with patch.object(inspect_ai.util, "resource", side_effect=local_resource):
        tasks = importlib.import_module("model_spec_evals.tasks")
        scorers = importlib.import_module("model_spec_evals.scorers")
    if scorers.model_spec != (data_dir / "grader_model_spec.md").read_text():
        raise ValueError("Model Spec grader source differs from the locked asset")
    return tasks, scorers


def make_tasks(root: Path, data_dir: Path, judge, model_spec_grading="rubric-only", benchmarks=None) -> dict:
    selected = list(EXPECTED_COUNTS) if benchmarks is None else benchmarks
    if (not isinstance(selected, list) or not selected
            or any(name not in EXPECTED_COUNTS for name in selected)
            or len(selected) != len(set(selected))):
        raise ValueError("Select unique known benchmarks; the selection cannot be empty")
    from inspect_ai import Task
    from inspect_ai.dataset import MemoryDataset

    spec, _ = spec_modules(root, data_dir)
    xs = importlib.import_module("inspect_evals.xstest.xstest")
    strong = importlib.import_module("inspect_evals.strong_reject.strong_reject")
    if xs.XSTEST_DATASET_REVISION != XSTEST_REVISION or STRONG_REVISION not in strong.PATH_TO_STRONG_REJECT_CSV:
        raise ValueError("Upstream benchmark dataset revision changed")
    with (data_dir / "xstest.csv").open(newline="") as handle:
        xs_records = list(csv.DictReader(handle))
    with (data_dir / "strong_reject.csv").open(newline="") as handle:
        strong_records = list(csv.DictReader(handle))

    def local_xstest(*args, **kwargs):
        return MemoryDataset([kwargs["sample_fields"](record) for record in xs_records], name="XSTest")

    def local_strong(*args, **kwargs):
        return MemoryDataset([kwargs["sample_fields"](record) for record in strong_records], name="StrongREJECT")

    tasks = {"model_spec": spec.model_spec_section_task(dataset_dir=str(data_dir / "model_spec_dataset"), grader_model=judge, num_grader_samples=1)}
    if model_spec_grading == "rubric-only":
        from .model_spec import rubric_only_section_compliance
        tasks["model_spec"] = Task(dataset=tasks["model_spec"].dataset,
            scorer=rubric_only_section_compliance(judge), name="model_spec_evals_rubric_only",
            metadata={"model_spec_grading": model_spec_grading})
    elif model_spec_grading != "full-spec":
        raise ValueError("Unknown Model Spec grading mode")
    # Only public asset loading is redirected; the task/solver/scorer functions are upstream.
    with patch.object(xs, "hf_dataset", side_effect=local_xstest):
        tasks["xstest_safe"] = xs.xstest(subset="safe", scorer_model=judge)
        tasks["xstest_unsafe"] = xs.xstest(subset="unsafe", scorer_model=judge)
    with patch.object(strong, "load_csv_dataset", side_effect=local_strong):
        tasks["strong_reject"] = strong.strong_reject(jailbreak_method="None", judge_llm=judge)
    for name, task in tasks.items():
        if len(task.dataset) != EXPECTED_COUNTS[name]:
            raise ValueError(f"Unexpected {name} dataset count: {len(task.dataset)}")
    return {name: tasks[name] for name in selected}


def sample_record(sample) -> dict:
    # json_dataset assigns random internal message ids at every load. They are
    # not benchmark sample ids or tool-call ids. Exclude only that top-level
    # transport field; nested tool ids and tool_call_id must remain intact.
    inputs = sample.input if isinstance(sample.input, str) else [m.model_dump(mode="json", exclude_none=True, exclude={"id"}) for m in sample.input]
    return {"id": sample.id, "input": inputs, "target": sample.target, "metadata": sample.metadata}


def task_records(tasks: dict) -> dict:
    return {name: [sample_record(sample) for sample in task.dataset] for name, task in tasks.items()}


def installed_environment() -> dict[str, str]:
    # Freeze installed distributions, not unrelated *.egg-info directories
    # discovered through CWD/sys.path (which differ for -m and console scripts).
    paths = sorted({sysconfig.get_path("purelib"), sysconfig.get_path("platlib")})
    return dict(sorted((d.metadata["Name"].lower().replace("_", "-"), d.version)
                       for d in importlib.metadata.distributions(path=paths)))


def source_lock(root: Path, data_dir: Path, records: dict) -> dict:
    files = {}
    for folder in [root / "evaluation", root / "model_spec_evals/src/model_spec_evals"]:
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".md", ".json"} and "__pycache__" not in path.parts:
                files[str(path.relative_to(root))] = sha256(path)
    asset_files = {str(p.relative_to(data_dir)): sha256(p) for p in sorted(data_dir.rglob("*")) if p.is_file()}
    return {"packages": check_versions(), "environment_packages": installed_environment(), "source_files": files, "data_files": asset_files,
            "dataset_hashes": {name: canonical_hash(rows) for name, rows in records.items()},
            "dataset_counts": {name: len(rows) for name, rows in records.items()},
            "upstream_revisions": {"model_spec_dataset": SPEC_DATA_REVISION, "model_spec_evals": SPEC_CODE_REVISION, "grader_model_spec": SPEC_REVISION, "xstest_hf_reference": XSTEST_REVISION, "xstest_author_csv": XSTEST_AUTHOR_REVISION, "strong_reject": STRONG_REVISION}}
