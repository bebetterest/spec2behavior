"""The four fixed policy tests used by the reference epoch sweep."""

import json
from pathlib import Path
import shutil

from .core import SPEC_URL, public_get, sha256, spec_modules
from .datasets import verify

BENCHMARKS = ("model-spec-original", "model-spec-generic", "constitution-original", "constitution-generic")
RENDERER_DATE = "2026-10-24"
GRADER_SHA256 = "e7a70bc6ab363213d88f6c20a7b78c6949d3f5feba22ec054c97f333a71a8aeb"
DEFAULT_CONFIG = Path(__file__).with_name("config.reference.json")


def prepare_assets(root, directory):
    source = root / "data/evaluation"
    verify(source)
    manifest = json.loads((source / "manifest.json").read_text())
    dataset_dir = directory / "policy"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    names = {"manifest.json"}
    for collection in manifest["collections"].values():
        names.add(collection["path"])
        if "license" in collection:
            names.add(collection["license"]["path"])
    for name in sorted(names):
        shutil.copyfile(source / name, dataset_dir / name)
    spec = directory / "grader_model_spec.md"
    if not spec.exists():
        spec.write_bytes(public_get(SPEC_URL).content)
    if sha256(spec) != GRADER_SHA256:
        raise ValueError("The pinned upstream scorer resource failed its checksum")


def make_tasks(root, directory, judge):
    from inspect_ai import Task
    from inspect_ai.dataset import json_dataset
    from inspect_ai.solver import generate
    from .model_spec import rubric_only_section_compliance

    dataset_dir = directory / "policy"
    verify(dataset_dir)
    spec_modules(root, directory)
    return {name: Task(dataset=json_dataset(str(dataset_dir / f"{name}.jsonl")),
                       solver=generate(), scorer=rubric_only_section_compliance(judge),
                       name=name.replace("-", "_"))
            for name in BENCHMARKS}


def pilot_ids(tasks):
    """Two items per test, including instruction and tool-history cases."""
    result = {}
    for name, task in tasks.items():
        rows = list(task.dataset)
        if name == "model-spec-original":
            first = next(r for r in rows if any(m.role == "tool" or getattr(m, "tool_calls", None) for m in r.input))
            second = next(r for r in rows if r.input[-1].role == "system" and r.id != first.id)
        else:
            first = next(r for r in rows if {"system", "assistant"} <= {m.role for m in r.input})
            second = next(r for r in rows if all(m.role not in {"system", "assistant"} for m in r.input))
        result[name] = [first.id, second.id]
    return result
