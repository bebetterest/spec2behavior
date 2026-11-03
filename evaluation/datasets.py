"""Verify and read the four checked-in evaluation collections without API calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .core import canonical_hash, sha256

DEFAULT_DIRECTORY = Path(__file__).resolve().parents[1] / "data/evaluation"


def _file(directory, name):
    path = (directory / name).resolve()
    if path.parent != directory.resolve():
        raise ValueError("Dataset and license files must be inside the collection directory")
    return path


def load_collection(name, directory=DEFAULT_DIRECTORY):
    """Return unchanged Inspect records after checking the published manifest."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("purpose") != "evaluation":
        raise ValueError("Unsupported evaluation dataset manifest")
    if name not in manifest["collections"]:
        raise ValueError(f"Unknown collection: {name}")
    info = manifest["collections"][name]
    path = _file(directory, info["path"])
    if sha256(path) != info["sha256"]:
        raise ValueError(f"Dataset checksum changed: {name}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != info["rows"] or len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"Wrong row count or duplicate IDs: {name}")
    content = [dict(id=row["id"], input=row["input"], target=row["target"]) for row in rows]
    if canonical_hash(content) != info["content_sha256"]:
        raise ValueError(f"Evaluation inputs or rubrics changed: {name}")
    if info["kind"] not in {"upstream", "synthetic_reviewed"}:
        raise ValueError(f"Unknown collection kind: {name}")
    for row in rows:
        if (set(row) != {"id", "input", "target", "metadata"}
                or not isinstance(row["id"], str) or not row["id"]
                or not isinstance(row["target"], str) or not row["target"].strip()
                or not isinstance(row["input"], list) or not row["input"]
                or not isinstance(row["metadata"], dict)):
            raise ValueError(f"Invalid Inspect record: {name}")
        metadata = row["metadata"]
        if not metadata.get("section_id") or not metadata.get("top_level_section"):
            raise ValueError(f"Missing scoring metadata: {row['id']}")
        # Official histories can end with tool or instruction turns. Preserve
        # all roles, structured content and tool-call IDs without rewriting.
        if any(message.get("role") not in {"system", "developer", "user", "assistant", "tool"}
               or "content" not in message for message in row["input"]):
            raise ValueError(f"Invalid input message: {row['id']}")
        if info["kind"] == "synthetic_reviewed":
            if (row["input"][-1]["role"] != "user"
                    or metadata.get("review_method") not in {"human", "codex"}
                    or metadata.get("source") != info["source"]["path"]
                    or metadata.get("source_sha256") != info["source"]["sha256"]
                    or metadata.get("instruction_role_convention") != info["instruction_role_convention"]):
                raise ValueError(f"Invalid reviewed source binding: {row['id']}")
    if "license" in info:
        license_info = info["license"]
        if sha256(_file(directory, license_info["path"])) != license_info["sha256"]:
            raise ValueError(f"Upstream license changed: {name}")
    return rows


def verify(directory=DEFAULT_DIRECTORY):
    """Check every collection, including cross-collection IDs and total counts."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    counts = {}
    ids = set()
    custom_rows = 0
    for name, info in manifest["collections"].items():
        rows = load_collection(name, directory)
        current = {row["id"] for row in rows}
        if ids & current:
            raise ValueError("Evaluation IDs collide across collections")
        ids.update(current)
        counts[name] = len(rows)
        if info["kind"] == "synthetic_reviewed":
            custom_rows += len(rows)
    if len(ids) != manifest["total_rows"] or custom_rows != manifest["custom_rows"]:
        raise ValueError("Collection totals differ from the manifest")
    return dict(verified=True, collections=counts, total_rows=len(ids), custom_rows=custom_rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(verify(args.directory), indent=2))
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"Evaluation dataset error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
