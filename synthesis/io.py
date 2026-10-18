"""JSONL checkpoint files.

``jobs.jsonl`` is the source of truth for resume. ``train.jsonl`` is rebuilt
from it and is the file to fine-tune on. Both are append-only during a run.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def drop_incomplete_tail(path: Path) -> None:
    """Drop a trailing partial line left when a process dies mid-append.

    A finished record always ends with a newline. Resume ignores an unfinished
    tail so that job is run again instead of failing the whole checkpoint.
    """

    if not path.exists():
        return
    with path.open("rb") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            return
        handle.seek(-1, 2)
        if handle.read(1) == b"\n":
            return
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    head, _newline, _tail = data.rpartition(b"\n")
    path.write_bytes(head + (b"\n" if head else b""))


def read_jsonl(
    path: Path, *, ignore_incomplete_tail: bool = False
) -> Iterator[dict[str, Any]]:
    """Read JSONL, optionally ignoring an unfinished checkpoint line.

    Dataset files may have a valid final record without a newline. Checkpoint
    previews can ignore that tail to match resume, without changing the file.
    """

    if not path.exists():
        return
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if ignore_incomplete_tail and not line.endswith(b"\n"):
                break
            try:
                stripped = line.decode("utf-8").strip()
                if not stripped:
                    continue
                value = json.loads(stripped)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()


def load_ids(
    path: Path, key: str = "id", *, ignore_incomplete_tail: bool = False
) -> set[str]:
    return {
        str(record[key])
        for record in read_jsonl(path, ignore_incomplete_tail=ignore_incomplete_tail)
        if record.get(key)
    }


def train_row(job_id: str, example: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": example["id"],
        "job_id": job_id,
        "messages": example["messages"],
        "meta": example.get("meta") or {},
    }


def materialize_train(jobs_path: Path, train_path: Path) -> int:
    """Rewrite ``train.jsonl`` so it matches ``jobs.jsonl`` exactly."""

    train_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = train_path.with_suffix(train_path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for job in read_jsonl(jobs_path):
            job_id = str(job["id"])
            for example in job.get("examples") or []:
                handle.write(json.dumps(train_row(job_id, example), ensure_ascii=False) + "\n")
                count += 1
    temporary.replace(train_path)
    return count
