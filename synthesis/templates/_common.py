"""Shared read and request helpers for the default rule templates."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from synthesis.arguments import positive_int as _at_least_one, temperature
from synthesis.jsonutil import parse_json_object
from synthesis.portions import DEFAULT_MAX_PORTION_CHARS, split_portions
from synthesis.types import ChatRequest, Example, ParseError

_GROUNDING = (
    "Ground the response in the excerpt. For examples, choose a concrete scenario directly governed by it. "
    "Preserve its stated permissions, exceptions, and limits; do not add restrictions or permissions. "
    "Make generated conversations self-contained by including needed scenario facts in their messages. "
    "Do not claim an assistant identity, model family, developer, knowledge cutoff, or available tools "
    "unless those facts are explicitly supplied in the excerpt or task. "
    "Choose another scenario if it would require inventing those facts."
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", required=True, type=Path, help="Markdown spec or constitution.")
    parser.add_argument(
        "--max-portion-chars",
        type=_at_least_one,
        default=DEFAULT_MAX_PORTION_CHARS,
        help=(
            "Split a heading's direct body on paragraph boundaries when it exceeds this "
            "many characters. This is a soft character limit, not a token limit."
        ),
    )
    parser.add_argument(
        "--temperature", type=temperature, default=1.0, metavar="VALUE|default",
        help="Sampling temperature from 0 to 2; 'default' omits the API field.",
    )
    parser.add_argument(
        "--no-json-mode",
        action="store_true",
        help="Do not send response_format=json_object. Use this for servers that reject it.",
    )


def read(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    """Yield one record per rule excerpt."""

    source = Path(args.source)
    if not source.is_file():
        raise ValueError(f"source file not found: {source}")
    if args.max_portion_chars < 1:
        raise ValueError("--max-portion-chars must be >= 1")
    text = source.read_text(encoding="utf-8")
    for portion in split_portions(text, args.max_portion_chars):
        yield {
            "id": f"{source.name}/{portion.id}",
            "source": str(source),
            "portion_id": portion.id,
            "title": portion.title,
            "ancestors": list(portion.ancestors),
            "text": portion.text,
        }


def excerpt(item: dict[str, Any]) -> str:
    ancestors = " > ".join(item["ancestors"]) if item["ancestors"] else "(none)"
    return (
        f"Title: {item['title']}\n"
        f"Ancestors: {ancestors}\n\n"
        f"<excerpt>\n{item['text']}\n</excerpt>"
    )


def generation_request(item: dict[str, Any], args: argparse.Namespace, task: str) -> ChatRequest:
    extra = {} if args.no_json_mode else {"response_format": {"type": "json_object"}}
    return ChatRequest(
        messages=[{"role": "user", "content": f"{excerpt(item)}\n\n{_GROUNDING}\n\n{task}"}],
        temperature=args.temperature,
        extra=extra,
    )


def load_object(content: str) -> dict[str, Any]:
    return parse_json_object(content)


def require_text(data: dict[str, Any], key: str, raw: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ParseError(f"JSON object needs a non-empty {key}", raw=raw)
    return value.strip()


def saved(item: dict[str, Any], form: str, messages: list[dict[str, str]]) -> Example:
    return Example(
        messages=messages,
        meta={
            "source": item["source"],
            "portion_id": item["portion_id"],
            "sft_form": form,
        },
    )
