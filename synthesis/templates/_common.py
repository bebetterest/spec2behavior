"""Shared read and request helpers for the default rule templates."""

from __future__ import annotations

import argparse
import secrets
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from synthesis.arguments import positive_int as _at_least_one, temperature
from synthesis.jsonutil import parse_json_object
from synthesis.portions import DEFAULT_MAX_PORTION_CHARS, split_portions
from synthesis.types import ChatRequest, Example, ParseError

_GROUNDING = (
    "Use the excerpt as the source of behavioral rules. Preserve each rule's scope, conditions, "
    "exceptions, and allowed discretion. Keep requirements, recommendations, defaults, and permissions "
    "distinct; do not strengthen or weaken them, or turn unspecified matters into additional rules.\n\n"
    "Choose a concrete scenario in which a stated rule or exception determines the response. "
    "You may invent ordinary scenario details, but place any facts needed to justify a trained reply "
    "in the saved input messages. Do not rely on authoring context that will be absent from the "
    "training conversation, or import facts about the teacher's deployment into its fictional assistant. "
    "Keep facts consistent across turns. When a decision depends on missing information, express "
    "that uncertainty or ask for the needed information rather than assuming it.\n\n"
    "Check the assistant training targets for rule fidelity, supported factual claims, and whether "
    "they fulfill the user's request wherever the excerpt permits. A deliberately violating draft "
    "requested by the task should violate the selected rule, without changing the scenario facts. "
    "Use the random seed to vary appropriate scenario details and wording; omit the seed from your output. "
    "Return only the requested JSON object."
)

_INAPPLICABLE = (
    "If the excerpt cannot support this training form without inventing or changing a behavioral "
    'rule, return {"not_applicable":"brief reason grounded in the excerpt"} instead of the normal '
    "object. Use this for a mismatch between excerpt and training form, not merely because the "
    "example calls for a refusal."
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
    seed = secrets.token_hex(16)
    item["generation_seed"] = seed
    extra = {} if args.no_json_mode else {"response_format": {"type": "json_object"}}
    return ChatRequest(
        messages=[{
            "role": "user",
            "content": f"Random seed: {seed}\n\n{excerpt(item)}\n\n{_GROUNDING}\n\n{task}\n\n{_INAPPLICABLE}",
        }],
        temperature=args.temperature,
        extra=extra,
        generation_seed=seed,
    )


def load_object(content: str) -> dict[str, Any]:
    data = parse_json_object(content)
    if "not_applicable" in data:
        reason = require_text(data, "not_applicable", content)
        raise ParseError(f"template not applicable: {reason}", raw=content)
    return data


def require_text(data: dict[str, Any], key: str, raw: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ParseError(f"JSON object needs a non-empty {key}", raw=raw)
    return value.strip()


def saved(item: dict[str, Any], form: str, messages: list[dict[str, str]]) -> Example:
    meta = {
        "source": item["source"],
        "portion_id": item["portion_id"],
        "sft_form": form,
    }
    seed = item.get("generation_seed")
    if seed:
        if any(seed in message["content"] for message in messages):
            raise ParseError("generation seed must not appear in training messages")
        meta["generation_seed"] = seed
    return Example(
        messages=messages,
        meta=meta,
    )
