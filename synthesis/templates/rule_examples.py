"""From one excerpt, write requests and the replies that follow it.

One model call produces two kinds of training rows. The first asks for
examples and stores the examples as the reply. Each further row is one of
those requests with its reply, and does not include the excerpt.
"""

from __future__ import annotations

import argparse
from typing import Any

from synthesis.templates._common import (
    _at_least_one,
    add_arguments as add_shared_arguments,
    excerpt,
    generation_request,
    load_object,
    read as read_portions,
    require_text,
    saved,
)
from synthesis.types import ChatRequest, Example, ParseError

name = "rule_examples"
description = "From a rule excerpt, write requests and the replies that follow it."

ASK = (
    "Write user requests governed by this excerpt, and the assistant reply for each. "
    "Show the behavior. Do not mention this policy document inside the requests or replies."
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    add_shared_arguments(parser)
    parser.add_argument(
        "--examples",
        type=_at_least_one,
        default=1,
        help="Request and reply pairs to generate from each heading's direct body.",
    )


def read(args: argparse.Namespace):
    """Yield excerpts. Reject a bad ``--examples`` before any request is sent."""

    if args.examples < 1:
        raise ValueError("--examples must be >= 1")
    return read_portions(args)


def prompt(item: dict[str, Any], args: argparse.Namespace) -> ChatRequest:
    pair_label = "pair" if args.examples == 1 else "pairs"
    task = (
        f"{ASK}\n\nWrite {args.examples} {pair_label}.\n"
        'Return a JSON object: {"examples":[{"request":"...","reply":"..."}]}'
    )
    return generation_request(item, args, task)


def parse(item: dict[str, Any], content: str, args: argparse.Namespace) -> list[Example]:
    data = load_object(content)
    pairs = data.get("examples")
    if not isinstance(pairs, list) or not pairs:
        raise ParseError("JSON object has no examples", raw=content)
    if len(pairs) != args.examples:
        raise ParseError(f"expected {args.examples} examples, got {len(pairs)}", raw=content)
    cleaned: list[tuple[str, str]] = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            raise ParseError(f"example {index} is not an object", raw=content)
        cleaned.append((require_text(pair, "request", content), require_text(pair, "reply", content)))
    rendered = "\n\n".join(f"Request:\n{request}\n\nReply:\n{reply}" for request, reply in cleaned)
    examples = [
        saved(
            item,
            "generate",
            [
                {"role": "user", "content": f"{excerpt(item)}\n\n{ASK}"},
                {"role": "assistant", "content": rendered},
            ],
        )
    ]
    for request, reply in cleaned:
        examples.append(
            saved(
                item,
                "act",
                [
                    {"role": "user", "content": request},
                    {"role": "assistant", "content": reply},
                ],
            )
        )
    return examples
