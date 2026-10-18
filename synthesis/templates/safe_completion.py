"""From one excerpt, write a request that is only partly allowed.

The training row is that request and a reply that answers the allowed part
and refuses the rest.
"""

from __future__ import annotations

import argparse
from typing import Any

from synthesis.templates._common import (
    add_arguments,
    generation_request,
    load_object,
    read,
    require_text,
    saved,
)
from synthesis.types import ChatRequest, Example

name = "safe_completion"
description = "Answer the allowed part of a request and refuse the rest."

_TASK = (
    "Write one user request that this excerpt only partly allows, and the assistant reply. "
    "Make both parts concrete applications of this excerpt. "
    "The reply must answer the allowed part and refuse the disallowed part. "
    "Do not mention this policy document.\n\n"
    'Return a JSON object: {"request":"...","reply":"..."}'
)


def prompt(item: dict[str, Any], args: argparse.Namespace) -> ChatRequest:
    return generation_request(item, args, _TASK)


def parse(item: dict[str, Any], content: str, args: argparse.Namespace) -> list[Example]:
    del args
    data = load_object(content)
    return [
        saved(
            item,
            "partial",
            [
                {"role": "user", "content": require_text(data, "request", content)},
                {"role": "assistant", "content": require_text(data, "reply", content)},
            ],
        )
    ]
