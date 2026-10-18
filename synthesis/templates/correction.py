"""From one excerpt, write a request, a violating reply, and a corrected reply.

The training row shows the request and the violating reply, and the assistant
message is only the correction.
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

name = "correction"
description = "Correct a reply that violates a rule excerpt."

_TASK = (
    "Write one user request, one assistant reply that violates this excerpt, "
    "and one corrected reply that follows it. Do not mention this policy document.\n\n"
    'Return a JSON object: {"request":"...","wrong_reply":"...","correct_reply":"..."}'
)


def prompt(item: dict[str, Any], args: argparse.Namespace) -> ChatRequest:
    return generation_request(item, args, _TASK)


def parse(item: dict[str, Any], content: str, args: argparse.Namespace) -> list[Example]:
    del args
    data = load_object(content)
    request = require_text(data, "request", content)
    wrong = require_text(data, "wrong_reply", content)
    correct = require_text(data, "correct_reply", content)
    user = (
        f"Request:\n{request}\n\n"
        "The reply below violates the policy. Correct it.\n\n"
        f"Reply:\n{wrong}"
    )
    return [
        saved(
            item,
            "correct",
            [
                {"role": "user", "content": user},
                {"role": "assistant", "content": correct},
            ],
        )
    ]
