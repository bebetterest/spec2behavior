"""From one excerpt, write what the assistant takes from it.

The training row includes the excerpt. The assistant message is the reflection,
not the JSON wrapper.
"""

from __future__ import annotations

import argparse
from typing import Any

from synthesis.templates._common import (
    add_arguments,
    excerpt,
    generation_request,
    load_object,
    read,
    require_text,
    saved,
)
from synthesis.types import ChatRequest, Example

name = "reflection"
description = "Explain an excerpt's requirements, permissions, exceptions, and limits."

ASK = (
    "Explain the obligations, permissions, exceptions, and limits stated in this excerpt. "
    "Distinguish explicit rules from matters it leaves unspecified, then describe how you would apply it."
)


def prompt(item: dict[str, Any], args: argparse.Namespace) -> ChatRequest:
    task = f'{ASK}\n\nReturn a JSON object: {{"reflection":"..."}}'
    return generation_request(item, args, task)


def parse(item: dict[str, Any], content: str, args: argparse.Namespace) -> list[Example]:
    del args
    data = load_object(content)
    reflection = require_text(data, "reflection", content)
    return [
        saved(
            item,
            "reflect",
            [
                {"role": "user", "content": f"{excerpt(item)}\n\n{ASK}"},
                {"role": "assistant", "content": reflection},
            ],
        )
    ]
