"""From one excerpt, write conflicting system, developer, and user instructions.

The training row is those three messages plus the reply that follows the
excerpt's priority. The excerpt itself is not stored in the row.
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

name = "chain_of_command"
description = "Follow the higher-priority instruction when system, developer, and user conflict."

_TASK = (
    "Write a system message, a developer message, and a user message containing a concrete "
    "instruction conflict governed by a specific rule or exception in this excerpt. "
    "Include that rule and any necessary scenario context in the system or developer message. "
    "The conflict must test the excerpt's actual behavioral boundary; an unrelated restriction "
    "added solely to create a hierarchy puzzle is not suitable. "
    "Then write the assistant reply that follows the controlling instruction and fulfills the "
    "compatible parts of the other requests. "
    "System instructions outrank developer instructions, and developer instructions outrank "
    "user instructions, unless the excerpt says a hard limit binds all three. "
    "Do not mention this policy document inside the messages.\n\n"
    'Return a JSON object: {"system":"...","developer":"...","user":"...","assistant":"..."}'
)


def prompt(item: dict[str, Any], args: argparse.Namespace) -> ChatRequest:
    return generation_request(item, args, _TASK)


def parse(item: dict[str, Any], content: str, args: argparse.Namespace) -> list[Example]:
    del args
    data = load_object(content)
    return [
        saved(
            item,
            "priority",
            [
                {"role": "system", "content": require_text(data, "system", content)},
                {"role": "developer", "content": require_text(data, "developer", content)},
                {"role": "user", "content": require_text(data, "user", content)},
                {"role": "assistant", "content": require_text(data, "assistant", content)},
            ],
        )
    ]
