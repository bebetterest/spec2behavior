"""From one excerpt, write a multi-turn conversation that keeps the same boundary.

The user is refused or only partly answered, then rephrases, jokes, or swaps
the concept. Each assistant reply still follows the excerpt. The training row
is that conversation.
"""

from __future__ import annotations

import argparse
from typing import Any

from synthesis.templates._common import (
    add_arguments,
    generation_request,
    load_object,
    read,
    saved,
)
from synthesis.types import ChatRequest, Example, ParseError

name = "pressure"
description = "Keep the same boundary across turns after a refusal or a partial answer."

_TASK = (
    "Write a multi-turn conversation. The first assistant reply refuses or only partly answers. "
    "A later user turn rephrases the request, treats it as a joke, or swaps the concept. "
    "Keep every turn focused on that same rule, without drifting to unrelated topics. "
    "Every assistant reply still follows this excerpt. Include at least two user turns. "
    "Do not mention this policy document.\n\n"
    'Return a JSON object: {"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}'
)


def prompt(item: dict[str, Any], args: argparse.Namespace) -> ChatRequest:
    return generation_request(item, args, _TASK)


def parse(item: dict[str, Any], content: str, args: argparse.Namespace) -> list[Example]:
    del args
    data = load_object(content)
    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ParseError("JSON object has no messages", raw=content)
    cleaned: list[dict[str, str]] = []
    expect = "user"
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ParseError(f"message {index} is not an object", raw=content)
        role = message.get("role")
        text = message.get("content")
        if role != expect or not isinstance(text, str) or not text.strip():
            raise ParseError(
                f"message {index} must be a non-empty {expect} message alternating with the other role",
                raw=content,
            )
        cleaned.append({"role": role, "content": text.strip()})
        expect = "assistant" if expect == "user" else "user"
    user_turns = sum(message["role"] == "user" for message in cleaned)
    if user_turns < 2 or cleaned[-1]["role"] != "assistant":
        raise ParseError(
            "pressure dialogue needs at least two user turns and an assistant reply after the last one",
            raw=content,
        )
    return [saved(item, "pressure", cleaned)]
