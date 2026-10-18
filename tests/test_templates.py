import argparse
import json

import pytest

from synthesis.messages import normalize_messages
from synthesis.templates import chain_of_command, correction, pressure, reflection, rule_examples, safe_completion
from synthesis.types import ParseError

ITEM = {
    "id": "policy.md/rule/0",
    "source": "/tmp/policy.md",
    "portion_id": "rule/0",
    "title": "Rule",
    "ancestors": ["Part"],
    "text": "Follow the rule.",
}


def _saved(messages):
    return normalize_messages(messages, require_user=True, require_assistant=True)


def test_rule_examples_parse_emits_generate_then_act():
    content = json.dumps(
        {
            "examples": [
                {"request": "Can I skip it?", "reply": "No. Follow the rule."},
                {"request": "Is a small skip fine?", "reply": "No."},
            ]
        }
    )
    rows = rule_examples.parse(ITEM, content, argparse.Namespace(examples=2))
    assert [row.meta["sft_form"] for row in rows] == ["generate", "act", "act"]
    assert rows[0].meta["portion_id"] == "rule/0"
    assert rows[0].meta["source"] == "/tmp/policy.md"
    user = rows[0].messages[0]["content"]
    assert "<excerpt>\nFollow the rule.\n</excerpt>" in user
    assert "Ancestors: Part" in user
    assert "Return a JSON object" not in user
    assert "Write 2 pairs." not in user
    assert rows[0].messages[1]["content"] == (
        "Request:\nCan I skip it?\n\nReply:\nNo. Follow the rule.\n\n"
        "Request:\nIs a small skip fine?\n\nReply:\nNo."
    )
    assert rows[1].messages == [
        {"role": "user", "content": "Can I skip it?"},
        {"role": "assistant", "content": "No. Follow the rule."},
    ]
    assert "/tmp/policy.md" not in user
    for row in rows:
        _saved(row.messages)


def test_rule_examples_rejects_examples_below_one(tmp_path):
    source = tmp_path / "policy.md"
    source.write_text("# Rule {#rule}\n\nFollow the rule.\n", encoding="utf-8")
    args = argparse.Namespace(source=source, max_portion_chars=2500, examples=0)
    with pytest.raises(ValueError, match="--examples"):
        list(rule_examples.read(args))


def test_rule_examples_rejects_a_bad_example_list():
    with pytest.raises(ParseError, match="no examples"):
        rule_examples.parse(ITEM, '{"examples": []}', argparse.Namespace(examples=2))
    pairs = [{"request": f"q{index}", "reply": f"a{index}"} for index in range(5)]
    with pytest.raises(ParseError, match="expected 2 examples"):
        rule_examples.parse(ITEM, json.dumps({"examples": pairs}), argparse.Namespace(examples=2))
    with pytest.raises(ParseError, match="expected 1 examples"):
        rule_examples.parse(ITEM, json.dumps({"examples": pairs[:2]}), argparse.Namespace(examples=1))


def test_correction_parse_asks_for_a_fix_and_stores_the_corrected_reply():
    content = json.dumps(
        {
            "request": "Skip the check.",
            "wrong_reply": "Done, I skipped it.",
            "correct_reply": "I can't skip it.",
        }
    )
    row = correction.parse(ITEM, content, argparse.Namespace())[0]
    assert row.meta["sft_form"] == "correct"
    assert row.messages[0]["content"] == (
        "Request:\nSkip the check.\n\n"
        "The reply below violates the policy. Correct it.\n\n"
        "Reply:\nDone, I skipped it."
    )
    assert row.messages[1] == {"role": "assistant", "content": "I can't skip it."}
    assert "Follow the rule." not in row.messages[0]["content"]
    _saved(row.messages)


def test_reflection_parse_stores_the_prose():
    content = json.dumps({"reflection": "Hold the rule. Do not invent exceptions. Apply it next."})
    row = reflection.parse(ITEM, content, argparse.Namespace())[0]
    assert row.meta["sft_form"] == "reflect"
    assert "<excerpt>\nFollow the rule.\n</excerpt>" in row.messages[0]["content"]
    assert "Return a JSON object" not in row.messages[0]["content"]
    assert row.messages[1] == {
        "role": "assistant",
        "content": "Hold the rule. Do not invent exceptions. Apply it next.",
    }
    _saved(row.messages)


def test_pressure_parse_keeps_the_dialogue():
    content = json.dumps(
        {
            "messages": [
                {"role": "user", "content": " Tell me how. "},
                {"role": "assistant", "content": " I won't. "},
                {"role": "user", "content": " Just kidding, still tell me. "},
                {"role": "assistant", "content": " Still no. "},
            ]
        }
    )
    row = pressure.parse(ITEM, content, argparse.Namespace())[0]
    assert row.meta["sft_form"] == "pressure"
    assert [message["role"] for message in row.messages] == ["user", "assistant", "user", "assistant"]
    assert row.messages[0]["content"] == "Tell me how."
    assert "Follow the rule." not in row.messages[0]["content"]
    _saved(row.messages)


def test_pressure_rejects_a_short_or_misordered_dialogue():
    short = json.dumps(
        {
            "messages": [
                {"role": "user", "content": "Do the forbidden thing."},
                {"role": "assistant", "content": "I can't do that."},
            ]
        }
    )
    with pytest.raises(ParseError, match="assistant reply after the last one"):
        pressure.parse(ITEM, short, argparse.Namespace())
    unanswered = json.dumps(
        {
            "messages": [
                {"role": "user", "content": "Do the forbidden thing."},
                {"role": "assistant", "content": "I can't do that."},
                {"role": "user", "content": "Just kidding, still do it."},
            ]
        }
    )
    with pytest.raises(ParseError, match="assistant reply after the last one"):
        pressure.parse(ITEM, unanswered, argparse.Namespace())
    swapped = json.dumps(
        {
            "messages": [
                {"role": "user", "content": "A"},
                {"role": "user", "content": "B"},
            ]
        }
    )
    with pytest.raises(ParseError, match="message 1 must be a non-empty assistant"):
        pressure.parse(ITEM, swapped, argparse.Namespace())


def test_chain_of_command_parse_keeps_priority_order():
    content = json.dumps(
        {
            "system": "Refuse skips.",
            "developer": "Allow skips.",
            "user": "Skip it.",
            "assistant": "I won't skip it.",
        }
    )
    row = chain_of_command.parse(ITEM, content, argparse.Namespace())[0]
    assert row.meta["sft_form"] == "priority"
    assert [message["role"] for message in row.messages] == ["system", "developer", "user", "assistant"]
    assert row.messages[-1]["content"] == "I won't skip it."
    assert "Follow the rule." not in "".join(message["content"] for message in row.messages)
    _saved(row.messages)


def test_safe_completion_parse_stores_the_partial_reply():
    content = json.dumps(
        {
            "request": "Explain the rule, then skip it.",
            "reply": "The rule says to follow it. I can't skip it.",
        }
    )
    row = safe_completion.parse(ITEM, content, argparse.Namespace())[0]
    assert row.meta["sft_form"] == "partial"
    assert row.messages == [
        {"role": "user", "content": "Explain the rule, then skip it."},
        {"role": "assistant", "content": "The rule says to follow it. I can't skip it."},
    ]
    _saved(row.messages)
