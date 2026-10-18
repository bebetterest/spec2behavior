import argparse
import asyncio
import json
from pathlib import Path

import synthesis.pipelines  # noqa: F401  registers the default templates
from synthesis.pipeline import get_pipeline
from synthesis.runner import run_synthesis
from synthesis.types import ChatRequest


class RecordingClient:
    def __init__(self, content: str) -> None:
        self.model = "recorder"
        self.max_tokens = 16
        self.content = content
        self.requests: list[ChatRequest] = []

    async def complete(self, request: ChatRequest) -> str:
        self.requests.append(request)
        return self.content


def test_rule_examples_run_writes_generate_and_act(tmp_path: Path):
    source = tmp_path / "policy.md"
    source.write_text(
        "# Part\n\n## Rule {#rule}\n\nFollow the rule.\n",
        encoding="utf-8",
    )
    payload = {
        "examples": [
            {"request": "May I skip it?", "reply": "No."},
            {"request": "What about a small skip?", "reply": "Still no."},
        ]
    }
    args = argparse.Namespace(
        source=source,
        max_portion_chars=2500,
        examples=2,
        temperature=0.4,
        no_json_mode=False,
    )
    client = RecordingClient(json.dumps(payload))

    async def scenario() -> None:
        stats = await run_synthesis(
            get_pipeline("rule_examples"),
            args,
            client,
            tmp_path / "out",
            concurrency=1,
            batch_size=1,
        )
        assert stats.wrote == 1

    asyncio.run(scenario())
    request = client.requests[0]
    prompt = request.messages[0]["content"]
    assert "Follow the rule." in prompt
    assert "Write 2 pairs." in prompt
    assert str(source) not in prompt
    assert request.extra["response_format"]["type"] == "json_object"
    assert request.temperature == 0.4
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["id"] for row in rows] == [
        "rule_examples/policy.md/rule/0#0",
        "rule_examples/policy.md/rule/0#1",
        "rule_examples/policy.md/rule/0#2",
    ]
    assert [row["meta"]["sft_form"] for row in rows] == ["generate", "act", "act"]
    assert rows[0]["meta"]["portion_id"] == "rule/0"
    assert rows[0]["meta"]["pipeline"] == "rule_examples"
    assert "Write 2 pairs." not in rows[0]["messages"][0]["content"]
    assert "Return a JSON object" not in rows[0]["messages"][0]["content"]
    assert rows[0]["messages"][1]["content"].startswith("Request:\nMay I skip it?")
    assert rows[1]["messages"] == [
        {"role": "user", "content": "May I skip it?"},
        {"role": "assistant", "content": "No."},
    ]
    assert "Follow the rule." not in rows[1]["messages"][0]["content"]


def test_template_prompt_list_omits_trainable_and_parse_splits_rows(tmp_path: Path):
    path = tmp_path / "split.py"
    path.write_text(
        "\n".join(
            [
                "name = 'split_rows'",
                "description = 'Two conversations from one reply.'",
                "def read(args):",
                "    del args",
                "    yield {'id': 's', 'title': 'Be honest'}",
                "def prompt(item, args):",
                "    del args",
                "    return [{'role': 'user', 'content': item['title'], 'trainable': False}]",
                "def parse(item, content, args):",
                "    del content, args",
                "    return [",
                "        {'messages': [",
                "            {'role': 'user', 'content': 'a'},",
                "            {'role': 'assistant', 'content': 'b'},",
                "        ], 'meta': {'section': item['title']}},",
                "        [",
                "            {'role': 'user', 'content': 'c'},",
                "            {'role': 'assistant', 'content': 'd'},",
                "        ],",
                "    ]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    from synthesis.template import load_template_file

    load_template_file(path)
    client = RecordingClient("unused")

    async def scenario() -> None:
        stats = await run_synthesis(
            get_pipeline("split_rows"),
            argparse.Namespace(),
            client,
            tmp_path / "out",
            concurrency=1,
            batch_size=1,
        )
        assert stats.wrote == 1

    asyncio.run(scenario())
    assert client.requests[0].messages == [{"role": "user", "content": "Be honest"}]
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["id"] for row in rows] == ["split_rows/s#0", "split_rows/s#1"]
    assert rows[0]["meta"]["section"] == "Be honest"
    assert rows[1]["messages"][1]["content"] == "d"


def test_templates_from_the_same_source_can_be_combined_for_training_and_inference(tmp_path):
    from training.data import SFTSettings, plan_sft, write_combined_jsonl
    from training.prompts import jobs_from_file

    source = tmp_path / "policy.md"
    source.write_text("# Rule {#rule}\n\nFollow the rule.\n", encoding="utf-8")
    args = argparse.Namespace(source=source, max_portion_chars=16000, examples=1,
                              temperature=1.0, no_json_mode=False)
    payloads = {
        "rule_examples": '{"examples":[{"request":"May I skip?","reply":"No."}]}',
        "reflection": '{"reflection":"Follow the rule without inventing exceptions."}',
    }

    async def generate():
        for name, content in payloads.items():
            stats = await run_synthesis(get_pipeline(name), args, RecordingClient(content), tmp_path / name)
            assert stats.wrote == 1

    asyncio.run(generate())
    sources = tuple(tmp_path / name / "train.jsonl" for name in payloads)
    combined = tmp_path / "reviewed" / "train.jsonl"
    write_combined_jsonl(sources, combined)
    plan = plan_sft(SFTSettings(data=combined, log_dir=tmp_path / "logs", model="demo", batch_size=1))
    assert plan.rows == 3
    prompts = jobs_from_file(combined, None)
    assert len(prompts) == 3
    assert len({job.id for job in prompts}) == 3
    assert {job.meta["pipeline"] for job in prompts} == set(payloads)
