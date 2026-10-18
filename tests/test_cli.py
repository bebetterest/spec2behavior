import json
from pathlib import Path

import httpx
import pytest

from synthesis.cli import main
from synthesis.client import ChatClient


@pytest.fixture(autouse=True)
def isolate_cli_config(tmp_path, monkeypatch):
    """Never read a developer's real .env or credentials in CLI tests."""

    monkeypatch.chdir(tmp_path)
    for key in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY", "OPENAI_BASE_URL", "SYNTHESIS_MODEL"):
        monkeypatch.delenv(key, raising=False)


def test_env_file_is_literal_and_cli_then_environment_take_precedence(tmp_path, monkeypatch, capsys):
    source = tmp_path / "policy.md"
    source.write_text("# Rule\n\nFollow the rule.\n", encoding="utf-8")
    file_key = "file-${TOKEN}-key"
    (tmp_path / ".env").write_text(
        f'export OPENROUTER_API_KEY="{file_key}" # a comment\nSYNTHESIS_MODEL=file-model\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("TOKEN", "must-not-expand")
    sent = []

    def handler(request):
        sent.append((request.headers["Authorization"], json.loads(request.content)["model"]))
        return httpx.Response(200, json={"choices": [{
            "message": {"content": '{"reflection":"Follow the rule."}'}, "finish_reason": "stop",
        }]})

    monkeypatch.setattr("synthesis.cli.ChatClient", lambda **kwargs: ChatClient(
        **kwargs, transport=httpx.MockTransport(handler), trust_env=False,
    ))
    command = ["run", "reflection", "--source", str(source), "--provider", "openrouter"]
    assert main(command + ["--output", str(tmp_path / "from-file")]) == 0
    assert sent[-1] == (f"Bearer {file_key}", "file-model")
    import os
    assert "OPENROUTER_API_KEY" not in os.environ
    assert "SYNTHESIS_MODEL" not in os.environ

    monkeypatch.setenv("OPENROUTER_API_KEY", "process-key")
    monkeypatch.setenv("SYNTHESIS_MODEL", "process-model")
    assert main(command + ["--output", str(tmp_path / "from-environment")]) == 0
    assert sent[-1] == ("Bearer process-key", "process-model")

    custom = tmp_path / "custom.env"
    custom.write_text("OPENROUTER_API_KEY=custom-key\nSYNTHESIS_MODEL=custom-model\n", encoding="utf-8")
    assert main(command + ["--output", str(tmp_path / "from-flags"), "--env-file", str(custom),
                           "--api-key", "explicit-key", "--model", "explicit-model"]) == 0
    assert sent[-1] == ("Bearer explicit-key", "explicit-model")
    monkeypatch.delenv("OPENROUTER_API_KEY")
    monkeypatch.delenv("SYNTHESIS_MODEL")
    assert main(command + ["--output", str(tmp_path / "from-custom"), "--env-file", str(custom)]) == 0
    assert sent[-1] == ("Bearer custom-key", "custom-model")

    assert main(command + ["--output", str(tmp_path / "preview"), "--dry-run"]) == 0
    captured = capsys.readouterr()
    assert file_key not in captured.out + captured.err
    assert len(sent) == 4
    assert not (tmp_path / "preview").exists()


def test_explicit_missing_env_file_fails_without_writing(tmp_path, capsys):
    output = tmp_path / "out"
    assert main(["run", "reflection", "--source", "missing.md", "--output", str(output),
                 "--env-file", "missing.env", "--dry-run"]) == 2
    assert "env file not found" in capsys.readouterr().err
    assert not output.exists()


def test_list_includes_builtin_pipelines(capsys):
    assert main(["list"]) == 0
    output = capsys.readouterr().out
    names = {line.split()[0] for line in output.splitlines() if line.strip()}
    assert {
        "chain_of_command",
        "correction",
        "pressure",
        "reflection",
        "rule_examples",
        "safe_completion",
    } <= names
    assert "section_conversations" not in names
    assert "prompts" not in names


def test_dry_run_prints_a_request_and_writes_nothing(tmp_path: Path, capsys):
    source = tmp_path / "policy.md"
    source.write_text("# Part\n\n## Rule {#rule}\n\nFollow the rule.\n", encoding="utf-8")
    output = tmp_path / "out"
    code = main(
        [
            "run",
            "rule_examples",
            "--source",
            str(source),
            "--output",
            str(output),
            "--dry-run",
        ]
    )
    assert code == 0
    assert not output.exists()
    captured = capsys.readouterr()
    assert "policy.md/rule/0" in captured.err
    assert "Follow the rule." in captured.err
    body = json.loads(captured.err.split("request body:\n", 1)[1])
    assert body["max_tokens"] == 16384
    assert body["temperature"] == 1.0
    assert "thinking" not in body
    assert "Write 1 pair." in body["messages"][0]["content"]


def test_dry_run_can_switch_token_field_or_use_provider_defaults(tmp_path, capsys):
    source = tmp_path / "policy.md"
    source.write_text("# Rule\n\nFollow the rule.\n", encoding="utf-8")
    command = ["run", "reflection", "--source", str(source), "--output", str(tmp_path / "out"), "--dry-run"]
    assert main(command + ["--max-token-field", "max_completion_tokens", "--max-tokens", "512"]) == 0
    body = json.loads(capsys.readouterr().err.split("request body:\n", 1)[1])
    assert body["max_completion_tokens"] == 512
    assert "max_tokens" not in body
    assert main(command + ["--max-tokens", "default", "--temperature", "default"]) == 0
    body = json.loads(capsys.readouterr().err.split("request body:\n", 1)[1])
    assert "max_tokens" not in body
    assert "temperature" not in body


@pytest.mark.parametrize("mode,expected", [("default", None), ("on", "enabled"), ("off", "disabled")])
@pytest.mark.parametrize("provider,base_url,model", [
    ("deepseek", "https://api.deepseek.com", "deepseek-flash"),
    ("openrouter", "https://openrouter.ai/api/v1", "deepseek/deepseek-v3.2"),
])
def test_provider_thinking_matches_preview_and_saves_only_final_content(
    tmp_path, capsys, monkeypatch, mode, expected, provider, base_url, model,
):
    source = tmp_path / "policy.md"
    source.write_text("# Rule\n\nFollow the rule.\n", encoding="utf-8")
    output = tmp_path / "out"
    sent = []
    raw = json.dumps({"reflection": "Follow the rule without inventing exceptions."})

    def handler(request):
        assert str(request.url) == base_url + "/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-provider-key"
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{
            "message": {
                "content": raw,
                "reasoning_content": "Teacher reasoning, not training text.",
                "reasoning": "Teacher reasoning, not training text.",
                "reasoning_details": [{"type": "reasoning.text", "text": "Teacher reasoning, not training text."}],
            },
            "finish_reason": "stop",
        }]})

    monkeypatch.setattr("synthesis.cli.ChatClient", lambda **kwargs: ChatClient(
        **kwargs, transport=httpx.MockTransport(handler), trust_env=False,
    ))
    command = ["run", "reflection", "--source", str(source), "--output", str(output),
               "--model", model, "--provider", provider, "--api-key", "test-provider-key", "--thinking", mode]
    assert main(command + ["--dry-run"]) == 0
    preview = json.loads(capsys.readouterr().err.split("request body:\n", 1)[1])
    assert not sent
    assert not output.exists()
    if expected is None:
        assert "thinking" not in preview
        assert "reasoning" not in preview
    elif provider == "openrouter":
        assert preview["reasoning"] == {"enabled": mode == "on"}
        assert "thinking" not in preview
    else:
        assert preview["thinking"] == {"type": expected}
        assert "reasoning" not in preview
    assert "extra_body" not in preview
    assert main(command) == 0
    assert sent == [preview]
    row = json.loads((output / "train.jsonl").read_text())
    assert row["messages"][-1] == {"role": "assistant", "content": json.loads(raw)["reflection"]}
    assert "Teacher reasoning" not in (output / "jobs.jsonl").read_text()


@pytest.mark.parametrize("provider,endpoint,key_env", [
    ("openai", "https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY"),
    ("deepseek", "https://api.deepseek.com/chat/completions", "DEEPSEEK_API_KEY"),
    ("openrouter", "https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY"),
])
def test_provider_presets_select_their_own_endpoint_and_key(tmp_path, monkeypatch, provider, endpoint, key_env):
    source = tmp_path / "policy.md"
    source.write_text("# Rule\n\nFollow the rule.\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://unrelated.example/v1")
    for name in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setenv(name, "test-" + name)
    sent = []

    def handler(request):
        sent.append((str(request.url), request.headers["Authorization"]))
        return httpx.Response(200, json={"choices": [{
            "message": {"content": '{"reflection":"Follow the rule."}'}, "finish_reason": "stop",
        }]})

    monkeypatch.setattr("synthesis.cli.ChatClient", lambda **kwargs: ChatClient(
        **kwargs, transport=httpx.MockTransport(handler), trust_env=False,
    ))
    command = ["run", "reflection", "--source", str(source), "--provider", provider, "--model", "demo"]
    assert main(command + ["--output", str(tmp_path / "preset")]) == 0
    assert sent == [(endpoint, "Bearer test-" + key_env)]
    assert main(command + ["--output", str(tmp_path / "override"), "--base-url", "http://localhost:8000/v1",
                           "--api-key", "explicit-test-key"]) == 0
    assert sent[-1] == ("http://localhost:8000/v1/chat/completions", "Bearer explicit-test-key")


@pytest.mark.parametrize("use_environment", [False, True])
@pytest.mark.parametrize("url,key_env,field,setting", [
    ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY", "reasoning", {"enabled": False}),
    ("https://eu.openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY", "reasoning", {"enabled": False}),
    ("https://api.deepseek.com/chat/completions", "DEEPSEEK_API_KEY", "thinking", {"type": "disabled"}),
])
def test_auto_mode_detects_provider_from_the_url(tmp_path, monkeypatch, use_environment, url, key_env, field, setting):
    source = tmp_path / "policy.md"
    source.write_text("# Rule\n\nFollow the rule.\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-test-key")
    monkeypatch.setenv(key_env, "provider-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", url if use_environment else "https://unrelated.example/v1")
    sent = []

    def handler(request):
        assert str(request.url) == url
        assert request.headers["Authorization"] == "Bearer provider-test-key"
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"reflection":"Follow the rule."}'}}]})

    monkeypatch.setattr("synthesis.cli.ChatClient", lambda **kwargs: ChatClient(
        **kwargs, transport=httpx.MockTransport(handler), trust_env=False,
    ))
    command = ["run", "reflection", "--source", str(source), "--output", str(tmp_path / "out"),
               "--model", "deepseek/deepseek-v3.2", "--thinking", "off"]
    if not use_environment:
        command += ["--base-url", url]
    assert main(command) == 0
    assert sent[0][field] == setting
    assert ("thinking" in sent[0]) != ("reasoning" in sent[0])


def test_auto_mode_preserves_the_legacy_generic_key(tmp_path, monkeypatch):
    source = tmp_path / "policy.md"
    source.write_text("# Rule\n\nFollow the rule.\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-test-key")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    sent = []

    def handler(request):
        assert request.headers["Authorization"] == "Bearer legacy-test-key"
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"reflection":"Follow the rule."}'}}]})

    monkeypatch.setattr("synthesis.cli.ChatClient", lambda **kwargs: ChatClient(
        **kwargs, transport=httpx.MockTransport(handler), trust_env=False,
    ))
    assert main(["run", "reflection", "--source", str(source), "--output", str(tmp_path / "out"),
                 "--base-url", "https://api.deepseek.com", "--model", "demo", "--thinking", "on"]) == 0
    assert sent[0]["thinking"] == {"type": "enabled"}


def test_bad_endpoint_is_reported_without_writing(tmp_path, capsys):
    output = tmp_path / "out"
    assert main(["run", "reflection", "--source", "missing.md", "--output", str(output),
                 "--base-url", "http://[", "--dry-run"]) == 2
    assert "Invalid IPv6 URL" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("provider,key_env", [("deepseek", "DEEPSEEK_API_KEY"), ("openrouter", "OPENROUTER_API_KEY")])
def test_explicit_provider_does_not_fall_back_to_another_providers_key(tmp_path, monkeypatch, capsys, provider, key_env):
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-test-key")
    monkeypatch.delenv(key_env, raising=False)
    output = tmp_path / "out"
    assert main(["run", "reflection", "--source", "missing.md", "--output", str(output),
                 "--model", "demo", "--provider", provider]) == 2
    assert key_env in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("flag,value", [
    ("--max-tokens", "0"), ("--concurrency", "0"), ("--max-retries", "-1"),
    ("--timeout", "nan"), ("--retry-backoff", "inf"), ("--temperature", "nan"),
    ("--thinking", "enabled"),
    ("--provider", "unknown"),
])
def test_bad_generation_settings_fail_before_writing(tmp_path, flag, value):
    output = tmp_path / "out"
    with pytest.raises(SystemExit) as caught:
        main(["run", "reflection", "--source", "missing.md", "--output", str(output), "--dry-run", flag, value])
    assert caught.value.code == 2
    assert not output.exists()


def test_truncation_reports_failure_and_can_be_retried_without_losing_successes(tmp_path, monkeypatch):
    source = tmp_path / "policy.md"
    source.write_text("# Rule\n\nFollow the rule.\n", encoding="utf-8")
    output = tmp_path / "out"
    state = {"reason": "length", "calls": 0}
    raw = json.dumps({"examples": [{"request": "May I skip?", "reply": "No."}]})

    def handler(request):
        state["calls"] += 1
        return httpx.Response(200, json={"choices": [{
            "message": {"content": raw}, "finish_reason": state["reason"],
        }]})

    monkeypatch.setattr("synthesis.cli.ChatClient", lambda **kwargs: ChatClient(
        **kwargs, transport=httpx.MockTransport(handler), trust_env=False,
    ))
    command = ["run", "rule_examples", "--source", str(source), "--output", str(output),
               "--model", "demo", "--base-url", "https://example.com/v1", "--max-tokens", "512"]
    assert main(command) == 1
    error = json.loads((output / "errors.jsonl").read_text())
    assert "finish_reason=length" in error["error"]
    assert error["raw"] == raw
    assert not (output / "train.jsonl").read_text()
    assert main(command) == 1
    assert state["calls"] == 1
    state["reason"] = "stop"
    assert main(command + ["--retry-errors", "--max-tokens", "16384"]) == 0
    rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert [row["meta"]["sft_form"] for row in rows] == ["generate", "act"]
    assert state["calls"] == 2
    assert main(command) == 0
    assert state["calls"] == 2


def test_pipeline_file_is_loaded_before_the_command(tmp_path: Path, capsys):
    path = tmp_path / "file_chain.py"
    path.write_text(
        "\n".join(
            [
                "from synthesis.pipeline import Pipeline, register",
                "from synthesis.types import ChatRequest, Example, Job",
                "",
                "@register",
                "class FileChain(Pipeline):",
                "    name = 'file_chain'",
                "    description = 'Loaded from a file.'",
                "",
                "    def jobs(self, args):",
                "        del args",
                "        yield Job(id='only', input={})",
                "",
                "    def request(self, job, args):",
                "        del job, args",
                "        return ChatRequest(messages=[{'role': 'user', 'content': 'hello from file'}])",
                "",
                "    def parse(self, job, content, args):",
                "        del job, args",
                "        return [Example(messages=[{'role': 'user', 'content': 'hello from file'}, {'role': 'assistant', 'content': content}])]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert main(["list", "--pipeline-file", str(path)]) == 0
    assert "file_chain" in capsys.readouterr().out
    output = tmp_path / "out"
    code = main(
        [
            "run",
            "--pipeline-file",
            str(path),
            "file_chain",
            "--output",
            str(output),
            "--dry-run",
        ]
    )
    assert code == 0
    assert "hello from file" in capsys.readouterr().err
    assert not output.exists()


def test_template_file_defines_read_prompt_and_parse(tmp_path: Path, capsys):
    path = tmp_path / "gloss.py"
    path.write_text(
        "\n".join(
            [
                "name = 'gloss'",
                "description = 'One string in, one conversation out.'",
                "def add_arguments(parser):",
                "    parser.add_argument('--text', required=True)",
                "def read(args):",
                "    yield {'id': 'only', 'text': args.text}",
                "def prompt(item, args):",
                "    del args",
                "    return 'Gloss: ' + item['text']",
                "def parse(item, content, args):",
                "    del args",
                "    return [",
                "        {'role': 'user', 'content': item['text']},",
                "        {'role': 'assistant', 'content': content},",
                "    ]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "out"
    code = main(
        [
            "run",
            "--template",
            str(path),
            "--output",
            str(output),
            "--text",
            "hello",
            "--dry-run",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "gloss" in captured.err
    assert "Gloss: hello" in captured.err
    assert not output.exists()


def test_materialize_rebuilds_train_jsonl(tmp_path: Path, capsys):
    output = tmp_path / "out"
    output.mkdir()
    job = {
        "id": "job-1",
        "pipeline": "prompts",
        "examples": [
            {
                "id": "job-1#0",
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ],
                "meta": {},
            }
        ],
        "raw": "Hello",
    }
    (output / "jobs.jsonl").write_text(json.dumps(job) + "\n", encoding="utf-8")
    assert main(["materialize", str(output)]) == 0
    train = (output / "train.jsonl").read_text(encoding="utf-8")
    assert json.loads(train)["job_id"] == "job-1"
    assert "training rows" in capsys.readouterr().out
