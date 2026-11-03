"""Offline protocol checks. Public assets are opt-in and never fetched by tests."""

import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("inspect_ai")
pytest.importorskip("inspect_evals")
pytest.importorskip("tinker_cookbook")

from inspect_ai import eval_async
from inspect_ai.model import (ChatMessageAssistant, ChatMessageSystem, ChatMessageTool,
                              ChatMessageUser, GenerateConfig, Model)
from inspect_ai.scorer import Score, Target
from inspect_ai.solver import TaskState
from inspect_ai.model import ModelName, ModelOutput
from inspect_ai.tool import ToolCall
from tinker_cookbook.eval.inspect_utils import convert_inspect_messages

from evaluation.audit import audit, screen_overlap
from evaluation.cli import main, training_runs
from evaluation.compare import paired_summary, value_for_pair
from evaluation.core import (EXPECTED_COUNTS, canonical_hash, installed_environment, make_tasks,
                             sha256, task_records, validate_config, write_json)
from evaluation.costs import estimate_costs, input_messages, pilot_ids
from evaluation.provider import PairedTinkerAPI, TinkerJudgeAPI, offline_judge, register_local_tokenizer
from evaluation.runner import comparison_binding, local_inspect_runtime, make_judge, training_binding, verify_plan

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config():
    return json.loads((ROOT / "evaluation/config.example.json").read_text())


@pytest.fixture(scope="module")
def upstream():
    cache = os.environ.get("SPEC2BEHAVIOR_EVAL_ASSETS")
    if not cache:
        pytest.skip("Set SPEC2BEHAVIOR_EVAL_ASSETS to a prepared local data cache")
    judge = offline_judge()
    return make_tasks(ROOT, Path(cache).resolve(), judge), judge


@pytest.fixture(scope="module")
def frozen_target():
    run = os.environ.get("SPEC2BEHAVIOR_EVAL_TRAINING_RUN")
    if not run:
        pytest.skip("Set SPEC2BEHAVIOR_EVAL_TRAINING_RUN to a completed local SFT run")
    binding = training_binding(ROOT, Path(run).resolve())
    tokenizer = register_local_tokenizer(binding["model"], ROOT / binding["tokenizer_snapshot"])
    return binding, tokenizer


@pytest.fixture(scope="module")
def frozen_judge(config):
    path = os.environ.get("SPEC2BEHAVIOR_EVAL_JUDGE")
    if not path:
        pytest.skip("Set SPEC2BEHAVIOR_EVAL_JUDGE to a prepared judge tokenizer cache")
    prepared = Path(path).resolve()
    saved = json.loads((prepared / "judge_tokenizer/manifest.json").read_text())
    tokenizer = register_local_tokenizer(config["judge"]["model"], prepared / saved["path"])
    return prepared / saved["path"], tokenizer


class FakeSamplingClient:
    def __init__(self, tokenizer, text="Private offline reasoning.</think>Offline visible answer.<|im_end|>", stop="stop"):
        self.tokenizer, self.text, self.stop = tokenizer, text, stop
        self.calls = []

    async def sample_async(self, **kwargs):
        self.calls.append(kwargs)
        await asyncio.sleep(0)
        text = self.text(kwargs) if callable(self.text) else self.text
        return SimpleNamespace(sequences=[SimpleNamespace(
            tokens=self.tokenizer.encode(text, add_special_tokens=False), stop_reason=self.stop)])


def harmony_reply(text, stop=True):
    return ("<|channel|>analysis<|message|>Private judge reasoning<|end|>"
            "<|start|>assistant<|channel|>final<|message|>" + text + ("<|return|>" if stop else ""))


class FakeJudgeClient(FakeSamplingClient):
    def __init__(self, tokenizer):
        def respond(call):
            prompt = tokenizer.decode(call["prompt"].to_ints())
            if "Return exactly one JSON object" in prompt:
                text = '{"section_id":"offline","critique":"Offline verdict","compliance":7}'
            elif "There are three classes" in prompt:
                text = "Offline verdict. GRADE: C"
            else:
                text = "#overall_reasoning Offline verdict\n#scores\n1.b 1\n2.b 1\n3.b 1"
            return harmony_reply(text)
        super().__init__(tokenizer, respond)


def api_for(binding, client, task="model_spec", config=None):
    return PairedTinkerAPI(task_name=task, sampling_client=client, renderer_name=binding["renderer"],
                          model_name=binding["model"], config=config or GenerateConfig())


def write_test_plan(directory, plan):
    plan.setdefault("targets", {"base": None, "sft": "offline-checkpoint"})
    plan["review_artifacts"] = {name: sha256(directory / name) for name in ["samples.json", "overlap_summary.json"]}
    plan["plan_sha256"] = canonical_hash(plan)
    write_json(directory / "plan.json", plan)


def test_overlap_finds_history_and_paraphrase_candidates_but_separates_source():
    source = "one two three four five six seven eight nine ten eleven twelve thirteen"
    train = [{"id": "a", "messages": [{"role": "user", "content": "<excerpt>" + source + "</excerpt>\nNew scenario."},
                                        {"role": "assistant", "content": "Hello! How can I assist you today?"}]},
             {"id": "b", "messages": [{"role": "user", "content": "Long output: " + source}]}]
    records = {"test": [{"id": "1", "input": source, "target": "", "metadata": {}},
                        {"id": "2", "input": [{"role": "assistant", "content": "Hello! How can I assist you today?"},
                                                 {"role": "user", "content": "A different query"}], "target": "", "metadata": {}}]}
    result = screen_overlap(records, train, {"spec": source})
    assert {h["training_id"] for h in result["generated_text_12word_matches"]} == {"b"}
    assert result["source_12word_matches"][0]["eval_id"] == "1"
    assert result["exact_input_messages"][0]["eval_role"] == "assistant"


def test_overlap_includes_tool_history_arguments():
    query = "one two three four five six seven eight nine ten eleven twelve thirteen"
    records = {"model_spec": [{"id": "tools", "input": [
        {"role": "assistant", "content": "", "tool_calls": [{"arguments": {"query": query}}]},
        {"role": "user", "content": "New question"}], "target": "", "metadata": {}}]}
    train = [{"id": "train", "messages": [{"role": "user", "content": query}]}]
    result = screen_overlap(records, train, {})
    hit, = result["generated_text_12word_matches"]
    assert hit["eval_field"] == "tool_calls[0].arguments" and hit["eval_role"] == "assistant"
    assert hit["shared_words"] == 13 and hit["training_id"] == "train"


def test_reprepare_preserves_manual_overlap_unless_explicitly_replaced(tmp_path):
    (tmp_path / "generic").mkdir()
    for name in ["model_spec.md", "constitution.md"]:
        (tmp_path / "generic" / name).write_text("Unrelated source policies.")
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps({"id": "t", "messages": [{"role": "user", "content": "A different training request"}]}) + "\n")
    records = {"model_spec": [{"id": "reviewed", "input": "Benchmark scenario awaiting review", "target": "", "metadata": {}}]}
    manual = [{"benchmark": "model_spec", "eval_id": "reviewed", "kind": "Shared scenario",
               "interpretation": "Confirmed by manual review"}]
    output = tmp_path / "prepared"
    write_json(output / "overlap_summary.json", {"manual_findings": manual})
    report = audit(tmp_path, records, train, output)
    assert report["manual_findings"] == manual and report["flagged_ids"]["model_spec"] == ["reviewed"]
    empty = tmp_path / "replacement.json"
    write_json(empty, [])
    report = audit(tmp_path, records, train, output, empty)
    assert report["manual_findings"] == [] and report["flagged_ids"]["model_spec"] == []


def test_costs_count_two_targets_and_separate_reasoning_from_visible_answer(config):
    rows = {"test": [{"id": "a", "target_input_tokens": 100, "judge_fixed_input_tokens": 1000}]}
    report = estimate_costs(rows, config, {"full": {"test": ["a"]}})["full"]
    scenario = report["scenarios"][0]
    expected = (200 * .184 + 2 * 1024 * .495 + (2000 + 2 * 256) * .33 + 2 * 512 * .84) / 1e6
    assert report["target_calls"] == report["judge_calls_max"] == 2
    assert scenario["total_usd"] == pytest.approx(expected)
    assert scenario["with_review_contingency_usd"] == pytest.approx(expected * 1.2)
    four = estimate_costs(rows, config, {"full": {"test": ["a"]}}, target_count=4)["full"]
    assert four["target_count"] == four["target_calls"] == four["judge_calls_max"] == 4
    assert four["scenarios"][0]["total_usd"] == pytest.approx(2 * expected)


def test_protocol_rejects_extra_epochs_and_implicit_retry(config):
    for group, key, value in [("sampling", "epochs", 2), ("judge", "samples", 2), ("execution", "max_retries", 1), ("judge", "max_tokens", 0)]:
        invalid = deepcopy(config)
        invalid[group][key] = value
        with pytest.raises(ValueError):
            validate_config(invalid)


def test_config_adjustments_reject_typos_missing_keys_and_invalid_budgets(config):
    validate_config(config)
    invalids = []
    for group, key, value in [("judge", "model_spec_grading", "typo"), ("judge", "top_p", 0), ("judge", "max_tokens", 131072),
                              ("judge", "max_token", 512), ("pricing", "judge_input_usd_per_million", float("nan"))]:
        changed = deepcopy(config)
        changed[group][key] = value
        invalids.append(changed)
    changed = deepcopy(config)
    del changed["judge"]["model"]
    invalids.append(changed)
    for changed in invalids:
        with pytest.raises(ValueError):
            validate_config(changed)


def test_cost_scenarios_follow_adjusted_caps_and_contingency(config):
    changed = deepcopy(config)
    changed["sampling"]["max_tokens"] = 128
    changed["judge"]["max_tokens"] = 64
    changed["pricing"]["contingency_fraction"] = .1
    validate_config(changed)
    rows = {"test": [{"id": "a", "target_input_tokens": 100, "judge_fixed_input_tokens": 1000}]}
    estimates = estimate_costs(rows, changed, {"pilot": {"test": ["a"]}})["pilot"]["scenarios"]
    for row in estimates:
        assert row["target_total_output_tokens"] == row["target_final_answer_tokens"] == 128
        assert row["judge_total_output_tokens"] == 64
        assert row["with_review_contingency_usd"] == pytest.approx(row["total_usd"] * 1.1)


@pytest.mark.parametrize("selection", [[], ["unknown"], ["model_spec", "model_spec"]])
def test_invalid_benchmark_selection_fails_before_loading_assets(tmp_path, selection):
    with pytest.raises(ValueError, match="unique known benchmarks"):
        make_tasks(tmp_path, tmp_path, None, benchmarks=selection)


def test_model_spec_selection_preserves_upstream_samples(upstream):
    all_tasks, judge = upstream
    tasks = make_tasks(ROOT, Path(os.environ["SPEC2BEHAVIOR_EVAL_ASSETS"]).resolve(), judge,
                       benchmarks=["model_spec"])
    assert list(tasks) == ["model_spec"]
    assert task_records(tasks)["model_spec"] == task_records(all_tasks)["model_spec"]


def test_preview_does_not_enter_execute_or_load_model_clients(tmp_path, monkeypatch):
    import evaluation.runner as runner

    costs = {"pilot": {"target_calls": 2, "judge_calls_max": 2, "benchmarks": {"test": {"samples_per_model": 1}},
                       "scenarios": [{"name": "test", "total_usd": 1, "with_review_contingency_usd": 1.2}]}}
    (tmp_path / "costs.json").write_text(json.dumps({"profiles": costs}))
    monkeypatch.setattr(runner, "verify_plan", lambda *args: {"plan_sha256": "frozen", "config": {"judge": {"model": "native-judge", "model_spec_grading": "rubric-only"}}, "targets": {"base": None, "sft": "checkpoint"}})
    def forbidden(*args, **kwargs):
        raise AssertionError("Preview constructed a paid client")
    monkeypatch.setattr(runner, "execute", forbidden)
    assert main(["run", "--plan", str(tmp_path)]) == 0
    assert main(["run", "--plan", str(tmp_path), "--execute"]) == 2  # no output directory


def test_frozen_plan_detects_an_edit_before_constructing_tasks(tmp_path):
    plan = {"schema_version": 1, "config": {"changed": True}}
    plan["plan_sha256"] = canonical_hash({"schema_version": 1, "config": {"changed": False}})
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="Frozen plan changed"):
        verify_plan(ROOT, tmp_path)


def test_named_training_runs_bind_distinct_checkpoints_and_shared_setup(tmp_path, monkeypatch):
    import evaluation.runner as runner

    runs = training_runs([f"low={tmp_path}/low", f"high={tmp_path}/high"])
    assert list(runs) == ["low", "high"]
    def binding(root, path, tokenizer_runs=()):
        return {"model": "same-model", "renderer": "same-renderer", "project_id": "same-project",
                "training_data": "same-data", "tokenizer_revision": "same-tokenizer", "tokenizer_snapshot": "snapshot",
                "training_run": path.name, "training_files": {path.name: "digest"},
                "targets": {"base": None, "sft": "tinker://" + path.name}}
    monkeypatch.setattr(runner, "training_binding", binding)
    merged = comparison_binding(tmp_path, runs)
    assert merged["targets"] == {"base": None, "low": "tinker://low", "high": "tinker://high"}
    assert merged["training_runs"] == {"low": "low", "high": "high"}
    assert set(merged["training_files"]) == {"low", "high"}
    with pytest.raises(ValueError, match="distinct"):
        comparison_binding(tmp_path, {"a": tmp_path / "same", "b": tmp_path / "same"})
    with pytest.raises(ValueError, match="reserved"):
        comparison_binding(tmp_path, {"base": tmp_path})
    with pytest.raises(ValueError, match="unique"):
        training_runs(["a=one", "a=two"])
    def mismatched(root, path, tokenizer_runs=()):
        return {**binding(root, path), "renderer": path.name}
    monkeypatch.setattr(runner, "training_binding", mismatched)
    with pytest.raises(ValueError, match="must share"):
        comparison_binding(tmp_path, runs)


def test_training_bindings_reuse_only_the_declared_frozen_tokenizer(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text("one frozen training row\n")
    first, second = tmp_path / "first", tmp_path / "second"
    cfg = {"model": "Example/Model", "rows": 1, "data": "train.jsonl", "data_sha256": sha256(data),
           "renderer": "example", "project_id": "project"}
    for run in [first, second]:
        write_json(run / "config.json", cfg)
        write_json(run / "status.json", {"state": "complete", "trained_rows": 1})
        write_json(run / "tokenizer_revision.json", {"model": cfg["model"], "revision": "frozen-commit"})
        (run / "checkpoints.jsonl").write_text(json.dumps({"name": "final", "sampler_path": "tinker://" + run.name,
            "binding": {"dataset_sha256": cfg["data_sha256"], "config_sha256": sha256(run / "config.json")}}) + "\n")
    snapshot = first / "cache/huggingface/hub/models--Example--Model/snapshots/frozen-commit"
    write_json(snapshot / "tokenizer.json", {})
    with pytest.raises(ValueError, match="tokenizer is unavailable"):
        training_binding(tmp_path, second)
    binding = training_binding(tmp_path, second, [first])
    assert binding["tokenizer_snapshot"] == str(snapshot.relative_to(tmp_path))
    assert str((snapshot / "tokenizer.json").relative_to(tmp_path)) in binding["training_files"]
    write_json(second / "tokenizer_revision.json", {"model": cfg["model"], "revision": "different-commit"})
    with pytest.raises(ValueError, match="tokenizer is unavailable"):
        training_binding(tmp_path, second, [first])


def test_environment_lock_ignores_uninstalled_metadata_on_sys_path(tmp_path, monkeypatch):
    before = installed_environment()
    metadata = tmp_path / "uninstalled_eval_test-9.9.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: uninstalled-eval-test\nVersion: 9.9\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert installed_environment() == before
    assert "uninstalled-eval-test" not in before


def test_upstream_dataset_counts_ids_roles_and_rubrics(upstream):
    tasks, _ = upstream
    assert {name: len(t.dataset) for name, t in tasks.items()} == EXPECTED_COUNTS
    for name, rows in task_records(tasks).items():
        assert len({r["id"] for r in rows}) == len(rows)
    spec = tasks["model_spec"].dataset
    assert sum(any(m.role == "tool" for m in input_messages(s)) for s in spec) == 4
    assert sum(any(m.role == "assistant" for m in input_messages(s)) for s in spec) == 39
    assert all(s.metadata["label"] == "safe" for s in tasks["xstest_safe"].dataset)
    assert all(s.metadata["label"] == "unsafe" for s in tasks["xstest_unsafe"].dataset)
    selected = pilot_ids(tasks, 5)
    for sample in spec:
        if any(m.role == "tool" for m in input_messages(sample)):
            assert sample.id in selected["model_spec"]
    # Independent loads get random Inspect message ids, but the benchmark
    # fingerprint must be stable and still retain original tool-call ids.
    again = make_tasks(ROOT, Path(os.environ["SPEC2BEHAVIOR_EVAL_ASSETS"]).resolve(), offline_judge())
    assert task_records(again) == task_records(tasks)
    tools = [m for row in task_records(tasks)["model_spec"] for m in row["input"] if m["role"] == "tool"]
    assert tools and all(m.get("tool_call_id") for m in tools)


def test_official_conversion_retains_tool_call_links():
    calls = [ToolCall(id="call_canary", function="lookup", arguments={"q": "question"})]
    messages = [ChatMessageAssistant(content="", tool_calls=calls),
                ChatMessageTool(content="result", tool_call_id="call_canary", function="lookup")]
    converted = convert_inspect_messages(messages)
    assert converted[0]["tool_calls"][0].id == "call_canary"
    assert converted[1]["tool_call_id"] == "call_canary"
    assert converted[1]["name"] == "lookup"


@pytest.mark.parametrize("stop,text,visible", [
    ("stop", "Private offline reasoning.</think>Offline visible answer.<|im_end|>", "Offline visible answer."),
    ("length", "Private offline reasoning only", ""),
    ("length", "Private offline reasoning.</think>Truncated visible answer", "Truncated visible answer"),
])
def test_reasoning_separation_and_length_evidence(frozen_target, stop, text, visible):
    binding, tokenizer = frozen_target
    client = FakeSamplingClient(tokenizer, text, stop)
    api = api_for(binding, client)
    output = asyncio.run(api.generate([ChatMessageUser(content="A canary")], [], "auto", GenerateConfig(max_tokens=8192, temperature=0)))
    assert output.completion == visible
    assert output.message.text == visible
    assert output.stop_reason == ("max_tokens" if stop == "length" else "stop")
    assert "Private offline reasoning" in output.metadata["tinker_sequences"][0]["text"]
    assert output.usage.output_tokens == len(tokenizer.encode(text, add_special_tokens=False))


def test_native_developer_mapping_is_limited_to_model_spec(frozen_target):
    binding, tokenizer = frozen_target
    messages = [ChatMessageSystem(content="ROLE_CANARY"), ChatMessageUser(content="Hello")]
    for task, role in [("model_spec", "developer"), ("xstest_safe", "system")]:
        client = FakeSamplingClient(tokenizer)
        asyncio.run(api_for(binding, client, task).generate(messages, [], "auto", GenerateConfig(max_tokens=8192)))
        prompt = tokenizer.decode(client.calls[0]["prompt"].to_ints())
        assert f"<|im_start|>{role}\nROLE_CANARY" in prompt
        assert "<BEGIN_RUBRIC>" not in prompt
        assert messages[0].role == "system"


def test_concurrent_sampling_evidence_is_not_shared(frozen_target):
    binding, tokenizer = frozen_target
    def response(call):
        marker = "ALPHA" if "ALPHA" in tokenizer.decode(call["prompt"].to_ints()) else "BETA"
        return f"Private reasoning.</think>{marker}<|im_end|>"
    api = api_for(binding, FakeSamplingClient(tokenizer, response))
    async def generate():
        return await asyncio.gather(*(api.generate([ChatMessageUser(content=x)], [], "auto", GenerateConfig(max_tokens=8192)) for x in ["ALPHA", "BETA"]))
    first, second = asyncio.run(generate())
    assert first.completion == "ALPHA" and second.completion == "BETA"
    assert "ALPHA" in first.metadata["tinker_sequences"][0]["text"]
    assert "BETA" in second.metadata["tinker_sequences"][0]["text"]


def test_judge_configuration_survives_all_upstream_scorers(upstream):
    tasks, judge = upstream
    judge.config = GenerateConfig(max_retries=0, max_tokens=4096, temperature=0, top_p=1, max_connections=4)
    async def score():
        for name, task in tasks.items():
            sample = task.dataset[0]
            state = TaskState(model=ModelName(judge), sample_id=sample.id, epoch=1, input=sample.input,
                              messages=input_messages(sample) + [ChatMessageAssistant(content="Visible only")],
                              output=ModelOutput.from_content("offline", "Visible only"), metadata=sample.metadata)
            await task.scorer[0](state, Target(sample.target))
            captured_messages, captured_config = judge.api.calls[-1]
            assert captured_config.max_tokens == 4096
            assert captured_config.temperature == 0
            assert captured_config.max_retries == 0
            assert captured_config.max_connections == 4
            if name == "model_spec":
                assert captured_config.seed == 42 and captured_config.response_schema
            assert "Private offline reasoning" not in " ".join(m.text for m in captured_messages)
    asyncio.run(score())


def test_model_spec_modes_change_only_the_judge_prompt(upstream, config):
    from evaluation.core import spec_modules

    _, judge = upstream
    data = Path(os.environ["SPEC2BEHAVIOR_EVAL_ASSETS"]).resolve()
    _, source = spec_modules(ROOT, data)
    results, prompts, fingerprints = {}, {}, {}
    async def check():
        for mode in ["rubric-only", "full-spec"]:
            task = make_tasks(ROOT, data, judge, mode)["model_spec"]
            fingerprints[mode] = task_records({"model_spec": task})
            sample = task.dataset[0]
            state = TaskState(model=ModelName(judge), sample_id=sample.id, epoch=1, input=sample.input,
                messages=input_messages(sample) + [ChatMessageAssistant(content="Visible candidate")], metadata=sample.metadata,
                output=ModelOutput.from_content("offline", "Visible candidate"))
            results[mode] = await task.scorer[0](state, Target(sample.target))
            prompts[mode] = judge.api.calls[-1][0][-1].text
            assert sample.target in prompts[mode]
    asyncio.run(check())
    assert fingerprints["rubric-only"] == fingerprints["full-spec"]
    assert results["rubric-only"].value == results["full-spec"].value == 1
    assert source.model_spec not in prompts["rubric-only"] and source.model_spec in prompts["full-spec"]
    assert len(prompts["rubric-only"]) < 10000
    assert config["judge"]["model_spec_grading"] == "rubric-only"


def test_real_inspect_pipeline_with_fake_tinker_and_judge(upstream, frozen_target, tmp_path):
    tasks, judge = upstream
    binding, tokenizer = frozen_target
    selected = pilot_ids(tasks, 5)
    async def run():
        for name, task in tasks.items():
            client = FakeSamplingClient(tokenizer)
            cfg = GenerateConfig(max_retries=0, max_tokens=8192, temperature=0, top_p=1, seed=20261020, cache=False)
            model = Model(api_for(binding, client, name, cfg), cfg)
            # Include every Model Spec tool history; one item per other task.
            ids = selected[name] if name == "model_spec" else selected[name][:1]
            logs = await eval_async(task, model=model, sample_id=ids, epochs=1, max_samples=4,
                                    log_dir=str(tmp_path / name), log_format="json", retry_on_error=0,
                                    fail_on_error=True, ctl_server=False, acp_server=False, **cfg.model_dump(exclude_none=True))
            assert len(logs) == 1 and logs[0].status == "success"
            assert logs[0].results.completed_samples == len(ids)
            assert len(client.calls) == len(ids)
            for call in client.calls:
                assert call["sampling_params"].temperature == 0
                assert call["sampling_params"].max_tokens == 8192
                assert call["sampling_params"].seed == 20261020
                assert "<BEGIN_RUBRIC>" not in tokenizer.decode(call["prompt"].to_ints())
            for sample in logs[0].samples:
                assert sample.output.completion == "Offline visible answer."
                assert sample.output.metadata["tinker_sequences"][0]["parse_success"] is True
    with local_inspect_runtime(tmp_path):
        asyncio.run(run())


def test_pair_metrics_preserve_scales_and_invalid_scores():
    assert value_for_pair("model_spec", Score(value=0, explanation="Could not extract any critiques.")) is None
    assert value_for_pair("model_spec", Score(value=1, metadata={"critiques": [{"compliance": 6}]})) == 1
    assert value_for_pair("xstest_safe", Score(value="P")) == 100
    assert value_for_pair("xstest_unsafe", Score(value="C")) == 0
    assert value_for_pair("strong_reject", Score(value={"refusal": 0, "specificity": 5, "convincingness": 3})) == 4
    assert value_for_pair("strong_reject", Score(value=float("nan"))) is None
    paired = paired_summary([(0, 1), (1, 1), (0, 0)])
    assert paired["target_minus_base"] == pytest.approx(1/3)
    assert paired["ci95"][0] <= 1/3 <= paired["ci95"][1]


@pytest.mark.parametrize("text,stop,visible", [
    (harmony_reply("Visible verdict"), "stop", "Visible verdict"),
    ("<|channel|>analysis<|message|>Private judge reasoning only", "length", ""),
    (harmony_reply("Partial verdict", stop=False), "length", "Partial verdict"),
])
def test_harmony_thinking_and_truncated_final_are_separated(config, frozen_judge, text, stop, visible):
    _, tokenizer = frozen_judge
    client = FakeSamplingClient(tokenizer, text, stop)
    api = TinkerJudgeAPI(task_name="judge", sampling_client=client, model_name=config["judge"]["model"],
        renderer_name=config["judge"]["renderer"], renderer_date="2000-01-02", context_window=131072)
    output = asyncio.run(api.generate([ChatMessageUser(content="Question")], [], "auto", GenerateConfig(max_tokens=4096)))
    assert output.completion == visible
    assert "Private judge reasoning" in output.metadata["tinker_sequences"][0]["text"]
    assert output.stop_reason == ("max_tokens" if stop == "length" else "stop")


def test_native_judge_schema_seed_date_budget_and_invalid_verdict(config, frozen_judge, upstream):
    _, tokenizer = frozen_judge
    tasks, _ = upstream
    client = FakeSamplingClient(tokenizer, harmony_reply("Not JSON"))
    class Service:
        def create_sampling_client(self, **kwargs):
            assert kwargs["base_model"] == config["judge"]["model"]
            assert kwargs["retry_config"].enable_retry_logic is False
            return client
    judge = make_judge(config, Service(), "2000-01-02")
    sample = tasks["model_spec"].dataset[0]
    native_tasks = make_tasks(ROOT, Path(os.environ["SPEC2BEHAVIOR_EVAL_ASSETS"]).resolve(), judge)
    state = TaskState(model=ModelName(judge), sample_id=sample.id, epoch=1, input=sample.input,
        messages=input_messages(sample) + [ChatMessageAssistant(content="Visible candidate")], metadata=sample.metadata,
        output=ModelOutput.from_content("offline", "Visible candidate"))
    score = asyncio.run(native_tasks["model_spec"].scorer[0](state, Target(sample.target)))
    assert score.value == 0 and "Could not extract" in score.explanation
    assert len(client.calls) == 1
    call = client.calls[0]; prompt = tokenizer.decode(call["prompt"].to_ints())
    assert "Return exactly one JSON object" in prompt and '"compliance"' in prompt
    assert '"minimum": 1' in prompt and '"maximum": 7' in prompt
    assert '"additionalProperties": false' in prompt
    assert "Current date: 2000-01-02" in prompt and "Reasoning: low" in prompt
    assert "<|start|>developer" in prompt
    assert "scenario-specific rubric" in prompt
    assert "# The model spec" not in prompt
    assert call["sampling_params"].max_tokens == 4096
    assert call["sampling_params"].temperature == 0 and call["sampling_params"].seed == 42
    async def check_other_scorers():
        for name in ["xstest_safe", "strong_reject"]:
            sample = native_tasks[name].dataset[0]
            state = TaskState(model=ModelName(judge), sample_id=sample.id, epoch=1, input=sample.input,
                messages=input_messages(sample) + [ChatMessageAssistant(content="Visible candidate")], metadata=sample.metadata,
                output=ModelOutput.from_content("offline", "Visible candidate"))
            verdict = await native_tasks[name].scorer[0](state, Target(sample.target))
            assert value_for_pair(name, verdict) is None
            if name == "xstest_safe":
                # Upstream deliberately treats an empty candidate as refusal,
                # without a judge call. Keep that separate from a bad verdict.
                before = len(client.calls)
                state.output = ModelOutput.from_content("offline", "")
                verdict = await native_tasks[name].scorer[0](state, Target(sample.target))
                assert value_for_pair(name, verdict) == 100 and len(client.calls) == before
    asyncio.run(check_other_scorers())


def test_context_overflow_is_rejected_before_sampling(frozen_target):
    binding, tokenizer = frozen_target
    client = FakeSamplingClient(tokenizer)
    api = PairedTinkerAPI(task_name="model_spec", sampling_client=client, model_name=binding["model"],
                         renderer_name=binding["renderer"], context_window=16)
    with pytest.raises(ValueError, match="context window"):
        asyncio.run(api.generate([ChatMessageUser(content="A question")], [], "auto", GenerateConfig(max_tokens=8192)))
    assert client.calls == []


@pytest.mark.parametrize("verdict_valid", [True, False])
@pytest.mark.parametrize("target_count,grading_mode,benchmarks", [(2, "full-spec", None), (4, "rubric-only", None), (4, "rubric-only", ["model_spec"])])
def test_paired_runner_and_saved_log_comparison_without_services(config, upstream, frozen_target, frozen_judge, tmp_path, monkeypatch, verdict_valid, target_count, grading_mode, benchmarks):
    import tinker
    import evaluation.runner as runner
    from evaluation.compare import compare_run
    from evaluation.core import write_json

    config = deepcopy(config)
    config["judge"]["model_spec_grading"] = grading_mode
    tasks, _ = upstream
    if benchmarks is not None:
        tasks = {name: tasks[name] for name in benchmarks}
    binding, tokenizer = frozen_target
    judge_path, judge_tokenizer = frozen_judge
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "data").symlink_to(Path(os.environ["SPEC2BEHAVIOR_EVAL_ASSETS"]).resolve(), target_is_directory=True)
    (prepared / "judge_tokenizer").symlink_to(judge_path, target_is_directory=True)
    selected = {name: [task.dataset[0].id] for name, task in tasks.items()}
    plan = {**binding, "config": config, "pilot_ids": selected, "benchmarks": list(tasks),
            "judge_tokenizer": {"path": "judge_tokenizer"}, "renderer_date": "2000-01-02",
            "lock": {"dataset_counts": {name: EXPECTED_COUNTS[name] for name in tasks}}}
    if target_count == 4:
        plan["targets"] = {**plan["targets"], "sft_low": "tinker://offline-low", "sft_high": "tinker://offline-high"}
    write_json(prepared / "samples.json", task_records(tasks))
    write_json(prepared / "overlap_summary.json", {"flagged_ids": {name: [] for name in tasks}})
    write_test_plan(prepared, plan)
    monkeypatch.setattr(runner, "verify_plan", lambda *args: plan)
    monkeypatch.setenv("TINKER_API_KEY", "offline-test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    clients = []
    class Service:
        def __init__(self, **kwargs):
            assert kwargs["project_id"] == binding["project_id"]
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def create_sampling_client(self, **kwargs):
            assert kwargs["retry_config"].enable_retry_logic is False
            clients.append(kwargs)
            if kwargs.get("base_model") == config["judge"]["model"]:
                return FakeJudgeClient(judge_tokenizer) if verdict_valid else FakeSamplingClient(judge_tokenizer, harmony_reply("Invalid verdict"))
            return FakeSamplingClient(tokenizer)
    monkeypatch.setattr(tinker, "ServiceClient", Service)
    output = tmp_path / "run"
    run = asyncio.run(runner.execute(ROOT, prepared, output, "pilot"))
    assert clients[0]["base_model"] == config["judge"]["model"]
    assert clients[1]["base_model"] == binding["model"]
    assert clients[2]["model_path"] == binding["targets"]["sft"]
    assert len(clients) == target_count + 1
    assert all(set(logs) == set(tasks) for logs in run["logs"].values())
    assert all(entry[0]["status"] == "success" for logs in run["logs"].values() for entry in logs.values())
    result = compare_run(prepared, output, output / "comparison.json")
    assert result["model_spec_grading"] == grading_mode
    for name, report in result["benchmarks"].items():
        assert len(report["comparisons"]) == target_count - 1
        for comparison in report["comparisons"].values():
            assert comparison["paired"]["paired_samples"] == int(verdict_valid)
            assert comparison["paired"]["target_minus_base"] == (0 if verdict_valid else None)
        assert report["diagnostics"]["base"]["judge_invalid"] == int(not verdict_valid)
        assert report["diagnostics"]["base"]["judge_calls"] == 1
        assert report["diagnostics"]["base"]["judge_parse_failures"] == 0
        if name == "model_spec":
            # FakeJudgeClient returns a schema-valid "offline" section, rather
            # than the requested section ID. Count it without changing the
            # historical pass score or the valid-pair denominator above.
            assert report["diagnostics"]["base"]["judge_section_id_checked"] == int(verdict_valid)
            assert report["diagnostics"]["base"]["judge_section_mismatch"] == int(verdict_valid)
        else:
            assert report["diagnostics"]["base"]["judge_section_id_checked"] == 0
            assert report["diagnostics"]["base"]["judge_section_mismatch"] == 0

    # A task that has not run is missing coverage, not a zero score.
    if "xstest_safe" not in tasks:
        return
    del run["logs"]["sft"]["xstest_safe"]
    write_json(output / "run.json", run)
    partial = compare_run(prepared, output, output / "partial-comparison.json")["benchmarks"]["xstest_safe"]
    assert partial["comparisons"]["sft"]["paired"]["paired_samples"] == 0
    assert partial["comparisons"]["sft"]["paired"]["target_mean"] is None
    assert partial["diagnostics"]["sft"] == {"status": "not_run", "logged_samples": 0, "missing_samples": 1}


def test_comparison_distinguishes_generation_failure_from_judge_failure(tmp_path, monkeypatch):
    from evaluation.compare import compare_run
    from evaluation.core import write_json

    write_json(tmp_path / "overlap_summary.json", {"flagged_ids": {"model_spec": []}})
    write_json(tmp_path / "samples.json", {"model_spec": [{"id": "a"}, {"id": "b"}]})
    plan = {"pilot_ids": {"model_spec": ["a", "b"]}, "lock": {"dataset_counts": {"model_spec": 2}}}
    write_test_plan(tmp_path, plan)
    write_json(tmp_path / "run.json", {"plan_sha256": plan["plan_sha256"], "profile": "pilot", "logs": {
        target: {"model_spec": [{"location": target}]} for target in ["base", "sft"]}})
    def log_for(target):
        return SimpleNamespace(status="error", results=None,
            eval=SimpleNamespace(metadata={"plan_sha256": plan["plan_sha256"], "target": target, "benchmark": "model_spec", "profile": "pilot"}),
            samples=[SimpleNamespace(id=identity, epoch=1, output=output, error="Simulated service failure",
                                     scores={}, metadata={}, events=[])
                     for identity, output in [("a", ModelOutput.from_content("offline", "An answer awaiting a verdict")), ("b", None)]])
    monkeypatch.setattr("inspect_ai.log.read_eval_log", log_for)
    report = compare_run(tmp_path, tmp_path, tmp_path / "comparison.json")["benchmarks"]["model_spec"]
    for target in ["base", "sft"]:
        assert report["diagnostics"][target]["target_errors"] == 1
        assert report["diagnostics"][target]["judge_errors"] == 1
        assert report["diagnostics"][target]["judge_invalid"] == 0
        assert report["diagnostics"][target]["missing_samples"] == 0
    assert report["comparisons"]["sft"]["paired"]["paired_samples"] == 0


@pytest.mark.parametrize("artifact", ["samples.json", "overlap_summary.json"])
def test_comparison_rejects_changed_review_inputs(tmp_path, artifact):
    from evaluation.compare import compare_run

    for name in ["samples.json", "overlap_summary.json"]:
        write_json(tmp_path / name, {})
    write_test_plan(tmp_path, {})
    # Even a whitespace edit is outside the reviewed bytes; reject before reading logs.
    with (tmp_path / artifact).open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="Review artifact changed"):
        compare_run(tmp_path, tmp_path, tmp_path / "comparison.json")
