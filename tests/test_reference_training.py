"""Offline reference-training checks; model calls are replaced by a fake service."""

from argparse import Namespace
import json
from types import SimpleNamespace

import pytest

from evaluation.core import sha256, write_json
from training import reference
from training.checkpoints import load_checkpoints
from training.cli import _resolve_target


@pytest.mark.parametrize("options,epochs,interval", [([], 10, 1), (["--epochs", "3", "--save-every-epochs", "0"], 3, 0)])
def test_reference_preview_defaults_and_overrides(tmp_path, monkeypatch, capsys, options, epochs, interval):
    monkeypatch.setattr(reference, "ROOT", tmp_path)
    monkeypatch.setattr(reference, "reviewed_data", lambda _: (tmp_path / "data.jsonl", "digest", [{"id": "row"}]))
    output = tmp_path / "runs" / "preview"
    assert reference.main(["--collection", "model-spec-original", "--output", str(output), *options]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["num_epochs"] == epochs
    assert plan["save_every_epochs"] == interval
    assert not output.exists()


@pytest.fixture
def offline_training(tmp_path, monkeypatch):
    tinker = pytest.importorskip("tinker")
    pytest.importorskip("tinker_cookbook.supervised.common")
    rows = [{"id": f"row-{i}"} for i in range(3)]
    data = tmp_path / "data.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "run"
    output.mkdir()
    config = {"recipe": "reference-dimuon", "model": reference.MODEL, "renderer": reference.RENDERER,
              "project_id": "offline-project", "rows": len(rows), "data": "data.jsonl",
              "data_sha256": sha256(data), "batch_size": 2, "num_epochs": 10, "save_every_epochs": 1}
    write_json(output / "status.json", {"state": "training"})
    datums = [SimpleNamespace(model_input=SimpleNamespace(length=1),
              loss_fn_inputs={"weights": tinker.TensorData([1.0], dtype="float32")}) for _ in rows]

    class Client:
        steps = 0
        fail_at = None

        def __init__(self):
            self.saved = []

        def get_info(self):
            return SimpleNamespace(model_data=SimpleNamespace(model_name=reference.MODEL),
                                   lora_rank=reference.RANK, optimizer_config=tinker.types.DimuonOptimizerConfig(version=1))

        def forward_backward(self, batch, loss_fn):
            if self.steps + 1 == self.fail_at:
                raise RuntimeError("offline simulated interruption")
            result = SimpleNamespace(loss_fn_outputs=[
                {"logprobs": tinker.TensorData([-1.0], dtype="float32")} for _ in batch])
            return SimpleNamespace(result=lambda: result)

        def optim_step(self, params):
            def finish():
                self.steps += 1
            return SimpleNamespace(result=finish)

        def save_state(self, name, ttl_seconds):
            self.saved.append(("state", name, self.steps, ttl_seconds))
            return SimpleNamespace(result=lambda: SimpleNamespace(path=f"tinker://offline/weights/{name}"))

        def save_weights_for_sampler(self, name, ttl_seconds):
            self.saved.append(("sampler", name, self.steps, ttl_seconds))
            return SimpleNamespace(result=lambda: SimpleNamespace(path=f"tinker://offline/sampler_weights/{name}"))

    client = Client()

    class Service:
        def __init__(self, *, project_id, display_name):
            assert project_id == config["project_id"]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def create_lora_training_client(self, *, base_model, rank, optimizer):
            assert base_model == reference.MODEL and rank == reference.RANK
            return client

    monkeypatch.setattr(tinker, "ServiceClient", Service)
    return output, config, rows, datums, client


@pytest.mark.parametrize("interval,expected_epochs", [(1, list(range(1, 11))), (3, [3, 6, 9, 10]), (0, [10])])
def test_reference_checkpoints_preserve_epochs_and_final_binding(offline_training, interval, expected_epochs):
    from evaluation.runner import training_binding

    output, config, rows, datums, client = offline_training
    config["save_every_epochs"] = interval
    write_json(output / "config.json", config)
    reference.train(output, config, rows, datums)
    records = load_checkpoints(output)
    assert [row["epoch"] for row in records] == expected_epochs
    assert [row["step"] for row in records] == [epoch * 2 for epoch in expected_epochs]
    assert [row["name"] for row in records] == [f"epoch-{epoch:02d}" if epoch < 10 else "final" for epoch in expected_epochs]
    for row in records:
        assert row["binding"] == {"dataset_sha256": config["data_sha256"], "config_sha256": sha256(output / "config.json")}
        assert row["batch"] == 0
        assert ("state", row["name"], row["step"], None) in client.saved
        assert ("sampler", row["name"], row["step"], None) in client.saved
    assert len(client.saved) == 2 * len(expected_epochs)
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert len(metrics) == 20
    assert sum(row["batch_rows"] for row in metrics) == 30
    assert json.loads((output / "status.json").read_text()) == {
        "state": "complete", "completed_steps": 20, "trained_rows": 30}
    args = Namespace(log_dir=output, base=False, model_path=None, model=None, renderer=None)
    assert _resolve_target(args) == (reference.MODEL, reference.RENDERER, records[-1]["sampler_path"])
    # The final checkpoint remains consumable by the existing evaluation binder.
    write_json(output / "tokenizer_revision.json", {"model": reference.MODEL, "revision": "offline-tokenizer"})
    snapshot = output / "cache/huggingface/hub" / ("models--" + reference.MODEL.replace("/", "--")) / "snapshots/offline-tokenizer"
    write_json(snapshot / "tokenizer.json", {})
    assert training_binding(output.parent, output)["targets"]["sft"] == records[-1]["sampler_path"]


def test_completed_epoch_checkpoint_survives_later_interruption(offline_training):
    output, config, rows, datums, client = offline_training
    write_json(output / "config.json", config)
    client.fail_at = 3
    with pytest.raises(RuntimeError, match="simulated interruption"):
        reference.train(output, config, rows, datums)
    records = load_checkpoints(output)
    assert len(records) == 1 and records[0]["epoch"] == 1 and records[0]["step"] == 2
    assert records[0]["state_path"] and records[0]["sampler_path"]
    assert json.loads((output / "status.json").read_text())["state"] != "complete"
    args = Namespace(log_dir=output, base=False, model_path=None, model=None, renderer=None)
    assert _resolve_target(args)[2] == records[0]["sampler_path"]
