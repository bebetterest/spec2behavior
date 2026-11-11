"""Run one configuration of the reference Nemotron DiMuon SFT experiment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys

from evaluation.core import sha256, write_json
from synthesis.arguments import nonnegative_int
from synthesis.io import append_jsonl
from training.data import load_conversations

ROOT = Path(__file__).resolve().parents[1]
MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
RENDERER = "nemotron3_ultra_preserve_thinking"
TOKENIZER_REVISION = "a9904d24bcc1d289a1950fa9d2b978c47cf903b9"
COLLECTIONS = ("model-spec-original", "model-spec-generic", "constitution-original", "constitution-generic")
VERSIONS = {"tinker": "0.34.0", "tinker-cookbook": "0.5.7", "transformers": "5.5.4", "torch": "2.14.1"}
MAX_LENGTH = 32768
TRAIN_ON = "all_assistant_messages"
INITIAL_LR = 0.001
RANK = 64


def reviewed_data(collection):
    directory = ROOT / "data/variants"
    info = json.loads((directory / "manifest.json").read_text())["collections"][collection]["reviewed"]
    path = (directory / info["path"]).resolve()
    if not path.is_relative_to(directory) or sha256(path) != info["sha256"]:
        raise ValueError(f"Reviewed dataset checksum changed: {collection}")
    rows = load_conversations(path, train_on_what=TRAIN_ON)
    if (len(rows) != info["rows"] or len({r["id"] for r in rows}) != len(rows)
            or any(r["meta"].get("review_status") != "accepted" for r in rows)):
        raise ValueError(f"Reviewed dataset manifest does not match the accepted rows: {collection}")
    return path, info["sha256"], rows


def epoch_batches(size, batch_size, epoch):
    order = list(range(size))
    random.Random(epoch).shuffle(order)
    for start in range(0, size, batch_size):
        yield order[start:start + batch_size]


def tokenize(rows, output):
    from huggingface_hub import snapshot_download
    from tinker_cookbook.renderers import TrainOnWhat, get_renderer
    from tinker_cookbook.supervised.common import datum_from_model_input_weights
    from transformers import AutoTokenizer

    versions = {key: importlib.metadata.version(key) for key in VERSIONS}
    if versions != VERSIONS:
        raise ValueError(f"Use the reference dependency versions: {VERSIONS}; installed: {versions}")
    snapshot = snapshot_download(
        MODEL, revision=TOKENIZER_REVISION, cache_dir=output / "cache/huggingface/hub",
        allow_patterns=["*.json", "*.jinja", "tokenizer.model", "vocab.json", "merges.txt"],
    )
    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=False, local_files_only=True)
    renderer = get_renderer(RENDERER, tokenizer, model_name=MODEL)
    if not renderer.has_extension_property:
        raise ValueError("The reference renderer must preserve historical assistant prefixes")
    datums = []
    for row in rows:
        inputs, weights = renderer.build_supervised_example(row["messages"], train_on_what=TrainOnWhat(TRAIN_ON))
        if inputs.length > MAX_LENGTH or not weights.any():
            raise ValueError(f"Truncation or empty loss mask in {row['id']}")
        datums.append(datum_from_model_input_weights(inputs, weights, max_length=MAX_LENGTH, reduction="mean"))
    return datums


def train(output, config, rows, datums):
    import tinker
    from tinker_cookbook.supervised.common import compute_mean_nll
    from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier

    steps = math.ceil(len(rows) / config["batch_size"]) * config["num_epochs"]
    save_every = config.get("save_every_epochs", 1)
    binding = {"dataset_sha256": config["data_sha256"], "config_sha256": sha256(output / "config.json")}
    with tinker.ServiceClient(project_id=config["project_id"], display_name="Spec2Behavior reference SFT") as service:
        client = service.create_lora_training_client(
            base_model=MODEL, rank=RANK, optimizer=tinker.types.DimuonOptimizerConfig(version=1),
        )
        info = client.get_info()
        if (info.model_data.model_name != MODEL or info.lora_rank != RANK
                or info.optimizer_config.model_dump() != {"type": "dimuon", "version": 1}):
            raise RuntimeError("The Tinker training client does not match the reference recipe")
        step = 0
        with (output / "metrics.jsonl").open("x", buffering=1) as metrics:
            for epoch in range(config["num_epochs"]):
                loss_sum = 0.0
                for indices in epoch_batches(len(rows), config["batch_size"], epoch):
                    batch = [datums[i] for i in indices]
                    lr = INITIAL_LR * compute_schedule_lr_multiplier("linear", step, steps)
                    forward = client.forward_backward(batch, "cross_entropy")
                    update = client.optim_step(tinker.types.DimuonParams(learning_rate=lr, beta1=.9, grad_clip_norm=0))
                    result = forward.result()
                    update.result()
                    loss = compute_mean_nll([row["logprobs"] for row in result.loss_fn_outputs],
                                            [datum.loss_fn_inputs["weights"] for datum in batch])
                    if not math.isfinite(loss):
                        raise RuntimeError("Non-finite training loss")
                    step += 1
                    loss_sum += len(batch) * loss
                    record = {"epoch_number": epoch + 1, "step": step, "batch_rows": len(batch),
                              "learning_rate": lr, "train_example_mean_nll": loss,
                              "input_tokens": sum(datum.model_input.length for datum in batch),
                              "row_ids": [rows[i]["id"] for i in indices]}
                    metrics.write(json.dumps(record, allow_nan=False) + "\n")
                print(json.dumps({"epoch": epoch + 1, "mean_training_nll": loss_sum / len(rows)}), flush=True)
                final = epoch + 1 == config["num_epochs"]
                if final or (save_every and (epoch + 1) % save_every == 0):
                    name = "final" if final else f"epoch-{epoch + 1:02d}"
                    state = client.save_state(name, ttl_seconds=None).result()
                    sampler = client.save_weights_for_sampler(name, ttl_seconds=None).result()
                    append_jsonl(output / "checkpoints.jsonl", {
                        "name": name, "epoch": epoch + 1, "batch": 0, "step": step,
                        "state_path": state.path, "sampler_path": sampler.path, "binding": binding,
                    })
        write_json(output / "status.json", {"state": "complete", "completed_steps": step,
                                             "trained_rows": len(rows) * config["num_epochs"]})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True, choices=COLLECTIONS)
    parser.add_argument("--batch-size", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument("--epochs", type=int, choices=(1, 3, 5, 10), default=10,
                        help="Training epochs (default: 10); shorter schedules reproduce earlier comparisons")
    parser.add_argument("--save-every-epochs", type=nonnegative_int, default=1,
                        help="Save training state and sampler every N epochs (default: 1); 0 saves only final")
    parser.add_argument("--output", type=Path, required=True, help="Fresh run directory under runs/ or local/")
    parser.add_argument("--execute", action="store_true", help="Start paid training after reviewing the plan")
    args = parser.parse_args(argv)
    try:
        output = args.output.resolve()
        if not any(output.is_relative_to(ROOT / name) and output != ROOT / name for name in ("runs", "local")):
            raise ValueError("--output must be a run subdirectory of runs/ or local/")
        path, digest, rows = reviewed_data(args.collection)
        config = {"recipe": "reference-dimuon", "model": MODEL, "renderer": RENDERER, "collection": args.collection,
                  "data": str(path.relative_to(ROOT)), "data_sha256": digest, "rows": len(rows),
                  "num_epochs": args.epochs, "batch_size": args.batch_size, "lora_rank": RANK, "optimizer_version": 1,
                  "learning_rate": INITIAL_LR, "lr_schedule": "linear", "train_on_what": TRAIN_ON,
                  "max_length": MAX_LENGTH, "shuffle_seed": 0, "save_every_epochs": args.save_every_epochs}
        plan = {**config, "optimizer_updates": math.ceil(len(rows) / args.batch_size) * args.epochs,
                "tokenizer_revision": TOKENIZER_REVISION, "dependencies": VERSIONS}
        print(json.dumps(plan, indent=2), flush=True)
        if not args.execute:
            return 0

        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
        if not os.environ.get("TINKER_API_KEY") or not os.environ.get("TINKER_PROJECT_ID"):
            raise ValueError("Execution requires TINKER_API_KEY and TINKER_PROJECT_ID")
        config["project_id"] = os.environ["TINKER_PROJECT_ID"]
        output.mkdir(parents=True, exist_ok=False)
        datums = tokenize(rows, output)
        write_json(output / "config.json", config)
        write_json(output / "tokenizer_revision.json", {"model": MODEL, "revision": TOKENIZER_REVISION})
        write_json(output / "status.json", {"state": "training"})
        train(output, config, rows, datums)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Reference SFT: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
