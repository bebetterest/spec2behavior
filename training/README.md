# Training

Supervised fine-tuning and inference on [Tinker](https://tinker-docs.thinkingmachines.ai/). The dataset is conversation JSONL from [synthesis](../synthesis/README.md), reviewed for rule compliance and factual correctness. Tinker Cookbook trains on `messages` and ignores `id` and `meta`. A metadata field such as `review_decision` does not filter rows: pass a file containing only the accepted conversations.

Install the training extra and export an API key from the Tinker console:

```bash
pip install -e ".[train]"
export TINKER_API_KEY=...
```

`python -m training sft --dry-run` validates the dataset and prints the run plan without Tinker.

## Fine-tune

```bash
python -m training sft \
  --data runs/model-spec/reviewed/train.jsonl \
  --log-dir logs/model-spec \
  --model Qwen/Qwen3-8B \
  --batch-size 16 \
  --num-epochs 1 \
  --save-every 20
```

`--data` accepts several paths and may be repeated, for example `--data runs/spec/rule-examples --data runs/constitution/reflection`. Each value is a `train.jsonl` file or a synthesis output directory. Several inputs are concatenated in that order into `log_dir/combined.jsonl` when training starts. The renderer defaults to `get_recommended_renderer_name(model)`. Pass `--renderer` to override it, for example `qwen3_disable_thinking`.

`--batch-size` must be no larger than the number of training rows. Tinker drops a partial last batch. Loss is on every assistant message (`--train-on-what all_assistant_messages`). `--max-length` truncates tokenized conversations (default 32768).

`--train-on-what customized` reads a boolean `trainable` field on every message. Other loss modes reject that field. Synthesis keeps `trainable` when a pipeline sets it.

The same command resumes. Tinker reads the last `state_path` in `logs/model-spec/checkpoints.jsonl` and continues. The model, renderer, and dataset must match `spec2behavior_run.json`. `--overwrite` deletes the log directory first. `--load-checkpoint tinker://...` supplies starting weights only when that directory has no resume checkpoint.

Held-out NLL uses a shuffle split, not a separate file:

```bash
python -m training sft \
  --data runs/model-spec/reviewed/train.jsonl \
  --log-dir logs/model-spec \
  --model Qwen/Qwen3-8B \
  --test-size 32 \
  --eval-every 10
```

`--eval-every` requires `--test-size`. Weights & Biases is off unless `--wandb-project` is set.

`logs/<name>/spec2behavior_run.json` records the model, renderer, and data path. `checkpoints.jsonl` is Tinker's own log. Periodic sampler checkpoints expire after seven days in the Tinker service; the final checkpoint is kept.

## Inference

Sampling uses the same renderer as training. A trailing assistant message is stripped and stored as `meta.gold`, so `train.jsonl` can be scored without the model continuing after its own answer.

```bash
python -m training checkpoints logs/model-spec

python -m training infer \
  --log-dir logs/model-spec \
  --input runs/model-spec/reviewed/train.jsonl \
  --output logs/model-spec/infer \
  --concurrency 8 \
  --batch-size 32 \
  --temperature 0
```

`--log-dir` selects the newest `sampler_path`. A log directory with no sampler checkpoint is an error. `--model-path tinker://...` overrides the log directory. `--base` samples the base model instead of the checkpoint. Pass either `--base` or `--model-path`. A one-shot prompt prints to stdout:

```bash
python -m training infer \
  --model Qwen/Qwen3-8B \
  --prompt "Explain what a user message is allowed to override."
```

Re-running a file inference command skips ids already in `predictions.jsonl`. Empty completions are recorded in `errors.jsonl` and skipped until `--retry-errors`. Error history is retained, including failures outside `--limit`; successful predictions take precedence over earlier errors. A request that raises stays unfinished and is tried again. `--limit` caps how many pending prompts this invocation samples.

File inference exits `1` when requests fail, remain pending, or have recorded errors skipped on resume. A successful selected batch exits `0`; later prompts outside `--limit` may still remain.

```text
logs/<name>/infer/
  predictions.jsonl
  errors.jsonl
  failures.jsonl
```

A prediction line:

```json
{
  "id": "rule_examples/model_spec.md/protect_privileged_information/0#1",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "meta": {
    "model": "Qwen/Qwen3-8B",
    "model_path": "tinker://...",
    "renderer": "qwen3",
    "gold": "reference answer, if the input ended with one"
  }
}
```

JSONL rows may use `prompt` instead of `messages`. `--system` is added for `--prompt` and for those `prompt` rows. Rows that already have `messages` are sent as written, aside from the stripped final assistant turn.

## Adjust

The main defaults are:

| Command | Flag | Default |
|---|---|---|
| `sft` | `--learning-rate`, `--lr-schedule` | `1e-4`, `linear` |
| `sft` | `--num-epochs`, `--batch-size`, `--lora-rank` | `1`, `16`, `32` |
| `sft` | `--max-length` | `32768` tokens for the entire tokenized conversation |
| `sft` | `--train-on-what` | `all_assistant_messages` |
| `sft` | `--test-size`, `--eval-every`, `--save-every` | `0`, `0`, `20` |
| `infer` | `--max-tokens`, `--temperature` | `1024`, `0.0` |

Synthesis has a separate output budget of 16384 tokens by default. Its `--max-portion-chars` is an input character target; neither setting changes SFT's total sequence length. Input token length plus output must fit the chosen model's context window. Inspect the rows with the target tokenizer before training and adjust `--max-length` for long conversations.

`run_sft(settings, dataset_builder=...)` keeps that log directory, resume check, and hyperparameter setup, and uses your builder instead of `FromConversationFileBuilder`. The builder is the object Tinker expects on `train.Config`. The conversation files are still validated first.

`run_inference` takes any object with `async complete(messages) -> Completion`. A prediction file is chat JSONL, so a later synthesis pipeline can read it.

## What this module does not do

It does not merge LoRA weights, download a checkpoint, or run the Model Spec eval harness. Those stay separate from training and sampling.
