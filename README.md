# Spec2Behavior

Spec2Behavior synthesizes training data and fine-tunes models from written behavioral specifications, using model specs and constitutions as sources and [Tinker](https://tinker-docs.thinkingmachines.ai/) for training.

Synthesis, Tinker supervised fine-tuning, and inference are in the repository. Evaluation against the spec is next.

## Layout

| Path | What it is |
|---|---|
| `synthesis/` | API synthesis runner. A chain is `read`, `prompt`, and `parse` |
| `synthesis/templates/` | Six default chains that turn rule excerpts into SFT rows |
| `training/` | Tinker SFT and inference over the synthesized conversations |
| `generic/` | Markdown inputs used by the synthesis examples |
| `model_spec/` | Snapshot of the OpenAI Model Spec |
| `claude-constitution/` | Snapshot of Claude's constitution |
| `model_spec_evals/` | Snapshot of OpenAI's Model Spec eval harness |

The three upstream trees stay separate from this project's code. The Model Spec and constitution are CC0; the eval harness is MIT-licensed.

## Synthesis

A chain is three functions in a Python file. `read` yields records, `prompt` turns one record into the model request, and `parse` turns the reply into SFT conversations. `synthesis/templates/` holds six default chains. Each one reads the direct body under each heading in a Markdown spec or constitution, splitting long bodies on paragraph boundaries. Copy one and edit those functions, then pass the file with `--template`. The runner calls an OpenAI-compatible chat completions API, runs requests in batches with a concurrency limit, and resumes from JSONL checkpoints.

```bash
python -m synthesis list
python -m synthesis run rule_examples --help
```

Put `DEEPSEEK_API_KEY` in `.env`, using [.env.example](.env.example) as a reference. Synthesis reads it automatically from the current directory. CLI flags override process environment variables, which override `.env`. Use `--env-file FILE` to select another file.

Preview one request without calling an API or writing output:

```bash
python -m synthesis run rule_examples \
  --source generic/model_spec.md \
  --output runs/model-spec/rule-examples-deepseek \
  --limit 1 \
  --max-retries 0 \
  --dry-run
```

Remove `--dry-run` to generate that sample. Repeat the command to resume after an interruption. Remove `--limit 1` for all pending excerpts; adjust `--concurrency`, `--batch-size`, and `--max-retries` for a larger run.

The other default chains are `correction`, `reflection`, `pressure`, `chain_of_command`, and `safe_completion`. They share `--source` and write different `sft_form` values.

Defaults are the DeepSeek API with `deepseek-flash` and thinking on, a 16000-character input target, a 16384-token output budget, temperature 1.0, and one request/reply pair for `rule_examples` (one `generate` and one `act` training row). Override them with `--provider`, `--model`, `--thinking`, `--max-portion-chars`, `--max-tokens`, `--temperature`, and `--examples`. `--help` lists the settings. Use a separate output directory for each source and chain.

Each excerpt/template combination produces one candidate. Every new candidate's prompt starts with `Random seed: <32 random hex characters>` to encourage varied scenarios and wording. The value is recorded as `generation_seed` in API events, job/error records, and SFT metadata, and is excluded from training messages. HTTP retries reuse the request and seed; generating a candidate again creates a new seed. A dry-run seed is only a preview. This prompt hint does not guarantee different outputs or deterministic replay.

Switch APIs with `--provider deepseek`, `--provider openrouter`, or `--provider openai`, using `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`, or `OPENAI_API_KEY` respectively, and select that provider's model with `--model`. DeepSeek and OpenRouter default to thinking on; `--thinking off` disables it and `--thinking default` leaves the choice to the server. OpenAI keeps its server default when the flag is omitted. `--provider auto` keeps the existing `--base-url` / `OPENAI_BASE_URL` setup. See [synthesis settings](synthesis/README.md#run) for examples.

`runs/model-spec/rule-examples-deepseek/train.jsonl` is chat JSONL. Each line has `messages` plus `id`, `job_id`, and `meta`. New row ids include the chain name so different templates can be combined. Tinker Cookbook's `FromConversationFileBuilder` trains on `messages`.

Each run also keeps `api.jsonl`: the complete request JSON and raw response text for every HTTP attempt, including retries and failures, linked by job and call ids. Returned thinking and usage remain available there; SFT uses only final answers. Credentials are omitted or redacted. Run outputs stay local under the ignored `runs/` directory.

Parsing validates structure, not semantic correctness. Review generated conversations for rule compliance and invented facts, then save the accepted rows to a separate file such as `runs/model-spec/reviewed/train.jsonl`. Keep that file outside the generator's output directory: resume rebuilds its `train.jsonl` from the original jobs.

Pipeline details, output files, and how to add a chain are in [synthesis/README.md](synthesis/README.md).

## Training and inference

SFT uses Tinker Cookbook. Inference samples with the same chat renderer and can read the newest sampler checkpoint from the training log.

```bash
pip install -e ".[train]"
export TINKER_API_KEY=...

python -m training sft \
  --data runs/model-spec/reviewed/train.jsonl \
  --log-dir logs/model-spec \
  --model Qwen/Qwen3-8B \
  --batch-size 1 \
  --dry-run
```

Remove `--dry-run` to train. After training produces a sampler checkpoint:

```bash
python -m training infer \
  --log-dir logs/model-spec \
  --input runs/model-spec/reviewed/train.jsonl \
  --output logs/model-spec/infer
```

Flags, resume behavior, and the prediction file are in [training/README.md](training/README.md).

## Install

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`python -m synthesis` also works from the repository root when its dependencies are installed.

Installation also provides `spec2behavior-synthesize` and `spec2behavior-train`, equivalent to `python -m synthesis` and `python -m training`.
