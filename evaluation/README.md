# Evaluation

Compare a Tinker base model and one or more final SFT samplers with [Inspect AI](https://inspect.aisi.org.uk/). Candidates and the judge use the native Tinker SDK. Model Spec defaults to **rubric-only** grading.

| Benchmark | Items/model | Metric |
|---|---:|---|
| [Model Spec](https://github.com/openai/model_spec_evals) | 587, after 9 upstream skips | Section compliance >=6/7; higher is better |
| [XSTest](https://github.com/paul-rottger/xstest) | 250 safe + 200 unsafe | Safe refusal lower; unsafe refusal higher |
| [StrongREJECT](https://github.com/alexandrasouly/strongreject) | 313 in the pinned CSV | Direct-request composite 0–5; lower is better |

## Fixed policy test collections

[The checked-in evaluation datasets](../data/evaluation/README.md) provide the official Model Spec set (587 records) and three reviewed synthetic collections (194 Model Spec generic, 71 Constitution original, 86 Constitution generic). Run `python -m evaluation.datasets` to verify all **938** records offline. Each `target` is a rubric shown only to the judge. The synthetic datasets load with `evaluation.builder.inspect_task.build_task(path, judge)`; supply a configured judge and preserve the documented developer-role convention. The comparison CLI below selects its upstream benchmarks and does not automatically include these synthetic files.

## Install and prepare

Run from this checkout. Keep evaluation dependencies in a separate local environment.

```bash
python -m venv local/evaluation/.venv
local/evaluation/.venv/bin/python -m pip install -e '.[eval,dev]'

local/evaluation/.venv/bin/python -m evaluation prepare \
  --training-run b128_lr6e-3=runs/my-first-sft \
  --training-run b32_lr1e-3=runs/my-second-sft \
  --training-run b32_lr5e-4=runs/my-third-sft \
  --config evaluation/config.example.json \
  --output local/evaluation/my-comparison
```

Preparation downloads public benchmark/tokenizer files, screens training overlap and calculates costs **without model inference**. No model weights are downloaded. The training run must contain its completed config/status/checkpoint log and frozen tokenizer.

Repeat `--training-run [NAME=]PATH` for multiple SFT checkpoints; base is included once. A single unnamed run retains the `sft` label. Names must be unique; `base` is reserved. Compared runs must share the base model, renderer, project, training data and tokenizer revision, so sampling and tariffs apply consistently. A frozen tokenizer cache in any named run can serve runs declaring the same model/revision; its exact bytes are bound in the plan.

Add `prepare --benchmark model_spec` to evaluate only the official Model Spec
dataset. Repeat `--benchmark` to select several tasks; omission retains all four.
The selection is frozen in the plan and applies to overlap screening, costs,
pilot IDs, execution and comparison. Both `pilot` and `full` use that selection.

Read the generated `REVIEW.md`. `config.json` is the editable configuration; `plan.json` freezes the checkpoint, settings, code/packages, data, tokenizers, renderer date and pilot ids. Full evidence lives in `overlap_summary.json`, `overlap_candidates.json`, `prompt_tokens.json` and `costs.json`. Keep these experiment artifacts under ignored `local/` or `runs/`.

Optional `--manual-findings FILE` supplies reviewed scenario-overlap findings (an object or list with `benchmark`, `eval_id`, `kind` and `interpretation`). Rebuilding the same directory retains existing manual findings automatically. An explicit file replaces them; use `[]` to clear them deliberately.

## Adjust one config

| Keys in config.json | Effect |
|---|---|
| `judge.model_spec_grading` | `rubric-only` (default): conversation + scenario rubric + fixed scale; `full-spec`: upstream prompt with the complete policy |
| `judge.model`, `judge.renderer`, `judge.context_window` | Exact Tinker model id, native chat format and context capacity |
| `judge.temperature`, `judge.top_p`, `judge.max_tokens` | Judge decoding and total thinking + verdict budget |
| `sampling.*` | Common decoding for base and every SFT; never set different caps per target |
| `judge.max_connections`, `sampling.max_connections`, `execution.max_samples` | Judge, target and in-flight sample concurrency |
| `execution.pilot_min_per_task` | Minimum; coverage of metadata groups and role/tool histories can select more |
| `pricing.judge_*_usd_per_million` | Judge input/output tariffs; update with the model choice |
| `pricing.scenarios`, `pricing.contingency_fraction` | Cost assumptions and contingency, not a spending limit |

The default judge is **openai/gpt-oss-120b:peft:131072**, renderer `gpt_oss_low_reasoning`. Its 128K context supports either grading mode. Rubric-only omits the full policy; the standard 32K variant cannot fit `full-spec`. It is a different model family from the current Nemotron targets. Only `TINKER_API_KEY` is required; `openai/` in this model id names the Tinker base model, not the OpenAI API.

Rubric-only uses `rubric_only_prompt.md` with the original conversation and item rubric. It retains the pinned upstream JSON schema, verdict parser, 1–7 scale, >=6 compliance threshold and grouped metric. This is a custom grading protocol; agreement with the full-policy judge has not been measured. The mode is frozen in the plan and recorded in logs/reports. Select `full-spec` to use the original upstream scorer and prompt.

Changing the judge requires matching its renderer, context capacity and tariffs. Preparation resolves and freezes its public tokenizer revision; optional `judge.tokenizer_revision` selects an explicit HF commit. Edit config.json, then repeat `prepare --config PATH/config.json` with the same named training runs and output. Changes invalidate the old plan. Unknown/missing settings are rejected. Keep a separate prepared directory when retaining an older run for comparison.

This version supports one candidate and one verdict/item (`epochs=1`, `samples=1`). Automatic retries are disabled. Seed parameters do not guarantee deterministic API replay. Costs use native prompt tokens, include reasoning output, clamp scenarios to configured caps and assume uncached rates. Final-answer tokenization differs between models, so the output-limit estimate is not a hard spending ceiling. The example freezes tariffs checked on 2026-10-20, including a temporary Nemotron promotion; recheck [Tinker tariffs](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/) before execution.

## Preview, run, compare

```bash
# Preview the frozen pilot and cost estimates; no model clients.
local/evaluation/.venv/bin/python -m evaluation run \
  --plan local/evaluation/my-comparison --profile pilot

# After review and authorization of this paid run:
export TINKER_API_KEY=...
local/evaluation/.venv/bin/python -m evaluation run \
  --plan local/evaluation/my-comparison --profile pilot \
  --output local/evaluation/my-pilot --execute

# Read saved Inspect logs; no API calls.
local/evaluation/.venv/bin/python -m evaluation compare \
  --plan local/evaluation/my-comparison --run local/evaluation/my-pilot \
  --output local/evaluation/my-pilot/comparison.json
```

Review real pilot answers, verdicts, invalid/empty/truncated outputs and actual usage before authorizing `--profile full` in a fresh directory. Pilot and full costs are additive. Runs do not auto-resume or overwrite previous output. Comparison verifies the frozen review files and log metadata without loading model clients or enforcing the generation environment lock. Reports contain Inspect metrics and one `comparisons` entry per SFT against base: paired deltas/bootstrap intervals, metadata slices and a flagged-excluded auxiliary view. Coverage and target/judge/parse/truncation diagnostics are reported per target. Do not average the three benchmarks together.

## Protocol details

- Data and rubric text are pinned upstream. Model Spec uses the selected grading mode above; XSTest/StrongREJECT use pinned `inspect-evals` tasks and scorers. Target decoding and the judge override upstream defaults; this is not a paper reproduction.
- Native Tinker sampling has no constrained JSON decoder. For Model Spec, the adapter appends the **actual upstream response schema** as a format instruction. The upstream parser validates verdicts without repair/retry. Invalid Model Spec critiques retain upstream score 0 and are separately counted. XSTest/StrongREJECT keep their textual verdict formats.
- Cookbook's official Inspect adapter converts roles/tool history and parses replies. Thin extensions map Model Spec's Inspect system turns to native developer, preserve actual length termination, and keep truncated thinking out of visible answers. Harmony analysis and final channels are also separated on unclean termination. Both input + output budgets are checked before sampling.
- The optional upstream full-policy judge uses **2025-12-18**, earlier than the current synthesis source. Both modes score the requested section of the last answer. Rubrics and any full policy go to the judge only.
- XSTest's HF mirror currently requires authentication; we use its author's pinned original CSV with the upstream converter/solver/scorer. Byte equality with that mirror is not claimed. Empty/partial responses retain upstream refusal handling. StrongREJECT uses direct prompts, not jailbreak attacks.
- Overlap screening covers input/history text, serialized tool-call arguments and complete Model Spec rubrics against retained training messages; source excerpts are checked separately from generated text. Exact/contained spans, 12-word matches and lexical neighbours cannot certify semantic independence. Shared scenarios and teacher/base pretraining exposure remain limitations; flags do not remove official items.

## Offline checks

```bash
SPEC2BEHAVIOR_EVAL_ASSETS=local/evaluation/my-comparison/data \
SPEC2BEHAVIOR_EVAL_TRAINING_RUN=runs/my-completed-sft \
SPEC2BEHAVIOR_EVAL_JUDGE=local/evaluation/my-comparison \
  local/evaluation/.venv/bin/python -m pytest -q
```

Optional local paths exercise real upstream tasks, frozen native tokenizers/renderers, tool histories, schema formatting, truncation, both Tinker model paths and Inspect logging/comparison with fake clients. No paid inference runs in tests. Offline checks establish integration behavior, not judge quality, API availability or an SFT improvement.
