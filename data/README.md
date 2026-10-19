# Synthesized candidates

[candidates.jsonl](candidates.jsonl) contains 3,170 SFT conversations from 2,712 convertible candidates across four generation rounds. Sources are [the Model Spec](../generic/model_spec.md) and [the constitution](../generic/constitution.md). Generation used the DeepSeek API with `deepseek-flash`, thinking on, and a 16,384-token output budget. Each excerpt/template combination was requested once per round; `rule_examples` exports both `generate` and `act` conversations.

**This dataset is unreviewed.** It contains known semantic issues, including unsupported facts and incorrect interpretations of instruction authority. JSON/message validation and duplicate checks do not establish rule compliance. Review the conversations and save accepted rows to a separate file before training. The training loader does not filter by `meta.review_status`.

Each line contains `id`, `job_id`, `messages`, and `meta`. Row IDs include the round and template. Metadata retains the source path, excerpt ID, generation seed, model, SFT form, and original row/job IDs. Source paths are relative to the repository; machine-specific output directories are omitted. Conversation messages, row IDs, and seeds are unchanged from the local merged export. Seeds stay out of training messages. Twelve rows carry metadata for JSON whitespace-escaping normalization, which preserved field values.

[manifest.json](manifest.json) records the dataset checksum, source checksums, generation code commit, requested settings, and row counts. No train/test split is assigned. Raw API requests, replies, and runtime records remain local.

To validate the file and SFT settings without contacting Tinker:

```bash
python -m training sft \
  --data data/candidates.jsonl \
  --log-dir logs/candidates-check \
  --model Qwen/Qwen3-8B \
  --batch-size 1 \
  --dry-run
```

For training, pass a file containing only accepted conversations, as described in [the training guide](../training/README.md).
