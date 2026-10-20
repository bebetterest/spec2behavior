# SFT datasets

Use [reviewed.jsonl](reviewed.jsonl) for SFT: it contains **2,829 accepted conversations**. [candidates.jsonl](candidates.jsonl) preserves all **3,170 original conversations**, including excluded examples. The training loader reads every row in the selected file; it does not filter by `meta.review_status`.

Both files come from four synthesis rounds over [the Model Spec](../generic/model_spec.md) and [the constitution](../generic/constitution.md). Generation used the DeepSeek API with `deepseek-flash`, thinking on, and a 16,384-token output budget. Each excerpt/template combination was requested once per round; `rule_examples` exports both `generate` and `act` conversations.

## Semantic review

All 3,170 rows received an individual semantic decision against their original source excerpt and actual saved conversation roles. The review retained 2,829 rows, rejected 338, and excluded another 3 whose correctness remained uncertain. It examined every trained assistant turn, source-rule fidelity, instruction authority, factual support, requested formats, and completion of permitted work. [review_policy.md](review_policy.md) defines the criteria.

The primary reviewer was `deepseek-flash` with thinking on and a 16,384-token output limit. A targeted second pass covered 949 provisional acceptances with authority, completion, or deployment-assertion risk indicators. Codex independently adjudicated 82 rows, including unresolved evidence checks, selected acceptances and rejections, and all second-pass rejection proposals. Review decisions require anchored source and conversation evidence; a failed API or evidence check is not an acceptance.

The reviewed file is a selection, not a rewrite. Accepted `messages`, row IDs, job IDs, and generation metadata are unchanged; review metadata marks acceptance and identifies the final reviewer. A `rule_examples` pair is judged per exported row, so its `generate` and `act` forms can receive different decisions. A multi-turn row is excluded if any trained assistant turn has a material defect.

| Template | Original rows | Reviewed rows |
|---|---:|---:|
| `rule_examples` | 916 | 842 |
| `correction` | 461 | 434 |
| `reflection` | 454 | 415 |
| `pressure` | 436 | 367 |
| `chain_of_command` | 453 | 400 |
| `safe_completion` | 450 | 371 |
| **Total** | **3,170** | **2,829** |

This is model-assisted curation, not human certification. The generator and primary reviewer use the same model family, so correlated errors can remain. The review does not establish target-model token lengths or training performance.

## Provenance and use

Each line contains `id`, `job_id`, `messages`, and `meta`. Row IDs include the round and template. Metadata retains the repository-relative source path, excerpt ID, generation seed, model, SFT form, and original row/job IDs. Seeds stay out of training messages. Twelve original rows carry JSON whitespace-escaping normalization metadata; field values were preserved.

[manifest.json](manifest.json) records original generation settings, source checksums, and counts. [reviewed_manifest.json](reviewed_manifest.json) binds the reviewed file to the original candidate checksum and review policy, and records decisions and counts by template, source, form, and round. Raw API requests, replies, reasoning, per-row review evidence, exclusions, and runtime records remain local.

No train/test split is assigned. When making a split, keep variants from the same source excerpt together to avoid placing closely related generation rounds on both sides.

To validate the reviewed file and SFT settings without contacting Tinker:

```bash
python -m training sft \
  --data data/reviewed.jsonl \
  --log-dir logs/reviewed-check \
  --model Qwen/Qwen3-8B \
  --batch-size 1 \
  --dry-run
```

Remove `--dry-run` only when ready to train. See [the training guide](../training/README.md) for training configuration and resume behavior.
