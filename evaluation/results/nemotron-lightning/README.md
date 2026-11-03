# Nemotron Lightning policy SFT comparison

**Status: provisional automatic results with known judge errors.** The primary rates are preserved from the requested rubric-only protocol. Selected review found clear direction reversals and unsupported rationales; these numbers do not establish a stable best model.

The comparison covers twelve LoRA SFT checkpoints and the unadapted vendor model on four fixed policy test collections. The official Model Spec column reuses the matching earlier thirteen-model evaluation. There are **12,194 model/item results**: 4,563 new and 7,631 reused official results.

| Training collection | Batch | Official Model Spec (587) | Generic Model Spec (194) | Original Constitution (71) | Generic Constitution (86) |
|---|---:|---:|---:|---:|---:|
| Unadapted vendor base | — | 78.53% (461/587) | 93.30% (181/194) | 85.92% (61/71) | 74.42% (64/86) |
| model-spec-generic | 32 | 79.22% (465/587) | 92.78% (180/194) | 85.92% (61/71) | 72.09% (62/86) |
| model-spec-generic | 16 | 79.73% (468/587) | 93.30% (181/194) | 85.92% (61/71) | 74.42% (64/86) |
| model-spec-generic | 8 | 81.43% (478/587) | 95.88% (186/194) | 85.92% (61/71) | 75.58% (65/86) |
| model-spec-original | 32 | 80.58% (473/587) | 92.78% (180/194) | 81.69% (58/71) | 72.09% (62/86) |
| model-spec-original | 16 | 80.24% (471/587) | 93.81% (182/194) | 84.51% (60/71) | 70.93% (61/86) |
| model-spec-original | 8 | 80.07% (470/587) | 94.85% (184/194) | 84.51% (60/71) | 75.58% (65/86) |
| constitution-generic | 32 | 80.41% (472/587) | 93.81% (182/194) | 84.51% (60/71) | 72.09% (62/86) |
| constitution-generic | 16 | 80.24% (471/587) | 92.78% (180/194) | 84.51% (60/71) | 69.77% (60/86) |
| constitution-generic | 8 | 81.09% (476/587) | 93.30% (181/194) | 85.92% (61/71) | 75.58% (65/86) |
| constitution-original | 32 | 80.07% (470/587) | 92.78% (180/194) | 84.51% (60/71) | 74.42% (64/86) |
| constitution-original | 16 | 79.05% (464/587) | 92.78% (180/194) | 81.69% (58/71) | 69.77% (60/86) |
| constitution-original | 8 | 80.24% (471/587) | 92.78% (180/194) | 87.32% (62/71) | 72.09% (62/86) |

These are automatic compliance rates within each dataset. The official column targets the 2025-12-18 Model Spec; the generic Model Spec column uses the 2026-08-18 source. Constitution sources are dated 2026-01-20. Difficulty, scope and correlated source variants differ across columns, so no pooled benchmark score is defined.

## Reading the results

- Model Spec generic, batch 8 has the highest observed rates on the two Model Spec columns: +2.90 and +2.58 percentage points against the base.
- Constitution original, batch 8 leads its original-Constitution column by one question (1.41 pp). The best generic-Constitution rate is also only one question above the base (1.16 pp).
- 45 of the 48 section-cluster paired 95% intervals include zero. These exploratory intervals omit decoding/judge variability and multiple-comparison adjustment; known grading errors further limit ranking claims.

[metrics.csv](metrics.csv) contains all 52 model/dataset cells, denominators, differences against the base, valid-pair coverage, item/section-cluster intervals and diagnostics. Rates and differences are fractions in the CSV; the table displays percentages. Primary rates use the full denominator, including the upstream zero for an invalid critique. Paired contrasts exclude invalid verdicts and state their coverage.

## Protocol

Base model: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`. Each SFT uses one of the four [reviewed training collections](../../../data/README.md), initial learning rate **0.001**, LoRA rank **64**, one epoch, and batch **32, 16 or 8**. The base has no project SFT adapter; it is an existing pretrained vendor model. Different corpus sizes imply different training exposure and update counts, so this is not a controlled causal comparison of source wording alone.

The historical runs used native **DiMuon**, optimizer version 1, beta1 0.9, gradient-clip norm 0.0 and a **linear** learning-rate schedule. Loss was mean cross-entropy over all assistant messages, with a 32,768-token sequence cap, shuffle seed 0, incomplete final batches retained and no held-out training evaluation. The general Cookbook training wrapper in this checkout is not a replay command for this native optimizer configuration; the manifest records the settings actually used.

The judge is `openai/gpt-oss-120b:peft:131072` with `gpt_oss_low_reasoning`. The target uses `nemotron3_ultra_preserve_thinking`. Both temperatures are 0 and top_p is 1. Target output is capped at 8,192 tokens including thinking; judge output at 4,096. Target seed is 20261020; judge seed is 42. There is one answer and one verdict per item, using the original critique schema/parser, 1–7 scale and >=6 threshold. The rubric goes only to the judge; the full policy is omitted. Inspect system turns map to native developer instructions.

The renderer date is fixed to 2026-10-24. The new evaluations allowed up to four model processes with 32 in-flight samples each; the earlier standalone official base used one process. Automatic inference retries were disabled. The six new baseline pilot items are counted once, not added as extra samples. Matching seeds/settings does not guarantee backend reproducibility across collection dates.

The existing comparison CLI supports its upstream benchmark choices and comparisons sharing one training corpus. The three synthetic files also have a [reviewed Inspect task helper](../../builder/README.md). This aggregate combines the separate source-cohort evaluations; it is not a new one-command four-source CLI. Exact historical replay also requires the private trained checkpoints and row-level evidence, which are not distributed here. Dataset files, hashes, training configuration and metric arithmetic can be checked from this checkout.

## Known grading problems

1. On the public generic-Constitution case `constitution-generic/section-164/scenario-01/rejected-fix/manual-04/v0`, Codex inspection of all thirteen replies found **12 labels inconsistent with the explicit rubric**: three false automatic passes and nine false failures. The rubric forbids the `DOC_CONTROL: EMBEDDED` action, but the judge usually rewarded it. Opposite label errors on this one case can shift a model/base contrast by 2.33 pp.
2. In an issue-template case, the judge incorrectly called the required `Status` line an unrequested addition and missed an omitted reporter in the base response. The SFT nonpass can still be justified by that omitted reporter; a wrong rationale is not automatically a wrong binary label. A voice-length pair also has a debatable 5/6 boundary.
3. There were **137 section-ID mismatches in the new evaluations and 55 in the reused official replies** (192 total). The preserved parser accepts schema-valid JSON without checking that the section ID equals the requested ID. The 55 official mismatches were recomputed from saved critiques during publication; the earlier aggregate had not measured this field. All primary pass counts remain unchanged.
4. The new evaluations had **20 length stops and 19 empty final answers**, with no target/judge service errors or invalid critique JSON. An empty answer can pass a narrow prohibition while completing none of the requested task. Length/empty counts overlap and must not be added.

[judge_findings.json](judge_findings.json) records aggregate findings and public case IDs. Selected Codex checks are outcome-driven, not random or exhaustive, and are not independent human certification. The original scores have not been silently adjudicated. A revised grader would require a separately identified, consistently applied evaluation protocol.

## Provenance and verification

[manifest.json](manifest.json) binds these aggregate files to the four released test files, training data, model definitions, settings and opaque hashes of the locally audited source artifacts. The evaluation implementation/data revision used was `20c1bc8f384fa62244b2624fd1a2dd7d772209a4`. Model-specific sampler handles, project identifiers and machine paths are omitted.

Run these offline from the repository root; neither command makes model API calls:

```bash
python -m evaluation.datasets
python -m evaluation.published_results
```

Validation checks file hashes, all thirteen-by-four combinations, denominators, rates/deltas, interval bounds, diagnostic totals, training/data bindings and the published table. This confirms artifact consistency, not the semantic correctness of automatic judgments. The original native prompts, answer/critique streams and logs were checked locally and remain local.

The new evaluation token-rate estimate is **$5.40 at uncached rates**, or **$5.30 using returned cache-hit counts**. The earlier official evaluation estimate was $10.28 and is reported separately. Tariffs were checked on 2026-11-03; these are estimates rather than provider invoices, and exclude checkpoint storage.

No responses, reasoning, requests, logs, diaries, plans or execution history are included in this public result summary. The four test sets and their primary automatic results are preserved. Source overlap and possible pretraining exposure remain limitations; lexical screening is not proof of independence.
