# Evaluation datasets

Four fixed conversation/rubric collections are available, containing **938 evaluation records**. They are separate from the SFT conversations in `data/variants/`.

| Collection | Records | Specification | Origin |
| --- | ---: | --- | --- |
| [Model Spec original](model-spec-original.jsonl) | **587** | OpenAI Model Spec 2025-12-18 | Pinned OpenAI public dataset, applying upstream skips |
| [Model Spec generic](model-spec-generic.jsonl) | **194** | Generic Model Spec 2026-08-18 | Source-grounded synthesis and review |
| [Constitution original](constitution-original.jsonl) | **71** | Claude Constitution 2026-01-20 | Source-grounded synthesis and review |
| [Constitution generic](constitution-generic.jsonl) | **86** | Generic Constitution 2026-01-20 | Source-grounded synthesis and review |
| **Total** | **938** | | **587 upstream + 351 reviewed synthetic** |

[manifest.json](manifest.json) is the index: file paths, counts, file/content checksums, source versions and checksums, upstream revisions, construction settings, and review scope. Keep the four collections separate when reporting scores. The two Model Spec collections target different source versions; they are not an original/generic matched test pair.

## Format and loading

Each line is an Inspect AI record with `id`, `input`, `target`, and `metadata`. `input` is the conversation **before** the evaluated final answer. `target` is the grading rubric, **not** a reference answer. Send the conversation to the candidate model and the rubric to the judge only. The grading scale is 1–7, with compliance at **>=6**.

From a checkout, verify all files without dependencies beyond the base project and without any API calls:

```bash
python -m evaluation.datasets
```

Read a verified collection as dictionaries, preserving roles and tool histories:

```python
from evaluation.datasets import load_collection

records = load_collection("model-spec-generic")
conversation = records[0]["input"]
rubric = records[0]["target"]  # Judge only.
```

All four JSONL files also load directly with `inspect_ai.dataset.json_dataset`. For example:

```python
from inspect_ai.dataset import json_dataset

dataset = json_dataset("data/evaluation/model-spec-original.jsonl")
```

In the upstream convention, Inspect `system` messages represent **developer/operator** instructions. Preserve this mapping when using another model adapter. The official file includes tool-call histories and inputs ending in tool or instruction turns; do not force every official input to end in a user message. Internal transport message IDs are omitted; actual tool-call IDs, structured contents and message order are preserved.

For the three synthetic collections, `evaluation.builder.inspect_task.build_task(path, judge)` constructs a rubric-only Inspect task with an explicitly supplied judge. See [evaluation](../../evaluation/README.md) for the native Tinker adapter and [the builder](../../evaluation/builder/README.md) for generation and review. The existing `evaluation prepare/run` comparison CLI selects its upstream benchmarks; it does not automatically evaluate these three synthetic files. Loading or verifying data makes no inference calls.

## Official Model Spec collection

The official file is the pinned [OpenAI Model Spec Dataset](https://github.com/openai/model_spec_dataset/tree/89c260767c5a0a3a91a06dec77e3ab0306de768f), interpreted with the pinned [upstream loader](https://github.com/openai/model_spec_evals/tree/1d5df6d651c1e33161e507c8890af7ae00218941). The source has **596 records**, of which **9 carry upstream skip flags**, leaving the same **587** used in this project's official evaluations. Conversation, rubric and scoring metadata are unchanged by this export; JSON serialization is normalized by Inspect.

Its target is the **2025-12-18** Model Spec. It is not regenerated from the repository's newer 2026-08-18 specification. Dataset rights are provided under upstream **CC0-1.0**; the exact notice is preserved in [LICENSE-model-spec-dataset.txt](LICENSE-model-spec-dataset.txt). The upstream evaluation implementation has its separate [MIT notice](../../model_spec_evals/LICENSE).

## Reviewed synthetic collections

The synthetic collections use each nonempty heading's **direct body** as the primary source unit, keeping the full document available for context and exceptions. They use direct DeepSeek `deepseek-flash`, thinking enabled, and a **131,072-token output cap including thinking**. Each physical request has a fresh recorded random prompt seed; seeds are variation hints, not reproducibility guarantees.

Construction follows the [OpenAI method](https://alignment.openai.com/model-spec-evals/): extract an observable focus, verify its source evidence, author a conversation/rubric, independently author two compliant and two noncompliant probes, review the complete case, and grade the probes with intended labels hidden. Explicit hash-bound Codex review checks the source, rubric, exceptions, authority and all probe labels before retention. Some repairs include direct Codex corrections followed by fresh independent model stages. Separate calls do not imply independent model families.

The retained set has **351 records**, with **0 unresolved original candidates**. All 95 previously uncertain originals have final resolutions. Of 49 historical rejected originals, **42 have retained repaired successors and 7 are excluded**. The final rejected-case pass contributed 36 repairs to the unchanged 315-record baseline. Exclusions cover unsupported or unobservable grading boundaries, and cases whose selected failure could not be reliably instantiated by the construction method. Optional behavior was not converted into a duty, and placeholder output was not treated as an actual violation.

Published records preserve the reviewed IDs, conversations, rubrics and content provenance hashes. Machine-specific local run paths are removed from metadata. Candidate/plan hashes identify the original reviewed artifacts; they are not checksums of the sanitized public row. The manifest separately binds every public file and its `id`/`input`/`target` contents. Source snapshots remain available at their repository-relative paths.

Original candidates, revision decisions, probes, requests, replies, thinking, seeds, costs, failed attempts, and execution records remain **local only**. They are not included in this dataset release.

## Evaluation results and judge reliability

[The public thirteen-model comparison](../../evaluation/results/nemotron-lightning/README.md) reports the first aggregate evaluation on these fixed files. Selected case review found clear rubric misapplication, so data acceptance must not be treated as certification of an external judge. The report preserves the automatic primary rates and documents the known grading problems separately. Per-item answers, judge replies and execution evidence remain local.

The optional `build_task` helper enforces the returned section ID using its local `Grade` contract. The historical published comparison used the upstream critique parser without that equality check for every column. Use the documented protocol when comparing results; these validation choices can change scores.

## Validation and limits

- The official records match the pinned upstream loader, including its skip selection and tool histories.
- The final 36 added repairs passed cached offline reconstruction without model requests. All 315 earlier retained records remain unchanged in ID, input and rubric.
- The synthetic review screened against all **5,624 reviewed SFT records**. Final repair overlap and similarity screens had no unresolved flags; lexical screening does not establish semantic independence.
- Source-grounded model and Codex review are **not independent human certification** or measured benchmark reliability. Target-model score stability and behavioral generalization have not been newly established by this release.
- Aligned original/generic sources, reused policy principles, generator/reviewer model correlation, and possible pretraining exposure limit independence. Do not use these files as training input or interpret a pooled score as one benchmark.
