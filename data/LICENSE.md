# Dataset license

bebetterest licenses the rights it holds in the following Spec2Behavior datasets, including their selection, arrangement, and review metadata, under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode.en).

| Dataset | Covered files |
| --- | --- |
| Generic SFT mixture | [candidates.jsonl](candidates.jsonl) and [reviewed.jsonl](reviewed.jsonl) |
| Four source-specific SFT collections | The candidate and reviewed JSONL files indexed in [variants/manifest.json](variants/manifest.json) |
| Three synthetic evaluation collections | [Model Spec generic](evaluation/model-spec-generic.jsonl), [Constitution original](evaluation/constitution-original.jsonl), and [Constitution generic](evaluation/constitution-generic.jsonl) |

These datasets contain AI-generated material. The [SFT data guide](README.md) and [evaluation data guide](evaluation/README.md) document their sources and review.

Suggested attribution when sharing the data:

> [Spec2Behavior datasets](https://github.com/bebetterest/spec2behavior/tree/main/data) by bebetterest, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Identify the collection and release or commit used where practical, and indicate any modifications when sharing.

The [official OpenAI Model Spec evaluation export](evaluation/model-spec-original.jsonl) retains its [upstream CC0-1.0 notice](evaluation/LICENSE-model-spec-dataset.txt). The source snapshots have their own CC0 notices for the [Model Spec](../model_spec/LICENSE) and the [Claude Constitution](../claude-constitution/LICENSE).
