# Generate policy evaluation data

Build fresh conversation/rubric candidates for Model Spec generic and the original/generic Claude constitutions. This is separate from SFT synthesis. The example selects two sections per source, producing at most six final candidates. It follows the [OpenAI Model Spec Evals construction method](https://alignment.openai.com/model-spec-evals/): one concrete focus per case, source review, and calibration with hypothetical compliant/noncompliant answers.

OpenAI also used researcher review and human response labels. This implementation provides separate model calls and hash-bound human or Codex decisions; model checks and Codex decisions must not be presented as human certification or as an official benchmark.

The finalized exports are in [data/evaluation/](../../data/evaluation/README.md): 194 Model Spec generic, 71 original Constitution and 86 generic Constitution cases. Their manifest and dataset card describe the published selection. The pinned official Model Spec collection is maintained separately in that same directory. Use these fixed exports for evaluation; generation below constructs new candidates.

## Run

Use a local environment with `.[evalgen]` installed. Add `dev` for tests and `eval` for native Tinker/Inspect evaluation. For example:

```bash
uv venv --python 3.14 local/eval-builder/.venv
uv pip install --python local/eval-builder/.venv/bin/python -e '.[evalgen,dev]'

local/eval-builder/.venv/bin/python -m evaluation.builder prepare \
  --config evaluation/builder/config.example.json \
  --output local/eval-builder/smoke

local/eval-builder/.venv/bin/python -m evaluation.builder run \
  --plan local/eval-builder/smoke

# Makes paid API calls. Configure DEEPSEEK_API_KEY in the ignored root .env.
local/eval-builder/.venv/bin/python -m evaluation.builder run \
  --plan local/eval-builder/smoke --execute
```

`prepare` and the default `run` are offline. A changed source, training corpus, implementation, dependency or configuration invalidates the frozen plan. Use a fresh directory for a changed experiment. Changing a stage model also requires checking its tariffs. No key is written to the plan.

## Full three-source generation

The small example remains bounded to two selected sections per document. To attempt one case for every nonempty heading's direct body, use the full configuration:

```bash
local/eval-builder/.venv/bin/python -m evaluation.builder prepare \
  --config evaluation/builder/config.full.json \
  --output local/eval-builder/full-128k

local/eval-builder/.venv/bin/python -m evaluation.builder run \
  --plan local/eval-builder/full-128k --execute
```

`config.full.json` uses 128K output including thinking, concurrency 32, up to one content revision, at most 2,544 calls and a $50 conservative peak-rate estimate limit. The checked-in sources contain 79 + 40 + 40 direct-body units; untestable units can be omitted, so this is not a promise of 159 accepted cases. Review the frozen plan and pricing when sources change. Raw evidence and every failed/rejected attempt remain local. The accepted export still requires explicit hash-bound review decisions.

## Construction and review

1. Extract one observable focus from each selected section, with exact quotes, original line ranges, strength, applicability, exceptions and priority. The primary unit is each heading's direct body, so parent sections do not repeat their descendants; a nonempty preamble is retained. The full frozen document remains available to the extractor and source reviewers.
2. Review the focus before authoring a case. Unsupported identity/deployment assumptions and untestable points are omitted, not converted into invented generic facts.
3. Generate a fresh conversation and short rubric. Do not copy source examples or put the answer, source policy or rubric in the target input.
4. In a separate call, hide the rubric and write four source-grounded probes: minimal pass, alternative pass, near-miss fail and clear fail.
5. Review source fidelity, all probe labels and rubric scope. Grade probes in shuffled order with labels and source justifications hidden. By default the grader receives only conversation/answer/rubric, matching the project's rubric-only protocol; `grading_mode: full-spec` also supplies the source. The 1–7 scale passes at >=6.
6. A fixable review or calibration failure can trigger one revision by default. Every version and fresh check is retained. Rejection, uncertainty, truncation and malformed output are not silently repaired or accepted.
7. Screen conversations and rubrics against all four reviewed SFT collections. Lexical matches require review; no lexical flag does not establish semantic independence.
8. Inspect `REVIEW.md` and `cards/`, then record a decision and independently checked probe labels. Only reviewed candidates that completed the automated checks and have no unresolved lexical flag can be exported.

```bash
local/eval-builder/.venv/bin/python -m evaluation.builder review-template \
  --run local/eval-builder/smoke --output local/eval-builder/smoke/decisions.json

# Fill decision, reviewer, reviewer_type, rationale and all four probe_labels.
local/eval-builder/.venv/bin/python -m evaluation.builder export \
  --run local/eval-builder/smoke \
  --decisions local/eval-builder/smoke/decisions.json \
  --output local/eval-builder/smoke/reviewed.jsonl
```

Reviewer type is `human` or `codex`; leave uncertain cases out. A changed candidate invalidates its decision. The template deliberately does not prefill independent labels. Exported records contain `id`, `input`, `target` and `metadata`; probes and source evidence stay in the review artifacts, not the target input.

## Bounded corrections after independent review

The main pipeline can restore omitted Markdown footnote markers only when every other character matches uniquely inside the declared lines. It records normalization separately and preserves raw replies and cached results. If an independent reviewer finds a semantic defect in a first-version case, use the optional curation helper. It reuses completed calls, preserves the raw response and creates a child plan with the feedback and implementation hashes. It shares the original call and cost caps; a failed/uncertain API request is never implicitly replayed.

Create a local changes JSON file containing `resume` (existing job IDs) and/or `revisions`. Each revision requires `id`, the canonical `case_sha256` from its original `v0/case` result, `reviewer`, `reviewer_type`, source-grounded `issues`, and exact source `evidence` with original line ranges. It can use the still-unused first content revision, not overwrite a prior revision. If the original plan forbids revisions, this action is refused.

```bash
local/eval-builder/.venv/bin/python -m evaluation.builder.curate \
  --run local/eval-builder/smoke --changes local/eval-builder/changes.json

# Review the child plan first, then execute only the selected corrections.
local/eval-builder/.venv/bin/python -m evaluation.builder.curate \
  --run local/eval-builder/smoke --changes local/eval-builder/changes.json --execute
```

For new source/case review calls, this helper defaults to `--review-effort low` while keeping the original thinking toggle and output cap. DeepSeek and OpenRouter use their respective effort fields. Other stages retain their frozen settings. The child plan records this override. Citation normalization can only restore footnote markers inside the declared source lines; it cannot change words, punctuation, strength or exceptions. New content receives fresh blind probes and grades. Review the resulting candidates and use the normal decision/export steps; a correction does not imply acceptance.

## Configuration

`config.example.json` is the editable entry point:

- `sources[].sections`: title regexes. An empty list selects all nonempty sections, bounded by `max_sections_per_source`; increase that limit explicitly for wider coverage. Each selected section yields at most one focus/case.
- `model` and optional `stage_models.author/reviewer/judge`: select compatible model aliases on the configured endpoint. Defaults use `deepseek-flash` for all stages. Separate calls do not imply independent model families.
- `provider`, `base_url`, `api_key_env`: use the existing DeepSeek/OpenRouter/OpenAI-compatible HTTP client. Thinking fields follow the selected provider; generic endpoints require `thinking: null`.
- `thinking`, `max_tokens`, `temperature`: default thinking on and 131,072 output tokens (128K) including thinking. Input context is separate and remains the provider/model's context window. The default request timeout is 1,800 seconds to accommodate long thinking outputs. DeepSeek ignores temperature in thinking mode; the request omits it. New requests get different recorded prompt seeds, which are provenance/variation hints, not deterministic sampling guarantees.
- `concurrency`, `max_revisions`, `max_calls`, `max_estimated_usd`: bounded execution. The example allows six concurrent jobs and at most 96 API calls including content revisions. Conservative in-flight token reservations stop new requests before the estimated allowance is exhausted.
- The three token tariffs are configurable and frozen. Recorded usage yields a peak-rate estimate; missing usage is accounted for using the conservative request reservation. This is not a provider invoice or account-level spending cap.

There is no automatic HTTP/inference retry. Run-level locking prevents concurrent execution against one plan. Re-running resumes completed calls/outcomes without rebilling them. A failed or uncertain call is not implicitly replayed; inspect its saved API record before choosing a new run. A crash lock requires checking its saved PID before removal.

On macOS, HTTPX can inherit the system HTTP/HTTPS proxy even when proxy environment variables are empty. If direct access to the provider works more reliably, bypass proxy inheritance for that host when launching a run, for example `NO_PROXY=api.deepseek.com local/eval-builder/.venv/bin/python -m evaluation.builder run --plan local/eval-builder/full-128k --execute`. The Python `ChatClient`/`StreamingChatClient` API also accepts `trust_env=False`. These choices do not modify global OS proxy settings. Validate the route before issuing paid calls.

## Explicit network recovery

A completed run can contain network-failed jobs without discarding successful work. Inspect their saved API records first. To freeze one explicit recovery attempt for jobs classified as `RetryableError`:

```bash
local/eval-builder/.venv/bin/python -m evaluation.builder.recover \
  --run local/eval-builder/full-128k --concurrency 8

local/eval-builder/.venv/bin/python -m evaluation.builder.recover \
  --run local/eval-builder/full-128k --concurrency 8 --execute
```

The child plan binds the failed call keys, code and parent ledger snapshots. Completed requests are reused; the original failed attempt is preserved under its original key and the new attempt has a separate key and seed. Unknown earlier usage continues to consume its conservative reservation. The original call/cost limits and content-revision limit remain in force. Schema and source-fidelity errors are not network retries.

This helper permits one explicit recovery attempt per failed network call. A further network, fatal API or budget error stops new requests; already in-flight replies are retained. Inspect `manual-recovery-1/result.json` for `finished` or `paused`. Repeating a finished/paused recovery does not rebill it; further action requires a new, separately reviewed recovery plan rather than an implicit loop.

## Streaming recovery for long replies

If the non-streaming path repeatedly loses connections, keep the same model, prompts, token cap and global billing limits while using SSE. Streaming preserves received text/thinking deltas locally and requests the provider's final usage statistics. An interrupted stream is not accepted as a complete response. This path is currently verified for DeepSeek.

```bash
# First freeze a source-balanced plan and validate one unfinished job.
local/eval-builder/.venv/bin/python -m evaluation.builder.recover_stream \
  --run local/eval-builder/full-128k --probe-id constitution-generic/section-337
local/eval-builder/.venv/bin/python -m evaluation.builder.recover_stream \
  --run local/eval-builder/full-128k --phase probe --execute

# Execute only after inspecting a finished transport probe.
local/eval-builder/.venv/bin/python -m evaluation.builder.recover_stream \
  --run local/eval-builder/full-128k --phase remaining --execute
```

Choose an actual unfinished job for `--probe-id`; it is frozen in the plan and need not be repeated later. Concurrency defaults to 4. Sources are interleaved so one document cannot monopolize the queue. This is a separate explicit recovery plan, not an automatic retry policy. Earlier successful aliases are reused, prior unknown-usage reservations remain charged to the same estimate limit, and any new network/API/budget failure stops further requests. Partial SSE evidence is stored as `stream_chunk` events; normal `response` events are clearly marked as assembled from that stream. The transport probe can finish while the candidate fails a quality gate; quality and network validation are separate.

## Inspect format and scope

Install `.[eval,evalgen]` to use `evaluation.builder.inspect_task.load_dataset(path)` or `build_task(path, judge)`. The caller supplies an explicitly configured Inspect judge, including a native Tinker judge. This does not automatically add custom data to the existing official-benchmark CLI.

Like the pinned OpenAI dataset, an Inspect `system` message represents a developer/operator instruction. When using this project's `PairedTinkerAPI`, configure `task_name="model_spec"` to apply that documented mapping. The export records `instruction_role_convention` explicitly. Do not interpret these as true system-over-developer tests. Distinguishing those two levels, tool execution and multimodal behavior are outside this first text-only builder and should be recorded as omissions.

All sources, plans, raw requests/replies/thinking, errors, costs, revisions, candidate cards and decisions should remain under ignored `local/` or `runs/`. Keep original candidates even when excluding them. Code/docs and explicitly reviewed portable evaluation exports belong in the repository; execution history does not.

## Add scenarios to a reviewed run

The supplemental builder reuses only exact, accepted source extractions and source reviews. It adds three fresh scenarios per eligible focus by default and can repair explicitly selected excluded candidates in a new run. Raw cases and decisions from the parent stay unchanged. Each new case still receives fresh challenges, source/case review, blind grades and an explicit Codex/human export decision.

Create an options JSON with `cases_per_focus`, `concurrency`, `max_estimated_usd`, optional `excluded_focuses` (ID-to-reason mapping), and optional `repairs` (excluded parent candidate IDs). The helper excludes focuses labelled identity-dependent, aspirational or `may`. Use `excluded_focuses` to record other unsupported or untestable boundaries, including permissions incorrectly extracted as obligations. A different focus requires its own source review; this helper does not invent that review. Repair feedback is visible only to the case author, never to the independent challenge writer or rubric-only grader. The general construction contract preserves source strength and forbids conflicting binary grading scales.

```bash
local/eval-builder/.venv/bin/python -m evaluation.builder.supplement prepare \
  --parent local/eval-builder/full-128k \
  --options local/eval-builder/supplement-options.json \
  --output local/eval-builder/supplement-3x

# Paid three-source probe; inspect quality and transport before continuing.
local/eval-builder/.venv/bin/python -m evaluation.builder.supplement run \
  --plan local/eval-builder/supplement-3x --phase probe --execute
local/eval-builder/.venv/bin/python -m evaluation.builder.supplement run \
  --plan local/eval-builder/supplement-3x --phase remaining --execute
```

Execution uses direct SSE with proxy inheritance disabled, fresh prompt seeds and zero HTTP retries. The first new transport/API/budget failure stops new requests. An explicit numbered `--phase continue-1` can process unstarted or interrupted jobs after a remaining-phase result; it freezes a ledger snapshot, reuses successful calls, skips failed jobs and keeps the original supplemental caps. Repeating a completed phase makes no paid calls. A new phase does not authorize retrying a failed request. Output/cost limits and all inputs/code are frozen before execution. The supplemental $30 default is separate from the parent's ledger and must be covered by the current run authorization.

Use normal review-template/export commands for the new run. A final aggregate should keep the earlier reviewed files intact, preserve unique IDs and source/decision hashes, and screen both SFT overlap and overlap with earlier evaluation cases. Report attempted units, authored candidates and accepted rows separately; adding scenarios does not guarantee more accepted rows.

For a user-requested pause or a ledger with explicit `user_paused` outcomes, use the snapshot-bound continuation helper. Its snapshot contains `parent_plan_sha256`, `calls_sha256`, `outcomes_sha256`, and a `jobs` array of unfinished IDs. Preparation validates that this is exactly the unfinished set and excludes failed physical requests. It preserves the ledger snapshot and freezes the continuation implementation in a child plan:

```bash
local/eval-builder/.venv/bin/python -m evaluation.builder.resume_supplement prepare \
  --plan local/eval-builder/supplement-3x \
  --snapshot local/eval-builder/supplement-3x/pause/resume-jobs.json --segment resume-1
local/eval-builder/.venv/bin/python -m evaluation.builder.resume_supplement run \
  --plan local/eval-builder/supplement-3x --segment resume-1 --execute

# Stop dispatching new requests; let already-started replies finish normally.
touch local/eval-builder/supplement-3x/pause.requested
```

Wait for `segments/resume-1/result.json` and `inflight: 0` before treating the pause as complete. Successful partial calls stay cached, and the segment writes `resume-jobs.json` for another explicit continuation. On an authorized resume, remove `pause.requested` and prepare a fresh numbered segment from that snapshot. Do not remove the stop file or resume automatically. Re-running an ended segment makes no new calls. The parent configuration, total cost/call caps and failed-request exclusions remain binding across all segments. The ordinary `continue-N` command does not resume outcomes already marked `user_paused`.
