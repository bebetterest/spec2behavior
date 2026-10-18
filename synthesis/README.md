# Synthesis

The runner calls a chat completions API. A chain is a template with three functions: `read` yields one record per model call, `prompt` builds that call, and `parse` turns the reply into SFT conversations. Six defaults live in `synthesis/templates/`. A job is usually one model call. A class that overrides `produce` is for a job that makes several calls.

## Run

From the repository root:

```bash
python -m synthesis list
python -m synthesis run rule_examples --help
```

Put `DEEPSEEK_API_KEY` in `.env` (see [the example](../.env.example)), then preview one request. The defaults use the DeepSeek API, `deepseek-flash`, and thinking on:

```bash
python -m synthesis run rule_examples \
  --source generic/model_spec.md \
  --output runs/model-spec/rule-examples-deepseek \
  --limit 1 \
  --max-retries 0 \
  --dry-run
```

Inspect the request, then remove `--dry-run` to generate that sample. The same command resumes. Finished job ids in `jobs.jsonl`, and unresolved parse or fatal API failures in `errors.jsonl`, are skipped. A trailing partial line left by a crash is discarded and that job runs again. Remove `--limit 1` to process all pending excerpts. `--help` shows the defaults for the selected chain.

Configuration precedence is CLI flags, then process environment variables, then `.env` in the current directory. `--env-file FILE` selects another file. Only the selected file is read; parent directories are not searched, and the process environment is not modified. Quoted values and comments use [python-dotenv syntax](https://github.com/theskumar/python-dotenv#file-format); variable interpolation and shell commands are not executed. Provider selection remains an explicit `--provider` flag. Model, API keys, and `OPENAI_BASE_URL` can be set in the file.

| Flag | Meaning |
|---|---|
| `--provider` | API preset: `openai`, `deepseek` (default), or `openrouter`. Explicit `auto` uses `--base-url` / `OPENAI_BASE_URL` and detects known hosts. |
| `--env-file` | Key/model/endpoint settings file. Default: `.env` in the current directory, if present. An explicitly selected file must exist. |
| `--max-tokens` | Output token budget for the entire response, including its JSON wrapper and every dialogue turn. Default 16384. Use `default` to omit the API field. |
| `--max-token-field` | API field for that budget: `max_tokens` (default) or `max_completion_tokens`. |
| `--thinking` | DeepSeek/OpenRouter thinking mode: `on` (default), `off`, or `default` (omit the field and keep the server default). OpenAI omits the field when this flag is absent. |
| `--temperature` | Sampling temperature from 0 to 2. Default 1.0. Use `default` to omit the API field. |
| `--max-portion-chars` | Soft character target for the heading's direct body. Default 16000. This controls input splitting. |
| `--examples` | `rule_examples` only: exactly this many request/reply pairs. Default 1, yielding one `generate` and one `act` row. |
| `--concurrency` | Maximum requests in flight. Default 8. |
| `--batch-size` | How many pending jobs are scheduled together. Default 32. Use a value at least as large as `--concurrency`. The next batch starts only after the current one has been written. |
| `--limit` | Cap how many pending jobs this invocation runs. |
| `--retry-errors` | Run ids in `errors.jsonl` again. Successful jobs are still skipped. |
| `--overwrite` | Delete this output directory's checkpoints and `api.jsonl`, then start over. |
| `--dry-run` | Print the first pending request. With `--overwrite`, preview a fresh run. No API call and no files written. |
| `--max-retries` | Extra attempts for HTTP 408, 409, 429, and 5xx, and for network errors. Default 5. |
| `--timeout` | HTTP timeout in seconds. Default 120. |
| `--retry-backoff` | Initial retry delay in seconds. Default 1.0; doubles on each retry. |
| `--base-url` | Override the selected provider's API root; a full `.../chat/completions` URL also works. In `auto` mode, reads `OPENAI_BASE_URL` or defaults to OpenAI. |
| `--api-key` | Override the selected provider's key environment variable. Omit for a local server. |
| `--model` | Model name. Uses `SYNTHESIS_MODEL`, then `deepseek-flash` for DeepSeek. Other providers require their own model name. |

A retryable failure is not marked done. Run the same command to try it again. A parse failure or a non-retryable HTTP error is skipped so a bad row does not spin forever. Fix the pipeline or the request, then pass `--retry-errors`. Error history is retained, including jobs outside `--limit`; a successful retry takes precedence over its earlier error records.

Dry-run uses the same completed-job selection as execution. It ignores an unfinished trailing checkpoint line without repairing the file; execution discards that line before resuming. With `--overwrite --dry-run`, existing checkpoints are ignored for the preview and stay untouched.

Truncated responses (`finish_reason=length`), filtered responses, tool calls, and explicit refusals are recorded as errors with the returned text. They do not become training rows or trigger automatic HTTP retries. To retry a truncated response, increase `--max-tokens` and repeat the command with `--retry-errors`; completed jobs are still skipped. The token budget is a configurable starting value, not a measured optimum or a guarantee of completion.

For an endpoint that requires the newer token field, add `--max-token-field max_completion_tokens`. In OpenAI's API that budget includes visible output and reasoning tokens; `max_tokens` is deprecated there. See the [Chat Completions API reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create). `--temperature default` supports endpoints that reject an explicit temperature. Other model-specific fields can be set in the template's `ChatRequest.extra`.

Select a provider to switch its default URL, key variable, and thinking request format together:

| `--provider` | Default API root | Key variable | `--thinking on` / `off` |
|---|---|---|---|
| `deepseek` | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` | `thinking.type`: `enabled` / `disabled` |
| `openrouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | `reasoning.enabled`: `true` / `false` |
| `openai` | `https://api.openai.com/v1` | `OPENAI_API_KEY` | Leave at `default`; set model-specific reasoning fields in the template. |

Explicit `--base-url` and `--api-key` override the preset. An explicit provider ignores `OPENAI_BASE_URL` and uses only its own key variable, so switching providers does not reuse another provider's configuration. For a proxy with a custom URL, select the provider whose request format it accepts, then pass `--base-url`.

Explicit `--provider auto` uses the existing URL/key settings: URL precedence is `--base-url`, then `OPENAI_BASE_URL`, then the OpenAI root. It recognizes `api.deepseek.com`, `openrouter.ai`, and `eu.openrouter.ai`; other hosts use the OpenAI format. It tries the recognized provider's key variable first, then falls back to `OPENAI_API_KEY` for compatibility with older commands. Select `--provider deepseek` or `openrouter` explicitly to use `--thinking on/off` through a custom proxy URL.

When `--thinking` is omitted, DeepSeek and OpenRouter enable thinking, while OpenAI keeps its server default. Use `--thinking default` to leave the choice to the endpoint, or `--thinking off` to disable it on DeepSeek/OpenRouter. On OpenRouter, choose a model that supports the requested mode; models with mandatory reasoning cannot disable it. A template's explicit `ChatRequest.extra["thinking"]` (DeepSeek) or `["reasoning"]` (OpenRouter) replaces the CLI setting for that whole object.

```bash
export DEEPSEEK_API_KEY=...
python -m synthesis run rule_examples \
  --source generic/model_spec.md \
  --output runs/model-spec/rule-examples-thinking \
  --provider deepseek \
  --model deepseek-flash \
  --thinking on \
  --dry-run
```

For OpenRouter, pass `--provider openrouter --model deepseek/deepseek-v3.2` and configure `OPENROUTER_API_KEY`. The CLI does not rewrite model names or replace an explicit `SYNTHESIS_MODEL`. See the [OpenRouter quickstart](https://openrouter.ai/docs/quickstart). Temperature stays configurable at its default of 1.0; DeepSeek ignores it while thinking is enabled.

Only the final `message.content` is parsed into SFT and saved as raw job text. The separate `reasoning_content`, `reasoning`, and `reasoning_details` fields stay in the API trace, outside SFT rows and raw job text; a reply with no final content is rejected. OpenRouter reasoning usually shares the total `--max-tokens` budget with the visible answer. `reasoning.exclude` only hides reasoning in the response; the `--thinking off` toggle uses `reasoning.enabled=false` to request disabling it. See [DeepSeek thinking mode](https://api-docs.deepseek.com/guides/thinking_mode/) and [OpenRouter reasoning tokens](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens) for API behavior.

Exit status is `0` when the selected jobs succeed, `1` when jobs fail, remain pending, or have recorded errors skipped on resume, and `2` for invalid arguments or input. Ctrl-C exits `130`. Completed jobs from that batch are kept. A `--limit` run can exit `0` while later, unselected jobs remain.

## Output

```text
runs/<name>/
  api.jsonl       request/response events for every HTTP attempt, including retries
  jobs.jsonl      one line per completed job; this is the resume checkpoint
  train.jsonl     one line per training conversation
  errors.jsonl    parse and fatal API error history; unresolved jobs skipped on resume
  failures.jsonl  retryable failures that will be tried again; safe to delete
```

The built-in API client records each request in `api.jsonl` before sending it and the full response body before validating or parsing it. Request events contain the complete JSON body, including source excerpt, prompt, model, and sampling settings. Response events contain the HTTP status and original response text in `body`, including any reasoning, usage, and provider metadata. Non-JSON and failed HTTP responses are retained too. Transport failures and cancellations have an event without an invented response. A hard interruption can leave a request without a response event.

Events share `pipeline`, `job_id`, `call_id`, and `attempt` to associate concurrent jobs, multiple calls inside one job, and HTTP retries. Match `job_id` and `pipeline` to the SFT metadata. Normal resume and retries append events; `--overwrite` clears the trace. Dry-run writes nothing. Headers are omitted, URL credentials and query parameters are stripped, and occurrences of the configured API key are redacted in trace values. A custom client supplied to `run_synthesis` must provide its own HTTP tracing.

The six default templates also record `generation_seed` on API events, completed jobs, parse/fatal errors, retryable failures, and each SFT row's `meta`. It identifies the random string embedded in that candidate's generation prompt. Training messages exclude it; an echoed seed is a parse error and the raw reply is retained for inspection.

The trace is for inspection, not training or resume selection. `jobs.jsonl` remains the checkpoint. Existing runs retain only what was saved at the time; earlier requests and full responses are not reconstructed retroactively. Keep run outputs under the ignored `runs/` directory.

A training line:

```json
{
  "id": "rule_examples/model_spec.md/protect_privileged_information/0#1",
  "job_id": "model_spec.md/protect_privileged_information/0",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "meta": {
    "source": "generic/model_spec.md",
    "portion_id": "protect_privileged_information/0",
    "sft_form": "act",
    "pipeline": "rule_examples",
    "generation_seed": "a9d7c4e1b2f083659ce4d71a038f6b25"
  }
}
```

`train.jsonl` is rebuilt from `jobs.jsonl` at the start and end of a run. `python -m synthesis materialize runs/<name>` rebuilds it on its own.

New row ids are `<pipeline>/<job_id>#<example index>`, so different templates over the same source have distinct ids. Existing checkpoints keep their saved ids and resume keys; older exports may need their row ids prefixed before combining them. Continue using separate output directories for each source and template.

Tinker SFT reads `messages`. `system`, `developer`, `user`, `assistant`, and `tool` are accepted. Every saved conversation must include a user message and an assistant message.

Before training, review rule compliance, exceptions, and unsupported factual claims. Keep accepted rows in a separate location such as `runs/model-spec/reviewed/train.jsonl`; editing the generator's `train.jsonl` would be undone by resume or materialization. Validate the reviewed dataset without contacting Tinker:

```bash
python -m training sft \
  --data runs/model-spec/reviewed/train.jsonl \
  --log-dir logs/model-spec \
  --model Qwen/Qwen3-8B \
  --batch-size 1 \
  --dry-run
```

This checks the dataset structure and training settings. It does not grade content or verify token lengths with the target tokenizer. See [training](../training/README.md) for the full settings.

## Templates

`list` shows the registered chains. A built-in name runs as before:

```bash
python -m synthesis run rule_examples --help
```

A new chain is a copy of a file in `synthesis/templates/`. Edit `read`, `prompt`, and `parse`. `add_arguments` adds flags that belong to that file. Set `name` to a lower-case identifier when the filename is not already one. Files whose names start with `_` are helpers and are not registered.

```bash
python -m synthesis run --template path/to/copy.py --output runs/name --source policy.md
```

`read` yields a dict with a string `id`, or a `Job`. `prompt` returns a `ChatRequest`, a user-message string, or a list of messages. `parse` returns `Example`s, a list of message lists, or one message list. Each saved conversation still needs a user message and an assistant message.

`prompt` and `parse` receive the same dict `read` yielded, including `id`.

### Shared excerpt flags

Every default chain takes `--source`, a Markdown file. The basic excerpt is a heading's direct body: all paragraphs after that heading, stopping at the next heading of any level. Child sections are read separately, so their text is not repeated in their parent. A direct body stays in one excerpt when it fits `--max-portion-chars` (default 16000).

Longer bodies are grouped into excerpts on paragraph boundaries. A one-line lead-in stays with the next paragraph. A lead-in ends with a colon, is an example label such as `**Example**: name`, is a bold label such as `**Types of uncertainty**`, or is an admonition marker such as `!!! meta "Commentary"`. A block that is only a table of contents is skipped. A fenced block is one paragraph. An individual paragraph longer than the character target is cut near a newline or a space, and a leftover shorter than half that target stays with the previous piece. The character target is therefore a soft limit. Headings inside fenced code are ignored.

The job id is `<filename>/<heading id>/<piece index>`. Heading ids prefer an explicit `{#anchor}`. Text before the first heading is omitted. Use a separate `--output` directory per source file and chain. Use a new output directory after changing the source, template, provider, model, thinking mode, or splitting settings: completed job ids do not encode those settings. Adjusting a token budget to retry failed jobs can reuse the output directory.

`--temperature` defaults to 1.0. `--no-json-mode` drops `response_format=json_object` for servers that reject it. The model is asked for JSON. The saved conversation stores the unwrapped text, and `meta.sft_form` records which shape it is. The runner also sets `meta.pipeline`, and `meta.model` when a model name is set.

Each excerpt/template combination makes one logical generation call for one candidate. `rule_examples` uses that candidate for both its `generate` row and the `act` row for each pair; `--examples` controls pair count, not candidate count. The other templates each produce one row.

Every new candidate's prompt begins with a line such as `Random seed: a9d7c4e1b2f083659ce4d71a038f6b25`. The shared helper generates 16 random bytes with `secrets.token_hex(16)`, yielding a 32-character string, and asks the teacher to vary appropriate scenario details and wording while preserving the excerpt's rules. This is a prompt diversity hint, not an API sampling `seed`; it does not guarantee unique output or deterministic replay. The string is saved as provenance, outside training messages.

Automatic HTTP retries reuse the same prompt and seed. A fresh attempt at an unfinished job, `--retry-errors`, or a new run builds a fresh seed. Completed jobs stay skipped on resume. Dry-run creates an unreserved preview seed without saving it, so a later real invocation gets another seed.

```bash
python -m synthesis run rule_examples \
  --source generic/model_spec.md \
  --output runs/model-spec/rule-examples

python -m synthesis run reflection \
  --source generic/constitution.md \
  --output runs/constitution/reflection
```

| Chain | `sft_form` | Training row |
|---|---|---|
| `rule_examples` | `generate`, then one `act` per pair | `generate`: the excerpt plus "write requests and replies", and the reply is the rendered pairs. `act`: the example request, and the reply is the example answer. `--examples` defaults to 1. Replies with a different pair count are recorded as errors. |
| `correction` | `correct` | The request, the violating reply, and an instruction to correct it. The assistant message is the corrected reply. |
| `reflection` | `reflect` | The excerpt plus a request to explain its requirements, recommendations, permissions, defaults, and conditions. The assistant message is that reflection. |
| `pressure` | `pressure` | A multi-turn dialogue testing a stated boundary. Replies resist pressure while adapting to genuinely changed conditions and allowed revised requests. |
| `chain_of_command` | `priority` | `system`, `developer`, `user`, and `assistant`, in that order. The conflict tests the excerpt's rule, included in the supplied messages. The reply follows the controlling instruction and fulfills compatible requests. |
| `safe_completion` | `partial` | A coherent request the excerpt only partly allows. The reply substantively answers the allowed part and refuses only the disallowed part. |

`generate` and `reflect` include the excerpt. `act`, `correct`, `pressure`, `priority`, and `partial` do not. The local file path is stored in `meta.source` and left out of the messages.

Parsing checks the schema and message structure. It does not independently grade rule compliance. Review a small generated sample before scaling a dataset.

Shared generation instructions preserve rule scope, strength, conditions, exceptions, and allowed discretion. Scenario facts needed to justify a reply must appear in the saved inputs; the teacher's own deployment is not context for the generated assistant. Reflection preserves conditional rules and unspecified cases. Pressure dialogues distinguish insistence from genuinely changed conditions. Instruction conflicts and partial completions must test the excerpt's actual boundary rather than add unrelated restrictions or harmless filler.

When an excerpt cannot support a template's training form without inventing or changing a rule, the teacher may return `{"not_applicable":"reason"}`. The parser records this as `template not applicable: ...` in `errors.jsonl`, retains the raw reason, and emits no SFT rows. It counts as an error and is skipped on resume unless `--retry-errors` is passed. This is an inspectable exclusion, not a successful generation or an automatic semantic grade. Normal output shapes and row counts stay the same. These prompt constraints still require content review: valid JSON can misstate a rule or invent facts.

## Add a chain

Copy a file in `synthesis/templates/`. The three functions are the whole chain. `--template` may be repeated. When one file is passed and no chain name is given, that file is the chain.

```python
name = "gloss"
description = "Turn each line into one SFT conversation."

def add_arguments(parser):
    parser.add_argument("--input", required=True)

def read(args):
    for index, line in enumerate(open(args.input, encoding="utf-8"), start=1):
        text = line.strip()
        if text:
            yield {"id": f"line-{index}", "text": text}

def prompt(item, args):
    del args
    return f"Translate to French:\n{item['text']}"

def parse(item, content, args):
    del args
    if not content.strip():
        raise ValueError("empty completion")
    return [
        {"role": "user", "content": item["text"]},
        {"role": "assistant", "content": content.strip()},
    ]
```

```bash
python -m synthesis run --template chains/gloss.py --output runs/gloss --input lines.txt
```

`name` is a lower-case identifier and becomes the CLI command when you pass it explicitly. Do not reuse runner flags (`--output`, `--model`, `--concurrency`, and the rest listed on `Pipeline`).

Job ids must be stable across runs. The runner refuses duplicate ids. `parse` may return several conversations from one reply; they are stored as `<job id>#0`, `<job id>#1`, and so on.

`parse_json_object` in `synthesis.jsonutil` accepts a JSON object, a fenced object, or an object surrounded by prose. Raise `ParseError` from `parse` when the reply cannot be used. The runner records that and skips the job until `--retry-errors`.

A `Pipeline` subclass loaded with `--pipeline-file` is for a job that calls the model more than once. A module dropped in `synthesis/pipelines/` is imported on startup.

### Several model calls in one job

Override `produce` and call `client.complete` once per step. The job is saved only after `produce` returns, so a later step that fails runs the whole job again. `--concurrency` limits `complete` calls across jobs. Return a list of `Example`, or a `Produced` when you want the step outputs kept in `jobs.jsonl` as `raw`.

```python
async def produce(self, job, args, client):
    draft = await client.complete(ChatRequest(messages=[
        {"role": "user", "content": f"Draft a reply:\n{job.input['text']}"},
    ]))
    final = await client.complete(ChatRequest(messages=[
        {"role": "user", "content": f"Revise this reply:\n{draft}"},
    ]))
    return self.parse(job, final, args)
```

`preview` controls the dry-run printout. The default prints `request()`. Return `None` when the first call is built inside `produce`.

A later stage can also be its own command. Its `read()` walks the previous `train.jsonl` with `read_jsonl` from `synthesis.io`. Use a new `--output` directory so the two checkpoints stay separate. A boolean `trainable` on a message returned from `prompt()` is left out of the API request. `parse()` can keep it on the training row.
