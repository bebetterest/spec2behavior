"""Command line for synthesis runs.

Run from the repository root:

    python -m synthesis list
    python -m synthesis run <pipeline> --help
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values

from synthesis.arguments import nonnegative_float, nonnegative_int, optional_positive_int, positive_float, positive_int
from synthesis.client import DEFAULT_MAX_TOKENS, PROVIDER_DEFAULTS, TOKEN_LIMIT_FIELDS, ChatClient, resolve_provider
from synthesis.io import materialize_train
from synthesis.pipeline import PIPELINES, get_pipeline, load_pipeline_file
from synthesis.runner import _log, run_synthesis
from synthesis.template import load_template_file

_HELP = """\
usage: python -m synthesis <command>

commands:
  list                         Show available chains
  run NAME [options]           Run one chain. Use --help after the name for its flags
  run --template FILE [options]
                               Run a file that defines read, prompt, and parse
  materialize OUTPUT_DIR       Rebuild train.jsonl from jobs.jsonl

--template FILE and --pipeline-file FILE may appear anywhere and may be repeated.
"""


def main(argv: list[str] | None = None) -> int:
    import synthesis.pipelines  # noqa: F401  loads @register side effects

    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        pipeline_files, template_files, argv = _take_chain_files(argv)
        for path in pipeline_files:
            load_pipeline_file(path)
        template_names = [load_template_file(path) for path in template_files]
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    if not argv or argv[0] in {"-h", "--help"}:
        print(_HELP, end="")
        return 0
    command, rest = argv[0], argv[1:]
    implied = template_names[0] if len(template_names) == 1 else None
    if command == "list":
        return _list(rest)
    if command == "run":
        return _run(rest, implied)
    if command == "materialize":
        return _materialize(rest)
    print(f"unknown command {command!r}\n", file=sys.stderr)
    print(_HELP, end="", file=sys.stderr)
    return 2


def _take_chain_files(argv: list[str]) -> tuple[list[Path], list[Path], list[str]]:
    pipeline_files: list[Path] = []
    template_files: list[Path] = []
    rest: list[str] = []
    index = 0
    while index < len(argv):
        flag = argv[index]
        if flag in {"--pipeline-file", "--template"}:
            if index + 1 >= len(argv):
                raise ValueError(f"{flag} needs a Python file")
            bucket = template_files if flag == "--template" else pipeline_files
            bucket.append(Path(argv[index + 1]))
            index += 2
            continue
        rest.append(flag)
        index += 1
    return pipeline_files, template_files, rest


def _list(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m synthesis list", description="Show pipelines.")
    parser.parse_args(argv)
    if not PIPELINES:
        print("no pipelines registered")
        return 0
    width = max(len(name) for name in PIPELINES)
    for name in sorted(PIPELINES):
        print(f"{name:<{width}}  {PIPELINES[name].description}")
    return 0


def _materialize(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m synthesis materialize",
        description="Rebuild train.jsonl from jobs.jsonl.",
    )
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    jobs_path = args.output_dir / "jobs.jsonl"
    if not jobs_path.is_file():
        print(f"no jobs file at {jobs_path}", file=sys.stderr)
        return 2
    count = materialize_train(jobs_path, args.output_dir / "train.jsonl")
    print(f"wrote {count} training rows to {args.output_dir / 'train.jsonl'}")
    return 0


def _run(argv: list[str], implied_name: str | None = None) -> int:
    if not argv or argv[0] in {"-h", "--help"}:
        if implied_name and argv and argv[0] in {"-h", "--help"}:
            name, pipeline_argv = implied_name, argv
        else:
            print(
                "usage: python -m synthesis run NAME [options]\n"
                "       python -m synthesis run --template FILE [options]\n",
                file=sys.stderr,
            )
            _list([])
            return 0 if argv else 2
    elif argv[0].startswith("-"):
        if implied_name is None:
            print(
                "Pass a chain name, or one --template file.\n",
                file=sys.stderr,
            )
            return 2
        name, pipeline_argv = implied_name, argv
    else:
        name, pipeline_argv = argv[0], argv[1:]
    try:
        pipeline = get_pipeline(name)
    except KeyError as exc:
        print(exc, file=sys.stderr)
        return 2
    parser = _run_parser(pipeline)
    args = parser.parse_args(pipeline_argv)
    try:
        env_file = args.env_file or Path(".env")
        if args.env_file is not None and not env_file.is_file():
            raise ValueError(f"env file not found: {env_file}")
        config = {
            **(dotenv_values(env_file, interpolate=False, encoding="utf-8-sig") if env_file.is_file() else {}),
            **os.environ,
        }
        model = args.model or config.get("SYNTHESIS_MODEL")
        if not args.dry_run and not model:
            raise ValueError("Pass --model or set SYNTHESIS_MODEL.")
        if args.provider == "auto":
            base_url = args.base_url or config.get("OPENAI_BASE_URL") or PROVIDER_DEFAULTS["openai"][0]
            provider = resolve_provider("auto", base_url)
            key_env = PROVIDER_DEFAULTS[provider][1]
            api_key = args.api_key or config.get(key_env) or config.get("OPENAI_API_KEY")
        else:
            provider = args.provider
            default_url, key_env = PROVIDER_DEFAULTS[provider]
            base_url = args.base_url or default_url
            api_key = args.api_key or config.get(key_env)
        if not args.dry_run and not api_key and urlsplit(base_url).hostname in {
            "api.openai.com", "api.deepseek.com", "openrouter.ai", "eu.openrouter.ai",
        }:
            print(f"Pass --api-key or set {key_env}.", file=sys.stderr)
            return 2
        client = ChatClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_backoff=args.retry_backoff,
            max_tokens=args.max_tokens,
            max_token_field=args.max_token_field,
            thinking={"default": None, "on": True, "off": False}[args.thinking],
            provider=provider,
        )
    except (OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2

    async def _execute():
        try:
            return await run_synthesis(
                pipeline,
                args,
                client,
                args.output,
                concurrency=args.concurrency,
                batch_size=args.batch_size,
                limit=args.limit,
                retry_errors=args.retry_errors,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
        finally:
            await client.aclose()

    try:
        stats = asyncio.run(_execute())
    except KeyboardInterrupt:
        _log("Interrupted. Completed jobs are saved. Run the same command to resume.")
        return 130
    except (OSError, ValueError, KeyError) as exc:
        print(exc, file=sys.stderr)
        return 2
    if args.dry_run:
        return 0
    _log(
        f"done: wrote {stats.wrote}, errors {stats.errors}, pending {stats.pending}, "
        f"skipped {stats.skipped} (errors {stats.skipped_errors}), train rows {stats.train_rows}"
    )
    _log(f"output: {args.output}")
    return 1 if stats.pending or stats.errors or stats.skipped_errors else 0


def _run_parser(pipeline) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"python -m synthesis run {pipeline.name}",
        description=pipeline.description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", required=True, type=Path, help="Directory for JSONL checkpoints.")
    parser.add_argument("--model", default=None, help="Model name. Defaults to SYNTHESIS_MODEL.")
    parser.add_argument(
        "--env-file", type=Path, default=None,
        help="Read key/model settings from this file, or .env in the current directory if present. Environment wins.",
    )
    parser.add_argument(
        "--provider", choices=("auto", *PROVIDER_DEFAULTS), default="auto",
        help="Endpoint and key preset; auto uses --base-url or OPENAI_BASE_URL and detects known hosts.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override the provider's API root. In auto mode, defaults to OPENAI_BASE_URL or OpenAI.",
    )
    parser.add_argument("--api-key", default=None, help="Bearer token; otherwise use the selected provider's key environment variable.")
    parser.add_argument("--concurrency", type=positive_int, default=8, help="Maximum in-flight requests.")
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=32,
        help="Pending jobs scheduled together. Use a value >= concurrency.",
    )
    parser.add_argument("--limit", type=positive_int, default=None, help="Run at most this many pending jobs.")
    parser.add_argument(
        "--max-tokens", type=optional_positive_int, default=DEFAULT_MAX_TOKENS, metavar="N|default",
        help="Output token budget for the whole response; 'default' omits the API field.",
    )
    parser.add_argument(
        "--max-token-field", choices=TOKEN_LIMIT_FIELDS, default="max_tokens",
        help="API field used for the output token budget.",
    )
    parser.add_argument(
        "--thinking", choices=("default", "on", "off"), default="default",
        help="DeepSeek/OpenRouter thinking toggle; 'default' omits the API field.",
    )
    parser.add_argument("--timeout", type=positive_float, default=120, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--max-retries",
        type=nonnegative_int,
        default=5,
        help="Retries after the first attempt for rate limits and server errors.",
    )
    parser.add_argument("--retry-backoff", type=nonnegative_float, default=1.0, help="Initial retry delay in seconds.")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Run jobs recorded in errors.jsonl again. Successful jobs are still skipped.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete this output directory's checkpoints first.")
    parser.add_argument("--dry-run", action="store_true", help="Print the first pending request and call no API.")
    pipeline.add_arguments(parser)
    return parser
