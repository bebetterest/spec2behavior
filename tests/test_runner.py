import argparse
import asyncio
import json
from pathlib import Path

import pytest
import httpx

from synthesis.client import ChatClient
from synthesis.runner import run_synthesis
from synthesis.types import ChatRequest, Example, Job, ParseError, Produced, RetryableError
from synthesis.pipeline import Pipeline


class EchoPipeline(Pipeline):
    name = "echo"
    description = "Test pipeline."

    def jobs(self, args: argparse.Namespace):
        for index in range(args.n):
            yield Job(id=f"job-{index}", input={})

    def request(self, job: Job, args: argparse.Namespace) -> ChatRequest:
        del args
        return ChatRequest(messages=[{"role": "user", "content": job.id}])

    def parse(self, job: Job, content: str, args: argparse.Namespace) -> list[Example]:
        del args
        if content.startswith("BAD"):
            raise ParseError("unusable", raw=content)
        return [
            Example(
                messages=[
                    {"role": "user", "content": job.id},
                    {"role": "assistant", "content": content},
                ]
            )
        ]


class FakeClient:
    def __init__(self, replies: dict[str, str | Exception] | None = None) -> None:
        self.model = "fake-model"
        self.max_tokens = None
        self.replies = replies or {}
        self.calls: list[str] = []
        self.inflight = 0
        self.max_inflight = 0

    async def complete(self, request: ChatRequest) -> str:
        job_id = request.messages[-1]["content"]
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        self.calls.append(job_id)
        await asyncio.sleep(0.02)
        self.inflight -= 1
        reply = self.replies.get(job_id, f"answer {job_id}")
        if isinstance(reply, Exception):
            raise reply
        return reply


def _args(n: int) -> argparse.Namespace:
    return argparse.Namespace(n=n)


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_batch_concurrency_resume_and_errors(tmp_path: Path):
    client = FakeClient(
        {
            "job-1": ParseError("unusable"),
            "job-2": RetryableError("HTTP 503: down"),
        }
    )
    output = tmp_path / "run"

    async def first() -> None:
        stats = await run_synthesis(
            EchoPipeline(),
            _args(4),
            client,
            output,
            concurrency=2,
            batch_size=2,
        )
        assert stats.wrote == 2
        assert stats.errors == 1
        assert stats.pending == 1
        assert stats.selected == 4

    asyncio.run(first())
    assert client.max_inflight == 2
    assert len(client.calls) == 4
    jobs = _read(output / "jobs.jsonl")
    assert {row["id"] for row in jobs} == {"job-0", "job-3"}
    train = _read(output / "train.jsonl")
    assert [row["id"] for row in train] == ["echo/job-0#0", "echo/job-3#0"]
    assert train[0]["messages"][1]["content"] == "answer job-0"
    assert train[0]["meta"]["model"] == "fake-model"
    assert _read(output / "errors.jsonl")[0]["id"] == "job-1"
    assert _read(output / "failures.jsonl")[0]["id"] == "job-2"

    async def resume() -> None:
        before = len(client.calls)
        stats = await run_synthesis(
            EchoPipeline(),
            _args(4),
            client,
            output,
            concurrency=2,
            batch_size=2,
        )
        assert stats.skipped == 3
        assert stats.selected == 1
        assert stats.pending == 1
        assert client.calls[before:] == ["job-2"]

    asyncio.run(resume())

    async def retry_errors() -> None:
        client.replies["job-1"] = "fixed"
        client.replies["job-2"] = "fixed too"
        stats = await run_synthesis(
            EchoPipeline(),
            _args(4),
            client,
            output,
            concurrency=2,
            batch_size=2,
            retry_errors=True,
        )
        assert stats.wrote == 2
        assert stats.errors == 0
        assert stats.pending == 0

    asyncio.run(retry_errors())
    assert {row["id"] for row in _read(output / "jobs.jsonl")} == {"job-0", "job-1", "job-2", "job-3"}
    assert _read(output / "errors.jsonl")[0]["id"] == "job-1"
    train = _read(output / "train.jsonl")
    assert len(train) == 4


def test_limited_error_retry_keeps_unselected_errors_and_original_responses(tmp_path):
    output = tmp_path / "run"
    client = FakeClient({"job-0": "BAD first", "job-1": "BAD second"})

    async def scenario():
        first = await run_synthesis(EchoPipeline(), _args(2), client, output)
        assert first.errors == 2
        history = (output / "errors.jsonl").read_bytes()
        client.replies = {"job-0": "fixed", "job-1": "fixed too"}
        retried = await run_synthesis(EchoPipeline(), _args(2), client, output, retry_errors=True, limit=1)
        assert retried.wrote == 1
        assert (output / "errors.jsonl").read_bytes() == history

        before = len(client.calls)
        skipped = await run_synthesis(EchoPipeline(), _args(2), client, output)
        assert skipped.skipped_errors == 1
        assert skipped.selected == 0
        assert len(client.calls) == before

        last = await run_synthesis(EchoPipeline(), _args(2), client, output, retry_errors=True)
        assert last.wrote == 1
        assert last.skipped == 1
        done = await run_synthesis(EchoPipeline(), _args(2), client, output)
        assert done.skipped_errors == 0
        assert done.selected == 0
        assert (output / "errors.jsonl").read_bytes() == history

    asyncio.run(scenario())


def test_limit_then_continue(tmp_path: Path):
    client = FakeClient()
    output = tmp_path / "run"

    async def scenario() -> None:
        first = await run_synthesis(
            EchoPipeline(),
            _args(3),
            client,
            output,
            concurrency=1,
            batch_size=1,
            limit=1,
        )
        assert first.wrote == 1
        second = await run_synthesis(
            EchoPipeline(),
            _args(3),
            client,
            output,
            concurrency=1,
            batch_size=8,
        )
        assert second.skipped == 1
        assert second.wrote == 2

    asyncio.run(scenario())
    assert len(_read(output / "train.jsonl")) == 3


def test_duplicate_job_ids_fail(tmp_path: Path):
    class Dupes(EchoPipeline):
        def jobs(self, args):
            del args
            yield Job(id="same", input={})
            yield Job(id="same", input={})

    async def scenario() -> None:
        with pytest.raises(ValueError, match="duplicate"):
            await run_synthesis(Dupes(), _args(0), FakeClient(), tmp_path / "run")

    asyncio.run(scenario())


def test_resume_ignores_a_torn_trailing_line(tmp_path: Path):
    output = tmp_path / "run"
    output.mkdir()
    torn = (
        json.dumps(
            {
                "id": "job-0",
                "pipeline": "echo",
                "examples": [
                    {
                        "id": "job-0#0",
                        "messages": [
                            {"role": "user", "content": "job-0"},
                            {"role": "assistant", "content": "kept"},
                        ],
                        "meta": {},
                    }
                ],
                "raw": "kept",
            }
        )
        + "\n"
        + '{"id": "job-1", "raw": "cut'
    )
    (output / "jobs.jsonl").write_bytes(torn.encode())
    client = FakeClient()

    async def scenario() -> None:
        stats = await run_synthesis(
            EchoPipeline(),
            _args(2),
            client,
            output,
            concurrency=1,
            batch_size=1,
        )
        assert stats.skipped == 1
        assert stats.wrote == 1
        assert client.calls == ["job-1"]

    asyncio.run(scenario())
    ids = [row["id"] for row in _read(output / "jobs.jsonl")]
    assert ids == ["job-0", "job-1"]


class TwoStepPipeline(Pipeline):
    name = "two_step"
    description = "Draft, then revise."

    def jobs(self, args):
        del args
        yield Job(id="only", input={"text": "Hello"})

    async def produce(self, job, args, client):
        del args
        draft, revised = await asyncio.gather(
            client.complete(ChatRequest(messages=[{"role": "user", "content": "draft"}])),
            client.complete(ChatRequest(messages=[{"role": "user", "content": "revise"}])),
        )
        return Produced(
            examples=[
                Example(
                    messages=[
                        {"role": "user", "content": job.input["text"], "trainable": False},
                        {"role": "assistant", "content": revised, "trainable": True},
                    ]
                )
            ],
            raw=[draft, revised],
        )


def test_multi_step_job_limits_calls_and_resumes(tmp_path: Path):
    client = FakeClient()
    output = tmp_path / "run"

    async def scenario() -> None:
        dry = await run_synthesis(
            TwoStepPipeline(),
            _args(0),
            client,
            output,
            concurrency=1,
            batch_size=1,
            dry_run=True,
        )
        assert dry.selected == 1
        assert client.calls == []
        assert not output.exists()
        stats = await run_synthesis(
            TwoStepPipeline(),
            _args(0),
            client,
            output,
            concurrency=1,
            batch_size=1,
        )
        assert stats.wrote == 1
        assert client.max_inflight == 1
        assert sorted(client.calls) == ["draft", "revise"]
        again = await run_synthesis(
            TwoStepPipeline(),
            _args(0),
            client,
            output,
            concurrency=1,
            batch_size=1,
        )
        assert again.selected == 0

    asyncio.run(scenario())
    job = _read(output / "jobs.jsonl")[0]
    assert job["raw"] == ["answer draft", "answer revise"]
    row = _read(output / "train.jsonl")[0]
    assert row["messages"][1]["trainable"] is True
    assert row["messages"][0]["trainable"] is False


@pytest.mark.parametrize("multi_step", [False, True])
def test_api_trace_associates_concurrent_and_multi_step_calls_with_jobs(tmp_path, multi_step):
    output = tmp_path / "run"
    pipeline = TwoStepPipeline() if multi_step else EchoPipeline()

    async def handler(request):
        text = json.loads(request.content)["messages"][-1]["content"]
        await asyncio.sleep(0.01)
        return httpx.Response(200, json={"choices": [{"message": {"content": f"answer {text}"}}]})

    async def scenario():
        client = ChatClient(
            base_url="https://example.com/v1", api_key=None, model="demo",
            transport=httpx.MockTransport(handler), trust_env=False,
        )
        try:
            stats = await run_synthesis(pipeline, _args(3), client, output, concurrency=3)
        finally:
            await client.aclose()
        assert stats.wrote == (1 if multi_step else 3)

    asyncio.run(scenario())
    events = _read(output / "api.jsonl")
    calls = {event["call_id"] for event in events}
    assert len(calls) == (2 if multi_step else 3)
    jobs = {job["id"] for job in _read(output / "jobs.jsonl")}
    assert {event["job_id"] for event in events} == jobs
    for call in calls:
        request, response = [event for event in events if event["call_id"] == call]
        assert request["event"] == "request" and response["event"] == "response"
        assert request["job_id"] == response["job_id"]
        assert request["pipeline"] == response["pipeline"] == pipeline.name
        text = request["body"]["messages"][-1]["content"]
        assert json.loads(response["body"])["choices"][0]["message"]["content"] == f"answer {text}"


def test_limit_must_be_positive(tmp_path: Path):
    async def scenario() -> None:
        with pytest.raises(ValueError, match="limit"):
            await run_synthesis(EchoPipeline(), _args(1), FakeClient(), tmp_path / "run", limit=0)

    asyncio.run(scenario())


def test_dry_run_writes_nothing(tmp_path: Path):
    client = FakeClient()
    output = tmp_path / "run"

    async def scenario() -> None:
        stats = await run_synthesis(
            EchoPipeline(),
            _args(2),
            client,
            output,
            dry_run=True,
        )
        assert stats.selected == 2
        assert client.calls == []
        assert not output.exists()

    asyncio.run(scenario())


def test_overwrite_preview_selects_all_jobs_without_touching_checkpoints(tmp_path):
    output = tmp_path / "run"
    client = FakeClient()

    async def scenario():
        await run_synthesis(EchoPipeline(), _args(1), client, output)
        (output / "errors.jsonl").write_text('{"id":"job-1","error":"old error"}\n', encoding="utf-8")
        (output / "failures.jsonl").write_text('{"id":"job-1","error":"old failure"}\n', encoding="utf-8")
        (output / "api.jsonl").write_text('{"event":"request","job_id":"old"}\n', encoding="utf-8")
        before = {path.name: path.read_bytes() for path in output.iterdir()}
        preview = await run_synthesis(EchoPipeline(), _args(2), client, output, overwrite=True, dry_run=True)
        assert preview.selected == 2
        assert preview.skipped == preview.skipped_errors == 0
        assert client.calls == ["job-0"]
        assert {path.name: path.read_bytes() for path in output.iterdir()} == before

        # An overwrite preview must also ignore corrupt old checkpoints,
        # because execution would delete them rather than try to resume.
        (output / "jobs.jsonl").write_bytes(b"not JSON\n")
        corrupt = {path.name: path.read_bytes() for path in output.iterdir()}
        preview = await run_synthesis(EchoPipeline(), _args(2), client, output, overwrite=True, dry_run=True)
        assert preview.selected == 2
        assert {path.name: path.read_bytes() for path in output.iterdir()} == corrupt
        executed = await run_synthesis(EchoPipeline(), _args(2), client, output, overwrite=True)
        assert executed.selected == preview.selected
        assert executed.wrote == 2
        assert not (output / "api.jsonl").exists()

    asyncio.run(scenario())


@pytest.mark.parametrize("tail", [
    b'{"id":"job-2", "raw":"cut',
    b'{"id":"job-2", "raw":"\xe4\xb8',
    b'{"id":"job-2"}',
])
def test_dry_run_ignores_unfinished_checkpoint_lines_without_repairing_them(tmp_path, tail):
    output = tmp_path / "run"
    client = FakeClient()

    async def scenario():
        await run_synthesis(EchoPipeline(), _args(1), client, output)
        with (output / "jobs.jsonl").open("ab") as handle:
            handle.write(tail)
        (output / "errors.jsonl").write_bytes(b'{"id":"job-1","error":"old error"}\n' + tail)
        before = {path.name: path.read_bytes() for path in output.iterdir()}
        preview = await run_synthesis(EchoPipeline(), _args(3), client, output, dry_run=True)
        assert preview.selected == 1
        assert preview.skipped == 2
        assert preview.skipped_errors == 1
        assert client.calls == ["job-0"]
        assert {path.name: path.read_bytes() for path in output.iterdir()} == before

        executed = await run_synthesis(EchoPipeline(), _args(3), client, output)
        assert executed.selected == preview.selected
        assert executed.wrote == 1
        assert client.calls == ["job-0", "job-2"]
        assert (output / "errors.jsonl").read_bytes() == b'{"id":"job-1","error":"old error"}\n'

    asyncio.run(scenario())
