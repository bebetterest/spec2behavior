import json

import httpx
import pytest

from synthesis.client import ChatClient, build_chat_body
from synthesis.types import ChatRequest, FatalAPIError, ParseError, RetryableError


def test_api_trace_keeps_each_attempt_and_raw_response_without_credentials():
    import asyncio

    events = []
    sent = []
    secret = "private-api-key"
    seed = "0123456789abcdef0123456789abcdef"
    response_body = '{ "choices": [{"message": {"content": "done", "reasoning_content": "notes"}}], "usage": {"total_tokens": 7}}\n'

    def handler(request):
        sent.append(json.loads(request.content))
        if len(sent) == 1:
            return httpx.Response(503, text=f"upstream unavailable: {secret}")
        return httpx.Response(200, text=response_body, headers={"Authorization": f"Bearer {secret}"})

    async def scenario():
        client = ChatClient(
            base_url="https://username:password@example.com/v1", api_key=secret, model="demo",
            transport=httpx.MockTransport(handler), trust_env=False, max_retries=1, retry_backoff=0,
        )
        try:
            result = await client.complete(
                ChatRequest(
                    messages=[{"role": "user", "content": f"test echo: {secret}"}],
                    generation_seed=seed,
                ),
                trace=events.append,
            )
        finally:
            await client.aclose()
        assert result == "done"

    asyncio.run(scenario())
    assert [event["event"] for event in events] == ["request", "response", "error", "request", "response"]
    assert [event["attempt"] for event in events] == [1, 1, 1, 2, 2]
    assert len({event["call_id"] for event in events}) == 1
    assert {event["generation_seed"] for event in events} == {seed}
    assert sent[0] == sent[1]
    assert "seed" not in sent[0]
    assert "generation_seed" not in sent[0]
    assert events[-1]["body"] == response_body
    assert json.loads(events[-1]["body"])["usage"]["total_tokens"] == 7
    assert events[0]["body"]["messages"][0]["content"] == "test echo: [REDACTED]"
    assert sent[0]["messages"][0]["content"] == f"test echo: {secret}"
    serialized = json.dumps(events)
    for credential in (secret, "Authorization", "username", "password"):
        assert credential not in serialized
    assert all(event["url"] == "https://example.com/v1/chat/completions" for event in events)


@pytest.mark.parametrize("failure", ["status", "invalid_json", "length", "network"])
def test_api_trace_retains_failures_before_response_validation(failure):
    import asyncio

    events = []
    response_body = 'not JSON\n' if failure == "invalid_json" else json.dumps({"choices": [{
        "message": {"content": "partial"}, "finish_reason": "length" if failure == "length" else "stop",
    }]})

    def handler(request):
        if failure == "network":
            raise httpx.ConnectError("offline", request=request)
        return httpx.Response(401 if failure == "status" else 200, text=response_body)

    async def scenario():
        client = ChatClient(
            base_url="https://example.com/v1", api_key=None, model="demo", max_retries=0,
            transport=httpx.MockTransport(handler), trust_env=False,
        )
        try:
            with pytest.raises((FatalAPIError, RetryableError, ParseError)):
                await client.complete(ChatRequest(messages=[{"role": "user", "content": "Hi"}]), trace=events.append)
        finally:
            await client.aclose()

    asyncio.run(scenario())
    assert events[0]["event"] == "request"
    assert events[-1]["event"] == "error"
    if failure == "network":
        assert len(events) == 2
    else:
        assert events[1]["event"] == "response"
        assert events[1]["body"] == response_body


def test_api_trace_marks_a_cancelled_request_without_fabricating_a_response():
    import asyncio

    events = []

    async def handler(request):
        raise asyncio.CancelledError

    async def scenario():
        client = ChatClient(
            base_url="https://example.com/v1", api_key=None, model="demo",
            transport=httpx.MockTransport(handler), trust_env=False,
        )
        try:
            with pytest.raises(asyncio.CancelledError):
                await client.complete(ChatRequest(messages=[{"role": "user", "content": "Hi"}]), trace=events.append)
        finally:
            await client.aclose()

    asyncio.run(scenario())
    assert [event["event"] for event in events] == ["request", "cancelled"]


def test_http_error_does_not_expose_an_echoed_api_key_to_checkpoint_logs():
    import asyncio

    secret = "private-api-key"

    async def scenario():
        client = ChatClient(
            base_url="https://example.com/v1", api_key=secret, model="demo",
            transport=httpx.MockTransport(lambda request: httpx.Response(401, text=f"Invalid key: {secret}")),
            trust_env=False,
        )
        try:
            with pytest.raises(FatalAPIError) as caught:
                await client.complete(ChatRequest(messages=[{"role": "user", "content": "Hi"}]))
            assert secret not in str(caught.value)
            assert "[REDACTED]" in str(caught.value)
        finally:
            await client.aclose()

    asyncio.run(scenario())


def test_build_chat_body_omits_unset_fields_and_applies_extra_last():
    body = build_chat_body(
        ChatRequest(
            messages=[{"role": "user", "content": "Hi"}],
            extra={"response_format": {"type": "json_object"}},
        ),
        model="demo",
        max_tokens=None,
    )
    assert body == {
        "messages": [{"role": "user", "content": "Hi"}],
        "model": "demo",
        "response_format": {"type": "json_object"},
    }
    assert "temperature" not in body


def test_token_budget_field_and_template_override_do_not_send_both_fields():
    request = ChatRequest(messages=[{"role": "user", "content": "Hi"}], max_tokens=512)
    body = build_chat_body(request, model="demo", max_tokens=8192, max_token_field="max_completion_tokens")
    assert body["max_completion_tokens"] == 512
    assert "max_tokens" not in body
    request.extra = {"max_completion_tokens": 1024}
    body = build_chat_body(request, model="demo", max_tokens=8192)
    assert body["max_completion_tokens"] == 1024
    assert "max_tokens" not in body
    request.extra["max_tokens"] = 256
    with pytest.raises(ValueError, match="only one"):
        build_chat_body(request, model="demo", max_tokens=8192)


def test_template_thinking_overrides_client_default_without_mutating_the_request():
    request = ChatRequest(
        messages=[{"role": "user", "content": "Hi"}],
        extra={"thinking": {"type": "disabled"}, "response_format": {"type": "json_object"}},
    )
    body = build_chat_body(request, model="demo", max_tokens=None, thinking=True)
    assert body["thinking"] == {"type": "disabled"}
    assert body["response_format"] == {"type": "json_object"}
    assert request.extra == {"thinking": {"type": "disabled"}, "response_format": {"type": "json_object"}}


def test_template_reasoning_overrides_openrouter_toggle_without_changing_the_template():
    extra = {"reasoning": {"enabled": False, "exclude": True}, "response_format": {"type": "json_object"}}
    request = ChatRequest(messages=[{"role": "user", "content": "Hi"}], extra=extra)
    body = build_chat_body(request, model="demo", max_tokens=None, thinking=True, provider="openrouter")
    assert body["reasoning"] == {"enabled": False, "exclude": True}
    assert "thinking" not in body
    assert "extra_body" not in body
    assert request.extra == {
        "reasoning": {"enabled": False, "exclude": True}, "response_format": {"type": "json_object"},
    }


def test_mixed_thinking_formats_are_rejected():
    request = ChatRequest(messages=[], extra={"thinking": {"type": "disabled"}})
    with pytest.raises(ValueError, match="only one of thinking and reasoning"):
        build_chat_body(request, model="demo", max_tokens=None, thinking=True, provider="openrouter")


def test_openai_thinking_toggle_is_rejected_before_opening_a_client():
    with pytest.raises(ValueError, match="requires --provider"):
        ChatClient(base_url="https://api.openai.com/v1", api_key=None, model="demo", thinking=True)


@pytest.mark.parametrize("extra", [True, "enabled", {}, {"type": "on"}])
def test_invalid_template_thinking_is_rejected(extra):
    with pytest.raises(ValueError, match="thinking must be an object"):
        build_chat_body(ChatRequest(messages=[], extra={"thinking": extra}), model="demo", max_tokens=None)


def test_reasoning_without_a_final_answer_is_rejected():
    async def scenario():
        client = ChatClient(
            base_url="https://api.deepseek.com", api_key=None, model="demo", thinking=True,
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"choices": [{
                "message": {"content": None, "reasoning_content": '{"reflection":"not a final answer"}'},
                "finish_reason": "stop",
            }]})),
            trust_env=False,
        )
        try:
            with pytest.raises(FatalAPIError, match="empty completion"):
                await client.complete(ChatRequest(messages=[{"role": "user", "content": "Hi"}]))
        finally:
            await client.aclose()

    import asyncio
    asyncio.run(scenario())


@pytest.mark.parametrize("finish_reason", ["length", "content_filter", "tool_calls"])
def test_unfinished_or_filtered_completions_are_not_accepted_or_retried(finish_reason):
    calls = []
    raw = '{"reflection":"This looks like valid JSON."}'

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"choices": [{
            "message": {"content": raw}, "finish_reason": finish_reason,
        }]})

    async def scenario():
        client = ChatClient(
            base_url="https://example.com/v1", api_key=None, model="demo",
            max_retries=3, transport=httpx.MockTransport(handler), trust_env=False,
        )
        try:
            with pytest.raises(ParseError, match="finish_reason") as caught:
                await client.complete(ChatRequest(messages=[{"role": "user", "content": "Hi"}]))
            assert caught.value.raw == raw
        finally:
            await client.aclose()

    import asyncio
    asyncio.run(scenario())
    assert len(calls) == 1


def test_completions_url_accepts_a_root_or_a_full_path():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    async def run(base_url: str) -> str:
        client = ChatClient(
            base_url=base_url,
            api_key="test",
            model="demo",
            max_retries=0,
            transport=httpx.MockTransport(handler),
            trust_env=False,
        )
        try:
            return await client.complete(ChatRequest(messages=[{"role": "user", "content": "Hi"}]))
        finally:
            await client.aclose()

    assert pytest.importorskip("asyncio").run(run("https://example.com/v1")) == "ok"
    assert pytest.importorskip("asyncio").run(run("https://example.com/v1/chat/completions")) == "ok"
    assert calls["n"] == 2


def test_retry_then_success_and_fatal_status():
    state = {"n": 0}

    def retry_handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] < 3:
            return httpx.Response(429, text="slow down")
        body = json.loads(request.content.decode())
        assert body["model"] == "demo"
        return httpx.Response(200, json={"choices": [{"message": {"content": "done"}}]})

    def fatal_handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(401, text="nope")

    async def scenario() -> None:
        client = ChatClient(
            base_url="https://example.com/v1",
            api_key=None,
            model="demo",
            max_retries=2,
            retry_backoff=0,
            transport=httpx.MockTransport(retry_handler),
            trust_env=False,
        )
        try:
            text = await client.complete(ChatRequest(messages=[{"role": "user", "content": "Hi"}]))
        finally:
            await client.aclose()
        assert text == "done"
        assert state["n"] == 3

        fatal = ChatClient(
            base_url="https://example.com/v1",
            api_key=None,
            model="demo",
            max_retries=5,
            retry_backoff=0,
            transport=httpx.MockTransport(fatal_handler),
            trust_env=False,
        )
        try:
            with pytest.raises(FatalAPIError):
                await fatal.complete(ChatRequest(messages=[{"role": "user", "content": "Hi"}]))
        finally:
            await fatal.aclose()

        exhausted = ChatClient(
            base_url="https://example.com/v1",
            api_key=None,
            model="demo",
            max_retries=1,
            retry_backoff=0,
            transport=httpx.MockTransport(lambda request: httpx.Response(503, text="down")),
            trust_env=False,
        )
        try:
            with pytest.raises(RetryableError):
                await exhausted.complete(ChatRequest(messages=[{"role": "user", "content": "Hi"}]))
        finally:
            await exhausted.aclose()

    import asyncio

    asyncio.run(scenario())


def test_list_content_parts_are_joined():
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": [
                {"type": "reasoning", "text": "Hidden teacher reasoning."},
                {"type": "thinking", "text": "Also hidden."},
                {"type": "text", "text": "a"},
                "b",
                {"type": "output_text", "text": "c"},
            ]}}]},
        )

    async def scenario() -> str:
        client = ChatClient(
            base_url="https://example.com/v1",
            api_key=None,
            model="demo",
            max_retries=0,
            transport=httpx.MockTransport(handler),
            trust_env=False,
        )
        try:
            return await client.complete(ChatRequest(messages=[{"role": "user", "content": "Hi"}]))
        finally:
            await client.aclose()

    import asyncio

    assert asyncio.run(scenario()) == "abc"


@pytest.mark.parametrize("field,value", [
    ("tool_calls", [{"id": "call-1", "function": {"name": "tool", "arguments": "{}"}}]),
    ("function_call", {"name": "tool", "arguments": "{}"}),
])
def test_tool_call_is_rejected_even_when_finish_reason_is_stop(field, value):
    import asyncio

    async def scenario():
        client = ChatClient(
            base_url="https://example.com/v1", api_key=None, model="demo",
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"choices": [{
                "message": {"content": '{"reflection":"Looks complete."}', field: value},
                "finish_reason": "stop",
            }]})), trust_env=False,
        )
        try:
            with pytest.raises(ParseError, match="tool call"):
                await client.complete(ChatRequest(messages=[{"role": "user", "content": "Hi"}]))
        finally:
            await client.aclose()

    asyncio.run(scenario())
