"""OpenAI-compatible chat completions client."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from synthesis.types import ChatRequest, FatalAPIError, ParseError, RetryableError

# 408/409/429 are client status codes that are still worth retrying.
_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
TOKEN_LIMIT_FIELDS = ("max_tokens", "max_completion_tokens")
DEFAULT_MAX_TOKENS = 16384
DEFAULT_DEEPSEEK_MODEL = "deepseek-flash"
PROVIDER_DEFAULTS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}
TraceCallback = Callable[[dict[str, Any]], None]


def resolve_provider(provider: str, base_url: str) -> str:
    """Choose the request format, inferring known hosts in auto mode."""

    if provider == "auto":
        hostname = urlsplit(base_url).hostname
        if hostname in {"openrouter.ai", "eu.openrouter.ai"}:
            return "openrouter"
        if hostname == "api.deepseek.com":
            return "deepseek"
        return "openai"
    if provider not in PROVIDER_DEFAULTS:
        raise ValueError(f"unknown provider {provider!r}")
    return provider


def build_chat_body(
    request: ChatRequest,
    *,
    model: str | None,
    max_tokens: int | None,
    max_token_field: str = "max_tokens",
    thinking: bool | None = None,
    provider: str = "deepseek",
) -> dict[str, Any]:
    """Build the JSON body for ``/chat/completions``.

    Omitted optional fields stay out of the body. Keys in ``request.extra``
    are applied last. An explicit token field in ``extra`` replaces the
    configured token field so the request never sends both variants.
    ``thinking`` uses the selected provider's toggle; None keeps its default.
    """

    if max_token_field not in TOKEN_LIMIT_FIELDS:
        raise ValueError(f"max_token_field must be one of {TOKEN_LIMIT_FIELDS}")
    if thinking is not None and type(thinking) is not bool:
        raise ValueError("thinking must be a boolean or None")
    if provider not in PROVIDER_DEFAULTS:
        raise ValueError(f"unknown provider {provider!r}")
    if thinking is not None and provider == "openai":
        raise ValueError("--thinking on/off requires --provider deepseek or openrouter")
    body: dict[str, Any] = {"messages": request.messages}
    chosen_model = request.model or model
    if chosen_model:
        body["model"] = chosen_model
    if request.temperature is not None:
        body["temperature"] = request.temperature
    chosen_max_tokens = request.max_tokens if request.max_tokens is not None else max_tokens
    if chosen_max_tokens is not None and not any(key in request.extra for key in TOKEN_LIMIT_FIELDS):
        body[max_token_field] = chosen_max_tokens
    if thinking is not None:
        if provider == "openrouter":
            body["reasoning"] = {"enabled": thinking}
        else:
            body["thinking"] = {"type": "enabled" if thinking else "disabled"}
    body.update(request.extra)
    if "thinking" in body and "reasoning" in body:
        raise ValueError("Set only one of thinking and reasoning, using the selected provider's format")
    value = body.get("thinking")
    if value is not None and (
        not isinstance(value, dict) or value.get("type") not in ("enabled", "disabled")
    ):
        raise ValueError("thinking must be an object with type 'enabled' or 'disabled'")
    value = body.get("reasoning")
    if value is not None and (
        not isinstance(value, dict)
        or ("enabled" in value and type(value["enabled"]) is not bool)
    ):
        raise ValueError("reasoning must be an object with an optional boolean 'enabled'")
    if all(key in body for key in TOKEN_LIMIT_FIELDS):
        raise ValueError("Set only one of max_tokens and max_completion_tokens")
    for key in TOKEN_LIMIT_FIELDS:
        value = body.get(key)
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{key} must be a positive integer")
    value = body.get("temperature")
    if value is not None and (
        not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 2
    ):
        raise ValueError("temperature must be between 0 and 2")
    return body


class ChatClient:
    """Async client for one OpenAI-compatible ``/chat/completions`` endpoint.

    ``base_url`` may be either ``https://host/v1`` or the full completions URL.
    An empty API key omits the Authorization header for local servers.
    ``provider`` selects the thinking format; auto detects known hosts.
    Set it explicitly for a proxy with a custom URL.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str | None,
        timeout: float = 120,
        max_retries: int = 5,
        retry_backoff: float = 1.0,
        max_tokens: int | None = None,
        max_token_field: str = "max_tokens",
        thinking: bool | None = None,
        provider: str = "auto",
        transport: httpx.AsyncBaseTransport | None = None,
        trust_env: bool = True,
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and > 0")
        if type(max_retries) is not int or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if not math.isfinite(retry_backoff) or retry_backoff < 0:
            raise ValueError("retry_backoff must be finite and >= 0")
        if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 1):
            raise ValueError("max_tokens must be a positive integer or None")
        if max_token_field not in TOKEN_LIMIT_FIELDS:
            raise ValueError(f"max_token_field must be one of {TOKEN_LIMIT_FIELDS}")
        if thinking is not None and type(thinking) is not bool:
            raise ValueError("thinking must be a boolean or None")
        self.provider = resolve_provider(provider, base_url)
        if thinking is not None and self.provider == "openai":
            raise ValueError("--thinking on/off requires --provider deepseek or openrouter")
        self.model = model
        self.max_tokens = max_tokens
        self.max_token_field = max_token_field
        self.thinking = thinking
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._api_key = api_key
        self._url = _completions_url(base_url)
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers=headers,
            transport=transport,
            trust_env=trust_env,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def complete(self, request: ChatRequest, *, trace: TraceCallback | None = None) -> str:
        body = build_chat_body(
            request, model=self.model, max_tokens=self.max_tokens,
            max_token_field=self.max_token_field, thinking=self.thinking, provider=self.provider,
        )
        if "model" not in body:
            raise FatalAPIError("no model set; pass --model or set SYNTHESIS_MODEL")
        call_id = uuid4().hex

        def record(event: str, attempt: int, **fields: Any) -> None:
            if trace is not None:
                trace(_redact({
                    "event": event,
                    "call_id": call_id,
                    "attempt": attempt,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "provider": self.provider,
                    "url": _trace_url(self._url),
                    **({"generation_seed": request.generation_seed} if request.generation_seed else {}),
                    **fields,
                }, self._api_key))

        last_error: RetryableError | None = None
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            number = attempt + 1
            record("request", number, body=body)

            def received(response: httpx.Response) -> None:
                record("response", number, status_code=response.status_code, body=response.text)

            try:
                return await self._once(body, on_response=received if trace is not None else None)
            except asyncio.CancelledError:
                record("cancelled", number)
                raise
            except (FatalAPIError, ParseError) as exc:
                record("error", number, error_type=type(exc).__name__)
                raise
            except RetryableError as exc:
                record("error", number, error_type=type(exc).__name__)
                last_error = exc
                if attempt == attempts - 1:
                    break
                await asyncio.sleep(self.retry_backoff * (2**attempt))
        assert last_error is not None
        raise last_error

    async def _once(
        self, body: dict[str, Any], *, on_response: Callable[[httpx.Response], None] | None = None,
    ) -> str:
        try:
            response = await self._client.post(self._url, json=body)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise RetryableError(f"request failed: {_redact(str(exc), self._api_key)}") from exc
        if on_response is not None:
            on_response(response)
        if response.status_code in _RETRYABLE_STATUS or 500 <= response.status_code < 600:
            raise RetryableError(
                f"HTTP {response.status_code}: {_snippet(_redact(response.text, self._api_key))}"
            )
        if response.status_code >= 400:
            raise FatalAPIError(f"HTTP {response.status_code}: {_snippet(_redact(response.text, self._api_key))}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise RetryableError(f"response is not JSON: {exc}") from exc
        return _message_text(payload)


def _trace_url(url: str) -> str:
    parts = urlsplit(url)
    return parts._replace(netloc=parts.netloc.rsplit("@", 1)[-1], query="", fragment="").geturl()


def _redact(value: Any, secret: str | None) -> Any:
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, dict):
        return {_redact(key, secret): _redact(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    return value


def _completions_url(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if not base:
        raise ValueError("base_url is empty")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def _message_text(payload: Any) -> str:
    try:
        choice = payload["choices"][0]
        message = choice["message"]
        content = message.get("content")
        finish_reason = choice.get("finish_reason")
    except (AttributeError, KeyError, IndexError, TypeError) as exc:
        raise FatalAPIError(f"response has no message content: {exc}") from exc
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") in (None, "text", "output_text"):
                text = part.get("text") or part.get("content") or ""
                if isinstance(text, str):
                    parts.append(text)
        content = "".join(parts)
    raw = content if isinstance(content, str) else None
    if finish_reason == "length":
        raise ParseError(
            "completion reached its token limit (finish_reason=length); "
            "increase --max-tokens, then rerun with --retry-errors",
            raw=raw,
        )
    if finish_reason not in (None, "stop"):
        raise ParseError(f"completion ended with finish_reason={finish_reason!r}", raw=raw)
    if message.get("refusal"):
        raise ParseError("model refused the synthesis request", raw=str(message["refusal"]))
    if message.get("tool_calls") or message.get("function_call"):
        raise ParseError("completion requested a tool call instead of a final answer", raw=raw)
    if not isinstance(content, str) or not content.strip():
        raise FatalAPIError("model returned an empty completion")
    return content


def _snippet(text: str, limit: int = 500) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."
