"""Minimal client for OpenAI-compatible chat completion APIs (OpenAI, Ollama, vLLM, LM Studio, ...).

Only the standard library is used. The API key is passed as a bearer token, never echoed, and is
refused over plain http to non-local hosts. Redirects are not followed.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from . import __version__
from .client import _LOCAL_HOSTS, _NoRedirect, make_ssl_context, tls_hint
from .models import clean_text

MAX_BODY_BYTES = 5_000_000
_RETRY_STATUS = {429, 500, 502, 503, 504}
_SAFE_NAME = re.compile(r"[^a-zA-Z0-9_-]")


class LlmError(Exception):
    pass


@dataclass
class ToolCall:
    id: str
    name: str                      # name as the model wrote it (an alias, see ToolSet)
    arguments: dict[str, Any] | None  # None when the model sent something that is not a JSON object
    raw_arguments: str = ""
    error: str = ""


@dataclass
class Reply:
    content: str
    tool_calls: list[ToolCall]
    message: dict[str, Any]        # the assistant message, to be replayed in a follow-up turn
    latency: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class ToolSet:
    """MCP tools in OpenAI function format, with a mapping between safe aliases and real names."""

    functions: list[dict[str, Any]] = field(default_factory=list)
    alias_to_name: dict[str, str] = field(default_factory=dict)
    name_to_alias: dict[str, str] = field(default_factory=dict)
    schemas: dict[str, dict[str, Any]] = field(default_factory=dict)  # real name -> inputSchema

    def real_name(self, alias: str) -> str:
        return self.alias_to_name.get(alias, alias)


def build_toolset(tools: list[dict[str, Any]]) -> ToolSet:
    toolset = ToolSet()
    used: set[str] = set()
    for tool in tools:
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            continue
        alias = _SAFE_NAME.sub("_", name)[:64] or "tool"
        suffix = 1
        while alias in used:
            suffix += 1
            alias = f"{alias[:60]}_{suffix}"
        used.add(alias)
        schema = tool.get("inputSchema")
        if not isinstance(schema, dict):
            schema = {"type": "object", "properties": {}}
        description = tool.get("description")
        toolset.functions.append({
            "type": "function",
            "function": {
                "name": alias,
                "description": description if isinstance(description, str) else "",
                "parameters": schema,
            },
        })
        toolset.alias_to_name[alias] = name
        toolset.name_to_alias[name] = alias
        toolset.schemas[name] = schema
    return toolset


def _parse_tool_call(raw: Any, position: int) -> ToolCall:
    function = raw.get("function") if isinstance(raw, dict) else None
    if not isinstance(function, dict):
        return ToolCall(f"call_{position}", "", None, error="malformed tool call")
    name = function.get("name") if isinstance(function.get("name"), str) else ""
    arguments = function.get("arguments", "")
    call_id = raw.get("id") if isinstance(raw.get("id"), str) and raw.get("id") else f"call_{position}"
    if isinstance(arguments, dict):  # some local servers return an object instead of a JSON string
        return ToolCall(call_id, name, arguments, json.dumps(arguments))
    text = arguments if isinstance(arguments, str) else ""
    if not text.strip():
        return ToolCall(call_id, name, {}, text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return ToolCall(call_id, name, None, text, error=f"arguments are not valid JSON ({exc.msg})")
    if not isinstance(parsed, dict):
        return ToolCall(call_id, name, None, text, error="arguments are not a JSON object")
    return ToolCall(call_id, name, parsed, text)


class ChatClient:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
        retries: int = 2,
        allow_insecure: bool = False,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise LlmError("--base-url must start with http:// or https://")
        if parsed.scheme == "http" and api_key and parsed.hostname not in _LOCAL_HOSTS and not allow_insecure:
            raise LlmError("refusing to send an API key over plain http; use https or pass --allow-insecure")
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self._timeout = timeout
        self._retries = max(0, retries)
        self._sleep = sleep
        self._opener = urllib.request.build_opener(_NoRedirect, urllib.request.HTTPSHandler(context=make_ssl_context()))

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json", "User-Agent": f"mcp-audit/{__version__}"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        data = json.dumps(payload).encode("utf-8")
        for attempt in range(self._retries + 1):
            request = urllib.request.Request(self._url, data=data, headers=headers, method="POST")
            try:
                with self._opener.open(request, timeout=self._timeout) as response:
                    body = response.read(MAX_BODY_BYTES + 1)
            except urllib.error.HTTPError as exc:
                detail = clean_text(exc.read(300).decode("utf-8", "replace")) if exc.fp else ""
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                status = exc.code
                exc.close()
                if status in _RETRY_STATUS and attempt < self._retries:
                    try:
                        delay = min(float(retry_after), 10.0) if retry_after else 2.0**attempt
                    except ValueError:
                        delay = 2.0**attempt
                    self._sleep(delay)
                    continue
                raise LlmError(f"HTTP {status} from model API: {detail}".strip()) from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt < self._retries:
                    self._sleep(2.0**attempt)
                    continue
                raise LlmError(f"cannot reach model API: {exc}{tls_hint(exc)}") from exc
            except ValueError as exc:
                raise LlmError(f"invalid request: {exc}") from exc
            if len(body) > MAX_BODY_BYTES:
                raise LlmError("model API response is too large")
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError as exc:
                raise LlmError("model API did not return JSON") from exc
            if not isinstance(parsed, dict):
                raise LlmError("model API returned an unexpected response")
            return parsed
        raise LlmError("model API request failed")  # pragma: no cover

    def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float = 0.0,
    ) -> Reply:
        payload: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        started = time.monotonic()
        data = self._post(payload)
        latency = time.monotonic() - started
        choices = data.get("choices")
        message = choices[0].get("message") if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            error = data.get("error")
            detail = error.get("message") if isinstance(error, dict) else None
            raise LlmError(f"model API returned no message{': ' + clean_text(str(detail)) if detail else ''}")
        raw_calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        content = message.get("content")
        return Reply(
            content=content if isinstance(content, str) else "",
            tool_calls=[_parse_tool_call(c, i) for i, c in enumerate(raw_calls)],
            message=message,
            latency=latency,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )
