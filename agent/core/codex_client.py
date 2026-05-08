"""ChatGPT/Codex subscription transport for ML Intern.

This module intentionally does not send Codex OAuth tokens to the normal
``api.openai.com/v1`` API surface. ChatGPT subscription tokens lack the
``model.request`` API scope, which causes ``missing_scope`` errors there.

Instead it mirrors the Codex/pi transport: POST streamed Responses requests to
``https://chatgpt.com/backend-api/codex/responses`` with the ChatGPT account
headers Codex expects.
"""

from __future__ import annotations

import json
import os
import platform
import re
import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, AsyncIterator

import httpx

from agent.core.codex_auth import CodexSubscriptionAuth, load_codex_subscription_auth, refresh_codex_subscription_auth

DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api"
CODEX_BASE_URL_ENV_VAR = "ML_INTERN_CODEX_BASE_URL"
JWT_CLAIM_PATH = "https://api.openai.com/auth"


@dataclass
class CodexLLMResult:
    content: str | None
    tool_calls_acc: dict[int, dict]
    token_count: int
    finish_reason: str | None
    response: Any
    usage_response: Any | None = None


def is_codex_subscription_params(llm_params: dict[str, Any]) -> bool:
    return isinstance(llm_params.get("_ml_intern_codex_auth"), CodexSubscriptionAuth)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _safe_id_part(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", value or "")[:64].rstrip("_")
    return cleaned or f"call_{uuid.uuid4().hex[:8]}"


def _extract_account_id(auth: CodexSubscriptionAuth) -> str:
    if auth.account_id:
        return auth.account_id
    # Fallback for older auth files where account_id was not persisted.
    parts = auth.access_token.split(".")
    if len(parts) == 3:
        import base64

        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        try:
            claims = json.loads(base64.urlsafe_b64decode(payload.encode("ascii")))
            account_id = claims.get(JWT_CLAIM_PATH, {}).get("chatgpt_account_id")
            if isinstance(account_id, str) and account_id:
                return account_id
        except Exception:
            pass
    raise RuntimeError("Codex auth token does not contain a ChatGPT account id; rerun codex login.")


def _codex_url() -> str:
    raw = os.environ.get(CODEX_BASE_URL_ENV_VAR, DEFAULT_CODEX_BASE_URL).rstrip("/")
    if raw.endswith("/codex/responses"):
        return raw
    if raw.endswith("/codex"):
        return f"{raw}/responses"
    return f"{raw}/codex/responses"


def build_codex_headers(auth: CodexSubscriptionAuth, session_id: str | None = None) -> dict[str, str]:
    account_id = _extract_account_id(auth)
    request_id = session_id or str(uuid.uuid4())
    return {
        "Authorization": f"Bearer {auth.access_token}",
        "chatgpt-account-id": account_id,
        "originator": os.environ.get("ML_INTERN_CODEX_ORIGINATOR", "codex_cli_rs"),
        "User-Agent": f"ml-intern ({platform.system()} {platform.release()}; {platform.machine()})",
        "OpenAI-Beta": "responses=experimental",
        "accept": "text/event-stream",
        "content-type": "application/json",
        "session_id": request_id,
        "session-id": request_id,
        "thread_id": request_id,
        "thread-id": request_id,
        "x-client-request-id": request_id,
    }


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def _convert_messages(messages: list[Any]) -> tuple[str, list[dict[str, Any]]]:
    instructions: list[str] = []
    items: list[dict[str, Any]] = []
    msg_index = 0
    for msg in messages:
        role = _get(msg, "role")
        content = _get(msg, "content")
        if role == "system":
            text = _content_to_text(content)
            if text:
                instructions.append(text)
            continue
        if role == "user":
            text = _content_to_text(content)
            if text:
                items.append({"role": "user", "content": [{"type": "input_text", "text": text}]})
        elif role == "assistant":
            text = _content_to_text(content)
            if text:
                items.append({
                    "type": "message",
                    "role": "assistant",
                    "id": f"msg_{msg_index}",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                })
            for tc in _get(msg, "tool_calls", None) or []:
                tc_id = _safe_id_part(str(_get(tc, "id", "")))
                fn = _get(tc, "function", {})
                name = _get(fn, "name", "")
                args = _get(fn, "arguments", "{}")
                if not isinstance(args, str):
                    args = _json_dumps(args)
                items.append({
                    "type": "function_call",
                    "id": f"fc_{tc_id}",
                    "call_id": tc_id,
                    "name": name,
                    "arguments": args,
                })
        elif role == "tool":
            call_id = _safe_id_part(str(_get(msg, "tool_call_id", "")))
            items.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": _content_to_text(content),
            })
        msg_index += 1
    return "\n\n".join(instructions), items


def _convert_tools(tools: list[Any] | None) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in tools or []:
        if _get(tool, "type") == "function":
            fn = _get(tool, "function", {})
            converted.append({
                "type": "function",
                "name": _get(fn, "name"),
                "description": _get(fn, "description", ""),
                "parameters": _get(fn, "parameters", {"type": "object", "properties": {}}),
                "strict": False,
            })
        else:
            converted.append(tool if isinstance(tool, dict) else dict(tool))
    return converted


def build_codex_request_body(messages: list[Any], tools: list[Any] | None, llm_params: dict[str, Any], *, stream: bool) -> dict[str, Any]:
    model = str(llm_params.get("model", "openai/gpt-5-codex")).removeprefix("openai/")
    instructions, input_items = _convert_messages(messages)
    body: dict[str, Any] = {
        "model": model,
        "store": False,
        "stream": stream,
        "instructions": instructions,
        "input": input_items,
        "text": {"verbosity": os.environ.get("ML_INTERN_CODEX_TEXT_VERBOSITY", "low")},
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": llm_params.get("_ml_intern_session_id"),
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "client_metadata": {"originator": "ml-intern"},
    }
    if tools:
        body["tools"] = _convert_tools(tools)
    effort = llm_params.get("reasoning_effort")
    if effort:
        body["reasoning"] = {"effort": effort, "summary": "auto"}
    if llm_params.get("temperature") is not None:
        body["temperature"] = llm_params["temperature"]
    return body


async def _iter_sse_lines(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    buffer: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            data_lines = [part[5:].strip() for part in buffer if part.startswith("data:")]
            buffer = []
            if not data_lines:
                continue
            data = "\n".join(data_lines).strip()
            if not data or data == "[DONE]":
                continue
            yield json.loads(data)
        else:
            buffer.append(line)


def _finish_reason(status: str | None, has_tool_calls: bool) -> str | None:
    if has_tool_calls:
        return "tool_calls"
    if status == "incomplete":
        return "length"
    if status in {"failed", "cancelled"}:
        return "error"
    return "stop"


def codex_result_to_litellm_response(result: CodexLLMResult) -> Any:
    tool_calls = [
        SimpleNamespace(
            id=tc.get("id"),
            type=tc.get("type", "function"),
            function=SimpleNamespace(
                name=(tc.get("function") or {}).get("name", ""),
                arguments=(tc.get("function") or {}).get("arguments", "{}"),
            ),
        )
        for _, tc in sorted(result.tool_calls_acc.items())
    ]
    message = SimpleNamespace(content=result.content, tool_calls=tool_calls or None)
    choice = SimpleNamespace(message=message, finish_reason=result.finish_reason)
    return SimpleNamespace(choices=[choice], usage=getattr(result.response, "usage", None))


async def call_codex_litellm_response(
    *,
    session: Any,
    messages: list[Any],
    tools: list[Any] | None,
    llm_params: dict[str, Any],
    stream: bool = False,
    timeout: float | None = None,
) -> Any:
    result = await call_codex_responses(
        session=session,
        messages=messages,
        tools=tools,
        llm_params=llm_params,
        stream=stream,
    )
    return codex_result_to_litellm_response(result)


async def call_codex_responses(
    *,
    session: Any,
    messages: list[Any],
    tools: list[Any] | None,
    llm_params: dict[str, Any],
    stream: bool,
    on_text_delta: Any | None = None,
) -> CodexLLMResult:
    auth = llm_params.get("_ml_intern_codex_auth")
    if not isinstance(auth, CodexSubscriptionAuth):
        raise RuntimeError("Missing Codex subscription auth")

    session_id = llm_params.get("_ml_intern_session_id") or getattr(session, "session_id", None) or str(uuid.uuid4())
    body = build_codex_request_body(messages, tools, {**llm_params, "_ml_intern_session_id": session_id}, stream=True)
    headers = build_codex_headers(auth, session_id=session_id)

    full_content = ""
    tool_calls_acc: dict[int, dict] = {}
    current_tool_index: int | None = None
    usage: dict[str, int] = {}
    finish_reason: str | None = None

    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        for attempt in range(2):
            async with client.stream("POST", _codex_url(), headers=headers, json=body) as response:
                if response.status_code == 401 and attempt == 0 and auth.auth_file:
                    text = await response.aread()
                    if "token_expired" in text.decode("utf-8", "replace"):
                        try:
                            raw = load_codex_subscription_auth(auth.auth_file, refresh=False)
                            refresh_token = None
                            if raw is not None:
                                # Read the file directly so we can force a refresh even
                                # when Codex's last_refresh timestamp looks recent but
                                # the backend has already expired/revoked the access token.
                                with auth.auth_file.open("r", encoding="utf-8") as f:
                                    auth_json = json.load(f)
                                tokens = auth_json.get("tokens") if isinstance(auth_json, dict) else None
                                refresh_token = tokens.get("refresh_token") if isinstance(tokens, dict) else None
                            if isinstance(refresh_token, str) and refresh_token:
                                refresh_codex_subscription_auth(auth.auth_file, refresh_token)
                                refreshed = load_codex_subscription_auth(auth.auth_file, refresh=False)
                            else:
                                refreshed = None
                        except Exception:
                            refreshed = None
                        if refreshed and refreshed.access_token != auth.access_token:
                            auth = refreshed
                            headers = build_codex_headers(auth, session_id=session_id)
                            continue
                    raise RuntimeError(f"Codex API error 401: {text.decode('utf-8', 'replace')}")
                if response.status_code >= 400:
                    text = await response.aread()
                    raise RuntimeError(f"Codex API error {response.status_code}: {text.decode('utf-8', 'replace')}")
                async for event in _iter_sse_lines(response):
                    etype = event.get("type")
                    if etype == "response.output_text.delta":
                        delta = event.get("delta") or ""
                        full_content += delta
                        if stream and on_text_delta and delta:
                            await on_text_delta(delta)
                    elif etype == "response.output_item.added" and isinstance(event.get("item"), dict):
                        item = event["item"]
                        if item.get("type") == "function_call":
                            current_tool_index = len(tool_calls_acc)
                            tool_calls_acc[current_tool_index] = {
                                "id": item.get("call_id") or f"call_{current_tool_index}",
                                "type": "function",
                                "function": {"name": item.get("name") or "", "arguments": item.get("arguments") or ""},
                            }
                    elif etype == "response.function_call_arguments.delta" and current_tool_index is not None:
                        tool_calls_acc[current_tool_index]["function"]["arguments"] += event.get("delta") or ""
                    elif etype == "response.function_call_arguments.done":
                        idx = current_tool_index if current_tool_index is not None else len(tool_calls_acc)
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {"id": f"call_{idx}", "type": "function", "function": {"name": "", "arguments": ""}}
                        if isinstance(event.get("arguments"), str):
                            tool_calls_acc[idx]["function"]["arguments"] = event["arguments"]
                    elif etype == "response.output_item.done" and isinstance(event.get("item"), dict):
                        item = event["item"]
                        if item.get("type") == "message":
                            text = "".join(
                                part.get("text") or part.get("refusal") or ""
                                for part in item.get("content", [])
                                if isinstance(part, dict)
                            )
                            if text and not full_content:
                                full_content = text
                        elif item.get("type") == "function_call":
                            idx = current_tool_index if current_tool_index is not None else len(tool_calls_acc)
                            tool_calls_acc[idx] = {
                                "id": item.get("call_id") or f"call_{idx}",
                                "type": "function",
                                "function": {"name": item.get("name") or "", "arguments": item.get("arguments") or "{}"},
                            }
                    elif etype in {"response.completed", "response.done", "response.incomplete"}:
                        resp = event.get("response") or {}
                        resp_usage = resp.get("usage") or {}
                        cached = (resp_usage.get("input_tokens_details") or {}).get("cached_tokens") or 0
                        usage = {
                            "prompt_tokens": max(0, (resp_usage.get("input_tokens") or 0) - cached),
                            "completion_tokens": resp_usage.get("output_tokens") or 0,
                            "total_tokens": resp_usage.get("total_tokens") or 0,
                            "cache_read_tokens": cached,
                            "cache_creation_tokens": 0,
                        }
                        finish_reason = _finish_reason(resp.get("status"), bool(tool_calls_acc))
                    elif etype == "error":
                        raise RuntimeError(f"Codex error: {event.get('message') or event.get('code') or event}")
                    elif etype == "response.failed":
                        err = (event.get("response") or {}).get("error") or {}
                        raise RuntimeError(f"Codex response failed: {err.get('message') or err.get('code') or event}")

    usage_obj = SimpleNamespace(
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
        prompt_tokens_details={"cached_tokens": usage.get("cache_read_tokens", 0)},
    )
    response_obj = SimpleNamespace(usage=usage_obj)
    return CodexLLMResult(
        content=full_content or None,
        tool_calls_acc=tool_calls_acc,
        token_count=usage_obj.total_tokens,
        finish_reason=finish_reason,
        response=response_obj,
        usage_response=response_obj,
    )
