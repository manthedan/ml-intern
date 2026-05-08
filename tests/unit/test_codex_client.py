from types import SimpleNamespace

from agent.core.codex_auth import CodexSubscriptionAuth
from agent.core.codex_client import build_codex_headers, build_codex_request_body


def test_codex_headers_use_chatgpt_backend_auth_shape(monkeypatch):
    monkeypatch.setenv("ML_INTERN_CODEX_ORIGINATOR", "codex_cli_rs")
    auth = CodexSubscriptionAuth(access_token="access", account_id="acct")

    headers = build_codex_headers(auth, session_id="session-1")

    assert headers["Authorization"] == "Bearer access"
    assert headers["chatgpt-account-id"] == "acct"
    assert headers["originator"] == "codex_cli_rs"
    assert headers["OpenAI-Beta"] == "responses=experimental"
    assert headers["session_id"] == "session-1"
    assert headers["thread_id"] == "session-1"


def test_codex_request_body_uses_responses_items_and_flat_tools():
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "hello"},
        SimpleNamespace(
            role="assistant",
            content=None,
            tool_calls=[
                SimpleNamespace(
                    id="call_1",
                    function=SimpleNamespace(name="bash", arguments='{"command":"pwd"}'),
                )
            ],
        ),
        {"role": "tool", "tool_call_id": "call_1", "content": "/tmp"},
    ]
    tools = [{"type": "function", "function": {"name": "bash", "description": "run", "parameters": {"type": "object"}}}]

    body = build_codex_request_body(messages, tools, {"model": "openai/gpt-5.5", "reasoning_effort": "high"}, stream=True)

    assert body["model"] == "gpt-5.5"
    assert body["instructions"] == "system prompt"
    assert body["input"][0] == {"role": "user", "content": [{"type": "input_text", "text": "hello"}]}
    assert body["input"][1]["type"] == "function_call"
    assert body["input"][2] == {"type": "function_call_output", "call_id": "call_1", "output": "/tmp"}
    assert body["tools"] == [{"type": "function", "name": "bash", "description": "run", "parameters": {"type": "object"}, "strict": False}]
    assert body["reasoning"] == {"effort": "high", "summary": "auto"}
