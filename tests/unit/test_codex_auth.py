import json
from pathlib import Path

import pytest

from agent.core.codex_auth import (
    CODEX_CLIENT_ID,
    default_codex_auth_file,
    load_codex_subscription_auth,
    refresh_codex_subscription_auth,
)


def test_load_codex_subscription_auth_reads_tokens(tmp_path: Path):
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "access-token",
                    "refresh_token": "refresh-token",
                    "account_id": "account-1",
                    "id_token": {"chatgpt_plan_type": "plus"},
                },
                "last_refresh": "2099-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    auth = load_codex_subscription_auth(auth_file, refresh=False)

    assert auth is not None
    assert auth.access_token == "access-token"
    assert auth.account_id == "account-1"
    assert auth.plan_type == "plus"
    assert auth.auth_file == auth_file


def test_load_codex_subscription_auth_ignores_api_key_only_auth(tmp_path: Path):
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({"OPENAI_API_KEY": "sk-test"}), encoding="utf-8")

    assert load_codex_subscription_auth(auth_file, refresh=False) is None


def test_default_codex_auth_file_honors_overrides(monkeypatch, tmp_path: Path):
    explicit = tmp_path / "custom-auth.json"
    monkeypatch.setenv("ML_INTERN_CODEX_AUTH_FILE", str(explicit))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))

    assert default_codex_auth_file() == explicit

    monkeypatch.delenv("ML_INTERN_CODEX_AUTH_FILE")
    assert default_codex_auth_file() == tmp_path / "codex-home" / "auth.json"


def test_refresh_codex_subscription_auth_persists_new_tokens(monkeypatch, tmp_path: Path):
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(
        json.dumps({"tokens": {"access_token": "old", "refresh_token": "refresh"}}),
        encoding="utf-8",
    )
    calls = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"access_token": "new-access", "refresh_token": "new-refresh"}

    def fake_post(url, *, json, headers, timeout):
        calls.append((url, json, headers, timeout))
        return Response()

    monkeypatch.setenv("ML_INTERN_CODEX_REFRESH_TOKEN_URL", "https://example.test/token")
    monkeypatch.setattr("agent.core.codex_auth.requests.post", fake_post)

    refreshed = refresh_codex_subscription_auth(auth_file, "refresh")

    assert calls == [
        (
            "https://example.test/token",
            {
                "client_id": CODEX_CLIENT_ID,
                "grant_type": "refresh_token",
                "refresh_token": "refresh",
            },
            {"Content-Type": "application/json"},
            30,
        )
    ]
    assert refreshed["tokens"]["access_token"] == "new-access"
    assert refreshed["tokens"]["refresh_token"] == "new-refresh"
    saved = json.loads(auth_file.read_text(encoding="utf-8"))
    assert saved["tokens"]["access_token"] == "new-access"
