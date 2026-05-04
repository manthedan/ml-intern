"""Read Codex CLI ChatGPT subscription auth for OpenAI-compatible calls.

Codex supports signing in with a ChatGPT subscription and stores the resulting
OAuth credentials under ``CODEX_HOME`` (default ``~/.codex``). This module gives
ML Intern a small, dependency-free bridge to reuse those credentials for direct
``openai/...`` LiteLLM calls.

This intentionally does not implement the login flow. Users should run
``codex login`` first, then ML Intern can read and refresh that auth.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

CODEX_HOME_ENV_VAR = "CODEX_HOME"
CODEX_AUTH_FILE_ENV_VAR = "ML_INTERN_CODEX_AUTH_FILE"
CODEX_SUBSCRIPTION_AUTH_ENV_VAR = "ML_INTERN_OPENAI_CODEX_AUTH"
CODEX_REFRESH_URL_ENV_VAR = "ML_INTERN_CODEX_REFRESH_TOKEN_URL"
DEFAULT_CODEX_REFRESH_URL = "https://auth.openai.com/oauth/token"
# Same public OAuth client id used by Codex CLI for token refresh.
CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_REFRESH_SKEW_SECONDS = 60


@dataclass(frozen=True)
class CodexSubscriptionAuth:
    """Bearer auth loaded from Codex's ChatGPT subscription credentials."""

    access_token: str
    account_id: str | None = None
    plan_type: str | None = None
    auth_file: Path | None = None


def codex_subscription_auth_enabled() -> bool:
    """Return whether OpenAI direct models should try Codex subscription auth."""
    value = os.environ.get(CODEX_SUBSCRIPTION_AUTH_ENV_VAR, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def default_codex_auth_file() -> Path:
    explicit = os.environ.get(CODEX_AUTH_FILE_ENV_VAR)
    if explicit:
        return Path(explicit).expanduser()
    codex_home = Path(os.environ.get(CODEX_HOME_ENV_VAR, "~/.codex")).expanduser()
    return codex_home / "auth.json"


def load_codex_subscription_auth(
    auth_file: str | os.PathLike[str] | None = None,
    *,
    refresh: bool = True,
) -> CodexSubscriptionAuth | None:
    """Load ChatGPT subscription bearer auth from Codex's ``auth.json``.

    Returns ``None`` when no usable Codex ChatGPT auth is present. API-key-only
    Codex auth is deliberately ignored because ML Intern already supports
    ``OPENAI_API_KEY`` directly through LiteLLM.
    """
    path = Path(auth_file).expanduser() if auth_file else default_codex_auth_file()
    try:
        raw = _load_auth_json(path)
    except (OSError, json.JSONDecodeError):
        return None

    tokens = raw.get("tokens") if isinstance(raw, dict) else None
    if not isinstance(tokens, dict):
        return None

    access_token = _nonempty_str(tokens.get("access_token"))
    refresh_token = _nonempty_str(tokens.get("refresh_token"))
    if refresh and refresh_token and _token_expired_or_stale(access_token, raw.get("last_refresh")):
        try:
            raw = refresh_codex_subscription_auth(path, refresh_token)
            tokens = raw.get("tokens") if isinstance(raw, dict) else None
            if isinstance(tokens, dict):
                access_token = _nonempty_str(tokens.get("access_token"))
        except Exception:
            # Fall through to existing access token; the API call will surface
            # auth errors if it is no longer accepted.
            pass

    if not access_token:
        return None

    id_token = tokens.get("id_token") if isinstance(tokens, dict) else None
    id_claims = id_token if isinstance(id_token, dict) else _decode_jwt_payload(id_token)
    return CodexSubscriptionAuth(
        access_token=access_token,
        account_id=_nonempty_str(tokens.get("account_id"))
        or _nonempty_str(id_claims.get("chatgpt_account_id")),
        plan_type=_nonempty_str(id_claims.get("chatgpt_plan_type")),
        auth_file=path,
    )


def refresh_codex_subscription_auth(path: Path, refresh_token: str) -> dict[str, Any]:
    """Refresh Codex ChatGPT OAuth tokens and persist the updated auth file."""
    response = requests.post(
        os.environ.get(CODEX_REFRESH_URL_ENV_VAR, DEFAULT_CODEX_REFRESH_URL),
        json={
            "client_id": CODEX_CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()

    auth = _load_auth_json(path)
    tokens = auth.setdefault("tokens", {})
    if payload.get("access_token"):
        tokens["access_token"] = payload["access_token"]
    if payload.get("refresh_token"):
        tokens["refresh_token"] = payload["refresh_token"]
    if payload.get("id_token"):
        tokens["id_token"] = _decode_jwt_payload(payload["id_token"])
    auth["last_refresh"] = _utc_now_rfc3339()
    _save_auth_json(path, auth)
    return auth


def _load_auth_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Codex auth file {path} must contain a JSON object")
    return data


def _save_auth_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _nonempty_str(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _decode_jwt_payload(token: Any) -> dict[str, Any]:
    if not isinstance(token, str) or token.count(".") < 2:
        return {}
    import base64

    payload = token.split(".", 2)[1]
    payload += "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload.encode("ascii"))
        data = json.loads(decoded)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _token_expired_or_stale(access_token: str | None, last_refresh: Any) -> bool:
    claims = _decode_jwt_payload(access_token)
    exp = claims.get("exp")
    if isinstance(exp, (int, float)):
        return exp <= time.time() + _REFRESH_SKEW_SECONDS
    # Some Codex auth files store parsed id_token claims, but the access token
    # may be opaque. Match Codex's conservative fallback: refresh after a day.
    if isinstance(last_refresh, str):
        try:
            from datetime import datetime, timezone

            normalized = last_refresh.replace("Z", "+00:00")
            refreshed_at = datetime.fromisoformat(normalized)
            if refreshed_at.tzinfo is None:
                refreshed_at = refreshed_at.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - refreshed_at).total_seconds() > 24 * 60 * 60
        except Exception:
            return False
    return False


def _utc_now_rfc3339() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
