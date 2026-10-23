"""
Sign-in settings.

Read from environment variables first, then from the ``[auth]`` section of
``.streamlit/secrets.toml`` (git-ignored). The SMTP password is never
logged or shown; put it in one of those two places, never in code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from backend.config import PROJECT_ROOT, ConfigError

SECURITY_MODES = ("starttls", "ssl", "none")
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")

# environment variable -> key in the [auth] section of secrets.toml
KEYS = {
    "INTERVIEW_AI_REQUIRE_SIGN_IN": "require_sign_in",
    "INTERVIEW_AI_SMTP_HOST": "smtp_host",
    "INTERVIEW_AI_SMTP_PORT": "smtp_port",
    "INTERVIEW_AI_SMTP_USERNAME": "smtp_username",
    "INTERVIEW_AI_SMTP_PASSWORD": "smtp_password",
    "INTERVIEW_AI_SMTP_SENDER": "smtp_sender",
    "INTERVIEW_AI_SMTP_SECURITY": "smtp_security",
    "INTERVIEW_AI_ALLOWED_EMAIL_DOMAINS": "allowed_email_domains",
    "INTERVIEW_AI_AUTH_DB_PATH": "auth_db_path",
}


@dataclass(frozen=True)
class AuthSettings:
    require_sign_in: bool = True
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = field(default=None, repr=False)  # never printed
    smtp_sender: Optional[str] = None
    smtp_security: str = "starttls"
    allowed_domains: frozenset[str] = frozenset()
    auth_db_path: Path = PROJECT_ROOT / "instance" / "auth.db"
    code_ttl_s: int = 600            # a code is valid for 10 minutes
    max_attempts: int = 5            # wrong guesses before the code is void
    resend_cooldown_s: int = 60      # wait between codes to the same address
    max_codes_per_hour: int = 5      # per address
    session_max_age_s: int = 12 * 3600

    @property
    def smtp_configured(self) -> bool:
        return bool(self.smtp_host and self.sender)

    @property
    def sender(self) -> Optional[str]:
        return self.smtp_sender or self.smtp_username


def _bool(value: str, key: str) -> bool:
    v = value.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"{key} must be true or false, got {value!r}")


def load_auth_settings(env: Optional[Mapping[str, str]] = None,
                       secrets: Optional[Mapping[str, Any]] = None) -> AuthSettings:
    env = os.environ if env is None else env
    secrets = secrets or {}

    def get(env_key: str) -> Optional[str]:
        value = env.get(env_key)
        if value is None or str(value).strip() == "":
            value = secrets.get(KEYS[env_key])
        if value is None or str(value).strip() == "":
            return None
        return str(value).strip()

    require = get("INTERVIEW_AI_REQUIRE_SIGN_IN")
    port_raw = get("INTERVIEW_AI_SMTP_PORT")
    security = (get("INTERVIEW_AI_SMTP_SECURITY") or "starttls").lower()
    host = get("INTERVIEW_AI_SMTP_HOST")
    domains = get("INTERVIEW_AI_ALLOWED_EMAIL_DOMAINS")
    db_path = get("INTERVIEW_AI_AUTH_DB_PATH")

    if security not in SECURITY_MODES:
        raise ConfigError(f"INTERVIEW_AI_SMTP_SECURITY must be one of {SECURITY_MODES}")
    if security == "none" and host and host not in LOCAL_HOSTS:
        raise ConfigError("Unencrypted SMTP (security 'none') is only allowed for localhost.")
    try:
        port = int(port_raw) if port_raw else (465 if security == "ssl" else 587)
    except ValueError as exc:
        raise ConfigError(f"INTERVIEW_AI_SMTP_PORT must be a number, got {port_raw!r}") from exc
    if not 1 <= port <= 65535:
        raise ConfigError("INTERVIEW_AI_SMTP_PORT must be between 1 and 65535")

    return AuthSettings(
        require_sign_in=_bool(require, "INTERVIEW_AI_REQUIRE_SIGN_IN") if require else True,
        smtp_host=host,
        smtp_port=port,
        smtp_username=get("INTERVIEW_AI_SMTP_USERNAME"),
        smtp_password=get("INTERVIEW_AI_SMTP_PASSWORD"),
        smtp_sender=get("INTERVIEW_AI_SMTP_SENDER"),
        smtp_security=security,
        allowed_domains=frozenset(d.strip().lower().lstrip("@")
                                  for d in (domains or "").split(",") if d.strip()),
        auth_db_path=Path(db_path) if db_path else AuthSettings.auth_db_path,
    )
