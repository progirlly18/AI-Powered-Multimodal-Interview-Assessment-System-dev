"""
Email address checks applied *before* a sign-in code is sent.

These catch typos, reserved/test domains and known throwaway-mail domains.
They cannot prove that a person owns an address — only the emailed
one-time code does that (backend/auth/service.py).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

from backend.config import PROJECT_ROOT

DEFAULT_DISPOSABLE_FILE = PROJECT_ROOT / "data" / "disposable_domains.txt"
MAX_EMAIL_LENGTH = 254
MAX_LOCAL_LENGTH = 64

_LOCAL_RE = re.compile(
    r"^[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*$"
)
_LABEL_RE = re.compile(r"^(?!-)[a-z0-9-]{1,63}(?<!-)$")
_TLD_RE = re.compile(r"^[a-z]{2,63}$|^xn--[a-z0-9-]{1,59}$")

# RFC 2606 / 6761 names that never belong to a real mailbox.
RESERVED_DOMAINS = frozenset({"example.com", "example.org", "example.net", "localhost"})
RESERVED_TLDS = frozenset({"test", "example", "invalid", "localhost", "local"})


class EmailRejected(ValueError):
    """The address cannot be used to sign in; ``str(exc)`` is safe to show."""


def normalize_email(raw: str) -> str:
    """Trim and lower-case. Mail providers treat addresses case-insensitively in
    practice, and one canonical form stops 'Ada@x.com' and 'ada@x.com' being
    two accounts."""
    return (raw or "").strip().lower()


def _domain_matches(domain: str, listed: str) -> bool:
    return domain == listed or domain.endswith("." + listed)


def load_disposable_domains(path: Path = DEFAULT_DISPOSABLE_FILE) -> frozenset[str]:
    """One domain per line; '#' starts a comment. Missing file -> empty set."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return frozenset()
    return frozenset(
        line.split("#", 1)[0].strip().lower()
        for line in lines
        if line.split("#", 1)[0].strip()
    )


def validate_email(
    raw: str,
    disposable: Iterable[str] = (),
    allowed_domains: Optional[Iterable[str]] = None,
) -> str:
    """Return the normalised address or raise ``EmailRejected``."""
    email = normalize_email(raw)
    if not email:
        raise EmailRejected("Enter your email address.")
    if len(email) > MAX_EMAIL_LENGTH:
        raise EmailRejected("That email address is too long.")
    if email.count("@") != 1:
        raise EmailRejected("That doesn't look like an email address.")
    local, domain = email.split("@")
    if not local or len(local) > MAX_LOCAL_LENGTH or not _LOCAL_RE.match(local):
        raise EmailRejected("That doesn't look like an email address.")
    labels = domain.split(".")
    if len(labels) < 2 or not all(_LABEL_RE.match(label) for label in labels) \
            or not _TLD_RE.match(labels[-1]):
        raise EmailRejected("The part after @ is not a valid domain.")
    if domain in RESERVED_DOMAINS or labels[-1] in RESERVED_TLDS \
            or any(_domain_matches(domain, d) for d in RESERVED_DOMAINS):
        raise EmailRejected("That is a reserved example domain, not a real mailbox.")
    if any(_domain_matches(domain, d) for d in disposable):
        raise EmailRejected("Disposable email addresses can't be used. Use your own email.")
    if allowed_domains is not None:
        allowed = [d.lower() for d in allowed_domains]
        if allowed and not any(_domain_matches(domain, d) for d in allowed):
            raise EmailRejected("This app only accepts addresses from: " + ", ".join(sorted(allowed)) + ".")
    return email


def mask_email(email: str) -> str:
    """For logs: 'ada.lovelace@uni.edu' -> 'a***@uni.edu'."""
    if "@" not in email:
        return "***"
    local, domain = email.split("@", 1)
    return f"{local[:1]}***@{domain}"
