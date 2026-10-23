"""One-time code generation and hashing (standard library only).

Codes are 6 random digits from ``secrets``. Only a salted SHA-256 hash is
stored, and comparison is constant-time. Brute force is limited by the
expiry and attempt limits in ``AuthService`` (5 guesses out of 1,000,000).
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets

CODE_LENGTH = 6
_CODE_RE = re.compile(rf"^\d{{{CODE_LENGTH}}}$")


def new_code() -> str:
    return f"{secrets.randbelow(10 ** CODE_LENGTH):0{CODE_LENGTH}d}"


def new_salt() -> str:
    return secrets.token_hex(16)


def hash_code(code: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{code}".encode("utf-8")).hexdigest()


def clean_code(raw: str) -> str:
    """Accept '123 456' or '123-456'; return '' if it is not a 6-digit code."""
    code = re.sub(r"[\s-]", "", raw or "")
    return code if _CODE_RE.match(code) else ""


def code_matches(code: str, salt: str, expected_hash: str) -> bool:
    return hmac.compare_digest(hash_code(code, salt), expected_hash)
