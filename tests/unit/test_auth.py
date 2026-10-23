"""Email sign-in: address rules, codes, settings and the sign-in service."""

import sqlite3

import pytest

from backend.auth.codes import clean_code, code_matches, hash_code, new_code, new_salt
from backend.auth.config import AuthSettings, load_auth_settings
from backend.auth.email_rules import (
    EmailRejected,
    load_disposable_domains,
    mask_email,
    normalize_email,
    validate_email,
)
from backend.auth.mailer import MailerError
from backend.auth.service import AuthError, AuthService, InvalidCode, RateLimited
from backend.auth.store import AuthStore
from backend.config import ConfigError

DISPOSABLE = load_disposable_domains()


# ---------------- email rules ----------------

@pytest.mark.parametrize("raw,expected", [
    ("Ada.Lovelace@Uni.EDU", "ada.lovelace@uni.edu"),
    ("  bob+interview@gmail.com ", "bob+interview@gmail.com"),
    ("o'brien@mail.co.uk", "o'brien@mail.co.uk"),
    ("x@xn--bcher-kva.de", "x@xn--bcher-kva.de"),
])
def test_valid_addresses(raw, expected):
    assert validate_email(raw, DISPOSABLE) == expected


@pytest.mark.parametrize("raw,message", [
    ("", "Enter your email"),
    ("plainaddress", "doesn't look like"),
    ("a@@b.com", "doesn't look like"),
    ("a@b@c.com", "doesn't look like"),
    (".ada@uni.edu", "doesn't look like"),
    ("ada..x@uni.edu", "doesn't look like"),
    ("ada@uni", "not a valid domain"),
    ("ada@-uni.edu", "not a valid domain"),
    ("ada@uni.c", "not a valid domain"),
    ("ada@uni.123", "not a valid domain"),
    ("a" * 65 + "@uni.edu", "doesn't look like"),
    ("ada@" + "a" * 250 + ".com", "too long"),
    ("test@example.com", "reserved"),
    ("x@mail.example.org", "reserved"),
    ("me@host.test", "reserved"),
    ("me@localhost.localdomain.local", "reserved"),
])
def test_invalid_addresses(raw, message):
    with pytest.raises(EmailRejected, match=message):
        validate_email(raw, DISPOSABLE)


@pytest.mark.parametrize("raw", ["x@mailinator.com", "x@YOPMAIL.com", "x@eu.10minutemail.com"])
def test_disposable_domains_rejected(raw):
    with pytest.raises(EmailRejected, match="Disposable"):
        validate_email(raw, DISPOSABLE)


def test_allowed_domains():
    assert validate_email("a@cmrit.ac.in", allowed_domains=["cmrit.ac.in"]) == "a@cmrit.ac.in"
    assert validate_email("a@cs.cmrit.ac.in", allowed_domains=["cmrit.ac.in"])  # subdomain
    with pytest.raises(EmailRejected, match="only accepts"):
        validate_email("a@gmail.com", allowed_domains=["cmrit.ac.in"])
    with pytest.raises(EmailRejected):
        validate_email("a@notcmrit.ac.in", allowed_domains=["cmrit.ac.in"])


def test_disposable_list_loads_and_ignores_comments(tmp_path):
    assert "mailinator.com" in DISPOSABLE and all(not d.startswith("#") for d in DISPOSABLE)
    f = tmp_path / "d.txt"
    f.write_text("# c\nA.com  # x\n\nb.org\n", encoding="utf-8")
    assert load_disposable_domains(f) == {"a.com", "b.org"}
    assert load_disposable_domains(tmp_path / "missing.txt") == frozenset()


def test_normalize_and_mask():
    assert normalize_email(" A@B.Com ") == "a@b.com"
    assert mask_email("ada.lovelace@uni.edu") == "a***@uni.edu"
    assert mask_email("nonsense") == "***"


# ---------------- codes ----------------

def test_codes_are_six_random_digits():
    codes = {new_code() for _ in range(200)}
    assert all(len(c) == 6 and c.isdigit() for c in codes)
    assert len(codes) > 150


@pytest.mark.parametrize("raw,expected", [("123456", "123456"), (" 123 456 ", "123456"),
                                          ("123-456", "123456"), ("12345", ""), ("abcdef", ""),
                                          ("1234567", ""), ("", "")])
def test_clean_code(raw, expected):
    assert clean_code(raw) == expected


def test_hash_and_match():
    salt = new_salt()
    h = hash_code("123456", salt)
    assert "123456" not in h and len(h) == 64
    assert code_matches("123456", salt, h)
    assert not code_matches("123457", salt, h)
    assert not code_matches("123456", new_salt(), h)


# ---------------- settings ----------------

def test_settings_defaults_and_precedence():
    s = load_auth_settings({}, {})
    assert s.require_sign_in is True and not s.smtp_configured and s.smtp_port == 587
    s = load_auth_settings({"INTERVIEW_AI_SMTP_HOST": "smtp.env.com"},
                           {"smtp_host": "smtp.secret.com", "smtp_username": "me@uni.edu",
                            "smtp_password": "pw", "allowed_email_domains": "@Uni.edu, cmrit.ac.in"})
    assert s.smtp_host == "smtp.env.com"            # environment wins
    assert s.sender == "me@uni.edu" and s.smtp_configured
    assert s.allowed_domains == {"uni.edu", "cmrit.ac.in"}
    assert "pw" not in repr(s)                      # password never printed
    assert load_auth_settings({"INTERVIEW_AI_SMTP_SECURITY": "ssl"}, {}).smtp_port == 465
    assert load_auth_settings({"INTERVIEW_AI_REQUIRE_SIGN_IN": "false"}, {}).require_sign_in is False


@pytest.mark.parametrize("env", [
    {"INTERVIEW_AI_SMTP_SECURITY": "plaintext"},
    {"INTERVIEW_AI_SMTP_SECURITY": "none", "INTERVIEW_AI_SMTP_HOST": "smtp.gmail.com"},
    {"INTERVIEW_AI_SMTP_PORT": "abc"},
    {"INTERVIEW_AI_SMTP_PORT": "70000"},
    {"INTERVIEW_AI_REQUIRE_SIGN_IN": "maybe"},
])
def test_invalid_settings(env):
    with pytest.raises(ConfigError):
        load_auth_settings(env, {})


def test_unencrypted_smtp_only_for_localhost():
    s = load_auth_settings({"INTERVIEW_AI_SMTP_SECURITY": "none",
                            "INTERVIEW_AI_SMTP_HOST": "127.0.0.1"}, {})
    assert s.smtp_security == "none"


# ---------------- service ----------------

class FakeMailer:
    def __init__(self, fail=False):
        self.sent = []
        self.fail = fail

    def send_code(self, to, code, valid_minutes):
        if self.fail:
            raise MailerError("The code could not be sent right now.")
        self.sent.append((to, code, valid_minutes))

    @property
    def last_code(self):
        return self.sent[-1][1]


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


@pytest.fixture
def svc(tmp_path):
    clock, mailer = Clock(), FakeMailer()
    settings = AuthSettings(smtp_host="smtp.test.invalid", smtp_username="sender@uni.edu",
                            auth_db_path=tmp_path / "auth.db")
    service = AuthService(settings, AuthStore(settings.auth_db_path), mailer, DISPOSABLE, clock)
    return service, mailer, clock


def test_full_sign_in(svc):
    service, mailer, clock = svc
    email = service.request_code(" Ada@Uni.EDU ")
    assert email == "ada@uni.edu" and mailer.sent[0][0] == "ada@uni.edu" and mailer.sent[0][2] == 10
    user = service.verify_code("ADA@uni.edu", mailer.last_code)
    assert user.email == "ada@uni.edu" and service.session_valid(user)
    clock.t += 12 * 3600 + 1
    assert not service.session_valid(user)
    assert not service.session_valid(None)


def test_code_is_single_use(svc):
    service, mailer, _ = svc
    service.request_code("ada@uni.edu")
    code = mailer.last_code
    service.verify_code("ada@uni.edu", code)
    with pytest.raises(AuthError, match="no active code"):
        service.verify_code("ada@uni.edu", code)


def test_wrong_codes_lock_the_code(svc):
    service, mailer, _ = svc
    service.request_code("ada@uni.edu")
    right = mailer.last_code
    wrong = "000000" if right != "000000" else "111111"
    for left in (4, 3, 2, 1):
        with pytest.raises(InvalidCode) as info:
            service.verify_code("ada@uni.edu", wrong)
        assert info.value.attempts_left == left
    with pytest.raises(AuthError, match="Too many wrong codes"):
        service.verify_code("ada@uni.edu", wrong)
    with pytest.raises(AuthError):
        service.verify_code("ada@uni.edu", right)   # even the right code is now void


def test_malformed_code_does_not_use_an_attempt(svc):
    service, mailer, _ = svc
    service.request_code("ada@uni.edu")
    for _ in range(10):
        with pytest.raises(AuthError, match="6-digit"):
            service.verify_code("ada@uni.edu", "12ab")
    assert service.verify_code("ada@uni.edu", mailer.last_code)


def test_code_expires(svc):
    service, mailer, clock = svc
    service.request_code("ada@uni.edu")
    clock.t += 601
    with pytest.raises(AuthError, match="expired"):
        service.verify_code("ada@uni.edu", mailer.last_code)


def test_resend_cooldown_and_new_code_replaces_old(svc):
    service, mailer, clock = svc
    service.request_code("ada@uni.edu")
    first = mailer.last_code
    with pytest.raises(RateLimited) as info:
        service.request_code("ada@uni.edu")
    assert 0 < info.value.retry_after_s <= 60
    clock.t += 61
    service.request_code("ada@uni.edu")
    second = mailer.last_code
    if first != second:
        with pytest.raises(InvalidCode):
            service.verify_code("ada@uni.edu", first)
    assert service.verify_code("ada@uni.edu", second)


def test_hourly_cap(svc):
    service, _mailer, clock = svc
    for _ in range(5):
        service.request_code("ada@uni.edu")
        clock.t += 61
    with pytest.raises(RateLimited, match="Too many codes"):
        service.request_code("ada@uni.edu")
    clock.t += 3600
    service.request_code("ada@uni.edu")                # window has passed


def test_fake_addresses_never_get_a_code(svc):
    service, mailer, _ = svc
    for bad in ("nobody@mailinator.com", "test@example.com", "not-an-email", "x@host.test"):
        with pytest.raises(EmailRejected):
            service.request_code(bad)
    assert mailer.sent == []


def test_cannot_verify_without_request(svc):
    service, _, _ = svc
    with pytest.raises(AuthError, match="no active code"):
        service.verify_code("ada@uni.edu", "123456")


def test_mail_failure_voids_the_code(svc):
    service, mailer, clock = svc
    mailer.fail = True
    with pytest.raises(MailerError):
        service.request_code("ada@uni.edu")
    with pytest.raises(AuthError, match="no active code"):
        service.verify_code("ada@uni.edu", "123456")
    with pytest.raises(RateLimited):          # counters kept: no rapid retry spam
        service.request_code("ada@uni.edu")


def test_allowed_domain_enforced_by_service(tmp_path):
    settings = AuthSettings(smtp_host="h", smtp_username="s@uni.edu",
                            allowed_domains=frozenset({"uni.edu"}), auth_db_path=tmp_path / "a.db")
    mailer = FakeMailer()
    service = AuthService(settings, AuthStore(settings.auth_db_path), mailer, DISPOSABLE)
    with pytest.raises(EmailRejected, match="only accepts"):
        service.request_code("ada@gmail.com")
    service.request_code("ada@uni.edu")
    assert len(mailer.sent) == 1


def test_database_never_contains_plain_codes(svc, tmp_path):
    service, mailer, _ = svc
    service.request_code("ada@uni.edu")
    code = mailer.last_code
    dump = "\n".join(sqlite3.connect(tmp_path / "auth.db").iterdump())
    assert code not in dump.replace("ada@uni.edu", "")
    service.verify_code("ada@uni.edu", code)
    user = AuthStore(tmp_path / "auth.db").get_user("ada@uni.edu")
    assert user is not None and user.created_at == user.last_login_at
