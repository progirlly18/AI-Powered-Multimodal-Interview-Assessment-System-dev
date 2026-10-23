"""SmtpMailer against a real SMTP conversation on localhost (no internet)."""

import smtplib
import socketserver
import threading
from email import message_from_bytes

import pytest

from backend.auth.config import AuthSettings
from backend.auth.mailer import MailerError, SmtpMailer, build_message


class _Handler(socketserver.StreamRequestHandler):
    """Just enough SMTP (RFC 5321) to accept one message."""

    def _send(self, line: str) -> None:
        self.wfile.write((line + "\r\n").encode())

    def handle(self) -> None:
        server = self.server
        self._send("220 localhost test SMTP")
        mail_from, rcpts = None, []
        while True:
            line = self.rfile.readline().decode(errors="replace").strip()
            if not line:
                return
            cmd = line.split(" ", 1)[0].upper()
            if cmd in ("EHLO", "HELO"):
                self._send("250 localhost")
            elif cmd == "MAIL":
                mail_from = line
                self._send("250 OK")
            elif cmd == "RCPT":
                if server.refuse_recipients:
                    self._send("550 No such user")
                else:
                    rcpts.append(line)
                    self._send("250 OK")
            elif cmd == "DATA":
                self._send("354 End data with <CR><LF>.<CR><LF>")
                data = []
                while True:
                    chunk = self.rfile.readline()
                    if chunk in (b".\r\n", b".\n"):
                        break
                    data.append(chunk)
                server.messages.append((mail_from, rcpts, b"".join(data)))
                self._send("250 Queued")
            elif cmd == "QUIT":
                self._send("221 Bye")
                return
            else:
                self._send("250 OK")


@pytest.fixture
def smtp_server():
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _Handler)
    server.daemon_threads = True
    server.messages = []
    server.refuse_recipients = False
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def settings_for(server) -> AuthSettings:
    return AuthSettings(smtp_host="127.0.0.1", smtp_port=server.server_address[1],
                        smtp_sender="humannext@uni.edu", smtp_security="none")


def test_code_email_is_delivered(smtp_server):
    SmtpMailer(settings_for(smtp_server)).send_code("ada@uni.edu", "482913", 10)
    assert len(smtp_server.messages) == 1
    _from, rcpts, raw = smtp_server.messages[0]
    assert any("ada@uni.edu" in r for r in rcpts)
    msg = message_from_bytes(raw)
    assert msg["To"] == "ada@uni.edu" and msg["From"] == "humannext@uni.edu"
    assert "482913" in msg["Subject"] and "482913" in msg.get_payload()
    assert "expires in 10 minutes" in msg.get_payload()


def test_refused_recipient_is_reported(smtp_server):
    smtp_server.refuse_recipients = True
    with pytest.raises(MailerError, match="refused that address"):
        SmtpMailer(settings_for(smtp_server)).send_code("ghost@uni.edu", "111111", 10)


def test_unreachable_server_is_reported():
    s = AuthSettings(smtp_host="127.0.0.1", smtp_port=1, smtp_sender="a@uni.edu",
                     smtp_security="none")
    with pytest.raises(MailerError, match="could not be sent"):
        SmtpMailer(s).send_code("ada@uni.edu", "111111", 10)


def test_not_configured():
    with pytest.raises(MailerError, match="not configured"):
        SmtpMailer(AuthSettings()).send_code("ada@uni.edu", "111111", 10)


def test_bad_password_is_reported(monkeypatch):
    class BadLogin:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self, **k):
            pass

        def login(self, *a):
            raise smtplib.SMTPAuthenticationError(535, b"bad credentials")

    monkeypatch.setattr(smtplib, "SMTP", BadLogin)
    s = AuthSettings(smtp_host="smtp.gmail.com", smtp_username="me@gmail.com",
                     smtp_password="wrong")
    with pytest.raises(MailerError, match="rejected its password"):
        SmtpMailer(s).send_code("ada@uni.edu", "111111", 10)


def test_message_contains_code_and_safety_note():
    text = build_message("me@uni.edu", "ada@uni.edu", "123456", 10).as_string()
    assert "123456" in text and "ignore this email" in text
