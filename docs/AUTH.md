# Sign-in (passwordless email codes)

HUMAN NEXT requires users to sign in before anything else is shown.

1. Enter an email address.
2. The app checks the address (below) and emails a **6-digit one-time code** to it.
3. Enter the code to sign in.

There are no passwords to create, store or leak.

## Why fake emails can't sign in

A code is only ever sent to the inbox of the address entered, so only someone who can read that inbox can finish signing in. A made-up address, a mistyped one or someone else's never receives the code.

Before anything is sent, obvious fakes are rejected immediately:

| Check | Rejects | Example |
|---|---|---|
| Format (RFC 5321 basics, length limits) | not an email address | `ada@@uni`, `.ada@x.com` |
| Domain shape | missing or invalid domain or top-level domain | `ada@uni`, `ada@uni.123` |
| Reserved domains (RFC 2606/6761) | test and example addresses | `test@example.com`, `me@host.test` |
| Disposable domains (`data/disposable_domains.txt`) | throwaway inboxes, incl. subdomains | `x@mailinator.com` |
| Optional allow-list (`INTERVIEW_AI_ALLOWED_EMAIL_DOMAINS`) | anything outside your organisation | only `@cmrit.ac.in` |

These checks are not enough on their own, which is why the code exists: the disposable list can never be complete, and a well-formed address can still belong to someone else.

## Security properties

**Codes:**
- 6 random digits from Python's `secrets` module.
- Only a salted SHA-256 hash is stored, never the code, and comparison is constant-time.
- Each code expires after **10 minutes**, works **once**, and allows **5** wrong tries before it is voided.
- Malformed input (e.g. letters) doesn't use up a try.

**Rate limits:**
- One code per address per **60 s**, and at most **5 codes per address per hour**.
- At most **10 requests per browser session**.
- A failed send keeps the counters, so a failing mail server can't be used for rapid retries.

**Privacy:**
- The UI never shows the code and never says whether an address has signed in before.
- Logs mask addresses (`a***@uni.edu`) and never contain codes or the SMTP password.

**Sessions:**
- A sign-in lasts at most **12 hours** in that browser session; it never expires in the middle of a recording.
- *Sign out* clears the interview session.

**Authorization:**
- Saved history is filtered by the signed-in email for every list, read, delete and clear.
- A session saved by one user can't be overwritten by another.
- History saved before sign-in existed has no owner and is hidden from everyone.

**Stored data** (`instance/auth.db`, git-ignored):
- `users`: email, first and last sign-in time;
- `login_codes`: code hash, salt, expiry, attempts and send counters.

**Fails closed:** if email sending is not configured, the app shows setup instructions and lets nobody in. Running without sign-in has to be switched on explicitly (`INTERVIEW_AI_REQUIRE_SIGN_IN=false`), and then history is stored under a single local owner.

## Setting up email sending (Gmail example)

1. Use a Google account with **2-Step Verification** turned on.
2. Create an **app password**: Google Account → Security → App passwords. You get 16 characters. Never use your normal password.
3. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` (git-ignored) and fill in:

   ```toml
   [auth]
   smtp_host = "smtp.gmail.com"
   smtp_port = 587
   smtp_security = "starttls"
   smtp_username = "your.address@gmail.com"
   smtp_password = "abcd efgh ijkl mnop"
   # allowed_email_domains = "cmrit.ac.in"   # optional
   ```

   Alternatively, use environment variables, which take precedence:

   ```powershell
   $env:INTERVIEW_AI_SMTP_HOST = "smtp.gmail.com"
   $env:INTERVIEW_AI_SMTP_USERNAME = "your.address@gmail.com"
   $env:INTERVIEW_AI_SMTP_PASSWORD = "abcd efgh ijkl mnop"
   ```

4. Restart the app and sign in with your own address.

**Other providers:**
- **Outlook/Office 365:** `smtp.office365.com`, port 587, starttls. Some organisations disable SMTP login.
- **Port 465 providers:** use `smtp_security = "ssl"`.
- Unencrypted SMTP (`"none"`) is refused for anything other than localhost.

## Limits and honest caveats

- **What sign-in proves:** that a person controls the inbox. It doesn't verify their identity beyond that, and there's no second factor.
- **Internet:** sending codes needs internet access and a working mail account. Gmail limits personal accounts to roughly 500 messages a day. At a venue without internet, sign in beforehand: a sign-in lasts 12 hours in that browser tab.
- **Scope:** the app is local and single-machine. Anyone with access to this computer's files can read `instance/*.db`. This protects the app's pages, not the computer.
- **Delivery:** codes can land in spam. If one doesn't arrive, wait 60 s, then press *Resend code*.

## Tests

- `tests/unit/test_auth.py`: rules, codes, settings, the full service lifecycle with a fake clock, and no plain codes in the database.
- `tests/unit/test_mailer.py`: a real SMTP conversation with a local test server, plus refused and unreachable servers and bad passwords.
- `tests/unit/test_auth_ui.py`:
  - the app stays locked until a code is verified;
  - fake addresses get no email;
  - wrong code, sign in and sign out;
  - per-user history;
  - unconfigured email blocks entry.
- Not tested here: delivery through a real provider (needs your credentials).
