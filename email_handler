"""
ERIC — Email Handler
eric.agi.robot@gmail.com

Pipeline:
  1. Check inbox (IMAP) — timer every 30min + voice command
  2. Read new emails — extract sender, subject, body
  3. Cosmos drafts a reply (in Eric's voice, mission-aware)
  4. Save draft to Eric's Drafts folder
  5. Forward draft to owner for approval
  6. Poll for approval — reply "APPROVED" from owner's email on matching thread
  7. On approval — send original draft via SMTP
  8. Log all activity to mission log

Security:
  - Approval must come from EMAIL_OWNER_ADDRESS only
  - Approval must reference matching subject thread ID
  - Approval expires after EMAIL_APPROVAL_TIMEOUT_SEC (default 24h)
  - Each draft can only be approved once

Gmail setup:
  - Enable IMAP in Gmail settings
  - Generate app password: Google Account → Security → App passwords
  - Add to .env: ERIC_EMAIL_PASSWORD=xxxx
"""

import imaplib
import smtplib
import email
import email.mime.text
import email.mime.multipart
import email.mime.base
import email.encoders
import logging
import threading
import time
import json
import pathlib
import datetime
import hashlib
from typing import Optional

log = logging.getLogger("eric.email")

from config import (
    ERIC_EMAIL_ADDRESS,
    ERIC_EMAIL_PASSWORD,
    EMAIL_OWNER_ADDRESS,
    EMAIL_IMAP_HOST,
    EMAIL_SMTP_HOST,
    EMAIL_SMTP_PORT,
    EMAIL_CHECK_INTERVAL_SEC,
    EMAIL_APPROVAL_TIMEOUT_SEC,
    EMAIL_ENABLED,
)

# ─── Pending drafts store ─────────────────────────────────────────────────────
# Keyed by draft_id (hash of subject + timestamp)
# { draft_id: { subject, to, body, created_at, forwarded, sent } }

_DRAFTS_FILE = pathlib.Path.home() / ".eric" / "email_drafts.json"
_drafts: dict = {}
_drafts_lock = threading.Lock()

_check_timer: Optional[threading.Timer] = None
_running = False


# ─── Init ─────────────────────────────────────────────────────────────────────

def init_email() -> bool:
    """Load pending drafts and verify Gmail credentials. Returns True if ready."""
    if not EMAIL_ENABLED:
        log.info("Email: disabled in config")
        return False
    if not ERIC_EMAIL_PASSWORD:
        log.warning("Email: ERIC_EMAIL_PASSWORD not set — email disabled")
        return False
    _load_drafts()
    # Quick credential check
    try:
        with _imap_connect() as m:
            log.info(f"Email: connected as {ERIC_EMAIL_ADDRESS}")
    except Exception as e:
        log.error(f"Email: credential check failed — {e}")
        return False
    return True


def email_available() -> bool:
    return EMAIL_ENABLED and bool(ERIC_EMAIL_PASSWORD)


# ─── Timer ────────────────────────────────────────────────────────────────────

def start_email_timer():
    """Start periodic email check. Call after init_email()."""
    global _running
    _running = True
    _schedule_next_check()
    log.info(f"Email: timer started — checking every {EMAIL_CHECK_INTERVAL_SEC // 60} min")


def stop_email_timer():
    global _running, _check_timer
    _running = False
    if _check_timer:
        _check_timer.cancel()


def _schedule_next_check():
    global _check_timer
    if not _running:
        return
    _check_timer = threading.Timer(EMAIL_CHECK_INTERVAL_SEC, _timed_check)
    _check_timer.daemon = True
    _check_timer.start()


def _timed_check():
    try:
        check_inbox()
        check_approvals()
    except Exception as e:
        log.error(f"Email: timed check error — {e}")
    _schedule_next_check()


# ─── IMAP connection ──────────────────────────────────────────────────────────

class _imap_connect:
    """Context manager for IMAP connection."""
    def __enter__(self):
        self.m = imaplib.IMAP4_SSL(EMAIL_IMAP_HOST)
        self.m.login(ERIC_EMAIL_ADDRESS, ERIC_EMAIL_PASSWORD)
        return self.m

    def __exit__(self, *args):
        try:
            self.m.logout()
        except Exception:
            pass


# ─── Check inbox ──────────────────────────────────────────────────────────────

def check_inbox() -> list[dict]:
    """
    Fetch unread emails. Returns list of message dicts.
    Marks fetched emails as read.
    Filters out approval emails and Eric's own forwards.
    """
    if not email_available():
        return []

    messages = []
    try:
        with _imap_connect() as m:
            m.select("INBOX")
            _, data = m.search(None, "UNSEEN")
            uids = data[0].split()
            if not uids:
                log.info("Email: inbox empty")
                return []

            log.info(f"Email: {len(uids)} unread message(s)")

            for uid in uids:
                _, msg_data = m.fetch(uid, "(RFC822)")
                raw = msg_data[0][1]
                msg = email.message_from_bytes(raw)

                sender  = _parse_address(msg.get("From", ""))
                subject = msg.get("Subject", "(no subject)")
                body    = _extract_body(msg)
                date    = msg.get("Date", "")

                # Skip approval emails — handled separately
                if _is_approval(sender, subject, body):
                    log.debug(f"Email: skipping approval email from {sender}")
                    continue

                # Skip Eric's own forwarded drafts bouncing back
                if sender.lower() == ERIC_EMAIL_ADDRESS.lower():
                    log.debug("Email: skipping self-sent message")
                    continue

                messages.append({
                    "uid":     uid.decode(),
                    "sender":  sender,
                    "subject": subject,
                    "body":    body,
                    "date":    date,
                })
                log.info(f"Email: new message from {sender} — {subject!r}")

    except Exception as e:
        log.error(f"Email: inbox check failed — {e}")

    # Process each new message
    for msg in messages:
        _handle_incoming(msg)

    return messages


def _handle_incoming(msg: dict):
    """Process a single incoming email — draft a reply and forward for approval."""
    log.info(f"Email: handling message from {msg['sender']} — {msg['subject']!r}")

    # Generate reply via Cosmos
    reply_body = _cosmos_draft_reply(msg)
    if not reply_body:
        log.warning("Email: Cosmos draft failed — skipping")
        return

    # Create draft
    draft_id = _make_draft_id(msg["subject"], msg["sender"])
    draft = {
        "draft_id":    draft_id,
        "to":          msg["sender"],
        "subject":     f"Re: {msg['subject']}",
        "body":        reply_body,
        "original":    msg,
        "created_at":  time.time(),
        "forwarded":   False,
        "sent":        False,
    }

    with _drafts_lock:
        _drafts[draft_id] = draft
    _save_drafts()

    # Forward to owner for approval
    _forward_for_approval(draft)


# ─── Cosmos reply drafting ─────────────────────────────────────────────────────

def _cosmos_draft_reply(msg: dict) -> Optional[str]:
    """Ask Cosmos to draft a reply email in Eric's voice."""
    try:
        from cosmos import ask_cosmos_plain
        prompt = f"""You are ERIC — Edge Robotics Innovation by Cosmos.
You have received an email and must draft a reply in your voice.
Be helpful, honest about your capabilities, and in character as an autonomous robot.
Keep the reply concise (3-6 sentences).

From: {msg['sender']}
Subject: {msg['subject']}
Message:
{msg['body'][:1000]}

Draft a reply email body only — no subject line, no greeting header, just the body text."""

        reply = ask_cosmos_plain(prompt, max_tokens=300)
        if reply:
            log.info("Email: Cosmos drafted reply")
        return reply
    except Exception as e:
        log.error(f"Email: Cosmos draft error — {e}")
        return None


# ─── Forward for approval ─────────────────────────────────────────────────────

def _forward_for_approval(draft: dict):
    """Send draft to owner email for review and approval."""
    try:
        approval_subject = f"[ERIC DRAFT] {draft['subject']} — reply APPROVED to send"

        body = f"""Eric has drafted a reply to an email and needs your approval to send it.

─── ORIGINAL MESSAGE ───────────────────────────────
From:    {draft['original']['sender']}
Subject: {draft['original']['subject']}
Date:    {draft['original']['date']}

{draft['original']['body'][:500]}

─── ERIC'S DRAFT REPLY ─────────────────────────────
To:      {draft['to']}
Subject: {draft['subject']}

{draft['body']}

────────────────────────────────────────────────────
Reply to this email with just: APPROVED
Eric will then send the draft.
Approval expires in {EMAIL_APPROVAL_TIMEOUT_SEC // 3600} hours.
Draft ID: {draft['draft_id']}
"""
        _send_email(
            to=EMAIL_OWNER_ADDRESS,
            subject=approval_subject,
            body=body,
        )
        with _drafts_lock:
            _drafts[draft["draft_id"]]["forwarded"] = True
        _save_drafts()
        log.info(f"Email: draft forwarded to {EMAIL_OWNER_ADDRESS} for approval")

    except Exception as e:
        log.error(f"Email: forward for approval failed — {e}")


# ─── Check approvals ──────────────────────────────────────────────────────────

def check_approvals():
    """
    Scan inbox for approval emails from owner.
    Approval criteria:
      - From EMAIL_OWNER_ADDRESS only
      - Body contains "APPROVED" (case insensitive)
      - Subject references a known pending draft
      - Not expired
    """
    if not email_available():
        return

    try:
        with _imap_connect() as m:
            m.select("INBOX")
            # Search for emails from owner only
            _, data = m.search(None, f'FROM "{EMAIL_OWNER_ADDRESS}" UNSEEN')
            uids = data[0].split()
            if not uids:
                return

            for uid in uids:
                _, msg_data = m.fetch(uid, "(RFC822)")
                raw = msg_data[0][1]
                msg = email.message_from_bytes(raw)

                sender  = _parse_address(msg.get("From", ""))
                subject = msg.get("Subject", "")
                body    = _extract_body(msg).strip()

                if sender.lower() != EMAIL_OWNER_ADDRESS.lower():
                    continue

                if "approved" not in body.lower():
                    continue

                # Match to a pending draft by subject
                draft_id = _find_draft_by_approval_subject(subject)
                if not draft_id:
                    log.debug(f"Email: approval received but no matching draft — {subject!r}")
                    continue

                with _drafts_lock:
                    draft = _drafts.get(draft_id)

                if not draft:
                    continue
                if draft["sent"]:
                    log.info(f"Email: draft {draft_id} already sent — ignoring duplicate approval")
                    continue

                # Check expiry
                age = time.time() - draft["created_at"]
                if age > EMAIL_APPROVAL_TIMEOUT_SEC:
                    log.warning(f"Email: approval for {draft_id} expired ({age/3600:.1f}h old)")
                    continue

                # Send the draft
                log.info(f"Email: approval confirmed from {sender} — sending draft {draft_id}")
                _send_approved_draft(draft)

                # Mark approval email as read
                m.store(uid, "+FLAGS", "\\Seen")

    except Exception as e:
        log.error(f"Email: approval check failed — {e}")


def _send_approved_draft(draft: dict):
    """Send an approved draft and mark it sent."""
    try:
        _send_email(
            to=draft["to"],
            subject=draft["subject"],
            body=draft["body"],
        )
        with _drafts_lock:
            _drafts[draft["draft_id"]]["sent"] = True
        _save_drafts()
        log.info(f"Email: draft sent to {draft['to']} — {draft['subject']!r}")

        # Notify owner that it was sent
        try:
            _send_email(
                to=EMAIL_OWNER_ADDRESS,
                subject=f"[ERIC SENT] {draft['subject']}",
                body=f"Eric has sent the approved reply to {draft['to']}.\n\nSubject: {draft['subject']}\n\n{draft['body']}",
            )
        except Exception:
            pass

    except Exception as e:
        log.error(f"Email: failed to send approved draft — {e}")


# ─── SMTP send ────────────────────────────────────────────────────────────────

def _send_email(to: str, subject: str, body: str, attachments: list = None):
    """Send an email via Gmail SMTP."""
    msg = email.mime.multipart.MIMEMultipart()
    msg["From"]    = ERIC_EMAIL_ADDRESS
    msg["To"]      = to
    msg["Subject"] = subject
    msg.attach(email.mime.text.MIMEText(body, "plain"))

    if attachments:
        for path in attachments:
            p = pathlib.Path(path)
            if not p.exists():
                continue
            with open(p, "rb") as f:
                part = email.mime.base.MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            email.encoders.encode_base64(part)
            part.add_header("Content-Disposition", f'attachment; filename="{p.name}"')
            msg.attach(part)

    with smtplib.SMTP_SSL(EMAIL_SMTP_HOST, EMAIL_SMTP_PORT) as server:
        server.login(ERIC_EMAIL_ADDRESS, ERIC_EMAIL_PASSWORD)
        server.sendmail(ERIC_EMAIL_ADDRESS, to, msg.as_string())


# ─── Voice command integration ────────────────────────────────────────────────

def handle_voice_email_command(text: str) -> Optional[str]:
    """
    Called from voice pipeline when utterance may be an email command.
    Returns spoken response for TTS, or None if not an email command.

    Recognised commands:
      "check email" / "any messages" / "any emails"
      "read email" / "read messages"
    """
    lower = text.lower()

    if any(p in lower for p in ["check email", "any messages", "any emails", "check messages"]):
        messages = check_inbox()
        check_approvals()
        if not messages:
            return "No new messages."
        count = len(messages)
        first = messages[0]
        summary = f"{count} new message{'s' if count > 1 else ''}. "
        summary += f"First from {first['sender'].split('@')[0]}: {first['subject']}. "
        if count > 1:
            summary += f"I have drafted {'replies' if count > 1 else 'a reply'} and forwarded {'them' if count > 1 else 'it'} to you for approval."
        return summary

    if any(p in lower for p in ["read email", "read message", "read my email"]):
        messages = check_inbox()
        if not messages:
            return "No new messages."
        first = messages[0]
        return f"Message from {first['sender'].split('@')[0]}: {first['body'][:200]}"

    return None


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _parse_address(header: str) -> str:
    """Extract email address from a From header."""
    import re
    match = re.search(r"[\w.+-]+@[\w.-]+\.\w+", header)
    return match.group(0).lower() if match else header.lower().strip()


def _extract_body(msg) -> str:
    """Extract plain text body from email message."""
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    body = part.get_payload(decode=True).decode("utf-8", errors="replace")
                    break
                except Exception:
                    pass
    else:
        try:
            body = msg.get_payload(decode=True).decode("utf-8", errors="replace")
        except Exception:
            body = str(msg.get_payload())
    return body.strip()


def _is_approval(sender: str, subject: str, body: str) -> bool:
    """Check if this email is an approval response."""
    return (
        sender.lower() == EMAIL_OWNER_ADDRESS.lower()
        and "[eric draft]" in subject.lower()
        and "approved" in body.lower()
    )


def _make_draft_id(subject: str, sender: str) -> str:
    raw = f"{subject}{sender}{time.time()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _find_draft_by_approval_subject(approval_subject: str) -> Optional[str]:
    """Find draft_id matching an approval email subject."""
    # Approval subject format: "[ERIC DRAFT] Re: Original Subject — reply APPROVED to send"
    lower = approval_subject.lower()
    with _drafts_lock:
        for draft_id, draft in _drafts.items():
            if not draft["sent"] and draft["subject"].lower() in lower:
                return draft_id
    return None


# ─── Draft persistence ────────────────────────────────────────────────────────

def _save_drafts():
    try:
        _DRAFTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_DRAFTS_FILE, "w") as f:
            json.dump(_drafts, f, indent=2)
    except Exception as e:
        log.error(f"Email: failed to save drafts — {e}")


def _load_drafts():
    global _drafts
    try:
        if _DRAFTS_FILE.exists():
            with open(_DRAFTS_FILE) as f:
                _drafts = json.load(f)
            log.info(f"Email: loaded {len(_drafts)} pending draft(s)")
    except Exception as e:
        log.error(f"Email: failed to load drafts — {e}")
        _drafts = {}
