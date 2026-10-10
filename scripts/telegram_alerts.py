#!/usr/bin/env python3
"""Standalone Telegram alert helper.

Handles two jobs:
  1. token_expiry_check  — auto-renew the Dhan token when <6h left (Dhan Web tokens only), else
                           warn when it expires within N hours
  2. send               — send an arbitrary message (used by other scripts)

Run by launchd or cron. Safe to call repeatedly — only alerts once per expiry cycle.

Usage:
  python scripts/telegram_alerts.py token_expiry_check
  python scripts/telegram_alerts.py send "Your message here"
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

STATE_DIR = Path(__file__).resolve().parent.parent / "data_engine" / "market_ai" / "state"
CREDS_FILE = STATE_DIR / "creds.json"
ALERT_STATE_FILE = STATE_DIR / "token_expiry_alert_state.json"

IST = timezone(timedelta(hours=5, minutes=30))
RENEW_URL = "https://api.dhan.co/v2/RenewToken"
RENEW_WHEN_HOURS_LEFT = 6.0   # job runs hourly; renewing inside this window survives a few missed (asleep) runs


def _load_creds() -> dict:
    try:
        return json.loads(CREDS_FILE.read_text())
    except Exception:
        return {}


def _send(bot_token: str, chat_id: str, text: str) -> bool:
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        return r.status_code == 200
    except Exception as e:
        print(f"[telegram] send failed: {e}", file=sys.stderr)
        return False


def _decode_token_expiry(token: str) -> datetime | None:
    try:
        payload = token.split(".")[1]
        payload += "=" * (4 - len(payload) % 4)
        data = json.loads(base64.b64decode(payload))
        return datetime.fromtimestamp(data["exp"], tz=timezone.utc)
    except Exception:
        return None


def _load_alert_state() -> dict:
    try:
        return json.loads(ALERT_STATE_FILE.read_text())
    except Exception:
        return {}


def _save_alert_state(state: dict) -> None:
    ALERT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    ALERT_STATE_FILE.write_text(json.dumps(state, indent=2))


def _find_jwt(obj) -> str | None:
    """The RenewToken response shape isn't pinned down in Dhan's docs — take the first JWT-looking string."""
    if isinstance(obj, str):
        return obj if obj.count(".") == 2 and len(obj) > 100 else None
    if isinstance(obj, dict):
        for k in ("accessToken", "access_token", "token"):
            hit = _find_jwt(obj.get(k))
            if hit:
                return hit
        obj = list(obj.values())
    if isinstance(obj, list):
        for v in obj:
            hit = _find_jwt(v)
            if hit:
                return hit
    return None


def _renew_token(creds: dict) -> tuple[bool, str]:
    """GET /v2/RenewToken (POST returns DH-905): Dhan KILLS the current token the moment it answers
    and returns one valid 24h more — so any JWT in the response is the only working token and MUST be
    persisted before anything else can fail (a test on 2026-10-09 discarded it and locked the account
    out until a manual re-generate). Backs up the old file first. DhanWrapper hot-reloads creds.json
    by mtime, so running processes pick it up without a restart."""
    old_tok = str(creds.get("access_token") or "").strip()
    cid = str(creds.get("client_id") or "").strip()
    old_exp = _decode_token_expiry(old_tok)
    if not (old_tok and cid and old_exp):
        return False, "missing token/client_id"
    try:
        r = requests.get(RENEW_URL, headers={"access-token": old_tok, "dhanClientId": cid,
                                             "Accept": "application/json"}, timeout=15)
        body = r.json() if r.content else {}
    except Exception as e:
        return False, f"request failed: {type(e).__name__}"
    new_tok = _find_jwt(body)
    if not new_tok or new_tok == old_tok:
        keys = list(body.keys()) if isinstance(body, dict) else type(body).__name__
        return False, f"HTTP {r.status_code}, no new token in response (keys={keys})"
    new_exp = _decode_token_expiry(new_tok) or old_exp
    shutil.copy2(CREDS_FILE, CREDS_FILE.with_suffix(".json.bak"))
    creds = dict(creds)
    creds["access_token"] = new_tok
    creds["verified_at"] = datetime.now(tz=IST).isoformat()
    tmp = CREDS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(creds, indent=2))
    os.chmod(tmp, 0o600)
    os.replace(tmp, CREDS_FILE)
    return True, new_exp.astimezone(IST).strftime("%d %b %Y %I:%M %p IST")


def cmd_token_expiry_check() -> None:
    creds = _load_creds()
    bot_token = str(creds.get("telegram_bot_token") or "").strip()
    chat_id = str(creds.get("telegram_chat_id") or "").strip()
    access_token = str(creds.get("access_token") or "").strip()

    if not bot_token or not chat_id:
        print("[token_expiry_check] Telegram not configured — skipping.")
        return
    if not access_token:
        print("[token_expiry_check] No access_token in creds.json — skipping.")
        return

    expiry_utc = _decode_token_expiry(access_token)
    if expiry_utc is None:
        print("[token_expiry_check] Could not decode token expiry.")
        return

    now_utc = datetime.now(tz=timezone.utc)
    hours_left = (expiry_utc - now_utc).total_seconds() / 3600
    expiry_ist = expiry_utc.astimezone(IST)
    expiry_str = expiry_ist.strftime("%d %b %Y %I:%M %p IST")

    if 0 < hours_left <= RENEW_WHEN_HOURS_LEFT:
        ok, detail = _renew_token(creds)
        if ok:
            print(f"[token_expiry_check] Renewed — new token valid until {detail}.")
            _send(bot_token, chat_id, f"🔄 <b>Dhan token auto-renewed</b>\nValid until {detail}. No action needed.")
            return
        print(f"[token_expiry_check] Auto-renew failed: {detail}")
        state = _load_alert_state()
        if state.get("renew_fail_alerted_for") != expiry_str:
            _send(bot_token, chat_id, f"⚠️ <b>Dhan token auto-renew failed</b> ({detail}).\n"
                                      f"Current token expires {expiry_str} — generate a new one manually.")
            state["renew_fail_alerted_for"] = expiry_str
            _save_alert_state(state)

    # Only alert if within 2 hours of expiry
    if hours_left > 2.0:
        print(f"[token_expiry_check] Token valid for {hours_left:.1f}h — no alert needed.")
        return

    # Deduplicate: don't send the same alert twice for the same expiry
    state = _load_alert_state()
    last_alerted_expiry = state.get("last_alerted_expiry")
    if last_alerted_expiry == expiry_str:
        print(f"[token_expiry_check] Already alerted for this expiry ({expiry_str}) — skipping.")
        return

    if hours_left <= 0:
        msg = (
            "🔴 <b>DHAN TOKEN EXPIRED</b>\n\n"
            f"Expired at: {expiry_str}\n"
            "The trading engine is now blocked.\n\n"
            "Action: Log into Dhan → API → Generate new token → paste at http://localhost:8000"
        )
    else:
        msg = (
            f"⚠️ <b>Dhan Token Expires in {hours_left:.0f} hour(s)</b>\n\n"
            f"Expiry: {expiry_str}\n\n"
            "👉 Log into Dhan → My Account → API Access → Generate Token\n"
            "👉 Paste at <b>http://localhost:8000</b> → Save Creds"
        )

    ok = _send(bot_token, chat_id, msg)
    if ok:
        _save_alert_state({**_load_alert_state(), "last_alerted_expiry": expiry_str, "alerted_at": now_utc.isoformat()})
        print(f"[token_expiry_check] Alert sent. Token expires in {hours_left:.1f}h.")
    else:
        print("[token_expiry_check] Failed to send alert.")


def cmd_send(message: str) -> None:
    creds = _load_creds()
    bot_token = str(creds.get("telegram_bot_token") or "").strip()
    chat_id = str(creds.get("telegram_chat_id") or "").strip()
    if not bot_token or not chat_id:
        print("[send] Telegram not configured.")
        return
    ok = _send(bot_token, chat_id, message)
    print("[send] OK" if ok else "[send] FAILED")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    command = sys.argv[1]
    if command == "token_expiry_check":
        cmd_token_expiry_check()
    elif command == "send":
        msg = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "(no message)"
        cmd_send(msg)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
