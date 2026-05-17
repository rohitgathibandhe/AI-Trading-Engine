#!/usr/bin/env python3
"""Standalone Telegram alert helper.

Handles two jobs:
  1. token_expiry_check  — warn when Dhan API token expires within N hours
  2. send               — send an arbitrary message (used by other scripts)

Run by launchd or cron. Safe to call repeatedly — only alerts once per expiry cycle.

Usage:
  python scripts/telegram_alerts.py token_expiry_check
  python scripts/telegram_alerts.py send "Your message here"
"""
from __future__ import annotations

import base64
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

STATE_DIR = Path(__file__).resolve().parent.parent / "data_engine" / "market_ai" / "state"
CREDS_FILE = STATE_DIR / "creds.json"
ALERT_STATE_FILE = STATE_DIR / "token_expiry_alert_state.json"

IST = timezone(timedelta(hours=5, minutes=30))


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
        _save_alert_state({"last_alerted_expiry": expiry_str, "alerted_at": now_utc.isoformat()})
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
