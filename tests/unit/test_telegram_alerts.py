from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

from market_ai.modules.agents.telegram_alerts import TelegramAlertConfig, TelegramAlertForwarder


def _ist_dt(day: int, hour: int = 10, minute: int = 0) -> datetime:
    tz = ZoneInfo("Asia/Kolkata") if ZoneInfo else None
    return datetime(2026, 1, day, hour, minute, tzinfo=tz)


class _StubForwarder(TelegramAlertForwarder):
    def __init__(self, *args, **kwargs):
        self.sent_messages = []
        super().__init__(*args, **kwargs)

    def _send_message(self, *, token: str, chat_id: str, text: str):  # type: ignore[override]
        self.sent_messages.append({"token": token, "chat_id": chat_id, "text": text})
        return {"ok": True, "result": {"message_id": len(self.sent_messages)}}


def _fw(tmp_path: Path, **cfg_kwargs) -> _StubForwarder:
    base = {
        "enabled": True,
        "min_severity": "ERROR",
        "live_only": True,
        "max_alert_age_sec": 180.0,
        "poll_interval_sec": 0.5,
        "timeout_sec": 5.0,
        "max_batch": 5,
        "max_message_chars": 3000,
        "failure_backoff_sec": 30.0,
        "disable_link_preview": True,
    }
    base.update(cfg_kwargs)
    cfg = TelegramAlertConfig(**base)
    return _StubForwarder(
        config=cfg,
        alerts_path=tmp_path / "agent_alerts.jsonl",
        status_path=tmp_path / "telegram_alert_status.json",
        logger=None,
    )


def test_process_pending_filters_severity_and_advances_offset(tmp_path: Path) -> None:
    fw = _fw(tmp_path, min_severity="ERROR")
    alerts_path = tmp_path / "agent_alerts.jsonl"
    alerts_path.write_text(
        json.dumps({"timestamp": "2026-01-05T10:00:00+05:30", "severity": "WARNING", "code": "W1", "message": "warn"}) + "\n"
        + json.dumps({"timestamp": "2026-01-05T10:00:05+05:30", "severity": "CRITICAL", "code": "C1", "message": "crit"}) + "\n"
    )
    creds = {"telegram_bot_token": "bot-token", "telegram_chat_id": "123456"}

    out = fw.process_pending(
        creds=creds,
        trade_mode="live",
        strategy_file="batman_bkm_monthly",
        force=True,
        when=_ist_dt(5, 10, 1),
    )
    assert out["ok"] is True
    assert out["sent"] == 1
    assert out["skipped"] == 1
    assert len(fw.sent_messages) == 1
    assert "C1" in fw.sent_messages[0]["text"]
    snap = fw.snapshot()
    assert snap["sent_total"] == 1
    assert snap["skipped_total"] == 1
    assert snap["last_offset"] == alerts_path.stat().st_size


def test_process_pending_live_only_skip_in_paper_mode(tmp_path: Path) -> None:
    fw = _fw(tmp_path, live_only=True)
    (tmp_path / "agent_alerts.jsonl").write_text(
        json.dumps({"timestamp": "2026-01-05T10:00:05+05:30", "severity": "CRITICAL", "code": "C1", "message": "crit"}) + "\n"
    )
    out = fw.process_pending(
        creds={"telegram_bot_token": "bot-token", "telegram_chat_id": "123456"},
        trade_mode="paper",
        strategy_file="batman_bkm_monthly",
        force=True,
        when=_ist_dt(5, 10, 1),
    )
    assert out["ok"] is True
    assert out["reason"] == "LIVE_ONLY_SKIP"
    assert len(fw.sent_messages) == 0


def test_send_test_message_requires_credentials(tmp_path: Path) -> None:
    fw = _fw(tmp_path)
    try:
        fw.send_test_message(creds={}, when=_ist_dt(5, 10, 1))
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "not configured" in str(exc).lower()


def test_process_pending_skips_stale_alerts(tmp_path: Path) -> None:
    fw = _fw(tmp_path, min_severity="CRITICAL", max_alert_age_sec=120.0)
    alerts_path = tmp_path / "agent_alerts.jsonl"
    alerts_path.write_text(
        json.dumps({"timestamp": "2026-01-05T09:55:00+05:30", "severity": "CRITICAL", "code": "OLD1", "message": "old"}) + "\n"
        + json.dumps({"timestamp": "2026-01-05T10:00:30+05:30", "severity": "CRITICAL", "code": "NEW1", "message": "new"}) + "\n"
    )
    creds = {"telegram_bot_token": "bot-token", "telegram_chat_id": "123456"}

    out = fw.process_pending(
        creds=creds,
        trade_mode="live",
        strategy_file="batman_bkm_monthly",
        force=True,
        when=_ist_dt(5, 10, 1),
    )
    assert out["ok"] is True
    assert out["sent"] == 1
    assert out["skipped"] == 1
    assert len(fw.sent_messages) == 1
    assert "NEW1" in fw.sent_messages[0]["text"]
    assert alerts_path.stat().st_size == fw.snapshot()["last_offset"]
