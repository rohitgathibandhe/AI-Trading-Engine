from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import request as urllib_request
from urllib import error as urllib_error

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


SEVERITY_ORDER = {"INFO": 10, "WARNING": 20, "ERROR": 30, "CRITICAL": 40}


def _now_ist() -> datetime:
    tz = ZoneInfo("Asia/Kolkata") if ZoneInfo else None
    return datetime.now(tz) if tz else datetime.now()


def _parse_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def _severity_score(value: Any) -> int:
    return SEVERITY_ORDER.get(str(value or "").upper(), 0)


def _mask_chat_id(chat_id: str) -> str:
    raw = str(chat_id or "").strip()
    if len(raw) <= 4:
        return raw
    return f"{raw[:2]}***{raw[-2:]}"


@dataclass
class TelegramAlertConfig:
    enabled: bool = False
    min_severity: str = "CRITICAL"
    live_only: bool = True
    max_alert_age_sec: float = 180.0
    poll_interval_sec: float = 5.0
    timeout_sec: float = 10.0
    max_batch: int = 5
    max_message_chars: int = 3000
    failure_backoff_sec: float = 30.0
    disable_link_preview: bool = True

    @classmethod
    def from_settings(cls, settings: Dict[str, Any]) -> "TelegramAlertConfig":
        return cls(
            enabled=bool(settings.get("telegram_alerts_enabled", False)),
            min_severity=str(settings.get("telegram_alert_min_severity", "CRITICAL")).upper(),
            live_only=bool(settings.get("telegram_alert_live_only", True)),
            max_alert_age_sec=max(0.0, float(settings.get("telegram_alert_max_age_sec", 180.0))),
            poll_interval_sec=max(0.5, float(settings.get("telegram_alert_poll_interval_sec", 5.0))),
            timeout_sec=max(1.0, float(settings.get("telegram_alert_timeout_sec", 10.0))),
            max_batch=max(1, int(settings.get("telegram_alert_max_batch", 5))),
            max_message_chars=max(256, int(settings.get("telegram_alert_max_message_chars", 3000))),
            failure_backoff_sec=max(1.0, float(settings.get("telegram_alert_failure_backoff_sec", 30.0))),
            disable_link_preview=bool(settings.get("telegram_alert_disable_link_preview", True)),
        )


@dataclass
class TelegramAlertStatus:
    enabled: bool = False
    configured: bool = False
    live_only: bool = True
    min_severity: str = "CRITICAL"
    chat_id_masked: Optional[str] = None
    last_run_at: Optional[str] = None
    last_sent_at: Optional[str] = None
    last_error_at: Optional[str] = None
    last_error: Optional[str] = None
    last_message_code: Optional[str] = None
    sent_total: int = 0
    skipped_total: int = 0
    last_offset: int = 0
    alerts_file_size: int = 0
    pending_bytes: int = 0
    consecutive_failures: int = 0
    backoff_until: Optional[str] = None
    next_poll_after: Optional[str] = None
    updated_at: str = ""

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "TelegramAlertStatus":
        return cls(
            enabled=bool(payload.get("enabled", False)),
            configured=bool(payload.get("configured", False)),
            live_only=bool(payload.get("live_only", True)),
            min_severity=str(payload.get("min_severity", "CRITICAL")),
            chat_id_masked=payload.get("chat_id_masked"),
            last_run_at=payload.get("last_run_at"),
            last_sent_at=payload.get("last_sent_at"),
            last_error_at=payload.get("last_error_at"),
            last_error=payload.get("last_error"),
            last_message_code=payload.get("last_message_code"),
            sent_total=int(payload.get("sent_total", 0)),
            skipped_total=int(payload.get("skipped_total", 0)),
            last_offset=int(payload.get("last_offset", 0)),
            alerts_file_size=int(payload.get("alerts_file_size", 0)),
            pending_bytes=int(payload.get("pending_bytes", 0)),
            consecutive_failures=int(payload.get("consecutive_failures", 0)),
            backoff_until=payload.get("backoff_until"),
            next_poll_after=payload.get("next_poll_after"),
            updated_at=str(payload.get("updated_at", "")),
        )

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TelegramAlertForwarder:
    def __init__(self, *, config: TelegramAlertConfig, alerts_path: Path, status_path: Path, logger: Any = None) -> None:
        self.config = config
        self.alerts_path = alerts_path
        self.status_path = status_path
        self.log = logger
        self.alerts_path.parent.mkdir(parents=True, exist_ok=True)
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.alerts_path.exists():
            try:
                self.alerts_path.write_text("")
            except Exception:
                pass
        self.state = self._load_state()
        self._sync_config_state()
        self._persist_state()

    def _log(self, level: str, msg: str, *args: Any) -> None:
        if self.log and hasattr(self.log, level):
            getattr(self.log, level)(msg, *args)

    def _default_state(self) -> TelegramAlertStatus:
        return TelegramAlertStatus()

    def _load_state(self) -> TelegramAlertStatus:
        if not self.status_path.exists():
            return self._default_state()
        try:
            payload = json.loads(self.status_path.read_text())
            if not isinstance(payload, dict):
                return self._default_state()
            return TelegramAlertStatus.from_payload(payload)
        except Exception:
            return self._default_state()

    def _persist_state(self) -> None:
        self.state.updated_at = _now_ist().isoformat(timespec="seconds")
        try:
            self.status_path.write_text(json.dumps(self.state.as_dict(), indent=2))
        except Exception:
            self._log("exception", "Failed to write telegram alert status: %s", self.status_path)

    def _sync_config_state(self) -> None:
        self.state.enabled = bool(self.config.enabled)
        self.state.live_only = bool(self.config.live_only)
        self.state.min_severity = str(self.config.min_severity).upper()

    def snapshot(self) -> Dict[str, Any]:
        return self.state.as_dict()

    def _extract_creds(self, creds: Dict[str, Any]) -> Tuple[str, str]:
        token = str(creds.get("telegram_bot_token") or creds.get("telegram_token") or "").strip()
        chat_id = str(creds.get("telegram_chat_id") or "").strip()
        self.state.configured = bool(token and chat_id)
        self.state.chat_id_masked = _mask_chat_id(chat_id) if chat_id else None
        return token, chat_id

    def _in_backoff(self, now: datetime) -> bool:
        until = _parse_dt(self.state.backoff_until)
        return bool(until and now < until)

    def _poll_due(self, now: datetime, force: bool) -> bool:
        if force:
            return True
        nxt = _parse_dt(self.state.next_poll_after)
        if nxt and now < nxt:
            return False
        return True

    def _read_new_alert_rows(self) -> List[Tuple[int, Dict[str, Any]]]:
        rows: List[Tuple[int, Dict[str, Any]]] = []
        if not self.alerts_path.exists():
            self.state.alerts_file_size = 0
            self.state.pending_bytes = 0
            return rows
        try:
            file_size = self.alerts_path.stat().st_size
        except Exception:
            file_size = 0
        self.state.alerts_file_size = int(file_size)
        if self.state.last_offset > file_size:
            # file truncated/rotated; restart from beginning
            self.state.last_offset = 0
        start = int(max(0, self.state.last_offset))
        if file_size <= start:
            self.state.pending_bytes = 0
            return rows
        try:
            with self.alerts_path.open("rb") as fh:
                fh.seek(start)
                offset = start
                for raw in fh:
                    offset += len(raw)
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line:
                        rows.append((offset, {}))
                        continue
                    try:
                        payload = json.loads(line)
                        if not isinstance(payload, dict):
                            payload = {}
                    except Exception:
                        payload = {}
                    rows.append((offset, payload))
        except Exception:
            self._log("exception", "Failed reading alerts file %s", self.alerts_path)
            return []
        self.state.pending_bytes = max(0, file_size - start)
        return rows

    def _is_stale_alert(self, alert: Dict[str, Any], *, now: datetime) -> bool:
        max_age = max(0.0, float(self.config.max_alert_age_sec))
        if max_age <= 0:
            return False
        ts = _parse_dt(alert.get("timestamp"))
        if ts is None:
            return False
        return (now - ts).total_seconds() > max_age

    def _format_message(
        self,
        alert: Dict[str, Any],
        *,
        trade_mode: str,
        strategy_file: Optional[str],
    ) -> str:
        severity = str(alert.get("severity") or "INFO").upper()
        code = str(alert.get("code") or "ALERT")
        message = str(alert.get("message") or "")
        ts = str(alert.get("timestamp") or "")
        source = str(alert.get("source") or "agent")
        details = alert.get("details") if isinstance(alert.get("details"), dict) else {}
        details = dict(details or {})
        details.setdefault("trade_mode", trade_mode)
        if strategy_file:
            details.setdefault("strategy_file", strategy_file)
        agent_action = str(details.get("agent_action") or "").strip()
        trade_impact = str(details.get("trade_impact") or "").strip()
        what_you_should_do = str(details.get("what_you_should_do") or "").strip()
        plain_reason = str(details.get("plain_reason") or "").strip()
        details_str = json.dumps(details, default=str, separators=(",", ":"), ensure_ascii=True)
        max_chars = max(256, int(self.config.max_message_chars))
        if agent_action or trade_impact or what_you_should_do or plain_reason:
            lines = [f"[{severity}] {code}", message]
            if agent_action:
                lines.append(f"Agent action: {agent_action}")
            if trade_impact:
                lines.append(f"Trade impact: {trade_impact}")
            if plain_reason:
                lines.append(f"Why: {plain_reason}")
            if what_you_should_do:
                lines.append(f"What you should do: {what_you_should_do}")
            lines.append(f"time: {ts}")
            lines.append(f"source: {source}")
            base = "\n".join(lines)
        else:
            base = (
                f"[{severity}] {code}\n"
                f"{message}\n"
                f"time: {ts}\n"
                f"source: {source}\n"
                f"details: {details_str}"
            )
        if len(base) <= max_chars:
            return base
        # Keep header + truncate details
        head = f"[{severity}] {code}\n{message}\n"
        budget = max_chars - len(head) - len("\n(details truncated)")
        if budget < 32:
            return head[: max_chars - 3] + "..."
        return head + details_str[:budget] + "\n(details truncated)"

    def _send_message(self, *, token: str, chat_id: str, text: str) -> Dict[str, Any]:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": bool(self.config.disable_link_preview),
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib_request.urlopen(req, timeout=float(self.config.timeout_sec)) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
        except urllib_error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else str(exc)
            raise RuntimeError(f"telegram_http_{exc.code}: {raw[:400]}") from exc
        except Exception as exc:
            raise RuntimeError(f"telegram_request_error: {exc}") from exc
        try:
            js = json.loads(raw)
        except Exception as exc:
            raise RuntimeError(f"telegram_non_json_response: {raw[:400]}") from exc
        if not isinstance(js, dict) or not js.get("ok"):
            raise RuntimeError(f"telegram_api_error: {str(js)[:400]}")
        return js

    def _record_send_success(self, *, now: datetime, code: Optional[str]) -> None:
        self.state.last_sent_at = now.isoformat(timespec="seconds")
        self.state.last_error = None
        self.state.last_error_at = None
        self.state.last_message_code = code
        self.state.sent_total = int(self.state.sent_total) + 1
        self.state.consecutive_failures = 0
        self.state.backoff_until = None

    def _record_send_failure(self, *, now: datetime, err: str) -> None:
        self.state.last_error = str(err)
        self.state.last_error_at = now.isoformat(timespec="seconds")
        self.state.consecutive_failures = int(self.state.consecutive_failures) + 1
        backoff = now + timedelta(seconds=float(self.config.failure_backoff_sec))
        self.state.backoff_until = backoff.isoformat(timespec="seconds")

    def process_pending(
        self,
        *,
        creds: Dict[str, Any],
        trade_mode: str,
        strategy_file: Optional[str],
        force: bool = False,
        when: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        now = when or _now_ist()
        self._sync_config_state()
        self.state.last_run_at = now.isoformat(timespec="seconds")
        self.state.next_poll_after = (now + timedelta(seconds=float(self.config.poll_interval_sec))).isoformat(timespec="seconds")

        token, chat_id = self._extract_creds(creds if isinstance(creds, dict) else {})
        if not self.config.enabled:
            self._persist_state()
            return {"ok": True, "sent": 0, "skipped": 0, "reason": "DISABLED"}
        if not token or not chat_id:
            self._persist_state()
            return {"ok": False, "sent": 0, "skipped": 0, "reason": "NOT_CONFIGURED"}
        if self.config.live_only and str(trade_mode).lower() != "live":
            self._persist_state()
            return {"ok": True, "sent": 0, "skipped": 0, "reason": "LIVE_ONLY_SKIP"}
        if self._in_backoff(now):
            self._persist_state()
            return {"ok": False, "sent": 0, "skipped": 0, "reason": "BACKOFF"}
        if not self._poll_due(now, force):
            self._persist_state()
            return {"ok": True, "sent": 0, "skipped": 0, "reason": "POLL_NOT_DUE"}

        rows = self._read_new_alert_rows()
        sent = 0
        skipped = 0
        threshold = _severity_score(self.config.min_severity)
        for next_offset, alert in rows:
            if sent >= int(self.config.max_batch):
                break
            if not alert:
                self.state.last_offset = int(next_offset)
                continue
            sev = str(alert.get("severity") or "").upper()
            if _severity_score(sev) < threshold:
                skipped += 1
                self.state.skipped_total = int(self.state.skipped_total) + 1
                self.state.last_offset = int(next_offset)
                continue
            if self._is_stale_alert(alert, now=now):
                skipped += 1
                self.state.skipped_total = int(self.state.skipped_total) + 1
                self.state.last_offset = int(next_offset)
                continue
            text = self._format_message(alert, trade_mode=trade_mode, strategy_file=strategy_file)
            try:
                self._send_message(token=token, chat_id=chat_id, text=text)
            except Exception as exc:
                self._record_send_failure(now=now, err=str(exc))
                self._persist_state()
                return {"ok": False, "sent": sent, "skipped": skipped, "reason": "SEND_FAILED", "error": str(exc)}
            self._record_send_success(now=now, code=str(alert.get("code") or ""))
            self.state.last_offset = int(next_offset)
            sent += 1

        # refresh pending bytes estimate after advancing cursor
        try:
            file_size = self.alerts_path.stat().st_size if self.alerts_path.exists() else 0
            self.state.alerts_file_size = int(file_size)
            self.state.pending_bytes = max(0, int(file_size) - int(self.state.last_offset))
        except Exception:
            pass
        self._persist_state()
        return {"ok": True, "sent": sent, "skipped": skipped, "reason": "DONE"}

    def send_test_message(
        self,
        *,
        creds: Dict[str, Any],
        text: Optional[str] = None,
        when: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        now = when or _now_ist()
        self._sync_config_state()
        token, chat_id = self._extract_creds(creds if isinstance(creds, dict) else {})
        if not token or not chat_id:
            self._persist_state()
            raise RuntimeError("Telegram bot token/chat_id not configured in creds.json")
        msg = text or (
            f"[TEST] TELEGRAM_ALERTS_OK\n"
            f"time: {now.isoformat(timespec='seconds')}\n"
            f"source: paper_pnl_server"
        )
        try:
            self._send_message(token=token, chat_id=chat_id, text=msg)
        except Exception as exc:
            self._record_send_failure(now=now, err=str(exc))
            self._persist_state()
            raise
        self._record_send_success(now=now, code="TELEGRAM_TEST")
        self._persist_state()
        return {"ok": True, "sent_at": self.state.last_sent_at, "chat_id_masked": self.state.chat_id_masked}

    def send_direct_message(
        self,
        *,
        creds: Dict[str, Any],
        text: str,
        code: str = "TELEGRAM_DIRECT",
        trade_mode: str = "live",
        when: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        now = when or _now_ist()
        self._sync_config_state()
        token, chat_id = self._extract_creds(creds if isinstance(creds, dict) else {})
        if not self.config.enabled:
            self._persist_state()
            return {"ok": False, "reason": "DISABLED"}
        if not token or not chat_id:
            self._persist_state()
            return {"ok": False, "reason": "NOT_CONFIGURED"}
        if self.config.live_only and str(trade_mode).lower() != "live":
            self._persist_state()
            return {"ok": False, "reason": "LIVE_ONLY_SKIP"}
        if self._in_backoff(now):
            self._persist_state()
            return {"ok": False, "reason": "BACKOFF"}
        try:
            self._send_message(token=token, chat_id=chat_id, text=str(text or ""))
        except Exception as exc:
            self._record_send_failure(now=now, err=str(exc))
            self._persist_state()
            return {"ok": False, "reason": "SEND_FAILED", "error": str(exc)}
        self._record_send_success(now=now, code=str(code or "TELEGRAM_DIRECT"))
        self._persist_state()
        return {"ok": True, "sent_at": self.state.last_sent_at, "chat_id_masked": self.state.chat_id_masked}
