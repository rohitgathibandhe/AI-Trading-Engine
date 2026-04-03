from __future__ import annotations

import io
import json
from pathlib import Path

from scripts import paper_pnl_server as server


def _run_get(path: str) -> tuple[int, dict]:
    handler = object.__new__(server.PaperHandler)
    handler.path = path
    captured: dict = {}

    def _capture(payload: dict, status: int = 200) -> None:
        captured["status"] = status
        captured["payload"] = payload

    handler._send_json = _capture  # type: ignore[assignment]
    server.PaperHandler.do_GET(handler)
    return int(captured["status"]), dict(captured["payload"])


def _run_post(path: str, payload: dict | None = None) -> tuple[int, dict]:
    body = b""
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    handler = object.__new__(server.PaperHandler)
    handler.path = path
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)
    captured: dict = {}

    def _capture(resp_payload: dict, status: int = 200) -> None:
        captured["status"] = status
        captured["payload"] = resp_payload

    handler._send_json = _capture  # type: ignore[assignment]
    server.PaperHandler.do_POST(handler)
    return int(captured["status"]), dict(captured["payload"])


def _configure_temp_state(monkeypatch, tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    static_dir = tmp_path / "static"
    state_dir.mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)
    (static_dir / "index.html").write_text("<html>ok</html>")
    (state_dir / "agent_settings.json").write_text(
        json.dumps(
            {
                "live_stage1_lot_multiplier": 1,
                "live_stage2_lot_multiplier": 2,
                "live_probation_sessions": 10,
                "live_probation_min_pass": 8,
                "batman_bkm_max_credit_pct": 6.0,
                "batman_bkm_tp_pct": 0.02,
                "batman_bkm_sl_pct": 0.025,
                "batman_bkm_balance_tolerance": 5000.0,
            },
            indent=2,
        )
    )
    (state_dir / "strategy_state.json").write_text(json.dumps({"BATMAN_BKM": {}}, indent=2))
    monkeypatch.setattr(server, "STATE_DIR", state_dir)
    monkeypatch.setattr(server, "SETTINGS_JSON", state_dir / "agent_settings.json")
    monkeypatch.setattr(server, "STRATEGY_STATE", state_dir / "strategy_state.json")
    monkeypatch.setattr(server, "LIVE_GATE_STATUS_JSON", state_dir / "live_gate_status.json")
    monkeypatch.setattr(server, "LIVE_GATE_SESSIONS_JSONL", state_dir / "live_gate_sessions.jsonl")
    monkeypatch.setattr(server, "POSITION_RECONCILE_STATUS_JSON", state_dir / "position_reconcile_status.json")
    monkeypatch.setattr(server, "EXECUTION_RECOVERY_STATUS_JSON", state_dir / "execution_recovery_status.json")
    monkeypatch.setattr(server, "EXECUTION_JOURNAL_JSONL", state_dir / "execution_journal.jsonl")
    monkeypatch.setattr(server, "AGENT_HEARTBEAT_JSON", state_dir / "agent_heartbeat.json")
    monkeypatch.setattr(server, "AGENT_ALERTS_JSONL", state_dir / "agent_alerts.jsonl")
    monkeypatch.setattr(server, "TELEGRAM_ALERT_STATUS_JSON", state_dir / "telegram_alert_status.json")
    monkeypatch.setattr(server, "BATMAN_BKM_TUNING_ADVICE_JSON", state_dir / "batman_bkm_tuning_advice.json")
    monkeypatch.setattr(server, "BATMAN_BKM_TUNING_HISTORY_JSONL", state_dir / "batman_bkm_tuning_history.jsonl")
    monkeypatch.setattr(server, "DECISION_COMMITTEE_STATUS_JSON", state_dir / "decision_committee_status.json")
    monkeypatch.setattr(server, "DECISION_COMMITTEE_HISTORY_JSONL", state_dir / "decision_committee_history.jsonl")
    monkeypatch.setattr(server, "DECISION_COMMITTEE_OUTCOMES_JSONL", state_dir / "decision_committee_outcomes.jsonl")
    monkeypatch.setattr(server, "PID_FILE", state_dir / "agent.pid")
    monkeypatch.setattr(server, "STATIC_DIR", static_dir)


def test_get_live_gate_status_default(monkeypatch, tmp_path: Path) -> None:
    _configure_temp_state(monkeypatch, tmp_path)
    code, payload = _run_get("/api/live_gate/status")
    assert code == 200
    assert payload["status"] == "PROBATION"
    assert payload["stage"] == "S1"
    assert payload["sessions_total"] == 0


def test_post_live_gate_reset_clears_status_and_sessions(monkeypatch, tmp_path: Path) -> None:
    _configure_temp_state(monkeypatch, tmp_path)
    status_path = tmp_path / "state" / "live_gate_status.json"
    sessions_path = tmp_path / "state" / "live_gate_sessions.jsonl"
    status_path.write_text(
        json.dumps(
            {
                "status": "LOCKED",
                "stage": "S1",
                "lot_multiplier_active": 1,
                "sessions_total": 4,
                "sessions_pass": 2,
                "sessions_fail": 2,
                "cum_mtm": -12000.0,
                "current_session_date": "2026-01-10",
                "last_fail_reason": "DATA_FAILSAFE_LOCK",
                "locked_for_date": "2026-01-10",
                "updated_at": "2026-01-10T15:30:00+05:30",
                "hard_lock": True,
                "consecutive_failures": 2,
            },
            indent=2,
        )
    )
    sessions_path.write_text('{"session_date":"2026-01-10","result":"FAIL"}\n')

    code, payload = _run_post("/api/live_gate/reset", payload={})
    assert code == 200
    assert payload["ok"] is True
    assert payload["live_gate"]["status"] == "PROBATION"
    assert payload["live_gate"]["stage"] == "S1"
    assert payload["live_gate"]["sessions_total"] == 0
    assert sessions_path.read_text() == ""


def test_control_status_includes_live_gate_summary(monkeypatch, tmp_path: Path) -> None:
    _configure_temp_state(monkeypatch, tmp_path)
    (tmp_path / "state" / "live_gate_status.json").write_text(
        json.dumps(
            {
                "status": "LOCKED",
                "stage": "S1",
                "lot_multiplier_active": 1,
                "sessions_total": 6,
                "sessions_pass": 4,
                "sessions_fail": 2,
                "cum_mtm": -2500.0,
                "current_session_date": "2026-01-15",
                "last_fail_reason": "DATA_FAILSAFE_LOCK",
                "locked_for_date": "2026-01-15",
                "updated_at": "2026-01-15T15:30:00+05:30",
            }
        )
    )
    (tmp_path / "state" / "decision_committee_status.json").write_text(
        json.dumps(
            {
                "status": "ACTIVE",
                "focus": "INTRADAY",
                "verdict": "READY",
                "consensus_bias": "BEARISH",
                "ensemble_confidence": 0.82,
                "advisor_signal": "ENTER_NOW",
                "advisor_strategy": "CALL_CREDIT_SPREAD",
                "expiry": "2026-04-30",
                "spot": 22850.0,
                "has_open_bkm": False,
                "current_session_date": "2026-04-03",
                "updated_at": "2026-04-03T10:15:00+05:30",
                "components": {
                    "market_structure": {"score": 0.81},
                    "option_chain": {"score": 0.76},
                    "trade_construction": {"score": 0.73},
                    "risk_critic": {"score": 0.95, "veto": False},
                },
                "reasons": ["TREND_ALIGNED", "PCR_SUPPORTIVE"],
            }
        )
    )

    code, payload = _run_get("/api/control/status")
    assert code == 200
    assert payload["running"] is False
    assert payload["live_gate_status"] == "LOCKED"
    assert payload["live_gate_stage"] == "S1"
    assert payload["locked_for_date"] == "2026-01-15"
    assert payload["sessions_total"] == 6
    assert payload["sessions_pass"] == 4
    assert payload["sessions_fail"] == 2
    assert payload["reconcile_status"] in {"OK", "LOCKED"}
    assert payload["execution_recovery_status"] in {"OK", "LOCKED"}
    assert payload["watchdog_status"] in {"MISSING", "OK", "STALE", "INVALID"}
    assert "alerts_recent_count" in payload
    assert "telegram_alerts_enabled" in payload
    assert "telegram_alerts_configured" in payload
    assert payload["intraday_committee_status"] == "ACTIVE"
    assert payload["intraday_committee_verdict"] == "READY"
    assert payload["intraday_committee_bias"] == "BEARISH"
    assert payload["intraday_committee_strategy"] == "CALL_CREDIT_SPREAD"
    assert payload["intraday_committee_reasons"] == ["TREND_ALIGNED", "PCR_SUPPORTIVE"]


def test_intraday_committee_endpoint_returns_status_and_history(monkeypatch, tmp_path: Path) -> None:
    _configure_temp_state(monkeypatch, tmp_path)
    state_dir = tmp_path / "state"
    (state_dir / "decision_committee_status.json").write_text(
        json.dumps(
            {
                "status": "ACTIVE",
                "focus": "INTRADAY",
                "verdict": "WAIT",
                "consensus_bias": "BULLISH",
                "ensemble_confidence": 0.61,
                "advisor_signal": "WAIT",
                "advisor_strategy": "PUT_CREDIT_SPREAD",
                "expiry": "2026-04-30",
                "spot": 22910.0,
                "has_open_bkm": False,
                "current_session_date": "2026-04-03",
                "updated_at": "2026-04-03T11:05:00+05:30",
                "components": {},
                "reasons": ["PERSISTENCE_PENDING"],
            }
        )
    )
    (state_dir / "decision_committee_history.jsonl").write_text(
        '{"timestamp":"2026-04-03T11:00:00+05:30","verdict":"WAIT","consensus_bias":"BULLISH"}\n'
        '{"timestamp":"2026-04-03T11:05:00+05:30","verdict":"WAIT","consensus_bias":"BULLISH"}\n'
    )

    code, payload = _run_get("/api/intraday_ai/committee")
    assert code == 200
    assert payload["ok"] is True
    assert payload["committee"]["verdict"] == "WAIT"
    assert payload["committee"]["consensus_bias"] == "BULLISH"
    assert len(payload["history"]) == 2
    assert payload["history"][-1]["timestamp"] == "2026-04-03T11:05:00+05:30"


def test_intraday_committee_review_summarizes_outcomes(monkeypatch, tmp_path: Path) -> None:
    _configure_temp_state(monkeypatch, tmp_path)
    state_dir = tmp_path / "state"
    (state_dir / "decision_committee_outcomes.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "trade_mode": "paper",
                        "strategy_label": "Call Credit Spread",
                        "committee_verdict_at_entry": "READY",
                        "committee_bias_at_entry": "BEARISH",
                        "committee_confidence_at_entry": 0.81,
                        "exit_reason": "PROFIT_PROTECT",
                        "result": "WIN",
                        "pnl_rs": 125.5,
                        "hold_minutes": 42.0,
                        "closed_at": "2026-04-03T11:30:00+05:30",
                    }
                ),
                json.dumps(
                    {
                        "trade_mode": "paper",
                        "strategy_label": "Put Credit Spread",
                        "committee_verdict_at_entry": "WAIT",
                        "committee_bias_at_entry": "BULLISH",
                        "committee_confidence_at_entry": 0.62,
                        "exit_reason": "PRICE_ACTION_REVERSAL",
                        "result": "LOSS",
                        "pnl_rs": -40.0,
                        "hold_minutes": 25.0,
                        "closed_at": "2026-04-03T12:15:00+05:30",
                    }
                ),
            ]
        )
        + "\n"
    )

    code, payload = _run_get("/api/intraday_ai/committee_review")
    assert code == 200
    assert payload["ok"] is True
    assert payload["review"]["total_outcomes"] == 2
    assert payload["review"]["wins"] == 1
    assert payload["review"]["losses"] == 1
    assert payload["review"]["realized_pnl_rs"] == 85.5
    assert payload["review"]["best_bias_label"] == "BEARISH"


def test_reconcile_status_and_reset(monkeypatch, tmp_path: Path) -> None:
    _configure_temp_state(monkeypatch, tmp_path)
    state_dir = tmp_path / "state"
    reconcile_path = state_dir / "position_reconcile_status.json"
    reconcile_path.write_text(
        json.dumps(
            {
                "status": "LOCKED",
                "hard_lock": True,
                "current_session_date": "2026-01-15",
                "locked_for_date": None,
                "mismatch_streak": 3,
                "last_mismatch_reason": "BROKER_LOCAL_POSITION_MISMATCH",
                "last_diff_summary": {"extra": {"x": 1}},
            }
        )
    )
    code, payload = _run_get("/api/reconcile/status")
    assert code == 200
    assert payload["status"] == "LOCKED"
    assert payload["hard_lock"] is True

    code, payload = _run_post("/api/reconcile/reset", payload={})
    assert code == 200
    assert payload["ok"] is True
    assert payload["reconcile"]["status"] == "OK"
    assert payload["reconcile"]["hard_lock"] is False


def test_execution_recovery_status_and_reset(monkeypatch, tmp_path: Path) -> None:
    _configure_temp_state(monkeypatch, tmp_path)
    state_dir = tmp_path / "state"
    exec_path = state_dir / "execution_recovery_status.json"
    exec_path.write_text(
        json.dumps(
            {
                "status": "LOCKED",
                "hard_lock": True,
                "current_session_date": "2026-01-15",
                "locked_for_date": None,
                "last_reason": "EXEC_JOURNAL_FAILED_OP",
                "last_details": {"count": 1},
            }
        )
    )
    code, payload = _run_get("/api/execution_recovery/status")
    assert code == 200
    assert payload["status"] == "LOCKED"
    assert payload["hard_lock"] is True

    code, payload = _run_post("/api/execution_recovery/reset", payload={})
    assert code == 200
    assert payload["ok"] is True
    assert payload["execution_recovery"]["status"] == "OK"
    assert payload["execution_recovery"]["hard_lock"] is False


def test_health_endpoint_reports_ok_and_stale(monkeypatch, tmp_path: Path) -> None:
    _configure_temp_state(monkeypatch, tmp_path)
    state_dir = tmp_path / "state"
    settings = json.loads((state_dir / "agent_settings.json").read_text())
    settings["ops_watchdog_stale_after_sec"] = 10
    (state_dir / "agent_settings.json").write_text(json.dumps(settings, indent=2))
    (state_dir / "agent_heartbeat.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-01-15T10:00:00+05:30",
                "status": "RUNNING",
                "phase": "loop_start",
                "trade_mode": "live",
            }
        )
    )

    import types

    class _FixedDateTime:
        @staticmethod
        def now(tz=None):
            if tz is None:
                return server.datetime(2026, 1, 15, 10, 0, 5)
            return server.datetime(2026, 1, 15, 10, 0, 5, tzinfo=tz)

    # monkeypatch server helper module imported function instead of datetime class internals
    from data_engine.market_ai.modules.agents import ops_monitor as opsm  # type: ignore

    monkeypatch.setattr(opsm, "_now_ist", lambda: server.datetime(2026, 1, 15, 10, 0, 5, tzinfo=server.ZoneInfo("Asia/Kolkata") if server.ZoneInfo else None))
    code, payload = _run_get("/api/health")
    assert code == 200
    assert payload["status"] == "OK"
    assert payload["age_sec"] == 5.0

    monkeypatch.setattr(opsm, "_now_ist", lambda: server.datetime(2026, 1, 15, 10, 0, 25, tzinfo=server.ZoneInfo("Asia/Kolkata") if server.ZoneInfo else None))
    code, payload = _run_get("/api/health")
    assert code == 200
    assert payload["status"] == "STALE"


def test_alerts_endpoint_and_clear(monkeypatch, tmp_path: Path) -> None:
    _configure_temp_state(monkeypatch, tmp_path)
    alerts_path = tmp_path / "state" / "agent_alerts.jsonl"
    alerts_path.write_text(
        json.dumps({"timestamp": "2026-01-15T10:00:00+05:30", "severity": "ERROR", "code": "X1", "message": "m1"}) + "\n"
        + json.dumps({"timestamp": "2026-01-15T10:00:10+05:30", "severity": "CRITICAL", "code": "X2", "message": "m2"}) + "\n"
    )

    code, payload = _run_get("/api/alerts?tail=1")
    assert code == 200
    assert payload["count"] == 1
    assert payload["alerts"][0]["code"] == "X2"

    code, payload = _run_post("/api/alerts/clear", payload={})
    assert code == 200
    assert payload["ok"] is True
    assert alerts_path.read_text() == ""


def test_telegram_alert_status_and_test_endpoint(monkeypatch, tmp_path: Path) -> None:
    _configure_temp_state(monkeypatch, tmp_path)
    state_dir = tmp_path / "state"
    (state_dir / "telegram_alert_status.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "configured": False,
                "min_severity": "CRITICAL",
                "live_only": True,
                "last_sent_at": None,
                "last_error": None,
            }
        )
    )

    code, payload = _run_get("/api/alerts/telegram/status")
    assert code == 200
    assert payload["enabled"] is True
    assert payload["configured"] is False

    monkeypatch.setattr(
        server,
        "_send_telegram_test_message",
        lambda text=None: {"ok": True, "sent_at": "2026-01-15T10:00:00+05:30", "chat_id_masked": "12***34", "text": text},
    )
    code, payload = _run_post("/api/alerts/telegram/test", payload={"text": "ping"})
    assert code == 200
    assert payload["ok"] is True
    assert payload["telegram"]["ok"] is True
    assert payload["telegram"]["chat_id_masked"] == "12***34"


def test_tuning_advice_refresh_and_get(monkeypatch, tmp_path: Path) -> None:
    _configure_temp_state(monkeypatch, tmp_path)
    (tmp_path / "state" / "live_gate_status.json").write_text(
        json.dumps(
            {
                "status": "PROMOTED",
                "stage": "S2",
                "sessions_total": 15,
                "sessions_pass": 10,
                "sessions_fail": 5,
                "cum_mtm": -5000.0,
            }
        )
    )
    (tmp_path / "state" / "live_gate_sessions.jsonl").write_text(
        json.dumps({"session_date": "2026-01-01", "result": "FAIL", "reasons": ["DAILY_LOCK_RED"]}) + "\n"
        + json.dumps({"session_date": "2026-01-02", "result": "FAIL", "reasons": ["DAILY_LOCK_RED"]}) + "\n"
    )

    code, payload = _run_post("/api/tuning/batman_bkm/refresh", payload={})
    assert code == 200
    assert payload["ok"] is True
    assert payload["advice"]["advisor"] == "BATMAN_BKM_TUNING_ADVISOR"
    assert payload["advice"]["summary"]["status"] in {"PROPOSALS_READY", "NO_CHANGE"}

    code, payload = _run_get("/api/tuning/batman_bkm/advice")
    assert code == 200
    assert payload["advisor"] == "BATMAN_BKM_TUNING_ADVISOR"


def test_tuning_apply_requires_manual_approval_flag(monkeypatch, tmp_path: Path) -> None:
    _configure_temp_state(monkeypatch, tmp_path)
    (tmp_path / "state" / "live_gate_status.json").write_text(
        json.dumps({"status": "PROMOTED", "stage": "S2", "sessions_total": 15, "sessions_pass": 11, "sessions_fail": 4, "cum_mtm": -7000.0})
    )
    (tmp_path / "state" / "live_gate_sessions.jsonl").write_text(
        json.dumps({"session_date": "2026-01-01", "result": "FAIL", "reasons": ["DAILY_LOCK_RED"]}) + "\n"
        + json.dumps({"session_date": "2026-01-02", "result": "FAIL", "reasons": ["DAILY_LOCK_RED"]}) + "\n"
    )
    refresh_code, refresh_payload = _run_post("/api/tuning/batman_bkm/refresh", payload={})
    assert refresh_code == 200
    proposal = next(iter(refresh_payload["advice"]["proposals"]))

    code, payload = _run_post("/api/tuning/batman_bkm/apply", payload={"proposal_id": proposal["id"]})
    assert code == 500
    assert "approve=true" in payload["error"]


def test_tuning_apply_updates_settings_when_approved(monkeypatch, tmp_path: Path) -> None:
    _configure_temp_state(monkeypatch, tmp_path)
    state_dir = tmp_path / "state"
    (state_dir / "live_gate_status.json").write_text(
        json.dumps({"status": "PROMOTED", "stage": "S2", "sessions_total": 15, "sessions_pass": 11, "sessions_fail": 4, "cum_mtm": -7000.0})
    )
    (state_dir / "live_gate_sessions.jsonl").write_text(
        json.dumps({"session_date": "2026-01-01", "result": "FAIL", "reasons": ["DAILY_LOCK_RED"]}) + "\n"
        + json.dumps({"session_date": "2026-01-02", "result": "FAIL", "reasons": ["DAILY_LOCK_RED"]}) + "\n"
    )
    refresh_code, refresh_payload = _run_post("/api/tuning/batman_bkm/refresh", payload={})
    assert refresh_code == 200
    proposal = next(iter(refresh_payload["advice"]["proposals"]))

    code, payload = _run_post(
        "/api/tuning/batman_bkm/apply",
        payload={"proposal_id": proposal["id"], "approve": True},
    )
    assert code == 200
    assert payload["ok"] is True
    settings = json.loads((state_dir / "agent_settings.json").read_text())
    assert settings[proposal["setting_key"]] == proposal["proposed_value"]
    assert (state_dir / "batman_bkm_tuning_history.jsonl").exists()


def test_tuning_apply_blocked_when_live_pid_file_present(monkeypatch, tmp_path: Path) -> None:
    _configure_temp_state(monkeypatch, tmp_path)
    state_dir = tmp_path / "state"
    (state_dir / "live_gate_status.json").write_text(
        json.dumps({"status": "PROMOTED", "stage": "S2", "sessions_total": 15, "sessions_pass": 11, "sessions_fail": 4, "cum_mtm": -7000.0})
    )
    (state_dir / "live_gate_sessions.jsonl").write_text(
        json.dumps({"session_date": "2026-01-01", "result": "FAIL", "reasons": ["DAILY_LOCK_RED"]}) + "\n"
        + json.dumps({"session_date": "2026-01-02", "result": "FAIL", "reasons": ["DAILY_LOCK_RED"]}) + "\n"
    )
    (state_dir / "agent.pid").write_text(json.dumps({"pid": 12345, "trade_mode": "live"}))

    refresh_code, refresh_payload = _run_post("/api/tuning/batman_bkm/refresh", payload={})
    assert refresh_code == 200
    proposal = next(iter(refresh_payload["advice"]["proposals"]))
    code, payload = _run_post(
        "/api/tuning/batman_bkm/apply",
        payload={"proposal_id": proposal["id"], "approve": True},
    )
    assert code == 500
    assert "Stop agent first" in payload["error"]


def _bkm_positions_payload_skewed() -> list[dict]:
    return [
        {"side": "SELL", "qty": 195, "entry": 73.4, "sec_id": "1", "expiry": "2026-03-30", "strike": "NIFTY-Mar2026-24500-PE"},
        {"side": "BUY", "qty": 65, "entry": 102.6, "sec_id": "2", "expiry": "2026-03-30", "strike": "NIFTY-Mar2026-24750-PE"},
        {"side": "BUY", "qty": 130, "entry": 15.9, "sec_id": "3", "expiry": "2026-03-30", "strike": "NIFTY-Mar2026-23000-PE"},
        {"side": "BUY", "qty": 65, "entry": 86.8, "sec_id": "4", "expiry": "2026-03-30", "strike": "NIFTY-Mar2026-26300-CE"},
        {"side": "SELL", "qty": 195, "entry": 53.05, "sec_id": "5", "expiry": "2026-03-30", "strike": "NIFTY-Mar2026-26500-CE"},
        {"side": "BUY", "qty": 130, "entry": 12.05, "sec_id": "6", "expiry": "2026-03-30", "strike": "NIFTY-Mar2026-27200-CE"},
    ]


def _bkm_positions_payload_ratio_121() -> list[dict]:
    return [
        {"side": "SELL", "qty": 260, "entry": 85.6, "sec_id": "1", "expiry": "2026-03-30", "strike": "NIFTY-Mar2026-22800-PE"},
        {"side": "BUY", "qty": 130, "entry": 109.2, "sec_id": "2", "expiry": "2026-03-30", "strike": "NIFTY-Mar2026-23000-PE"},
        {"side": "BUY", "qty": 130, "entry": 8.7, "sec_id": "3", "expiry": "2026-03-30", "strike": "NIFTY-Mar2026-20000-PE"},
        {"side": "BUY", "qty": 130, "entry": 153.2, "sec_id": "4", "expiry": "2026-03-30", "strike": "NIFTY-Mar2026-24100-CE"},
        {"side": "SELL", "qty": 260, "entry": 91.5, "sec_id": "5", "expiry": "2026-03-30", "strike": "NIFTY-Mar2026-24300-CE"},
        {"side": "BUY", "qty": 130, "entry": 7.7, "sec_id": "6", "expiry": "2026-03-30", "strike": "NIFTY-Mar2026-25500-CE"},
    ]


def test_importable_bkm_quality_warns_on_skewed_wings(monkeypatch, tmp_path: Path) -> None:
    _configure_temp_state(monkeypatch, tmp_path)
    monkeypatch.setattr(server, "_build_chain_map", lambda expiry: {"map": {}, "spot": 25446.35, "ts": 0.0})
    settings = json.loads((tmp_path / "state" / "agent_settings.json").read_text())
    settings["batman_bkm_import_enforce_quality"] = False
    out = server._build_importable_bkm_from_broker_positions(_bkm_positions_payload_skewed(), settings=settings)
    assert out["ok"] is True
    assert out["imported"] is True
    assert out["quality_warning"] is True
    quality = out.get("quality") or {}
    assert quality.get("ok") is False
    assert "OUTER_WING_ASYMMETRY_HIGH" in (quality.get("reasons") or [])


def test_importable_bkm_quality_can_block_when_enforced(monkeypatch, tmp_path: Path) -> None:
    _configure_temp_state(monkeypatch, tmp_path)
    monkeypatch.setattr(server, "_build_chain_map", lambda expiry: {"map": {}, "spot": 25446.35, "ts": 0.0})
    settings = json.loads((tmp_path / "state" / "agent_settings.json").read_text())
    settings["batman_bkm_import_enforce_quality"] = True
    out = server._build_importable_bkm_from_broker_positions(_bkm_positions_payload_skewed(), settings=settings)
    assert out["ok"] is False
    assert out["imported"] is False
    assert out["error"] == "BKM_QUALITY_CHECK_FAILED"


def test_importable_bkm_accepts_symmetric_ratio_121(monkeypatch, tmp_path: Path) -> None:
    _configure_temp_state(monkeypatch, tmp_path)
    monkeypatch.setattr(server, "_build_chain_map", lambda expiry: {"map": {}, "spot": 23777.8, "ts": 0.0})
    settings = json.loads((tmp_path / "state" / "agent_settings.json").read_text())
    settings["batman_bkm_import_enforce_quality"] = False
    out = server._build_importable_bkm_from_broker_positions(_bkm_positions_payload_ratio_121(), settings=settings)
    assert out["ok"] is True
    assert out["imported"] is True
    assert out["short_ratio"] == 2
