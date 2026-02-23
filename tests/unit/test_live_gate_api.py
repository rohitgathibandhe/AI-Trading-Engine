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
    monkeypatch.setattr(server, "BATMAN_BKM_TUNING_ADVICE_JSON", state_dir / "batman_bkm_tuning_advice.json")
    monkeypatch.setattr(server, "BATMAN_BKM_TUNING_HISTORY_JSONL", state_dir / "batman_bkm_tuning_history.jsonl")
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

    code, payload = _run_get("/api/control/status")
    assert code == 200
    assert payload["running"] is False
    assert payload["live_gate_status"] == "LOCKED"
    assert payload["live_gate_stage"] == "S1"
    assert payload["locked_for_date"] == "2026-01-15"
    assert payload["sessions_total"] == 6
    assert payload["sessions_pass"] == 4
    assert payload["sessions_fail"] == 2


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
