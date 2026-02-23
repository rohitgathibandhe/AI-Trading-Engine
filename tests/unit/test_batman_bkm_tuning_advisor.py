from __future__ import annotations

import json
from pathlib import Path

from market_ai.modules.agents.batman_bkm_tuning_advisor import (
    TuningPaths,
    apply_proposal,
    generate_advice,
    refresh_advice,
)


def _paths(tmp_path: Path) -> TuningPaths:
    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    return TuningPaths(
        settings_path=state / "agent_settings.json",
        strategy_state_path=state / "strategy_state.json",
        live_gate_status_path=state / "live_gate_status.json",
        live_gate_sessions_path=state / "live_gate_sessions.jsonl",
        advice_path=state / "batman_bkm_tuning_advice.json",
        history_path=state / "batman_bkm_tuning_history.jsonl",
    )


def _write_baseline_files(paths: TuningPaths) -> None:
    paths.settings_path.write_text(
        json.dumps(
            {
                "live_probation_sessions": 10,
                "batman_bkm_max_credit_pct": 6.0,
                "batman_bkm_tp_pct": 0.02,
                "batman_bkm_sl_pct": 0.025,
                "batman_bkm_balance_tolerance": 5000.0,
            },
            indent=2,
        )
    )
    paths.strategy_state_path.write_text(json.dumps({"BATMAN_BKM": {}}, indent=2))


def test_generate_advice_no_change_during_probation(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_baseline_files(paths)
    paths.live_gate_status_path.write_text(
        json.dumps(
            {
                "status": "PROBATION",
                "stage": "S1",
                "sessions_total": 3,
                "sessions_pass": 2,
                "sessions_fail": 1,
                "cum_mtm": 1500.0,
            }
        )
    )
    paths.live_gate_sessions_path.write_text("")

    report = generate_advice(paths)
    assert report["summary"]["status"] == "NO_CHANGE"
    assert report["proposals"] == []
    assert "Probation still in progress" in (report["summary"]["no_change_reason"] or "")


def test_generate_advice_proposes_de_risking_after_daily_locks(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_baseline_files(paths)
    paths.live_gate_status_path.write_text(
        json.dumps(
            {
                "status": "PROMOTED",
                "stage": "S2",
                "sessions_total": 14,
                "sessions_pass": 10,
                "sessions_fail": 4,
                "cum_mtm": -6500.0,
            }
        )
    )
    rows = [
        {"session_date": "2026-01-01", "result": "FAIL", "reasons": ["DAILY_LOCK_RED"]},
        {"session_date": "2026-01-02", "result": "PASS", "reasons": []},
        {"session_date": "2026-01-03", "result": "FAIL", "reasons": ["DAILY_LOCK_RED"]},
    ]
    paths.live_gate_sessions_path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    report = generate_advice(paths)
    assert report["summary"]["status"] == "PROPOSALS_READY"
    assert len(report["proposals"]) >= 1
    keys = {p["setting_key"] for p in report["proposals"]}
    assert "batman_bkm_max_credit_pct" in keys


def test_apply_proposal_updates_settings_and_history(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_baseline_files(paths)
    paths.live_gate_status_path.write_text(
        json.dumps(
            {
                "status": "PROMOTED",
                "stage": "S2",
                "sessions_total": 14,
                "sessions_pass": 10,
                "sessions_fail": 4,
                "cum_mtm": -6500.0,
            }
        )
    )
    paths.live_gate_sessions_path.write_text(
        json.dumps({"session_date": "2026-01-01", "result": "FAIL", "reasons": ["DAILY_LOCK_RED"]}) + "\n"
        + json.dumps({"session_date": "2026-01-02", "result": "FAIL", "reasons": ["DAILY_LOCK_RED"]}) + "\n"
    )

    report = refresh_advice(paths)
    proposal = next(p for p in report["proposals"] if p["setting_key"] == "batman_bkm_max_credit_pct")
    result = apply_proposal(paths=paths, proposal_id=proposal["id"])

    assert result["ok"] is True
    settings = json.loads(paths.settings_path.read_text())
    assert settings["batman_bkm_max_credit_pct"] == proposal["proposed_value"]
    history_lines = [line for line in paths.history_path.read_text().splitlines() if line.strip()]
    assert len(history_lines) == 1
