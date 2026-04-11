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
        learning_status_path=state / "batman_bkm_learning_status.json",
        learning_outcomes_path=state / "batman_bkm_learning_outcomes.jsonl",
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
                "batman_bkm_force_entry": True,
                "batman_bkm_catchup_not_before": "09:30",
                "batman_bkm_learning_opening_range_block_minutes": 15,
                "batman_bkm_ai_open_grace_sec": 180.0,
            },
            indent=2,
        )
    )
    paths.strategy_state_path.write_text(json.dumps({"BATMAN_BKM": {}}, indent=2))
    paths.learning_status_path.write_text(
        json.dumps(
            {
                "status": "ACTIVE",
                "total_outcomes": 0,
                "wins": 0,
                "losses": 0,
                "flats": 0,
                "realized_pnl_rs": 0.0,
                "last_entry_assessment": {},
                "top_negative_buckets": [],
                "top_positive_buckets": [],
            },
            indent=2,
        )
    )
    paths.learning_outcomes_path.write_text("")


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


def test_generate_advice_uses_learning_to_propose_timing_changes(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_baseline_files(paths)
    paths.live_gate_status_path.write_text(
        json.dumps(
            {
                "status": "PROBATION",
                "stage": "S1",
                "sessions_total": 1,
                "sessions_pass": 0,
                "sessions_fail": 1,
                "cum_mtm": -15522.0,
            }
        )
    )
    paths.live_gate_sessions_path.write_text("")
    paths.learning_status_path.write_text(
        json.dumps(
            {
                "status": "ACTIVE",
                "total_outcomes": 3,
                "wins": 0,
                "losses": 3,
                "flats": 0,
                "realized_pnl_rs": -12000.0,
                "last_entry_assessment": {},
                "top_negative_buckets": [{"bucket": "entry_phase", "label": "PRE_MARKET", "count": 3, "avg_pnl_rs": -4000.0, "win_rate": 0.0}],
                "top_positive_buckets": [],
            },
            indent=2,
        )
    )
    rows = [
        {
            "trade_id": f"T{i}",
            "result": "LOSS",
            "entry_phase": "PRE_MARKET",
            "held_minutes": 18.0,
            "quality_status": "PASS",
            "close_context": {"ai_reasons": ["LOSS_BUILDING"]},
            "pnl_rs": -4000.0,
        }
        for i in range(3)
    ]
    paths.learning_outcomes_path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    report = generate_advice(paths)

    keys = {p["setting_key"] for p in report["proposals"]}
    assert "batman_bkm_force_entry" in keys
    assert "batman_bkm_catchup_not_before" in keys
    assert "batman_bkm_ai_open_grace_sec" in keys
    assert report["metrics"]["batman_bkm_learning"]["premarket_loss_count"] == 3


def test_apply_proposal_supports_boolean_setting(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_baseline_files(paths)
    paths.live_gate_status_path.write_text(json.dumps({"status": "PROBATION", "sessions_total": 1}))
    paths.live_gate_sessions_path.write_text("")
    paths.learning_status_path.write_text(
        json.dumps({"status": "ACTIVE", "total_outcomes": 1, "wins": 0, "losses": 1, "flats": 0, "realized_pnl_rs": -1000.0})
    )
    paths.learning_outcomes_path.write_text(
        json.dumps(
            {
                "trade_id": "T1",
                "result": "LOSS",
                "entry_phase": "PRE_MARKET",
                "held_minutes": 15.0,
                "quality_status": "PASS",
                "close_context": {"ai_reasons": []},
                "pnl_rs": -1000.0,
            }
        ) + "\n"
    )

    report = refresh_advice(paths)
    proposal = next(p for p in report["proposals"] if p["setting_key"] == "batman_bkm_force_entry")
    result = apply_proposal(paths=paths, proposal_id=proposal["id"])

    assert result["ok"] is True
    settings = json.loads(paths.settings_path.read_text())
    assert settings["batman_bkm_force_entry"] is False
