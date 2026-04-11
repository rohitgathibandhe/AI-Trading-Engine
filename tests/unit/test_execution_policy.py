from __future__ import annotations

from data_engine.market_ai.modules.agents.execution_policy import build_execution_policy


def test_execution_policy_defaults_to_execution_for_backward_compatibility() -> None:
    policy = build_execution_policy({}, "paper")

    assert policy["mode"] == "EXECUTION"
    assert policy["new_entries_allowed"] is True
    assert policy["manage_existing_allowed"] is True
    assert policy["batman_new_entries_allowed"] is True
    assert policy["batman_manage_existing_allowed"] is True


def test_execution_policy_manage_only_blocks_new_entries_but_keeps_management() -> None:
    policy = build_execution_policy(
        {
            "agent_execution_mode": "MANAGE_ONLY",
            "intraday_ai_enabled": True,
            "intraday_ai_mode": "EXECUTION",
            "intraday_ai_auto_deploy_paper_enabled": True,
        },
        "paper",
    )

    assert policy["mode"] == "MANAGE_ONLY"
    assert policy["new_entries_allowed"] is False
    assert policy["manage_existing_allowed"] is True
    assert policy["intraday_new_entries_allowed"] is False
    assert policy["intraday_manage_existing_allowed"] is True
    assert policy["batman_new_entries_allowed"] is False
    assert policy["batman_manage_existing_allowed"] is True


def test_execution_policy_monitor_mode_disables_all_execution() -> None:
    policy = build_execution_policy(
        {
            "agent_execution_mode": "MONITOR",
            "intraday_ai_enabled": True,
            "intraday_ai_mode": "EXECUTION",
            "intraday_ai_auto_deploy_live_enabled": True,
        },
        "live",
    )

    assert policy["mode"] == "MONITOR"
    assert policy["new_entries_allowed"] is False
    assert policy["manage_existing_allowed"] is False
    assert policy["any_execution_allowed"] is False
    assert policy["intraday_new_entries_allowed"] is False
    assert policy["intraday_manage_existing_allowed"] is False
    assert policy["live_execution_allowed"] is False


def test_execution_policy_respects_intraday_local_mode() -> None:
    policy = build_execution_policy(
        {
            "agent_execution_mode": "EXECUTION",
            "intraday_ai_enabled": True,
            "intraday_ai_mode": "ADVISOR",
            "intraday_ai_auto_deploy_paper_enabled": True,
        },
        "paper",
    )

    assert policy["mode"] == "EXECUTION"
    assert policy["new_entries_allowed"] is True
    assert policy["intraday_new_entries_allowed"] is False
    assert policy["intraday_manage_existing_allowed"] is True
    assert policy["batman_new_entries_allowed"] is True
