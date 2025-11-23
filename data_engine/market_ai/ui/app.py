# -*- coding: utf-8 -*-
"""
Algo Agent Dashboard (Streamlit) — stable, end-to-end working build.

What this file does:
- Sidebar: save/verify credentials, start/stop agent, choose paper/live, set refresh.
- Trade tab: funds cards + NIFTY spot LTP (polls in background and shows latest).
- Positions tab:
    • Open positions table (SHORT → sellQty/sellAvg, LONG → buyQty/buyAvg; LTP=costPrice; PnL=unrealizedProfit)
    • Today’s closed trades table (Qty, Buy Avg, Sell Avg, LTP=costPrice, PnL=realizedProfit) + Overall row
- Agent Logs tab: shows last ~800 log lines.
- Settings tab: read/write JSON config.
"""

from __future__ import annotations

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime, timedelta, time as datetime_time
from typing import Iterable
from typing import Optional, Dict, Any, Tuple, List
from queue import Queue, Empty

# --- Bootstrap PYTHONPATH for VS Code/Streamlit launches ---
_ROOT = Path(__file__).resolve().parents[3]  # repo root
_DATA_ENGINE = _ROOT / "data_engine"
for _p in (str(_ROOT), str(_DATA_ENGINE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from math import erf

try:  # Streamlit 1.18+
    from streamlit_autorefresh import st_autorefresh
except ImportError:  # fallback for older builds
    st_autorefresh = None


# =============================================================================
# Auto-refresh helper
# =============================================================================
def _auto_refresh() -> None:
    """Trigger periodic reruns so LTP and positions update without manual refresh."""
    try:
        refresh_sec = max(1, int(st.session_state.get("refresh_sec", 5)))
    except Exception:
        refresh_sec = 5

    if st_autorefresh is not None:
        st_autorefresh(interval=refresh_sec * 1000, key="auto_refresh_tick")
    else:
        # fallback: rely on browser refresh every ~refresh_sec seconds via hidden script
        placeholder = st.empty()
        placeholder.markdown(
            f"""
            <script>
                setTimeout(function() {{ window.location.reload(); }}, {refresh_sec * 1000});
            </script>
            """,
            unsafe_allow_html=True,
        )


def _load_weekly_settings() -> Dict[str, Any]:
    defaults = {
        "min_prev_range": 0.003,
        "pnl_target": 6000.0,
        "pnl_stop": 4000.0,
        "max_gap_pct": 0.01,
        "directional_enabled": True,
    }
    if WEEKLY_SETTINGS_FILE.exists():
        try:
            data = json.loads(WEEKLY_SETTINGS_FILE.read_text())
            defaults.update(data or {})
        except Exception:
            pass
    return defaults


def _save_weekly_settings(data: Dict[str, Any]) -> None:
    WEEKLY_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    WEEKLY_SETTINGS_FILE.write_text(json.dumps(data, indent=2))


def _ensure_weekly_settings_state() -> None:
    ss = st.session_state
    if ss.get("_weekly_settings_loaded"):
        return
    data = _load_weekly_settings()
    ss["weekly_min_prev_range"] = data.get("min_prev_range", 0.003)
    ss["weekly_pnl_target"] = data.get("pnl_target", 6000.0)
    ss["weekly_pnl_stop"] = data.get("pnl_stop", 4000.0)
    ss["weekly_max_gap_pct"] = data.get("max_gap_pct", 0.01)
    ss["weekly_directional_enabled"] = bool(data.get("directional_enabled", True))
    ss["_weekly_settings_loaded"] = True

# =============================================================================
# Paths & constants
# =============================================================================
_UI_FILE = Path(__file__).resolve()            # .../data_engine/market_ai/ui/app.py
ENGINE_DIR = _UI_FILE.parents[1]               # .../data_engine/market_ai
# Repo root = parent of the top-level "data_engine" package (works for AI-Trading-Engine and older layouts)
ROOT = ENGINE_DIR.parents[1]
PYTHON = sys.executable

# Ensure repo root is on sys.path
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STATE_DIR = ENGINE_DIR / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
CONTROL_DIR = ENGINE_DIR / "control"
CONTROL_DIR.mkdir(parents=True, exist_ok=True)
RISK_RESET_FILE = CONTROL_DIR / "risk_reset.json"
ENTRY_BLOCK_STATE = STATE_DIR / "entry_block_status.json"
ACTION_STATUS_FILE = STATE_DIR / "manual_action_status.json"
PLAYBOOK_HISTORY_FILE = STATE_DIR / "playbook_history.jsonl"
ROLLING_SUMMARY_FILE = STATE_DIR / "rolling_option_summary.json"
SELECTOR_SUMMARY_FILE = STATE_DIR / "strategy_selector_training_summary.json"
BROADER_STATE_DIR = ROOT / "state"
BROADER_STATE_DIR.mkdir(parents=True, exist_ok=True)
WEEKLY_PLAN_FILE = BROADER_STATE_DIR / "weekly_plan.json"
WEEKLY_PLAN_SCRIPT = ROOT / "scripts" / "weekly_plan_cron.sh"
OPTION_CHAIN_SCRIPT = ROOT / "data_engine/market_ai/scripts/collect_live_option_chain.py"
WEEKLY_SETTINGS_FILE = BROADER_STATE_DIR / "weekly_plan_settings.json"
ORDER_INTENT_FILE = STATE_DIR / "order_intents.jsonl"
LAST_STRATEGY_FILE = STATE_DIR / "last_strategy.json"

AGENT_LOG = STATE_DIR / "agent.log"
BATMAN_LOG = STATE_DIR / "batman_daemon.log"
SETTINGS_JSON = STATE_DIR / "agent_settings.json"
PID_FILE = STATE_DIR / "agent.pid"
CREDS_FILE = STATE_DIR / "creds.json"
AGENT_ENTRY = ENGINE_DIR / "start_agent.py"

# Fallback for unusual layout (older copies)
if not AGENT_ENTRY.exists():
    _alt = ROOT / "data_engine" / "market_ai" / "start_agent.py"
    if _alt.exists():
        AGENT_ENTRY = _alt

# Thread-safe queue for LTP updates from background poller
LTP_QUEUE: Queue = Queue(maxsize=128)

DEFAULT_SETTINGS = {
    "max_legs": 1,
    "lot_size": 75,
    "nifty_lot_size": 75,
    "nifty_expiry_weekday": "Tuesday",
    "expiry_shift_if_holiday": True,
    "holiday_list": [],
    "gap_entry_threshold": 0.004,  # 0.4%
    "iv_floor_percentile": 0.2,
    "batman_long_offset": 300.0,
    "batman_short_offset": 500.0,
    "batman_qty_long": 1,
    "batman_qty_short": 3,
    "batman_weekly_hedge_enabled": True,
    "batman_weekly_hedge_price_cap": 12.0,
    "batman_weekly_hedge_min_distance": 300.0,
    "leg_sl_pct": 2.5,
    "profit_pct": 2.25,
    "smart_selector_enabled": True,
    "max_intraday_loss": -3000.0,
    "intraday_target": 4000.0,
    "allow_carry_forward": False,
    "max_carry_days": 0,
    "vix_carry_threshold": 12.0,
    "max_daily_trades": 5,
    "max_daily_loss_pct": 0.03,
    "avg_5m_volume": 100000.0,
    "per_leg_sl_mult": 1.6,
    "per_leg_tp_mult": 0.5,
    "hedge_enabled": True,
    "hedge_delta_breach": 0.30,
    "hedge_premium_hard_x": 2.0,
    "hedge_roll_distance": 150.0,
    "hedge_delta_max": 0.12,
    "hedge_price_max": 35.0,
    "hedge_salvage_wing_ltp": 5.0,
    "vix_adaptive_low": 12.0,
    "vix_adaptive_high": 20.0,
    "strangle_delta_low": 0.15,
    "strangle_delta_high": 0.15,
    "strangle_offset_low": 150.0,
    "strangle_offset_high": 150.0,
    "spread_short_delta_low": 0.25,
    "spread_short_delta_high": 0.25,
    "risk_max_portfolio_delta": 0.25,
    "risk_max_exposure_pct": 0.05,
    "risk_account_equity": 500000.0,
    "auto_playbook": {
        "hedge_on_delta": True,
        "flatten_on_delta": False,
        "flatten_on_exposure": False,
    },
}

LEGACY_HEDGE_SETTING_ALIASES = {
    "batman_enabled": "hedge_enabled",
    "batman_delta_breach": "hedge_delta_breach",
    "batman_premium_hard_x": "hedge_premium_hard_x",
    "batman_roll_distance": "hedge_roll_distance",
    "batman_hedge_delta_max": "hedge_delta_max",
    "batman_hedge_price_max": "hedge_price_max",
    "batman_salvage_wing_ltp": "hedge_salvage_wing_ltp",
}

BLOTTER_CSV = STATE_DIR / "trade_blotter.csv"
BLOTTER_SUMMARY = STATE_DIR / "trade_blotter_summary.json"
FEATURE_LOG_CSV = STATE_DIR / "feature_history.csv"
EQUITY_LOG_CSV = STATE_DIR / "equity_history.csv"

try:
    from market_ai.modules.strategies.monthly_strangle_with_weekly_hedge import BLOTTER_FIELDS as STRAT_BLOTTER_FIELDS  # type: ignore
except Exception:
    STRAT_BLOTTER_FIELDS = [
        "timestamp",
        "trade_mode",
        "warn_only",
        "executed",
        "side",
        "order_type",
        "exchange_seg",
        "product_type",
        "security_id",
        "quantity",
        "price",
        "strike",
        "tag",
        "notes",
    ]


def _ensure_csv_exists(path: Path, headers: List[str]) -> None:
    try:
        if path.exists() and path.stat().st_size > 0:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            handle.write(",".join(headers) + "\n")
    except Exception:
        pass


def _request_risk_reset() -> None:
    payload = {
        "action": "reset_risk",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        RISK_RESET_FILE.write_text(json.dumps(payload, indent=2))
    except Exception:
        pass


def _request_risk_action(action: str, reason: Optional[str] = None) -> None:
    payload = {
        "action": action,
        "reason": reason,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        RISK_RESET_FILE.write_text(json.dumps(payload, indent=2))
    except Exception:
        pass

# =============================================================================
# Dhan wrapper import
# =============================================================================
try:
    from data_engine.market_ai.dhan_wrapper import DhanWrapper  # runtime import
    _import_err = None
except Exception as e:
    # Keep type-checkers happy without breaking runtime
    DhanWrapper = Any  # type: ignore[assignment]
    _import_err = e

# =============================================================================
# Session init
# =============================================================================
def _init_state() -> None:
    ss = st.session_state
    saved_creds: Dict[str, Any] = {}
    if CREDS_FILE.exists():
        try:
            saved_creds = json.loads(CREDS_FILE.read_text())
        except Exception:
            saved_creds = {}

    ss.setdefault("client_id", saved_creds.get("client_id", os.getenv("DHAN_CLIENT_ID", "")))
    ss.setdefault("access_token", saved_creds.get("access_token", os.getenv("DHAN_ACCESS_TOKEN", "")))
    ss.setdefault("creds_verified", bool(saved_creds.get("creds_verified", False)))

    ss.setdefault("funds", {"available": None, "collateral": None, "utilized": None, "withdrawable": None})
    ss.setdefault("last_funds_at", None)

    ss.setdefault("agent_pid", None)
    ss.setdefault("agent_running", False)
    ss.setdefault("trade_mode", "paper")
    ss.setdefault("refresh_sec", 5)

    ss.setdefault("ltp_value", None)
    ss.setdefault("ltp_ts", None)

    ss.setdefault("dw", None)

    if not SETTINGS_JSON.exists():
        SETTINGS_JSON.write_text(json.dumps(DEFAULT_SETTINGS, indent=2))


def _load_settings() -> Dict[str, Any]:
    try:
        data = json.loads(SETTINGS_JSON.read_text())
        if not isinstance(data, dict):
            raise TypeError("settings not a dict")
    except Exception:
        data = {}
    for legacy, new in LEGACY_HEDGE_SETTING_ALIASES.items():
        if legacy in data and new not in data:
            data[new] = data[legacy]
    merged = {**DEFAULT_SETTINGS, **data}
    # ensure types
    merged["max_legs"] = int(merged.get("max_legs", DEFAULT_SETTINGS["max_legs"]))
    merged["lot_size"] = int(merged.get("lot_size", DEFAULT_SETTINGS["lot_size"]))
    merged["nifty_lot_size"] = int(merged.get("nifty_lot_size", merged["lot_size"]))
    # keep both lot size fields aligned for backward compatibility
    merged["lot_size"] = merged.get("nifty_lot_size", merged["lot_size"])
    merged["leg_sl_pct"] = float(merged.get("leg_sl_pct", DEFAULT_SETTINGS["leg_sl_pct"]))
    merged["profit_pct"] = float(merged.get("profit_pct", DEFAULT_SETTINGS["profit_pct"]))
    merged["smart_selector_enabled"] = bool(merged.get("smart_selector_enabled", DEFAULT_SETTINGS["smart_selector_enabled"]))
    merged["max_intraday_loss"] = float(merged.get("max_intraday_loss", DEFAULT_SETTINGS["max_intraday_loss"]))
    merged["intraday_target"] = float(merged.get("intraday_target", DEFAULT_SETTINGS["intraday_target"]))
    merged["allow_carry_forward"] = bool(merged.get("allow_carry_forward", DEFAULT_SETTINGS["allow_carry_forward"]))
    merged["max_carry_days"] = int(merged.get("max_carry_days", DEFAULT_SETTINGS["max_carry_days"]))
    merged["vix_carry_threshold"] = float(merged.get("vix_carry_threshold", DEFAULT_SETTINGS["vix_carry_threshold"]))
    merged["max_daily_trades"] = int(merged.get("max_daily_trades", DEFAULT_SETTINGS["max_daily_trades"]))
    merged["max_daily_loss_pct"] = float(merged.get("max_daily_loss_pct", DEFAULT_SETTINGS["max_daily_loss_pct"]))
    merged["avg_5m_volume"] = float(merged.get("avg_5m_volume", DEFAULT_SETTINGS["avg_5m_volume"]))
    merged["per_leg_sl_mult"] = float(merged.get("per_leg_sl_mult", DEFAULT_SETTINGS["per_leg_sl_mult"]))
    merged["per_leg_tp_mult"] = float(merged.get("per_leg_tp_mult", DEFAULT_SETTINGS["per_leg_tp_mult"]))
    merged["hedge_enabled"] = bool(merged.get("hedge_enabled", DEFAULT_SETTINGS["hedge_enabled"]))
    merged["hedge_delta_breach"] = float(merged.get("hedge_delta_breach", DEFAULT_SETTINGS["hedge_delta_breach"]))
    merged["hedge_premium_hard_x"] = float(merged.get("hedge_premium_hard_x", DEFAULT_SETTINGS["hedge_premium_hard_x"]))
    merged["hedge_roll_distance"] = float(merged.get("hedge_roll_distance", DEFAULT_SETTINGS["hedge_roll_distance"]))
    merged["hedge_delta_max"] = float(merged.get("hedge_delta_max", DEFAULT_SETTINGS["hedge_delta_max"]))
    merged["hedge_price_max"] = float(merged.get("hedge_price_max", DEFAULT_SETTINGS["hedge_price_max"]))
    merged["hedge_salvage_wing_ltp"] = float(merged.get("hedge_salvage_wing_ltp", DEFAULT_SETTINGS["hedge_salvage_wing_ltp"]))
    merged["risk_max_portfolio_delta"] = float(merged.get("risk_max_portfolio_delta", DEFAULT_SETTINGS["risk_max_portfolio_delta"]))
    merged["risk_max_exposure_pct"] = float(merged.get("risk_max_exposure_pct", DEFAULT_SETTINGS["risk_max_exposure_pct"]))
    merged["risk_account_equity"] = float(merged.get("risk_account_equity", DEFAULT_SETTINGS["risk_account_equity"]))
    merged["nifty_expiry_weekday"] = merged.get("nifty_expiry_weekday", DEFAULT_SETTINGS["nifty_expiry_weekday"])
    merged["expiry_shift_if_holiday"] = bool(merged.get("expiry_shift_if_holiday", DEFAULT_SETTINGS["expiry_shift_if_holiday"]))
    merged["gap_entry_threshold"] = float(merged.get("gap_entry_threshold", DEFAULT_SETTINGS.get("gap_entry_threshold", 0.004)))
    merged["iv_floor_percentile"] = float(merged.get("iv_floor_percentile", DEFAULT_SETTINGS.get("iv_floor_percentile", 0.2)))
    merged["batman_long_offset"] = float(merged.get("batman_long_offset", DEFAULT_SETTINGS["batman_long_offset"]))
    merged["batman_short_offset"] = float(merged.get("batman_short_offset", DEFAULT_SETTINGS["batman_short_offset"]))
    merged["batman_qty_long"] = int(merged.get("batman_qty_long", DEFAULT_SETTINGS["batman_qty_long"]))
    merged["batman_qty_short"] = int(merged.get("batman_qty_short", DEFAULT_SETTINGS["batman_qty_short"]))
    merged["batman_weekly_hedge_enabled"] = bool(merged.get("batman_weekly_hedge_enabled", DEFAULT_SETTINGS["batman_weekly_hedge_enabled"]))
    merged["batman_weekly_hedge_price_cap"] = float(merged.get("batman_weekly_hedge_price_cap", DEFAULT_SETTINGS["batman_weekly_hedge_price_cap"]))
    merged["batman_weekly_hedge_min_distance"] = float(merged.get("batman_weekly_hedge_min_distance", DEFAULT_SETTINGS["batman_weekly_hedge_min_distance"]))
    holidays_raw = merged.get("holiday_list", DEFAULT_SETTINGS["holiday_list"])
    if isinstance(holidays_raw, str):
        holidays_raw = [h.strip() for h in holidays_raw.replace(",", "\n").splitlines() if h.strip()]
    merged["holiday_list"] = holidays_raw if isinstance(holidays_raw, list) else []
    playbook = merged.get("auto_playbook") or DEFAULT_SETTINGS["auto_playbook"].copy()
    merged["auto_playbook"] = {
        "hedge_on_delta": bool(playbook.get("hedge_on_delta", True)),
        "flatten_on_delta": bool(playbook.get("flatten_on_delta", False)),
        "flatten_on_exposure": bool(playbook.get("flatten_on_exposure", False)),
    }
    return merged


def _load_blotter() -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    df: Optional[pd.DataFrame] = None
    summary: Dict[str, Any] = {}

    _ensure_csv_exists(BLOTTER_CSV, STRAT_BLOTTER_FIELDS)
    if BLOTTER_CSV.exists():
        try:
            df = pd.read_csv(BLOTTER_CSV)
            if not df.empty and "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                df = df.sort_values("timestamp")
        except Exception as exc:
            st.warning(f"Failed to read paper blotter: {exc}")
            df = None

    if BLOTTER_SUMMARY.exists():
        try:
            summary = json.loads(BLOTTER_SUMMARY.read_text())
        except Exception:
            summary = {}

    return df, summary


def _load_feature_history(limit: int = 200) -> Optional[pd.DataFrame]:
    if not FEATURE_LOG_CSV.exists():
        return None
    try:
        df = pd.read_csv(FEATURE_LOG_CSV)
    except Exception:
        return None
    if df.empty:
        return df
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp")
    if limit and len(df.index) > limit:
        df = df.tail(limit)
    return df.reset_index(drop=True)


def _load_equity_history(limit: Optional[int] = None) -> Optional[pd.DataFrame]:
    if not EQUITY_LOG_CSV.exists():
        return None
    try:
        df = pd.read_csv(EQUITY_LOG_CSV)
    except Exception:
        return None
    if df.empty or "timestamp" not in df.columns:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    numeric_cols = [
        "available",
        "collateral",
        "utilized",
        "withdrawable",
        "gross_exposure",
        "net_exposure",
        "unrealized",
        "realized",
        "equity_estimate",
        "net_delta",
        "total_notional",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "equity_estimate" not in df.columns:
        df["equity_estimate"] = np.nan
    zero_series = pd.Series(0.0, index=df.index)
    available = df["available"] if "available" in df.columns else zero_series
    collateral = df["collateral"] if "collateral" in df.columns else zero_series
    utilized = df["utilized"] if "utilized" in df.columns else zero_series
    fallback_equity = (available.fillna(0.0) + collateral.fillna(0.0) + utilized.fillna(0.0))
    df["equity_estimate"] = df["equity_estimate"].fillna(fallback_equity)
    df = df.sort_values("timestamp")
    if limit and len(df.index) > limit:
        df = df.tail(limit)
    return df.reset_index(drop=True)


def _load_rolling_summary() -> Dict[str, Any]:
    if not ROLLING_SUMMARY_FILE.exists():
        return {}
    try:
        return json.loads(ROLLING_SUMMARY_FILE.read_text())
    except Exception:
        return {}


def _load_selector_summary() -> Dict[str, Any]:
    if not SELECTOR_SUMMARY_FILE.exists():
        return {}
    try:
        return json.loads(SELECTOR_SUMMARY_FILE.read_text())
    except Exception:
        return {}


def _render_rolling_data_quality(summary: Dict[str, Any]) -> None:
    lookback = summary.get("lookback_days", 0)
    st.markdown(f"#### Rolling Option Data Quality (last {lookback} days)")
    cols = st.columns(4)
    cols[0].metric("Weekdays Covered", summary.get("days_with_data", 0), f"/ {summary.get('weekdays_expected', 0)}")
    cols[1].metric("Missing Days", summary.get("days_missing", 0))
    cols[2].metric("Latest Day", summary.get("latest_day") or "—", summary.get("latest_file_updated", ""))
    cols[3].metric("Rows Ingested", f"{summary.get('total_rows', 0):,}", f"avg {summary.get('average_rows_per_day', 0):.0f}/day")

    coverage = summary.get("coverage_ratio") or {}
    if coverage:
        cov_df = pd.DataFrame(
            [{"field": field.upper(), "coverage": round(float(val) * 100, 2)} for field, val in coverage.items()]
        )
        cov_df.set_index("field", inplace=True)
        st.caption("Field coverage (% of rows with value)")
        st.bar_chart(cov_df)

    per_day = summary.get("per_day_counts") or {}
    if per_day:
        series = pd.Series(per_day, dtype="float64")
        series.index = pd.to_datetime(series.index, errors="coerce")
        series = series.dropna().sort_index().tail(90)
        if not series.empty:
            st.caption("Rows per trading day (last 90 sessions)")
            st.line_chart(series)

    selectors = summary.get("selector_counts") or []
    if selectors:
        sel_df = pd.DataFrame(selectors, columns=["Selector", "Rows"])
        st.caption("Top selectors")
        st.dataframe(sel_df.head(10), hide_index=True, width="stretch")

    if summary.get("missing_dates"):
        st.caption("Recent missing trading days")
        st.write(", ".join(summary["missing_dates"][:10]))

    if summary.get("errors"):
            with st.expander("Ingestion warnings"):
                for err in summary["errors"]:
                    st.write(f"- {err}")


def _render_selector_training_summary(summary: Dict[str, Any]) -> None:
    st.markdown("#### Strategy Selector Training")
    cols = st.columns(3)
    cols[0].metric("Samples", summary.get("samples_total", 0))
    cols[1].metric("Strategies", summary.get("strategies_trained", 0))
    cols[2].metric("Val Fraction", f"{summary.get('val_fraction', 0.0):.2f}")
    if summary.get("generated_at"):
        st.caption(f"Trained at {summary['generated_at']}")

    metrics = summary.get("metrics") or []
    if metrics:
        df = pd.DataFrame(metrics)
        display_cols = ["strategy", "samples", "train_rmse", "val_rmse", "val_r2"]
        df = df[display_cols]
        st.dataframe(df.sort_values("val_rmse"), width="stretch")


def _parse_context_blob(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {"raw": value}
    return {}


def _is_missing(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    return False


def _fmt_optional(val: Any, digits: Optional[int] = 2, placeholder: str = "—") -> str:
    if val is None:
        return placeholder
    if isinstance(val, float) and np.isnan(val):
        return placeholder
    if isinstance(val, (pd.Timestamp, datetime)):
        return val.strftime("%Y-%m-%d %H:%M:%S")
    try:
        if isinstance(val, (int, np.integer)):
            return f"{int(val):,}"
        if digits is None:
            return str(val)
        return f"{float(val):,.{digits}f}"
    except Exception:
        return str(val)


def _fmt_rupee(val: Optional[float], digits: int = 0) -> str:
    if val is None:
        return "—"
    try:
        value = float(val)
    except Exception:
        return "—"
    if np.isnan(value):
        return "—"
    return f"₹ {value:,.{digits}f}"


def _format_context_summary(ctx: Dict[str, Any], limit: int = 6) -> str:
    if not ctx:
        return ""
    items = []
    for key, value in ctx.items():
        if value in (None, "", [], {}):
            continue
        if isinstance(value, bool):
            items.append(f"{key}={'yes' if value else 'no'}")
        elif isinstance(value, (int, float)) and not (isinstance(value, float) and np.isnan(value)):
            if isinstance(value, float) and abs(value) < 0.01:
                items.append(f"{key}={value:.3f}")
            else:
                items.append(f"{key}={value}")
        else:
            items.append(f"{key}={value}")
        if len(items) >= limit:
            break
    return ", ".join(items)


def _load_risk_limits() -> Dict[str, float]:
    settings = _load_settings()
    return {
        "delta_cap": float(settings.get("risk_max_portfolio_delta", DEFAULT_SETTINGS["risk_max_portfolio_delta"])),
        "exposure_pct": float(settings.get("risk_max_exposure_pct", DEFAULT_SETTINGS["risk_max_exposure_pct"])),
        "account_equity": float(settings.get("risk_account_equity", DEFAULT_SETTINGS["risk_account_equity"])),
    }


def _compute_equity_change(df: pd.DataFrame, days: float) -> Optional[Tuple[float, Optional[float]]]:
    if df is None or df.empty:
        return None
    if "timestamp" not in df.columns or "equity_estimate" not in df.columns:
        return None
    if df["timestamp"].empty:
        return None
    window_start = df["timestamp"].iloc[-1] - pd.Timedelta(days=days)
    window = df.loc[df["timestamp"] >= window_start]
    if window.empty:
        return None
    start_val = float(window["equity_estimate"].iloc[0])
    end_val = float(window["equity_estimate"].iloc[-1])
    if np.isnan(start_val) or np.isnan(end_val) or start_val == 0:
        return None
    delta = end_val - start_val
    pct = (delta / start_val) * 100 if start_val else None
    return delta, pct


def _recent_risk_events(feature_df: Optional[pd.DataFrame], limit: int = 5) -> List[Dict[str, Any]]:
    if feature_df is None or feature_df.empty or "strategy" not in feature_df.columns:
        return []
    risk_rows = feature_df.loc[feature_df["strategy"] == "risk_event"].copy()
    if risk_rows.empty:
        return []
    if "timestamp" in risk_rows.columns:
        risk_rows = risk_rows.dropna(subset=["timestamp"]).sort_values("timestamp")
    risk_rows = risk_rows.tail(limit)
    events: List[Dict[str, Any]] = []
    for _, row in risk_rows.iterrows():
        ctx = _parse_context_blob(row.get("context"))
        events.append(
            {
                "timestamp": row.get("timestamp"),
                "label": ctx.get("label", "Risk Event"),
                "severity": ctx.get("severity", "info"),
                "message": ctx.get("message", ""),
                "details": {
                    k: v for k, v in ctx.items() if k not in {"label", "message", "severity"} and v not in (None, "")
                },
            }
        )
    return list(reversed(events))


def _risk_events_in_range(feature_df: Optional[pd.DataFrame], start: datetime, end: datetime) -> List[Dict[str, Any]]:
    events = _recent_risk_events(feature_df, limit=500)
    filtered: List[Dict[str, Any]] = []
    for ev in events:
        ts = ev.get("timestamp")
        if not ts:
            continue
        try:
            ts_dt = pd.to_datetime(ts)
        except Exception:
            continue
        if start <= ts_dt <= end:
            enriched = dict(ev)
            enriched["timestamp_dt"] = ts_dt
            details = enriched.get("details") or {}
            enriched["net_delta"] = details.get("net_delta")
            enriched["total_notional"] = details.get("total_notional")
            enriched["block_reason"] = details.get("reason") or enriched.get("label")
            filtered.append(enriched)
    return filtered


def _predict_delta_breach(equity_df: Optional[pd.DataFrame], risk_limits: Dict[str, float]) -> Optional[Dict[str, Any]]:
    if equity_df is None or equity_df.empty or "net_delta" not in equity_df.columns:
        return None
    df = equity_df.dropna(subset=["net_delta"]).tail(30)
    if len(df) < 5:
        return None
    x = list(range(len(df)))
    y = df["net_delta"].astype(float)
    try:
        slope, intercept = np.polyfit(x, y, 1)
    except Exception:
        return None
    delta_cap = abs(float(risk_limits.get("delta_cap", 0.25)))
    if slope == 0:
        return None
    last_idx = len(df) - 1
    remaining = None
    try:
        target = delta_cap if slope > 0 else -delta_cap
        steps = (target - (slope * last_idx + intercept)) / slope
        if steps > 0:
            remaining = steps
    except Exception:
        remaining = None
    if remaining is None:
        return None
    forecast_minutes = remaining * float((st.session_state.get("refresh_sec") or 5)) / 60.0
    return {
        "slope": slope,
        "forecast_minutes": forecast_minutes,
        "target": target,
        "current": float(y.iloc[-1]),
    }


def _predict_notional_breach(equity_df: Optional[pd.DataFrame], risk_limits: Dict[str, float]) -> Optional[Dict[str, Any]]:
    if equity_df is None or equity_df.empty or "total_notional" not in equity_df.columns:
        return None
    df = equity_df.dropna(subset=["total_notional"]).tail(30)
    if len(df) < 5:
        return None
    series = df["total_notional"].astype(float)
    x = list(range(len(series)))
    try:
        slope, intercept = np.polyfit(x, series, 1)
    except Exception:
        return None
    cap = float(risk_limits.get("account_equity", 0.0)) * float(risk_limits.get("exposure_pct", 0.0))
    if cap <= 0 or slope == 0:
        return None
    last_idx = len(series) - 1
    steps = (cap - (slope * last_idx + intercept)) / slope
    if steps <= 0:
        return None
    forecast_minutes = steps * float((st.session_state.get("refresh_sec") or 5)) / 60.0
    return {
        "slope": slope,
        "forecast_minutes": forecast_minutes,
        "target": cap,
        "current": float(series.iloc[-1]),
    }


def _render_risk_alerts(feature_df: Optional[pd.DataFrame]) -> None:
    events = _recent_risk_events(feature_df, limit=5)
    st.caption("Recent Risk Alerts")
    if not events:
        st.write("None")
        return
    severity_colors = {
        "INFO": "#2563eb",
        "WARNING": "#b45309",
        "ERROR": "#b91c1c",
    }
    for ev in events:
        severity = str(ev["severity"]).upper()
        color = severity_colors.get(severity, "#374151")
        ts = ev.get("timestamp")
        header = f"{_fmt_optional(ts)} · {ev.get('label', 'Risk Event')} ({severity})"
        lines = [ev.get("message") or "—"]
        details = ev.get("details") or {}
        if details:
            detail_text = ", ".join(f"{k}={v}" for k, v in details.items())
            lines.append(detail_text)
        st.markdown(
            f"<div style='border-left:4px solid {color}; padding-left:8px; margin-bottom:6px;'>"
            f"<strong>{header}</strong><br>{'<br>'.join(lines)}</div>",
            unsafe_allow_html=True,
        )


def _load_playbook_history(limit: int = 10) -> List[Dict[str, Any]]:
    if not PLAYBOOK_HISTORY_FILE.exists():
        return []
    entries: List[Dict[str, Any]] = []
    try:
        with PLAYBOOK_HISTORY_FILE.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()[-limit:]
        for line in lines:
            try:
                entries.append(json.loads(line))
            except Exception:
                continue
    except Exception:
        return []
    return list(reversed(entries))


def _render_entry_block_status() -> None:
    st.caption("Entry Gate Status")
    if not ENTRY_BLOCK_STATE.exists():
        st.write("Entries enabled")
        return
    try:
        data = json.loads(ENTRY_BLOCK_STATE.read_text())
    except Exception:
        st.write("Entries enabled")
        return
    entry_enabled = bool(data.get("entry_enabled", True))
    label = data.get("label", "block").replace("_", " ").title()
    message = data.get("message", "")
    ts = data.get("timestamp")
    severity = str(data.get("severity", "info")).upper()
    color = "#22c55e" if entry_enabled else "#b91c1c"
    details = data.get("context") or {}
    detail_text = ", ".join(f"{k}={v}" for k, v in details.items() if v not in (None, ""))
    st.markdown(
        f"<div style='border-left:4px solid {color}; padding-left:8px;'>"
        f"<strong>{label} ({severity})</strong><br>"
        f"{_fmt_optional(ts)} · {message or '—'}"
        f"{('<br>'+detail_text) if detail_text else ''}"
        f"</div>",
        unsafe_allow_html=True,
    )
    action_cols = st.columns(2)
    with action_cols[0]:
        if st.button("Acknowledge Risk Event", key="ack_risk"):
            _request_risk_action("ack_risk", reason=label)
            st.success("Acknowledged risk event.")
    with action_cols[1]:
        if st.button("Snapshot State", key="snapshot_state"):
            _request_risk_action("snapshot_state")
        st.success("Snapshot requested.")
    action_cols2 = st.columns(2)
    with action_cols2[0]:
        if st.button("Force Hedge Refresh", key="force_hedge_refresh"):
            _request_risk_action("force_hedge_refresh")
            st.success("Hedge refresh signal sent.")
    with action_cols2[1]:
        if st.button("Flatten Positions", key="flatten_positions"):
            _request_risk_action("flatten_positions")
            st.warning("Flatten request sent.")
    if ACTION_STATUS_FILE.exists():
        try:
            status = json.loads(ACTION_STATUS_FILE.read_text())
        except Exception:
            status = None
        if status:
            st.caption(
                f"Latest manual action: {status.get('action')} ({status.get('status')})"
                f" · {_fmt_optional(status.get('timestamp'))}"
            )
            msg = status.get("message")
            if msg:
                st.write(msg)
def _fmt_size(num: Optional[int]) -> str:
    if num is None:
        return "—"
    step = 1024.0
    value = float(num)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < step:
            return f"{value:,.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= step
    return f"{value:,.1f} PB"


def _collect_file_info(path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "path": path,
        "exists": path.exists(),
        "size": None,
        "updated": None,
    }
    try:
        if path.exists():
            stat = path.stat()
            info["size"] = stat.st_size
            info["updated"] = datetime.fromtimestamp(stat.st_mtime)
    except Exception:
        pass
    return info


def _age_string(ts: Optional[datetime]) -> str:
    if ts is None:
        return "—"
    delta = datetime.now() - ts
    if delta.total_seconds() < 60:
        return f"{int(delta.total_seconds())}s ago"
    if delta.total_seconds() < 3600:
        return f"{int(delta.total_seconds()//60)}m ago"
    if delta.total_seconds() < 86400:
        return f"{int(delta.total_seconds()//3600)}h ago"
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _read_tail(path: Path, lines: int = 50) -> List[str]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            data = fh.readlines()
        return data[-lines:]
    except Exception:
        return []


def _clear_stale_pid_file() -> None:
    """Remove PID file and reset session flags."""
    try:
        if PID_FILE.exists():
            PID_FILE.unlink()
    except Exception:
        pass


def _seconds_since(ts: Optional[datetime]) -> Optional[float]:
    if ts is None:
        return None
    now = datetime.now(ts.tzinfo) if ts.tzinfo else datetime.now()
    return (now - ts).total_seconds()
    ss = st.session_state
    ss["agent_pid"] = None
    ss["agent_running"] = False
    ss.pop("agent_started_mode", None)

def _mark_creds_dirty() -> None:
    """Flag credentials as needing verification when the user edits them."""
    ss = st.session_state
    ss["creds_verified"] = False
    ss["dw"] = None
    ss["funds"] = {"available": None, "collateral": None, "utilized": None, "withdrawable": None}
    ss["last_funds_at"] = None
    try:
        if CREDS_FILE.exists():
            CREDS_FILE.unlink()
    except Exception:
        pass

def _pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False
    return True

def _sync_agent_state_from_pidfile() -> None:
    """Reload agent status from pid file when the page/session refreshes."""
    ss = st.session_state
    if ss.get("agent_running") and ss.get("agent_pid"):
        return
    if not PID_FILE.exists():
        return
    try:
        data = json.loads(PID_FILE.read_text())
        pid = int(data.get("pid"))
    except Exception:
        return
    if not _pid_alive(pid):
        try:
            if PID_FILE.exists():
                PID_FILE.unlink()
        except Exception:
            pass
        return
    ss["agent_pid"] = pid
    ss["agent_running"] = True
    # optional background helper pids (monitor, watcher)
    for key in ("monitor_pid", "watcher_pid"):
        try:
            val = int(data.get(key)) if data.get(key) is not None else None
            if val:
                ss[key] = val
        except Exception:
            pass
    if data.get("trade_mode"):
        ss["trade_mode"] = data["trade_mode"]
        ss["agent_started_mode"] = data["trade_mode"]

# =============================================================================
# Wrapper helpers
# =============================================================================
def _new_dw() -> Optional[Any]:
    """Create a DhanWrapper instance from sidebar creds."""
    if _import_err:
        st.error(f"Failed to import DhanWrapper: {_import_err}")
        return None
    cid = (st.session_state.get("client_id") or "").strip()
    tok = (st.session_state.get("access_token") or "").strip()
    if not cid or not tok:
        st.error("Enter Client ID and Access Token.")
        return None
    try:
        return DhanWrapper(dhan_client_id=cid, access_token=tok)
    except Exception as e:
        st.error(f"Failed to create Dhan client: {e}")
        return None

def _drain_ltp_queue_into_state() -> None:
    """Transfer any pending LTP updates from background thread to session_state."""
    ss = st.session_state
    while True:
        try:
            ltp, ts = LTP_QUEUE.get_nowait()
        except Empty:
            break
        ss["ltp_value"] = ltp
        ss["ltp_ts"] = ts

def _post_dw_setup(dw: Any) -> None:
    """Fetch funds, remember timestamp, and ensure NIFTY poller runs."""
    ss = st.session_state

    try:
        ss["funds"] = dw.get_funds()
        ss["last_funds_at"] = datetime.now()
    except Exception as e:
        st.warning(f"Funds fetch failed: {e}")

    try:
        seg, secid = "IDX_I", 13

        def _on_ltp_threadsafe(ltp: float, ts: datetime) -> None:
            try:
                LTP_QUEUE.put_nowait((ltp, ts))
            except Exception:
                pass

        dw.start_ltp_poller(
            seg,
            secid,
            max(1, int(ss.get("refresh_sec", 5))),
            _on_ltp_threadsafe,
            poll_when_closed=True,
        )
        # Prime once so the tile shows data immediately after verification
        try:
            ltp_once = dw.get_ltp_once(seg, secid)
            if ltp_once is not None:
                ss["ltp_value"] = float(ltp_once)
                ss["ltp_ts"] = datetime.now()
        except Exception:
            pass
    except Exception as e:
        st.info(f"No LTP from /marketfeed/ltp. Market may be closed or permission missing. ({e})")


def _verify_and_remember_creds() -> bool:
    """Verify creds, fetch funds once, and start/refresh NIFTY LTP poller."""
    ss = st.session_state
    os.environ["DHAN_CLIENT_ID"] = (ss.get("client_id") or "").strip()
    os.environ["DHAN_ACCESS_TOKEN"] = (ss.get("access_token") or "").strip()

    dw = _new_dw()
    if not dw:
        ss["creds_verified"] = False
        ss["dw"] = None
        return False

    ss["dw"] = dw
    ss["creds_verified"] = True
    try:
        CREDS_FILE.write_text(json.dumps({
            "client_id": ss.get("client_id", ""),
            "access_token": ss.get("access_token", ""),
            "creds_verified": True,
            "verified_at": datetime.now().isoformat(),
        }, indent=2))
    except Exception:
        pass
    _post_dw_setup(dw)

    return True

# =============================================================================
# Agent process control
# =============================================================================
def _agent_env() -> dict:
    base = os.environ.copy()
    base["DHAN_CLIENT_ID"] = st.session_state.get("client_id", "")
    base["DHAN_ACCESS_TOKEN"] = st.session_state.get("access_token", "")
    base["TRADE_MODE"] = st.session_state.get("trade_mode", "live")
    base["AGENT_SETTINGS_JSON"] = str(SETTINGS_JSON)
    base["PYTHONUNBUFFERED"] = "1"
    base["PYTHONPATH"] = os.pathsep.join([str(ROOT)] + ([base["PYTHONPATH"]] if base.get("PYTHONPATH") else []))
    settings = _load_settings()
    base["MAX_STRANGLES"] = str(settings.get("max_legs", 1))
    base["NIFTY_LOT"] = str(settings.get("nifty_lot_size", settings.get("lot_size", 75)))
    # convert % -> decimal
    base["SL_PCT"] = str(float(settings.get("leg_sl_pct", 2.5)) / 100.0)
    base["TP_PCT"] = str(float(settings.get("profit_pct", 2.25)) / 100.0)
    base["BLOTTER_PATH"] = str(STATE_DIR / "trade_blotter.csv")
    base["BLOTTER_SUMMARY_PATH"] = str(STATE_DIR / "trade_blotter_summary.json")
    base["FEATURE_LOG_PATH"] = str(STATE_DIR / "feature_history.csv")
    hedge_enabled = bool(settings.get("hedge_enabled", True))
    base["HEDGE_ENABLED"] = "1" if hedge_enabled else "0"
    base["HEDGE_DELTA_BREACH"] = str(settings.get("hedge_delta_breach", 0.30))
    base["HEDGE_PREMIUM_HARD_X"] = str(settings.get("hedge_premium_hard_x", 2.0))
    base["HEDGE_ROLL_DISTANCE"] = str(settings.get("hedge_roll_distance", 150.0))
    base["HEDGE_DELTA_MAX"] = str(settings.get("hedge_delta_max", 0.12))
    base["HEDGE_PRICE_MAX"] = str(settings.get("hedge_price_max", 35.0))
    base["HEDGE_SALVAGE_WING_LTP"] = str(settings.get("hedge_salvage_wing_ltp", 5.0))
    # legacy exports to support older start_agent builds
    base["BATMAN_ENABLED"] = base["HEDGE_ENABLED"]
    base["BATMAN_DELTA_BREACH"] = base["HEDGE_DELTA_BREACH"]
    base["BATMAN_PREMIUM_HARD_X"] = base["HEDGE_PREMIUM_HARD_X"]
    base["BATMAN_ROLL_DISTANCE"] = base["HEDGE_ROLL_DISTANCE"]
    base["BATMAN_HEDGE_DELTA_MAX"] = base["HEDGE_DELTA_MAX"]
    base["BATMAN_HEDGE_PRICE_MAX"] = base["HEDGE_PRICE_MAX"]
    base["BATMAN_SALVAGE_WING_LTP"] = base["HEDGE_SALVAGE_WING_LTP"]
    base["BATMAN_LONG_OFFSET"] = str(settings.get("batman_long_offset", 300.0))
    base["BATMAN_SHORT_OFFSET"] = str(settings.get("batman_short_offset", 500.0))
    base["BATMAN_QTY_LONG"] = str(settings.get("batman_qty_long", 1))
    base["BATMAN_QTY_SHORT"] = str(settings.get("batman_qty_short", 3))
    base["BATMAN_WEEKLY_HEDGE_ENABLED"] = "1" if settings.get("batman_weekly_hedge_enabled", True) else "0"
    base["BATMAN_WEEKLY_HEDGE_PRICE_CAP"] = str(settings.get("batman_weekly_hedge_price_cap", 12.0))
    base["BATMAN_WEEKLY_HEDGE_MIN_DISTANCE"] = str(settings.get("batman_weekly_hedge_min_distance", 300.0))
    base["STRATEGY_MODEL_PATH"] = str(STATE_DIR / "strategy_selector_model.json")
    base["STRATEGY_FILE"] = st.session_state.get("strategy_file", "monthly_strangle_with_weekly_hedge.py")
    # Weekly order params
    if base["STRATEGY_FILE"] == "weekly_theta_strangle.py":
        base["WEEKLY_ORDER_TYPE"] = st.session_state.get("weekly_order_type", "MARKET")
        try:
            slip = float(st.session_state.get("weekly_slip_pct", 0.2))
        except Exception:
            slip = 0.2
        base["WEEKLY_SLIPPAGE_PCT"] = str(slip / 100.0)
        if st.session_state.get("weekly_hedge_enabled"):
            base["WEEKLY_HEDGE_ENABLED"] = "1"
    return base
    return base

def start_agent() -> None:
    if st.session_state.get("agent_running"):
        return
    if not AGENT_ENTRY.exists():
        st.error(f"start_agent.py not found at: {AGENT_ENTRY}")
        return
    try:
        AGENT_LOG.parent.mkdir(parents=True, exist_ok=True)
        AGENT_LOG.touch(exist_ok=True)
        env = _agent_env()
        # also propagate lot_size to Batman daemon
        try:
            lot_size = int(st.session_state.get("lot_size", 75))
        except Exception:
            lot_size = 75
        env["BATMAN_LOT_SIZE"] = str(lot_size)
        def _spawn(path: Path, log_path: Path) -> Optional[int]:
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.touch(exist_ok=True)
                proc = subprocess.Popen(
                    [PYTHON, str(path)],
                    cwd=str(path.parent),
                    env=env,
                    stdout=open(log_path, "a"),
                    stderr=open(log_path, "a"),
                    start_new_session=True,
                )
                return proc.pid
            except Exception as exc:
                st.warning(f"Failed to start {path.name}: {exc}")
                return None

        strategy_file = st.session_state.get("strategy_file")
        if strategy_file == "batman":
            agent_pid = _spawn(ROOT / "data_engine" / "market_ai" / "scripts" / "ai_batman_daemon.py", BATMAN_LOG)
            monitor_pid = None
            watcher_pid = None
        else:
            agent_pid = _spawn(AGENT_ENTRY, AGENT_LOG)
            monitor_pid = _spawn(ROOT / "data_engine" / "market_ai" / "scripts" / "run_live_monitor.py", LOG_DIR / "live_monitor.log")
            watcher_pid = _spawn(ROOT / "data_engine" / "market_ai" / "scripts" / "watch_live_health.py", LOG_DIR / "live_watch.log")
        # one-shot event fetch on start to seed recent events
        try:
            evt_script = ROOT / "data_engine" / "market_ai" / "scripts" / "fetch_public_events.py"
            if evt_script.exists():
                subprocess.run([PYTHON, str(evt_script)], cwd=str(evt_script.parent), env=env, check=False)
        except Exception:
            pass
        if not agent_pid:
            st.error("Agent failed to start.")
            return
        st.session_state["agent_pid"] = agent_pid
        if strategy_file != "batman" and monitor_pid:
            st.session_state["monitor_pid"] = monitor_pid
        if strategy_file != "batman" and watcher_pid:
            st.session_state["watcher_pid"] = watcher_pid
        st.session_state["agent_running"] = True
        st.session_state["agent_started_mode"] = st.session_state.get("trade_mode", "live")
        try:
            PID_FILE.write_text(json.dumps({
                "pid": agent_pid,
                "monitor_pid": monitor_pid,
                "watcher_pid": watcher_pid,
                "trade_mode": st.session_state.get("trade_mode", "live"),
                "started_at": datetime.now().isoformat(),
            }, indent=2))
        except Exception:
            pass
    except Exception as e:
        st.error(f"Failed to start agent: {e}")

def stop_agent() -> None:
    pid = st.session_state.get("agent_pid")
    if not pid:
        st.session_state["agent_running"] = False
        return
    try:
        if sys.platform.startswith("win"):
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=False)
        else:
            os.kill(pid, 15)  # SIGTERM
        # stop background helpers if running
        for key in ("monitor_pid", "watcher_pid"):
            val = st.session_state.get(key)
            if val:
                try:
                    os.kill(val, 15)
                except Exception:
                    pass
    except Exception:
        pass
    finally:
        st.session_state["agent_pid"] = None
        st.session_state["agent_running"] = False
        st.session_state.pop("monitor_pid", None)
        st.session_state.pop("watcher_pid", None)
        st.session_state.pop("agent_started_mode", None)
        try:
            if PID_FILE.exists():
                PID_FILE.unlink()
        except Exception:
            pass

# =============================================================================
# Trade tab widgets
# =============================================================================
def _funds_cards(funds: Dict[str, Any]) -> None:
    cols = st.columns(4)
    labels = ["Available", "Collateral", "Utilized", "Withdrawable"]
    keys   = ["available", "collateral", "utilized", "withdrawable"]
    for col, label, key in zip(cols, labels, keys):
        with col:
            st.caption(label)
            val = funds.get(key)
            st.write("—" if val is None else f"₹ {val:,.0f}")

def _nifty_tile() -> None:
    _auto_refresh()
    _drain_ltp_queue_into_state()
    st.markdown("### NIFTY Spot Price")
    dw = st.session_state.get("dw")
    if not st.session_state.get("creds_verified") or not dw:
        st.info("Enter your credentials and click **Save & Verify** to start the LTP feed.")
        return
    v = st.session_state.get("ltp_value")
    ts = st.session_state.get("ltp_ts")

    ts_age = _seconds_since(ts)
    if dw and (v is None or (ts_age is not None and ts_age > max(5, int(st.session_state.get("refresh_sec", 5)) * 2))):
        try:
            ltp = dw.get_ltp_once("IDX_I", 13)
            if ltp is not None:
                st.session_state["ltp_value"] = ltp
                st.session_state["ltp_ts"] = datetime.now()
                v, ts = ltp, st.session_state["ltp_ts"]
        except Exception:
            pass

    if v is None:
        st.info("No LTP from /marketfeed/ltp.")
    else:
        st.markdown(f"**{v:,.2f}**")
        if ts:
            st.caption(f"Last update: {ts:%H:%M:%S}")

# =============================================================================
# Positions tab (open + today's closed)
# =============================================================================
def _positions_tab(dw) -> None:

    st.markdown("### Positions")

    if not st.session_state.get("creds_verified") or not dw:
        st.info("Save & verify credentials to view live positions.")
        return

    try:
        rows = dw.get_positions_live_with_ltp()  # type: ignore[attr-defined]
    except Exception as e:
        st.error(f"Failed to fetch positions: {e}")
        return

    def _raw(r: Dict[str, Any]) -> Dict[str, Any]:
        val = r.get("_raw")
        return val if isinstance(val, dict) else {}

    # ---- helpers over normalized rows (with raw fallback) ----
    def S(r, k, default=""):
        v = r.get(k)
        if v is None:
            v = _raw(r).get(k)  # fallback
        return v if v is not None else default

    def F(r, k, default=0.0):
        try:
            v = r.get(k)
            if v is None:
                v = _raw(r).get(k)
            return float(v) if v is not None else float(default)
        except Exception:
            return float(default)

    def I(r, k, default=0):
        try:
            v = r.get(k)
            if v is None:
                v = _raw(r).get(k)
            return int(float(v)) if v is not None else int(default)
        except Exception:
            return int(default)

    def _pt(r: Dict[str, Any]) -> str:
        return str(r.get("position_type") or r.get("side") or S(r, "positionType")).upper()

    def _net_qty(r: Dict[str, Any]) -> int:
        if r.get("qty") is not None:
            return int(r.get("qty"))
        return I(r, "netQty", 0)

    def _security_id(r: Dict[str, Any]) -> Optional[int]:
        raw = r.get("security_id") or r.get("securityId") or r.get("id")
        if raw is None:
            raw = _raw(r).get("securityId")
        try:
            return int(str(raw)) if raw is not None else None
        except Exception:
            return None

    # ================= OPEN POSITIONS (netQty != 0) =================
    open_src = [r for r in rows if _net_qty(r) != 0 and _pt(r) in ("LONG", "SHORT")]
    ltp_map: Dict[tuple[str, int], Optional[float]] = {}
    ltp_pairs: List[tuple[str, int]] = []
    seen_pairs: set[tuple[str, int]] = set()
    for r in open_src:
        seg = r.get("exchange_seg") or S(r, "exchangeSegment")
        sid = _security_id(r)
        if isinstance(seg, str) and sid is not None:
            key = (seg, sid)
            if key not in seen_pairs:
                seen_pairs.add(key)
                ltp_pairs.append(key)
    if ltp_pairs:
        try:
            ltp_map = dw.get_ltp_bulk(ltp_pairs)  # type: ignore[attr-defined]
        except Exception as exc:
            st.warning(f"Could not fetch live LTPs: {exc}")

    # ---- Payoff Analyzer Button ----
    with st.container():
        col1, col2 = st.columns([1, 1])
        with col2:
            if st.button("🔍 Analyze Payoff", key="analyze_payoff", help="Open payoff chart based on open positions"):
                st.session_state["payoff_open"] = True

    payoff_open = bool(st.session_state.get("payoff_open", False))
    if payoff_open:
        with st.expander("Payoff Chart (Live Positions)", expanded=True):
            c_close, _ = st.columns([1, 5])
            if c_close.button("Close analyzer", key="close_payoff"):
                st.session_state["payoff_open"] = False
            else:
                _render_payoff_analyzer(open_src, dw)

    if open_src:
        mapped_open = []
        total_unreal = 0.0
        for r in open_src:
            typ = _pt(r)
            sym = r.get("symbol") or S(r, "tradingSymbol")
            prod = r.get("product") or S(r, "productType")
            cost_price = F(r, "cost_price", F(r, "avg_price", 0.0))
            buy_avg = r.get("buy_avg")
            if buy_avg is None:
                buy_avg = F(r, "buyAvg", None)
            sell_avg = r.get("sell_avg")
            if sell_avg is None:
                sell_avg = F(r, "sellAvg", None)
            unreal = r.get("unrealized_profit")
            if unreal is None:
                unreal = F(r, "unrealizedProfit", 0.0)
            seg = r.get("exchange_seg") or S(r, "exchangeSegment")
            sid = _security_id(r)
            live_ltp = ltp_map.get((seg, sid)) if seg and sid is not None else None
            if live_ltp is None and seg and sid is not None:
                try:
                    live_single = dw.get_ltp_once(seg, sid)  # type: ignore[attr-defined]
                    if live_single is not None:
                        live_ltp = live_single
                except Exception:
                    pass

            avg_disp = sell_avg if typ == "SHORT" else buy_avg
            if avg_disp is None:
                avg_disp = cost_price
            qty_val = abs(_net_qty(r))
            if typ == "SHORT":
                qty_val = I(r, "sellQty", qty_val)
            else:
                qty_val = I(r, "buyQty", qty_val)
            qty_val = abs(int(qty_val or 0))

            ltp_val = live_ltp if live_ltp is not None else r.get("ltp")
            if ltp_val is None:
                ltp_val = cost_price

            mapped_open.append({
                "B/S": "S" if typ == "SHORT" else "B",
                "Name": sym,
                "Product": prod,
                "Qty": qty_val,
                "Avg Price": avg_disp,
                "LTP": ltp_val,
                "P&L": unreal,
            })
            total_unreal += unreal

        import pandas as pd
        df_open = pd.DataFrame(mapped_open)
        # Qty as nullable Int64 so Streamlit/Arrow is happy
        try:
            df_open["Qty"] = df_open["Qty"].astype("Int64")
        except Exception:
            pass

        def _pnl_style_col(col):
            return ["color: #d32f2f" if v < 0 else ("color: #2e7d32" if v > 0 else "") for v in col]
        st.markdown("#### Open Positions")
        styler_open = (
            df_open.style
            .format({"Avg Price": "₹ {:.2f}", "LTP": "₹ {:.2f}", "P&L": "₹ {:.2f}"})
            .apply(_pnl_style_col, subset=["P&L"], axis=0)
        )
        st.dataframe(styler_open, hide_index=True, width="stretch")
        total_color = "#2e7d32" if total_unreal > 0 else ("#d32f2f" if total_unreal < 0 else "inherit")
        st.markdown(f"<span style='color:{total_color};'>Total Unrealized P&L: ₹ {total_unreal:,.2f}</span>", unsafe_allow_html=True)
    else:
        st.caption("No open positions.")

    # ================= CLOSED (today) =================
    # Treat as closed if positionType == CLOSED or netQty == 0.
    # Show if there was intraday activity or realized PnL (like Dhan UI).
    closed_candidates = [r for r in rows if _pt(r) == "CLOSED" or _net_qty(r) == 0]
    closed_today = [
        r for r in closed_candidates
        if I(r, "dayBuyQty", 0) > 0 or I(r, "daySellQty", 0) > 0 or F(r, "realizedProfit", 0.0) != 0.0
    ]

    if closed_today:
        # de-dupe by tradingSymbol (keep the last occurrence)
        seen = {}
        for r in closed_today:
            key = S(r, "tradingSymbol").strip()
            if key:
                seen[key] = r
        uniq = list(seen.values()) if seen else closed_today

        mapped_closed = []
        total_realized = 0.0
        for r in uniq:
            realized = F(r, "realizedProfit", 0.0)
            total_realized += realized
            mapped_closed.append({
                "Symbol":   S(r, "tradingSymbol"),
                "Product":  S(r, "productType"),
                "Qty":      max(I(r, "buyQty", 0), I(r, "sellQty", 0)),  # show traded size
                "Buy Avg":  F(r, "buyAvg", 0.0),
                "Sell Avg": F(r, "sellAvg", 0.0),
                "LTP":      F(r, "costPrice", 0.0),
                "P&L":      realized,
            })

        import pandas as pd
        df_closed = pd.DataFrame(mapped_closed)
        try:
            df_closed["Qty"] = df_closed["Qty"].astype("Int64")
        except Exception:
            pass

        def _pnl_style_col_closed(col):
            return ["color: #d32f2f" if v < 0 else ("color: #2e7d32" if v > 0 else "") for v in col]
        st.markdown("#### Today’s Closed Trades")
        styler_closed = (
            df_closed.style
            .format({"Buy Avg": "₹ {:.2f}", "Sell Avg": "₹ {:.2f}", "LTP": "₹ {:.2f}", "P&L": "₹ {:.2f}"})
            .apply(_pnl_style_col_closed, subset=["P&L"], axis=0)
        )
        st.dataframe(styler_closed, hide_index=True, width="stretch")
        closed_color = "#2e7d32" if total_realized > 0 else ("#d32f2f" if total_realized < 0 else "inherit")
        st.markdown(f"<span style='color:{closed_color};'>Realized P&L (today): ₹ {total_realized:,.2f}</span>", unsafe_allow_html=True)
    else:
        st.caption("No closed trades today.")

# ---------- Payoff & Greeks Analyzer ----------
def _collect_open_option_legs(rows):
    legs = []
    for r in rows or []:
        try:
            raw = r.get("_raw") or {}
            pt = str(r.get("positionType") or raw.get("positionType") or "").upper()
            if pt not in ("LONG", "SHORT"):
                continue
            nqty = r.get("qty")
            if nqty is None:
                nqty = r.get("net_qty") or r.get("netQty") or raw.get("netQty")
            nqty = int(float(nqty or 0))
            if nqty == 0:
                continue
            side = pt
            qty = abs(nqty)
            strike = r.get("strike") or r.get("drvStrikePrice") or raw.get("drvStrikePrice")
            strike = float(strike or 0)
            otype = str(r.get("type") or r.get("drvOptionType") or raw.get("drvOptionType") or "").upper()
            if otype not in ("CALL", "PUT") or strike == 0:
                continue
            premium = r.get("sell_avg") if side == "SHORT" else r.get("buy_avg")
            if premium is None:
                premium = r.get("sellAvg") if side == "SHORT" else r.get("buyAvg")
            if premium is None:
                premium = raw.get("sellAvg") if side == "SHORT" else raw.get("buyAvg")
            premium = float(premium or 0.0)
            ltp = r.get("ltp") or r.get("cost_price") or r.get("costPrice") or raw.get("costPrice") or 0.0
            ltp = float(ltp)
            expiry = str(r.get("expiry") or r.get("drvExpiryDate") or raw.get("drvExpiryDate") or "")
            legs.append({
                "symbol": r.get("symbol") or raw.get("tradingSymbol"),
                "type": otype,          # CALL / PUT
                "side": side,           # LONG / SHORT
                "qty": qty,
                "strike": strike,
                "premium": premium,
                "ltp": ltp,
                "expiry": expiry,
            })
        except Exception:
            continue
    return legs


def _fallback_plan_from_positions() -> Optional[pd.DataFrame]:
    dw = st.session_state.get("dw")
    if not dw or not st.session_state.get("creds_verified"):
        return None
    try:
        rows = dw.get_positions_live_with_ltp()
    except Exception:
        return None
    legs = _collect_open_option_legs(rows)
    if not legs:
        return None
    combos: List[Dict[str, Any]] = []
    short_calls = [leg for leg in legs if leg["side"] == "SHORT" and leg["type"] == "CALL"]
    short_puts = [leg for leg in legs if leg["side"] == "SHORT" and leg["type"] == "PUT"]
    expiries = sorted({leg["expiry"] for leg in short_calls + short_puts if leg.get("expiry")})
    for expiry in expiries:
        ce_candidates = [leg for leg in short_calls if leg["expiry"] == expiry]
        pe_candidates = [leg for leg in short_puts if leg["expiry"] == expiry]
        if not ce_candidates or not pe_candidates:
            continue
        ce_leg = max(ce_candidates, key=lambda l: l["strike"])
        pe_leg = min(pe_candidates, key=lambda l: l["strike"])
        qty = min(ce_leg["qty"], pe_leg["qty"])
        net_credit = qty * ((ce_leg.get("premium") or 0.0) + (pe_leg.get("premium") or 0.0))
        combos.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "strategy": "live_positions",
            "expiry": expiry,
            "ce_strike": ce_leg["strike"],
            "pe_strike": pe_leg["strike"],
            "ce_ltp": ce_leg.get("ltp"),
            "pe_ltp": pe_leg.get("ltp"),
            "ce_delta": None,
            "pe_delta": None,
            "net_credit": net_credit,
            "context": json.dumps({"source": "open_positions", "qty": qty}, default=str),
        })
    if not combos:
        return None
    return pd.DataFrame(combos)


def _fallback_blotter_from_positions() -> Optional[pd.DataFrame]:
    dw = st.session_state.get("dw")
    if not dw or not st.session_state.get("creds_verified"):
        return None
    try:
        rows = dw.get_positions_live_with_ltp()
    except Exception:
        return None
    events: List[Dict[str, Any]] = []
    for row in rows:
        qty = row.get("qty")
        if not qty:
            continue
        side = "BUY" if qty > 0 else "SELL"
        strike = (row.get("_raw") or {}).get("drvStrikePrice")
        events.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "trade_mode": st.session_state.get("trade_mode", "live"),
            "side": side,
            "quantity": abs(int(qty)),
            "price": row.get("avg_price"),
            "strike": strike or "",
            "notes": row.get("symbol") or "",
        })
    if not events:
        return None
    return pd.DataFrame(events)

# --- Colours roughly matching Dhan ---
_DHAN_COLORS = {
    "today": "#2e7d32",   # green
    "target": "#1976d2",  # blue
    "expiry": "#d32f2f",  # red
    "breakeven": "#9e9e9e"
}

def _to_years(date_str: str) -> float:
    """Time to expiry in years from ISO date, fallback ~20 trading days."""
    try:
        dt = datetime.fromisoformat(date_str)
    except Exception:
        return 20/252.0
    days = max(0.0, (dt - datetime.now()).days)
    return days/252.0

def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """
    Standard normal CDF for array-like x.

    Uses math.erf (vectorized) to avoid relying on np.erf (which may not exist in some NumPy builds).
    """
    x = np.asarray(x, dtype=float)
    v_erf = np.vectorize(erf)
    return 0.5 * (1.0 + v_erf(x / np.sqrt(2.0)))

def _bs_price_greeks(S, K, r, q, sigma, T, is_call):
    """
    Black–Scholes price + greeks (Delta, Gamma, Theta/day, Vega per 1%).
    """
    if sigma <= 0 or T <= 0:
        # expiry-like fallback
        if is_call:
            price = max(0.0, S - K)
            delta = 1.0 if S > K else 0.0
        else:
            price = max(0.0, K - S)
            delta = -1.0 if S < K else 0.0
        return {"price": price, "delta": delta, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    Nd1 = _norm_cdf(d1)
    Nd2 = _norm_cdf(d2)
    n_pdf = (1.0/np.sqrt(2*np.pi)) * np.exp(-0.5*d1**2)

    if is_call:
        price = S*np.exp(-q*T)*Nd1 - K*np.exp(-r*T)*Nd2
        delta = np.exp(-q*T)*Nd1
    else:
        price = K*np.exp(-r*T)*(1-Nd2) - S*np.exp(-q*T)*(1-Nd1)
        delta = -np.exp(-q*T)*(1-Nd1)

    gamma = (np.exp(-q*T) * n_pdf) / (S * sigma * np.sqrt(T))
    theta = (- (S*np.exp(-q*T)*n_pdf*sigma) / (2*np.sqrt(T))
             - (r*K*np.exp(-r*T)*Nd2 if is_call else r*K*np.exp(-r*T)*(1-Nd2))
             + (q*S*np.exp(-q*T)*Nd1 if is_call else q*S*np.exp(-q*T)*(1-Nd1)))
    theta = theta / 365.0
    vega = S*np.exp(-q*T)*n_pdf*np.sqrt(T) / 100.0
    return {"price": float(price), "delta": float(delta), "gamma": float(gamma),
            "theta": float(theta), "vega": float(vega)}


def _render_payoff_analyzer(open_rows, dw):
    legs = _collect_open_option_legs(open_rows)
    if not legs:
        st.warning("No option legs available for payoff analysis.")
        return

    # --- left: selectable legs & settings ---
    st.markdown("&nbsp;", unsafe_allow_html=True)
    cols = st.columns([0.35, 0.65])

    with cols[0]:
        st.caption("Open Positions:")
        enabled = {}
        for i, leg in enumerate(legs):
            tag = "S" if leg["side"] == "SHORT" else "B"
            label = f"{tag}  {leg['symbol'] or (leg['type'][0] + ' ' + str(int(leg['strike'])))}"
            enabled[i] = st.checkbox(label, value=True, key=f"leg_{i}")

        st.divider()
        st.caption("Settings")
        spot_guess = None
        try:
            spot_guess = float(dw.get_ltp_once("IDX_I", 13))
        except Exception:
            pass
        spot = st.number_input("Spot (NIFTY)", min_value=0.0,
                               value=float(spot_guess or np.mean([l['strike'] for l in legs])),
                               step=1.0)
        iv = st.slider("Assumed IV (%)", 5.0, 60.0, 18.0, 0.5)
        r = st.slider("Risk-free (%)", 0.0, 10.0, 6.5, 0.1)
        q = st.slider("Div. yield (%)", 0.0, 3.0, 0.0, 0.1)

    active_legs = [leg for i, leg in enumerate(legs) if enabled.get(i)]
    if not active_legs:
        with cols[1]:
            st.info("Select at least one leg.")
        return

    with cols[1]:
        tabs = st.tabs(["Pay-Off", "Greeks"])

        # ===== Pay-Off (today vs expiry) =====
        with tabs[0]:
            xs = np.linspace(spot * 0.9, spot * 1.1, 250)
            total_today = np.zeros_like(xs)
            total_exp = np.zeros_like(xs)

            for leg in active_legs:
                K = float(leg["strike"])
                sigma = float(iv)/100.0
                T = _to_years(leg.get("expiry") or "")
                qty = int(leg["qty"])
                side_mult = -1 if leg["side"] == "SHORT" else 1
                is_call = (leg["type"] == "CALL")

                # Today: Black–Scholes mark-to-model vs entry premium
                today_vals = []
                for S in xs:
                    g = _bs_price_greeks(S, K, r/100.0, q/100.0, sigma, max(T, 1/252.0), is_call)
                    price = g["price"]
                    pnl = (price - leg["premium"]) if side_mult == 1 else (leg["premium"] - price)
                    today_vals.append(pnl * qty)
                total_today += np.array(today_vals)

                # Expiry: intrinsic vs premium
                intrinsic = np.maximum(0.0, xs - K) if is_call else np.maximum(0.0, K - xs)
                exp_pnl = (intrinsic - leg["premium"]) if side_mult == 1 else (leg["premium"] - intrinsic)
                total_exp += exp_pnl * qty

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xs, y=total_today, mode="lines", name="Today",
                                     line=dict(color=_DHAN_COLORS["today"]),
                                     hovertemplate="Spot ₹%{x:.2f}<br>P&L ₹%{y:,.0f}"))
            fig.add_trace(go.Scatter(x=xs, y=total_exp, mode="lines", name="Expiry",
                                     line=dict(color=_DHAN_COLORS["expiry"]),
                                     hovertemplate="Spot ₹%{x:.2f}<br>P&L ₹%{y:,.0f}"))
            fig.add_vline(x=float(spot), line_dash="dash", line_color="#455a64",
                          annotation_text=f"Spot ₹{spot:,.0f}")

            # Breakevens from expiry curve
            try:
                idx = np.where(np.diff(np.sign(total_exp)))[0]
                for i in idx:
                    x0, x1, y0, y1 = xs[i], xs[i+1], total_exp[i], total_exp[i+1]
                    be = x0 if y1 == y0 else x0 - y0 * (x1 - x0) / (y1 - y0)
                    fig.add_vline(x=float(be), line_dash="dot",
                                  line_color=_DHAN_COLORS["breakeven"],
                                  annotation_text=f"BE ₹{be:,.0f}")
            except Exception:
                pass

            fig.update_layout(title="Pay-Off Simulation",
                              xaxis_title="Spot Price", yaxis_title="Net P&L (₹)",
                              template="plotly_white", height=520,
                              legend=dict(orientation="h"))
            st.plotly_chart(fig, width="stretch")

            # Quick stats
            mx = np.nanmax(total_exp)
            mn = np.nanmin(total_exp)
            st.markdown(
                f"**Max Profit**: ₹ {mx:,.0f} &nbsp;&nbsp; "
                f"**Max Loss**: {'Unlimited' if mn < -1e6 else '₹ ' + format(mn, ',.0f')}"
            )

        # ===== Greeks (at current spot; signed by side/qty) =====
        with tabs[1]:
            rows = []
            for leg in active_legs:
                K = float(leg["strike"])
                T = _to_years(leg.get("expiry") or "")
                g = _bs_price_greeks(spot, K, r/100.0, q/100.0, float(iv)/100.0,
                                     max(T, 1/252.0), leg["type"] == "CALL")
                mult = (-1 if leg["side"] == "SHORT" else 1) * int(leg["qty"])
                rows.append({
                    "Leg": (leg["symbol"] or f"{leg['type'][0]} {int(K)}"),
                    "Side": "S" if leg["side"] == "SHORT" else "B",
                    "Qty": leg["qty"],
                    "Δ": round(mult * g["delta"], 4),
                    "Γ": round(mult * g["gamma"], 6),
                    "Θ/day": round(mult * g["theta"], 2),
                    "Vega": round(mult * g["vega"], 2),
                    "Model Px": round(g["price"], 2),
                })
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# =============================================================================
# Agent Logs tab
# =============================================================================
def _agent_logs_tab() -> None:
    st.markdown("### Agent Logs")
    start_dt, end_dt = _date_range_controls("agent_logs")
    st.caption(f"Showing log entries from {start_dt.date()} to {end_dt.date()}")
    try:
        if not AGENT_LOG.exists():
            st.info("No agent log found yet.")
            return
        with open(AGENT_LOG, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if not lines:
            st.info("Log file is empty.")
            return
        tail_raw = lines[-800:]
        filtered = []
        for line in tail_raw:
            try:
                ts_str = line.split(" ", 1)[0]
                ts = datetime.fromisoformat(ts_str)
            except Exception:
                ts = None
            if ts and start_dt <= ts <= end_dt:
                filtered.append(line)
        view_lines = filtered if filtered else tail_raw
        st.code("".join(view_lines), language="log")
    except Exception as e:
        st.error(f"Failed to read agent logs: {e}")


def _render_blotter_panel(blotter_df: Optional[pd.DataFrame], summary: Dict[str, Any]) -> None:
    st.markdown("#### Trade Blotter")
    if summary:
        metrics = [
            ("Total", summary.get("total_orders")),
            ("Executed", summary.get("executed_orders")),
            ("Warn-only", summary.get("warn_only_orders")),
        ]
        cols = st.columns(len(metrics))
        for col, (label, value) in zip(cols, metrics):
            with col:
                st.metric(label, _fmt_optional(value, digits=None))
        extra_metrics = [
            ("Rolls", summary.get("roll_orders")),
            ("Hedges", summary.get("hedge_orders")),
            ("Exits", summary.get("exit_orders")),
        ]
        cols_extra = st.columns(len(extra_metrics))
        for col, (label, value) in zip(cols_extra, extra_metrics):
            with col:
                st.metric(label, _fmt_optional(value, digits=None))
    if blotter_df is None:
        st.info("Blotter file not available yet. Start the agent to capture trades.")
        return
    if blotter_df.empty:
        fallback = _fallback_blotter_from_positions()
        if fallback is not None:
            st.info("No agent trades recorded yet. Showing live positions (read-only).")
            st.dataframe(fallback, hide_index=True, width="stretch")
        else:
            st.info("No trades recorded yet.")
        return
    table = blotter_df.copy()
    if "timestamp" in table.columns:
        table["timestamp"] = table["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    latest = table.tail(100)
    columns = [c for c in ["timestamp", "trade_mode", "side", "order_type", "product_type", "quantity", "price", "strike", "notes"] if c in latest.columns]
    st.dataframe(latest[columns], hide_index=True, width="stretch")
    st.caption("Showing the latest 100 blotter entries.")


def _render_agent_activity(feature_df: Optional[pd.DataFrame]) -> None:
    st.markdown("#### Agent Activity Feed")
    if feature_df is None:
        st.info("Decision feed not available yet. Ensure FEATURE_LOG_PATH is configured for the agent.")
        return
    if feature_df.empty:
        st.info("Waiting for the first strategy telemetry event.")
        return
    if "timestamp" not in feature_df.columns:
        st.info("Feature log missing timestamp column.")
        return
    recent = feature_df.dropna(subset=["timestamp"]).tail(25)
    recent = recent.sort_values("timestamp", ascending=False)
    for _, row in recent.iterrows():
        ts = row.get("timestamp")
        ts_text = _fmt_optional(ts)
        raw_strategy = str(row.get("strategy") or "unknown")
        strategy_name = raw_strategy.title()

        if raw_strategy == "environment":
            ctx = _parse_context_blob(row.get("context"))
            state = ctx.get("market_state", {})
            candidates = ctx.get("strategy_candidates", [])
            trend = str(state.get("trend", "—")).replace("_", " ").title()
            vol = str(state.get("volatility", "—")).replace("_", " ").title()
            ivr = state.get("iv_rank")
            bullet = [f"Trend {trend}", f"Vol {vol}"]
            if ivr is not None:
                bullet.append(f"IV rank {_fmt_optional(ivr)}")
            if candidates:
                top = candidates[0]
                bullet.append(f"Top: {top.get('name', '—')} (score {top.get('score')})")
            st.markdown(f"**{ts_text}** · Market Environment\n\n- " + " | ".join(bullet))
            continue

        if raw_strategy == "risk_event":
            ctx = _parse_context_blob(row.get("context"))
            label = str(ctx.get("label", "Risk Event")).title()
            severity = str(ctx.get("severity", "info")).upper()
            message = ctx.get("message") or ""
            extras = [
                f"{k}={v}"
                for k, v in ctx.items()
                if k not in {"label", "message", "severity"} and v not in (None, "", [], {})
            ]
            extra_text = (" | ".join(extras)) if extras else ""
            lines = [message or "—"]
            if extra_text:
                lines.append(extra_text)
            st.markdown(f"**{ts_text}** · {label} ({severity})\n\n" + "\n".join(f"- {line}" for line in lines))
            continue

        details = []
        spot_val = row.get("spot")
        if not _is_missing(spot_val):
            details.append(f"Spot ₹ {_fmt_optional(spot_val)}")
        ce_strike = row.get("ce_strike")
        pe_strike = row.get("pe_strike")
        ce_delta = row.get("ce_delta")
        pe_delta = row.get("pe_delta")
        ce_ltp = row.get("ce_ltp")
        pe_ltp = row.get("pe_ltp")
        if not _is_missing(ce_strike):
            details.append(f"CE { _fmt_optional(ce_strike, digits=0) } Δ {_fmt_optional(ce_delta, digits=3)} @ {_fmt_optional(ce_ltp)}")
        if not _is_missing(pe_strike):
            details.append(f"PE { _fmt_optional(pe_strike, digits=0) } Δ {_fmt_optional(pe_delta, digits=3)} @ {_fmt_optional(pe_ltp)}")
        net_credit = row.get("net_credit")
        if not _is_missing(net_credit):
            details.append(f"Net credit ₹ {_fmt_optional(net_credit)}")
        ctx_summary = _format_context_summary(_parse_context_blob(row.get("context")))
        bullet_lines = []
        if details:
            bullet_lines.append(" | ".join(details))
        if ctx_summary:
            bullet_lines.append(f"Context: {ctx_summary}")
        trade_mode = row.get("trade_mode")
        if trade_mode:
            bullet_lines.append(f"Mode: {trade_mode}")
        note = "\n".join(f"- {line}" for line in bullet_lines) if bullet_lines else "- —"
        st.markdown(f"**{ts_text}** · {strategy_name}\n\n{note}")


def _render_plan_snapshot(feature_df: Optional[pd.DataFrame]) -> None:
    st.markdown("#### Active Plan Snapshot")
    fallback_df: Optional[pd.DataFrame] = None
    if feature_df is None or "strategy" not in feature_df.columns:
        fallback_df = _fallback_plan_from_positions()
        feature_df = fallback_df
    if feature_df is None:
        st.info("Decision feed not available yet.")
        return
    feature_df = feature_df[feature_df["strategy"] != "environment"] if not feature_df.empty else feature_df
    if feature_df is not None and not feature_df.empty and {"ce_strike", "pe_strike"}.issubset(feature_df.columns):
        # Drop telemetry rows that carried no leg data (e.g. warning events)
        feature_df = feature_df.dropna(subset=["ce_strike", "pe_strike"], how="all")
    if feature_df is None or feature_df.empty:
        if fallback_df is None:
            fallback_df = _fallback_plan_from_positions()
        if fallback_df is not None and not fallback_df.empty:
            feature_df = fallback_df
        else:
            st.info("No active plan information recorded yet.")
            return
    latest = feature_df.sort_values("timestamp").groupby("strategy", as_index=False).last()
    for _, row in latest.iterrows():
        strategy_name = str(row.get("strategy") or "unknown").title()
        updated_at = _fmt_optional(row.get("timestamp"))
        ctx = _parse_context_blob(row.get("context"))

        lines = []
        lines.append(f"Updated {updated_at}")
        lines.append(
            f"CE { _fmt_optional(row.get('ce_strike'), digits=0) } Δ {_fmt_optional(row.get('ce_delta'), digits=3)} @ {_fmt_optional(row.get('ce_ltp'))}"
        )
        lines.append(
            f"PE { _fmt_optional(row.get('pe_strike'), digits=0) } Δ {_fmt_optional(row.get('pe_delta'), digits=3)} @ {_fmt_optional(row.get('pe_ltp'))}"
        )
        lines.append(f"Net credit ₹ {_fmt_optional(row.get('net_credit'))}")
        context_text = _format_context_summary(ctx)
        if context_text:
            lines.append(context_text)

        st.markdown(f"**{strategy_name}**\n\n" + "\n".join(f"- {line}" for line in lines))


def _render_environment_summary(feature_df: Optional[pd.DataFrame]) -> None:
    st.markdown("#### Market Regime")
    if feature_df is None or feature_df.empty:
        st.info("No regime data yet.")
        return
    if "strategy" not in feature_df.columns:
        st.info("No regime data yet.")
        return
    env_rows = feature_df.loc[feature_df["strategy"] == "environment"]
    if env_rows.empty:
        st.info("No regime data yet.")
        return
    latest = env_rows.sort_values("timestamp").iloc[-1]
    ctx = _parse_context_blob(latest.get("context"))
    state = ctx.get("market_state", {})
    candidates = ctx.get("strategy_candidates", [])

    cols = st.columns(3)
    cols[0].metric("Trend", str(state.get("trend", "—")).replace("_", " ").title())
    cols[1].metric("Volatility", str(state.get("volatility", "—")).replace("_", " ").title())
    cols[2].metric("IV Rank", _fmt_optional(state.get("iv_rank")))

    cols2 = st.columns(3)
    cols2[0].metric("Realized Vol", _fmt_optional(state.get("realized_vol")))
    cols2[1].metric("Drift", _fmt_optional(state.get("drift_pct")))
    cols2[2].metric("ATR %", _fmt_optional(state.get("atr_pct")))

    if candidates:
        df = pd.DataFrame(candidates)
        st.dataframe(df[[c for c in ["name", "score", "confidence", "sizing_hint", "rationale"] if c in df.columns]], width="stretch")
    else:
        st.caption("Strategy recommender idle – waiting for sufficient data.")


def _strategy_monitor_tab() -> None:
    st.markdown("### Strategy Monitor")
    blotter_df, summary = _load_blotter()
    feature_df = _load_feature_history(limit=250)
    start_dt, end_dt = _date_range_controls("strategy_monitor")
    st.caption(f"Showing data from {start_dt.date()} to {end_dt.date()}")
    blotter_filtered = _filter_df_by_range(blotter_df, start_dt, end_dt)
    feature_filtered = _filter_df_by_range(feature_df, start_dt, end_dt)

    left, right = st.columns([0.55, 0.45])
    with left:
        _render_blotter_panel(blotter_filtered, summary)
    with right:
        _render_environment_summary(feature_filtered)
        st.divider()
        _render_agent_activity(feature_filtered)
        st.divider()
        _render_plan_snapshot(feature_filtered)
    st.divider()
    _render_live_monitor_snapshot()


def _render_live_monitor_snapshot() -> None:
    st.markdown("### Live Monitor Snapshot")
    monitor_file = STATE_DIR / "live_monitor" / "latest.json"
    if not monitor_file.exists():
        st.info("No live monitor snapshot yet. Run scripts/run_live_monitor.py or enable the cron.")
        return
    try:
        data = json.loads(monitor_file.read_text())
    except Exception as e:
        st.error(f"Failed to read live monitor snapshot: {e}")
        return
    left, right = st.columns([0.35, 0.65])
    with left:
        st.metric("Spot", _fmt_optional(data.get("spot")))
        st.caption(f"As of {data.get('as_of')}")
        gap_pct = data.get("gap_pct")
        prev_close = data.get("prev_close")
        if prev_close is not None:
            st.caption(f"Prev close: {_fmt_optional(prev_close)}")
        if gap_pct is not None:
            st.metric("Gap %", f"{gap_pct*100:.2f}%")
    with right:
        positions = data.get("positions") or []
        if positions:
            df = pd.DataFrame(positions)
            df = df.rename(columns={"symbol": "Symbol", "pnl": "PnL", "pnl_pct": "PnL %", "risk_flag": "Risk"})
            show_cols = [c for c in ["Symbol", "PnL", "PnL %", "Risk", "bias", "distance_to_support_pct", "distance_to_resistance_pct", "notes"] if c in df.columns]
            st.dataframe(df[show_cols], width="stretch", hide_index=True)
        else:
            st.info("No positions in snapshot.")
    evs = data.get("events") or []
    if evs:
        st.markdown("#### Upcoming Events (last 2d)")
        df_e = pd.DataFrame(evs)
        cols = [c for c in ["timestamp", "source", "event_type", "headline", "importance"] if c in df_e.columns]
        st.dataframe(df_e[cols].sort_values("timestamp"), height=200, hide_index=True)

    status_path = STATE_DIR / "weekly_live_status.json"
    if status_path.exists():
        st.markdown("#### Weekly Live Status")
        try:
            status = json.loads(status_path.read_text())
            if status:
                cols = st.columns(3)
                cols[0].metric("Expiry", str(status.get("expiry", "—")))
                cols[1].metric("MTM", _fmt_optional(status.get("mtm")))
                cols[2].metric("MTM %", _fmt_optional(status.get("mtm_pct")))
                st.caption(f"Updated: {status.get('timestamp')}")
                st.json(status)
            else:
                st.caption("No weekly status yet.")
        except Exception:
            st.caption("Weekly status unreadable.")


def _render_capital_telemetry(
    equity_df: Optional[pd.DataFrame] = None,
    *,
    section_title: str = "Capital Telemetry",
    chart_key: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    st.markdown(f"#### {section_title}")
    df = equity_df if equity_df is not None else _load_equity_history(limit=1500)
    if df is None or df.empty or "equity_estimate" not in df.columns:
        st.info("No equity telemetry yet. Keep the agent running to build history.")
        return df
    df = df.dropna(subset=["equity_estimate"])
    if df.empty:
        st.info("No equity telemetry yet. Keep the agent running to build history.")
        return df
    equity_series = df["equity_estimate"].astype(float)
    current_equity = float(equity_series.iloc[-1])
    previous = float(equity_series.iloc[-2]) if len(equity_series) > 1 else None
    delta_label = None
    if previous is not None and previous != 0 and not np.isnan(previous):
        delta_val = current_equity - previous
        pct = (delta_val / previous) * 100 if previous else None
        delta_label = f"{delta_val:+,.0f}"
        if pct is not None and not np.isnan(pct):
            delta_label = f"{delta_val:+,.0f} ({pct:+.2f}%)"

    rolling_max = equity_series.cummax().replace(0, np.nan)
    drawdown_series = (equity_series / rolling_max) - 1.0
    max_drawdown_pct = float(drawdown_series.min()) * 100 if not drawdown_series.empty else None
    current_drawdown_pct = float(drawdown_series.iloc[-1]) * 100 if not drawdown_series.empty else None

    header_cols = st.columns(3)
    header_cols[0].metric("Current Equity", _fmt_rupee(current_equity, digits=0), delta=delta_label)
    header_cols[1].metric(
        "Max Drawdown",
        f"{max_drawdown_pct:.2f}%" if max_drawdown_pct is not None and not np.isnan(max_drawdown_pct) else "—",
    )
    header_cols[2].metric(
        "Current Drawdown",
        f"{current_drawdown_pct:.2f}%" if current_drawdown_pct is not None and not np.isnan(current_drawdown_pct) else "—",
    )

    timeframe_cols = st.columns(4)
    frames = [("Day Δ", 1), ("Week Δ", 7), ("Month Δ", 30), ("Year Δ", 365)]
    for col, (label, days) in zip(timeframe_cols, frames):
        change = _compute_equity_change(df, days)
        if not change:
            col.metric(label, "—", delta=None)
            continue
        delta_val, delta_pct = change
        col.metric(
            label,
            _fmt_rupee(delta_val, digits=0),
            delta=f"{delta_pct:+.2f}%" if delta_pct is not None else None,
        )

    chart_df = df.tail(720).copy()
    rolling_max_chart = chart_df["equity_estimate"].cummax().replace(0, np.nan)
    chart_df["drawdown_pct"] = ((chart_df["equity_estimate"] / rolling_max_chart) - 1.0) * 100

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_df["timestamp"],
            y=chart_df["equity_estimate"],
            name="Equity",
            mode="lines",
            line=dict(width=2, color="#1f77b4"),
            hovertemplate="Time %{x|%Y-%m-%d %H:%M}<br>Equity ₹%{y:,.0f}<extra></extra>",
        )
    )
    if chart_df["drawdown_pct"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=chart_df["timestamp"],
                y=chart_df["drawdown_pct"],
                name="Drawdown %",
                mode="lines",
                line=dict(width=1.5, color="#d62728", dash="dot"),
                fill="tozeroy",
                opacity=0.35,
                yaxis="y2",
                hovertemplate="Time %{x|%Y-%m-%d %H:%M}<br>Drawdown %{y:.2f}%<extra></extra>",
            )
        )
        fig.update_layout(
            yaxis=dict(title="Equity (₹)", rangemode="tozero"),
            yaxis2=dict(
                title="Drawdown %",
                overlaying="y",
                side="right",
                showgrid=False,
                rangemode="tozero",
            ),
        )
    else:
        fig.update_layout(yaxis=dict(title="Equity (₹)", rangemode="tozero"))

    fig.update_layout(
        xaxis=dict(title="Timestamp"),
        margin=dict(l=10, r=10, t=35, b=35),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    chart_kwargs = {"width": "stretch"}
    if chart_key:
        chart_kwargs["key"] = chart_key
    st.plotly_chart(fig, **chart_kwargs)
    return df


def _render_exposure_monitor(
    equity_df: Optional[pd.DataFrame],
    risk_limits: Dict[str, float],
    feature_df: Optional[pd.DataFrame],
    start: datetime,
    end: datetime,
) -> None:
    st.markdown("#### Exposure Monitor")
    if equity_df is None or equity_df.empty or "net_delta" not in equity_df.columns:
        st.info("No exposure telemetry yet.")
        return
    df = equity_df.dropna(subset=["net_delta"]).tail(720)
    if df.empty:
        st.info("No exposure telemetry yet.")
        return
    delta_cap = float(risk_limits.get("delta_cap", 0.25))
    exposure_pct = float(risk_limits.get("exposure_pct", 0.05))
    latest = df.iloc[-1]
    current_delta = float(latest.get("net_delta") or 0.0)
    current_equity = float(latest.get("equity_estimate") or 0.0)
    notional = float(latest.get("total_notional") or 0.0)
    notional_cap = current_equity * exposure_pct if current_equity else None

    cols = st.columns(3)
    cols[0].metric("Net Δ", f"{current_delta:.3f}", f"±{delta_cap}")
    cols[1].metric("Total Notional", _fmt_rupee(notional, digits=0), None if notional_cap is None else f"Cap {_fmt_rupee(notional_cap, digits=0)}")
    cols[2].metric("Equity Ref", _fmt_rupee(current_equity, digits=0))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["net_delta"],
            name="Net Δ",
            mode="lines",
            line=dict(width=2, color="#22c55e"),
            hovertemplate="Time %{x|%Y-%m-%d %H:%M}<br>Δ %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=delta_cap, line=dict(color="#f97316", dash="dash"), annotation_text=f"+{delta_cap:.2f}", annotation_position="top left")
    fig.add_hline(y=-delta_cap, line=dict(color="#f97316", dash="dash"), annotation_text=f"-{delta_cap:.2f}", annotation_position="bottom left")
    fig.update_layout(
        xaxis=dict(title="Timestamp"),
        yaxis=dict(title="Net Delta"),
        margin=dict(l=10, r=10, t=35, b=35),
        height=260,
    )
    # annotate risk events
    events = _risk_events_in_range(feature_df, start, end)
    y_min = float(np.nanmin(df["net_delta"])) if df["net_delta"].notna().any() else -delta_cap
    y_max = float(np.nanmax(df["net_delta"])) if df["net_delta"].notna().any() else delta_cap
    for ev in events:
        ts = ev.get("timestamp_dt")
        if not ts:
            continue
        try:
            ts = pd.to_datetime(ts).to_pydatetime()
        except Exception:
            continue
        label = ev.get("label", "")
        fig.add_trace(
            go.Scatter(
                x=[ts, ts],
                y=[y_min, y_max],
                mode="lines",
                line=dict(color="#f97316", width=1, dash="dot"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        net_delta = ev.get("net_delta")
        if net_delta is not None:
            fig.add_annotation(
                x=ts,
                y=net_delta,
                text=f"Δ {net_delta:.2f}",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40,
                bgcolor="rgba(249,115,22,0.2)",
            )

    st.plotly_chart(fig, width="stretch", key="exposure_monitor_chart")


def _date_range_controls(key_prefix: str = "date_range") -> Tuple[datetime, datetime]:
    today = datetime.now().date()
    default_range = (today, today)
    date_range = st.date_input(
        "Date range",
        default_range,
        key=f"{key_prefix}_picker",
        help="Filter telemetry/logs to the selected dates.",
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range
    if start_date > end_date:
        start_date, end_date = end_date, start_date
    start_dt = datetime.combine(start_date, datetime_time.min)
    end_dt = datetime.combine(end_date, datetime_time.max)
    return start_dt, end_dt


def _filter_df_by_range(df: Optional[pd.DataFrame], start: datetime, end: datetime, column: str = "timestamp") -> Optional[pd.DataFrame]:
    if df is None or df.empty or column not in df.columns:
        return df
    temp = df.copy()
    temp[column] = pd.to_datetime(temp[column], errors="coerce")
    mask = (temp[column] >= start) & (temp[column] <= end)
    return temp.loc[mask].reset_index(drop=True)


ORDER_AUDIT_LOG = STATE_DIR / "order_audit.jsonl"
ORDER_INTENT_QUEUE = STATE_DIR / "order_intents.jsonl"
PLAYBOOK_STATUS_LOG = STATE_DIR / "playbook_status.jsonl"


def _load_jsonl(path: Path, limit: int = 200) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as fh:
            rows = fh.readlines()
    except Exception:
        return []
    if limit and len(rows) > limit:
        rows = rows[-limit:]
    out = []
    for line in rows:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _render_order_audit_panel() -> None:
    st.markdown("#### Order Router / Circuit Status")
    intents = _load_jsonl(ORDER_INTENT_QUEUE, limit=200)
    audits = _load_jsonl(ORDER_AUDIT_LOG, limit=400)
    circuit = None
    try:
        cfg_state = json.loads(ENTRY_BLOCK_STATE.read_text()) if ENTRY_BLOCK_STATE.exists() else {}
        circuit = cfg_state.get("circuit_state")
    except Exception:
        circuit = None

    cols = st.columns(3)
    with cols[0]:
        st.metric("Pending Intents", len(intents))
    with cols[1]:
        if circuit and circuit.get("status") == "tripped":
            st.metric("Circuit", "Tripped", f"Resume {circuit.get('resume_at','—')}")
        else:
            st.metric("Circuit", "Clear")
    with cols[2]:
        last_audit = audits[-1] if audits else None
        st.metric("Last Audit", last_audit.get("stage") if isinstance(last_audit, dict) else "—")

    if intents:
        st.caption("Pending Order Intents")
        intent_df = pd.DataFrame(intents)
        if "timestamp" in intent_df.columns:
            intent_df["timestamp"] = pd.to_datetime(intent_df["timestamp"], errors="coerce")
        st.dataframe(intent_df.tail(50), width="stretch")
    else:
        st.info("No queued order intents.")

    if audits:
        st.caption("Recent Audit Events")
        audit_df = pd.DataFrame(audits)
        if "timestamp" in audit_df.columns:
            audit_df["timestamp"] = pd.to_datetime(audit_df["timestamp"], errors="coerce")
        st.dataframe(audit_df.tail(50), width="stretch")
    else:
        st.info("No audit entries yet.")


def _render_playbook_panel() -> None:
    st.markdown("#### Playbook Activity")
    history = _load_playbook_history(limit=10)
    status_rows = _load_jsonl(PLAYBOOK_STATUS_LOG, limit=100)
    if status_rows:
        status_df = pd.DataFrame(status_rows)
        if "timestamp" in status_df.columns:
            status_df["timestamp"] = pd.to_datetime(status_df["timestamp"], errors="coerce")
        st.dataframe(status_df.tail(20), width="stretch")
    if history:
        st.caption("Recent Steps")
        for entry in history:
            meta = entry.get("meta") or {}
            st.markdown(f"- {_fmt_optional(entry.get('timestamp'))} · {entry.get('step')} – {meta}")
    note = st.text_input("Playbook note", key="playbook_note")
    if st.button("Attach Note", key="playbook_note_btn"):
        if note:
            _log_manual_playbook_note(note)
            st.success("Note recorded")
        else:
            st.warning("Enter a note first")


def _log_manual_playbook_note(note: str) -> None:
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "action": "manual_note",
        "status": "info",
        "meta": {"note": note},
    }
    try:
        with PLAYBOOK_STATUS_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass


def _observability_tab() -> None:
    st.markdown("### Observability")
    ss = st.session_state
    feature_df = _load_feature_history(limit=400)
    equity_df = _load_equity_history(limit=1500)
    risk_limits = _load_risk_limits()
    start_dt, end_dt = _date_range_controls("obsv")
    st.caption(f"Showing data from {start_dt.date()} to {end_dt.date()}")
    feature_df = _filter_df_by_range(feature_df, start_dt, end_dt)
    equity_df = _filter_df_by_range(equity_df, start_dt, end_dt)
    forecast = _predict_delta_breach(equity_df, risk_limits)
    notional_forecast = _predict_notional_breach(equity_df, risk_limits)
    rolling_summary = _load_rolling_summary()
    if rolling_summary:
        _render_rolling_data_quality(rolling_summary)
        st.divider()
    selector_summary = _load_selector_summary()
    if selector_summary:
        _render_selector_training_summary(selector_summary)
        st.divider()

    pid = ss.get("agent_pid")
    running_flag = bool(ss.get("agent_running"))
    alive_flag = _pid_alive(pid)
    mode = ss.get("agent_started_mode") or ss.get("trade_mode", "live")

    status_icon = "🟢" if running_flag and alive_flag else ("🟡" if pid and not alive_flag else "🔴")
    status_text = "Running" if running_flag and alive_flag else ("Stale PID" if pid and not alive_flag else "Stopped")
    st.markdown(f"**Agent Status:** {status_icon} {status_text}")
    st.caption(f"Mode: {mode} | PID: {pid or '—'}")

    cols = st.columns(3)
    with cols[0]:
        st.metric("PID alive", "Yes" if alive_flag else "No")
    with cols[1]:
        uptime_delta: Optional[timedelta] = None
        try:
            if running_flag and PID_FILE.exists():
                data = json.loads(PID_FILE.read_text())
                started_at = data.get("started_at")
                if started_at:
                    started_dt = datetime.fromisoformat(started_at)
                    uptime_delta = datetime.now() - started_dt
        except Exception:
            uptime_delta = None
        if uptime_delta:
            total_seconds = int(uptime_delta.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            uptime_str = f"{hours}h {minutes}m"
        else:
            uptime_str = "—"
        st.metric("Uptime", uptime_str)
    with cols[2]:
        st.metric("Credentials", "Verified" if ss.get("creds_verified") else "Missing")

    st.divider()
    equity_df = _render_capital_telemetry(equity_df, chart_key="observability_capital_chart")
    if forecast:
        minutes = forecast.get("forecast_minutes")
        if minutes is not None and minutes < 60:
            st.warning(f"Net delta trending toward cap in ~{minutes:.1f} min (Δ {forecast['current']:.2f} → {forecast['target']:.2f}).")
    if notional_forecast:
        minutes = notional_forecast.get("forecast_minutes")
        if minutes is not None and minutes < 60:
            st.warning(
                "Total notional trending toward cap in ~{:.1f} min (₹ {:,.0f} → {:,.0f}).".format(
                    minutes,
                    notional_forecast["current"],
                    notional_forecast["target"],
                )
            )
    st.divider()
    _render_exposure_monitor(equity_df, risk_limits, feature_df, start_dt, end_dt)
    st.divider()
    _render_order_audit_panel()
    st.divider()
    _render_risk_alerts(feature_df)
    st.divider()
    _render_entry_block_status()
    st.divider()
    _render_playbook_panel()
    st.divider()
    action_cols = st.columns([1, 1, 1, 1])
    with action_cols[0]:
        restart_disabled = not ss.get("creds_verified")
        if st.button("Restart Agent", key="obs_restart", disabled=restart_disabled, help="Stop then start the agent process."):
            stop_agent()
            time.sleep(0.5)
            ok = True
            if not ss.get("creds_verified"):
                ok = _verify_and_remember_creds()
            if ok:
                start_agent()
            st.rerun()
    with action_cols[1]:
        stop_disabled = not running_flag
        if st.button("Stop Agent", key="obs_stop", disabled=stop_disabled):
            stop_agent()
            st.rerun()
    with action_cols[2]:
        stale = bool(pid and not alive_flag)
        clear_disabled = not (PID_FILE.exists() and (stale or not running_flag))
        if st.button("Clear Stale PID", key="obs_clear_pid", disabled=clear_disabled, help="Remove PID file when the process is already gone."):
            _clear_stale_pid_file()
            st.rerun()
    with action_cols[3]:
        if st.button("Reset Risk Block", key="obs_risk_reset", help="Instruct the agent to clear risk pause and re-enable entries."):
            _request_risk_reset()
            st.success("Risk reset signal sent.")

    st.divider()
    file_infos = [
        ("Agent Log", _collect_file_info(AGENT_LOG)),
        ("Trade Blotter", _collect_file_info(BLOTTER_CSV)),
        ("Feature Feed", _collect_file_info(FEATURE_LOG_CSV)),
        ("Equity Telemetry", _collect_file_info(EQUITY_LOG_CSV)),
        ("PID File", _collect_file_info(PID_FILE)),
    ]
    info_cols = st.columns(len(file_infos))
    for col, (label, info) in zip(info_cols, file_infos):
        with col:
            st.caption(label)
            st.write(f"Exists: {'Yes' if info['exists'] else 'No'}")
            st.write(f"Size: {_fmt_size(info.get('size'))}")
            st.write(f"Updated: {_age_string(info.get('updated'))}")

    st.divider()
    tail_lines = _read_tail(AGENT_LOG, 30)
    if tail_lines:
        st.caption("Recent Agent Log")
        st.code("".join(tail_lines), language="log")
    else:
        st.info("Agent log is empty or unavailable.")

# =============================================================================
# Settings tab
# =============================================================================
def _settings_tab() -> None:
    st.markdown("### Settings")
    current = _load_settings()

    # Global / index-level settings
    weekday_opts = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    cur_weekday = current.get("nifty_expiry_weekday", "Tuesday")
    if isinstance(cur_weekday, int):
        cur_weekday = weekday_opts[min(max(cur_weekday, 0), 4)]
    with st.expander("Global (index/expiry) settings", expanded=True):
        lot_size = st.number_input(
            "NIFTY lot size",
            50,
            1000,
            int(current.get("nifty_lot_size", current.get("lot_size", 75))),
            help="Used for position sizing across strategies.",
        )
        expiry_weekday = st.selectbox(
            "NIFTY expiry weekday",
            weekday_opts,
            index=weekday_opts.index(str(cur_weekday).title()) if str(cur_weekday).title() in weekday_opts else 1,
            help="Choose the scheduled weekly/monthly expiry day (default Tuesday).",
        )
        expiry_shift_if_holiday = st.checkbox(
            "Shift expiry if configured day is a holiday",
            value=bool(current.get("expiry_shift_if_holiday", True)),
            help="Skip the configured expiry day when it is a holiday and use the next available expiry.",
        )
        holiday_text = "\n".join(current.get("holiday_list", [])) if isinstance(current.get("holiday_list"), list) else str(current.get("holiday_list") or "")
        holiday_text = st.text_area(
            "Holiday list (YYYY-MM-DD, one per line)",
            value=holiday_text,
            help="Optional manual holiday overrides used to skip expiry if the configured day is a holiday.",
        )
        holiday_list = [h.strip() for h in holiday_text.replace(",", "\n").splitlines() if h.strip()]
        gap_entry_threshold = st.number_input(
            "Max allowed gap % for neutral entries",
            0.0,
            5.0,
            float(current.get("gap_entry_threshold", 0.4)),
            help="Skip entries when |gap| exceeds this percent (e.g., 0.4 for 0.4%).",
        )
        iv_floor_percentile = st.number_input(
            "IV floor percentile (0-1)",
            0.0,
            1.0,
            float(current.get("iv_floor_percentile", 0.2)),
            help="Require IV/VIX percentile above this before entering.",
        )

    c1, c2 = st.columns(2)
    with c1:
        max_legs = st.number_input("Max active strangles", 1, 10, int(current.get("max_legs", 1)))
    with c2:
        leg_sl_pct = st.number_input("Stop-loss per leg (%)", 0.5, 10.0, float(current.get("leg_sl_pct", 2.5)))
        profit_pct = st.number_input("Profit booking (%)", 0.5, 10.0, float(current.get("profit_pct", 2.25)))
    smart_selector_enabled = st.checkbox(
        "Enable smart selector (regime-aware)",
        value=bool(current.get("smart_selector_enabled", True)),
        help="Toggle the new trend-aware strangle/spread engine. Disable to fall back to neutral strangles only.",
    )

    with st.expander("Smart selector & risk"):
        r1, r2 = st.columns(2)
        with r1:
            max_intraday_loss = st.number_input(
                "Max intraday loss (₹)",
                -200000.0,
                -1000.0,
                float(current.get("max_intraday_loss", -3000.0)),
                help="Global SL for the basket; agent flattens when MTM <= this value.",
            )
            intraday_target = st.number_input(
                "Intraday target (₹)",
                1000.0,
                200000.0,
                float(current.get("intraday_target", 4000.0)),
            )
            partial_target_pct = st.number_input(
                "Partial target (% of TP)",
                0.1,
                0.9,
                float(current.get("partial_target_pct", 0.65)),
                help="Book profits early when basket MTM reaches this fraction of the day target.",
            )
            max_daily_trades = st.number_input(
                "Max daily trades",
                1,
                20,
                int(current.get("max_daily_trades", 5)),
            )
        with r2:
            max_daily_loss_pct = st.number_input(
                "Max daily loss (% of capital)",
                0.01,
                0.20,
                float(current.get("max_daily_loss_pct", 0.03)),
            )
            vix_carry_threshold = st.number_input(
                "VIX carry threshold",
                8.0,
                40.0,
                float(current.get("vix_carry_threshold", 12.0)),
            )
            avg_5m_volume = st.number_input(
                "Avg 5m volume (for open trend)",
                10000.0,
                500000.0,
                float(current.get("avg_5m_volume", 100000.0)),
            )
            per_leg_sl_mult = st.number_input(
                "Per-leg SL multiple",
                1.0,
                5.0,
                float(current.get("per_leg_sl_mult", 1.6)),
                help="Close the basket if any short leg premium grows to this multiple of its entry price.",
            )
            per_leg_tp_mult = st.number_input(
                "Per-leg TP multiple",
                0.05,
                1.0,
                float(current.get("per_leg_tp_mult", 0.5)),
                help="Close the basket if any short leg premium decays to this multiple of its entry price.",
            )
        allow_carry_forward = st.checkbox("Allow carry forward", value=bool(current.get("allow_carry_forward", False)))
        max_carry_days = st.number_input(
            "Max carry days",
            0,
            10,
            int(current.get("max_carry_days", 0)),
        )

    with st.expander("Adaptive strike selection"):
        a1, a2 = st.columns(2)
        with a1:
            vix_adaptive_low = st.number_input(
                "VIX lower band",
                8.0,
                25.0,
                float(current.get("vix_adaptive_low", 12.0)),
                help="When VIX is below this level, use the 'low' strike deltas.",
            )
            strangle_delta_low = st.number_input(
                "Strangle delta (low VIX)",
                0.05,
                0.3,
                float(current.get("strangle_delta_low", 0.12)),
            )
            strangle_offset_low = st.number_input(
                "Strangle offset (low VIX)",
                50.0,
                400.0,
                float(current.get("strangle_offset_low", 220.0)),
                help="Approx distance (pts) from spot for strangle shorts when vol is low.",
            )
            spread_short_delta_low = st.number_input(
                "Spread short delta (low VIX)",
                0.05,
                0.5,
                float(current.get("spread_short_delta_low", 0.22)),
            )
        with a2:
            vix_adaptive_high = st.number_input(
                "VIX upper band",
                10.0,
                35.0,
                float(current.get("vix_adaptive_high", 20.0)),
                help="Above this VIX, use the 'high' strike deltas (closer to ATM).",
            )
            strangle_delta_high = st.number_input(
                "Strangle delta (high VIX)",
                0.05,
                0.4,
                float(current.get("strangle_delta_high", 0.22)),
            )
            strangle_offset_high = st.number_input(
                "Strangle offset (high VIX)",
                50.0,
                400.0,
                float(current.get("strangle_offset_high", 120.0)),
                help="Approx distance (pts) from spot for strangle shorts when vol is high.",
            )
            spread_short_delta_high = st.number_input(
                "Spread short delta (high VIX)",
                0.05,
                0.5,
                float(current.get("spread_short_delta_high", 0.32)),
            )

    with st.expander("Hedge monitor adjustments"):
        hedge_enabled = st.checkbox(
            "Enable hedge monitor",
            value=bool(current.get("hedge_enabled", True)),
            help="Detect existing short-body structures, roll stressed legs, and add hedges automatically.",
        )
        bc1, bc2 = st.columns(2)
        with bc1:
            hedge_delta_breach = st.number_input(
                "Delta breach threshold",
                0.05,
                0.9,
                float(current.get("hedge_delta_breach", 0.30)),
                help="Roll a side when |delta| exceeds this value.",
            )
            hedge_roll_distance = st.number_input(
                "Roll distance (pts)",
                50.0,
                500.0,
                float(current.get("hedge_roll_distance", 150.0)),
                help="How far to move the body when rolling.",
            )
            hedge_salvage = st.number_input(
                "Wing salvage LTP",
                0.0,
                20.0,
                float(current.get("hedge_salvage_wing_ltp", 5.0)),
                help="Wings below this value can be considered for salvage/refresh (future use).",
            )
        with bc2:
            hedge_premium_hard_x = st.number_input(
                "Premium multiple trigger",
                1.0,
                4.0,
                float(current.get("hedge_premium_hard_x", 2.0)),
                help="Roll when current premium >= multiple × entry premium.",
            )
            hedge_delta_max = st.number_input(
                "Hedge target delta",
                0.01,
                0.5,
                float(current.get("hedge_delta_max", 0.12)),
                help="Select hedges with |delta| below this value.",
            )
            hedge_price_max = st.number_input(
                "Max hedge cost",
                1.0,
                200.0,
                float(current.get("hedge_price_max", 35.0)),
                help="Do not auto-buy hedges above this LTP.",
            )

    with st.expander("Batman strategy (monthly)"):
        bat_long_offset = st.number_input(
            "Long wing distance from spot (pts)",
            100.0,
            1200.0,
            float(current.get("batman_long_offset", 300.0)),
            step=100.0,
            help="Distance from spot for long hedges (both CE/PE).",
        )
        bat_short_offset = st.number_input(
            "Short body distance from spot (pts)",
            100.0,
            1500.0,
            float(current.get("batman_short_offset", 500.0)),
            step=100.0,
            help="Distance from spot for short bodies (both CE/PE).",
        )
        bat_qty_long = st.number_input(
            "Long wing quantity (lots)",
            1,
            10,
            int(current.get("batman_qty_long", 1)),
            help="Per-side long wings.",
        )
        bat_qty_short = st.number_input(
            "Short body quantity (lots)",
            1,
            10,
            int(current.get("batman_qty_short", 3)),
            help="Per-side short bodies.",
        )
        bat_weekly_hedge_enabled = st.checkbox(
            "Add cheap weekly hedges",
            value=bool(current.get("batman_weekly_hedge_enabled", True)),
            help="Buys far OTM weekly wings to cover the open monthly bodies for margin relief.",
        )
        bat_weekly_hedge_cap = st.number_input(
            "Weekly hedge max LTP (₹)",
            1.0,
            50.0,
            float(current.get("batman_weekly_hedge_price_cap", 12.0)),
            help="Skip weekly hedges above this premium.",
        )
        bat_weekly_hedge_min_dist = st.number_input(
            "Weekly hedge min distance from spot (pts)",
            50.0,
            1200.0,
            float(current.get("batman_weekly_hedge_min_distance", 300.0)),
            help="Only pick weekly hedges at least this far OTM.",
        )

    if st.button("💾 Save Settings"):
        SETTINGS_JSON.write_text(json.dumps({
            "max_legs": int(max_legs),
            "lot_size": int(lot_size),
            "leg_sl_pct": float(leg_sl_pct),
            "profit_pct": float(profit_pct),
            "smart_selector_enabled": bool(smart_selector_enabled),
            "max_intraday_loss": float(max_intraday_loss),
            "intraday_target": float(intraday_target),
            "partial_target_pct": float(partial_target_pct),
            "allow_carry_forward": bool(allow_carry_forward),
            "max_carry_days": int(max_carry_days),
            "vix_carry_threshold": float(vix_carry_threshold),
            "max_daily_trades": int(max_daily_trades),
            "max_daily_loss_pct": float(max_daily_loss_pct),
            "avg_5m_volume": float(avg_5m_volume),
            "per_leg_sl_mult": float(per_leg_sl_mult),
            "per_leg_tp_mult": float(per_leg_tp_mult),
            "vix_adaptive_low": float(vix_adaptive_low),
            "vix_adaptive_high": float(vix_adaptive_high),
            "strangle_delta_low": float(strangle_delta_low),
            "strangle_delta_high": float(strangle_delta_high),
            "strangle_offset_low": float(strangle_offset_low),
            "strangle_offset_high": float(strangle_offset_high),
            "spread_short_delta_low": float(spread_short_delta_low),
            "spread_short_delta_high": float(spread_short_delta_high),
            "hedge_enabled": bool(hedge_enabled),
            "hedge_delta_breach": float(hedge_delta_breach),
            "hedge_premium_hard_x": float(hedge_premium_hard_x),
            "hedge_roll_distance": float(hedge_roll_distance),
            "hedge_delta_max": float(hedge_delta_max),
            "hedge_price_max": float(hedge_price_max),
            "hedge_salvage_wing_ltp": float(hedge_salvage),
            "nifty_expiry_weekday": expiry_weekday,
            "expiry_shift_if_holiday": bool(expiry_shift_if_holiday),
            "holiday_list": holiday_list,
            "nifty_lot_size": int(lot_size),
            "lot_size": int(lot_size),
            "gap_entry_threshold": float(gap_entry_threshold) / 100.0,
            "iv_floor_percentile": float(iv_floor_percentile),
            "batman_long_offset": float(bat_long_offset),
            "batman_short_offset": float(bat_short_offset),
            "batman_qty_long": int(bat_qty_long),
            "batman_qty_short": int(bat_qty_short),
            "batman_weekly_hedge_enabled": bool(bat_weekly_hedge_enabled),
            "batman_weekly_hedge_price_cap": float(bat_weekly_hedge_cap),
            "batman_weekly_hedge_min_distance": float(bat_weekly_hedge_min_dist),
        }, indent=2))
        st.success("Settings saved.")
    with st.expander("Auto Playbook"):
        pb = current.get("auto_playbook", DEFAULT_SETTINGS["auto_playbook"])
        hedge_on_delta = st.checkbox("Auto refresh hedges on delta breach", value=pb.get("hedge_on_delta", True))
        flatten_on_delta = st.checkbox("Auto flatten on delta breach", value=pb.get("flatten_on_delta", False))
        flatten_on_exposure = st.checkbox("Auto flatten on exposure breach", value=pb.get("flatten_on_exposure", False))
        if st.button("Save Playbook Settings"):
            merged = _load_settings()
            merged["auto_playbook"] = {
                "hedge_on_delta": hedge_on_delta,
                "flatten_on_delta": flatten_on_delta,
                "flatten_on_exposure": flatten_on_exposure,
            }
            SETTINGS_JSON.write_text(json.dumps(merged, indent=2))
            st.success("Playbook settings saved.")
        st.caption(
            "These toggles control automated responses when risk events trigger (delta/exposure blocks).\n"
            "Hedge refresh uses weekly hedges; flatten closes all legs."
        )


def _paper_pnl_tab() -> None:
    df, summary = _load_blotter()

    # Backtest block is aligned to the currently selected strategy
    current_strategy_file = st.session_state.get("strategy_file", "")
    strategy_label = {
        "batman": "Batman",
        "weekly_theta_strangle.py": "Weekly Theta Strangle",
        "monthly_strangle_with_weekly_hedge.py": "Monthly Strangle w/ Hedge",
    }.get(current_strategy_file, "Selected Strategy")

    # Strategy-specific backtest panels
    if current_strategy_file == "batman":
        with st.expander(f"Run Backtest ({strategy_label})", expanded=False):
            cols_dates = st.columns(2)
            with cols_dates[0]:
                start_date = st.date_input("Start date", key="bat_bt_start")
            with cols_dates[1]:
                end_date = st.date_input("End date", key="bat_bt_end")
            btn_run, btn_reset = st.columns(2)
            with btn_run:
                if st.button("Run Batman Backtest"):
                    env = os.environ.copy()
                    env["DHAN_CLIENT_ID"] = st.session_state.get("client_id", "")
                    env["DHAN_ACCESS_TOKEN"] = st.session_state.get("access_token", "")
                    env["BACKTEST_START"] = start_date.isoformat() if start_date else ""
                    env["BACKTEST_END"] = end_date.isoformat() if end_date else ""
                    try:
                        import subprocess
                        out = subprocess.run(
                            [PYTHON, str(ROOT / "data_engine/market_ai/scripts/run_batman_backtest_live.py")],
                            cwd=str(ROOT),
                            env=env,
                            capture_output=True,
                            text=True,
                            timeout=600,
                        )
                        st.code(out.stdout + "\n" + out.stderr, language="log")
                        st.success("Backtest complete. Check the table below.")
                        st.session_state["bt_show_latest"] = True
                    except Exception as exc:
                        st.error(f"Backtest failed: {exc}")
            with btn_reset:
                if st.button("Reset Backtest View"):
                    for k in list(st.session_state.keys()):
                        if k.startswith("bat_bt_") or k.startswith("bt_pb_") or k.startswith("bt_tf"):
                            st.session_state.pop(k, None)
                    st.session_state.pop("bt_tf", None)
                    st.session_state["bt_show_latest"] = False
                    st.success("Backtest view reset. Run a new backtest to see results.")

            # Backtest results (collapsed with run controls)
            show_latest = st.session_state.get("bt_show_latest", True)
            from glob import glob
            reports_dir = ROOT / "reports"
            bt_files = sorted(glob(str(reports_dir / "batman_backtest_trades_*.csv")))
            pb_files = sorted(glob(str(reports_dir / "batman_backtest_playback_*.csv")))
            if bt_files and show_latest:
                latest_bt = Path(bt_files[-1])
                st.markdown(f"Latest backtest trades: `{latest_bt.name}`")
                latest_pb = Path(pb_files[-1]) if pb_files else None
                try:
                    import pandas as pd
                    bt_df = pd.read_csv(latest_bt)
                    pb_df = pd.read_csv(latest_pb) if latest_pb and latest_pb.exists() else pd.DataFrame()
                    if not bt_df.empty:
                        st.markdown("#### Entries and Strikes")
                        st.dataframe(bt_df, use_container_width=True)
                        leg_cols = ["long_call", "short_call", "long_put", "short_put"]
                        if all(c in bt_df.columns for c in leg_cols):
                            settings = _load_settings()
                            qty_long = settings.get("batman_qty_long", 1)
                            qty_short = settings.get("batman_qty_short", 3)
                            st.markdown("##### Legs by entry")
                            legs_rows = []
                            for _, r in bt_df.iterrows():
                                legs_rows.append({
                                    "entry_time": r.get("entry_time"),
                                    "long_call": r.get("long_call"),
                                    "short_call": r.get("short_call"),
                                    "long_put": r.get("long_put"),
                                    "short_put": r.get("short_put"),
                                    "qty_long": qty_long,
                                    "qty_short": qty_short,
                                })
                            st.dataframe(pd.DataFrame(legs_rows), use_container_width=True)
                        else:
                            st.info("Strikes not available in file (expected columns: long_call, short_call, long_put, short_put).")
                        if not pb_df.empty:
                            st.markdown("#### Playback timeframe")
                            pb_df["timestamp"] = pd.to_datetime(pb_df["timestamp"], errors="coerce")
                            pb_df = pb_df.dropna(subset=["timestamp"]).sort_values("timestamp")
                            if pb_df.empty:
                                st.info("Playback data empty after parsing timestamps.")
                            else:
                                max_idx = len(pb_df) - 1
                                cur_idx = st.session_state.get("bt_pb_idx", max_idx)
                                cur_idx = st.slider("Cursor", 0, max_idx, cur_idx, key="bt_pb_slider_live")
                                st.session_state["bt_pb_idx"] = cur_idx
                                row = pb_df.iloc[cur_idx]
                                pnl = row.get("pnl", 0.0)
                                st.metric("PNL at cursor", f"₹{pnl:,.2f}", help=str(row.get("timestamp")))
                                legs_live = []
                                legs_live.append({"leg": "Long Call", "strike": row.get("long_call"), "qty": row.get("qty_long"), "entry": row.get("long_call_entry"), "ltp": row.get("long_call_ltp")})
                                legs_live.append({"leg": "Short Call", "strike": row.get("short_call"), "qty": row.get("qty_short"), "entry": row.get("short_call_entry"), "ltp": row.get("short_call_ltp")})
                                legs_live.append({"leg": "Long Put", "strike": row.get("long_put"), "qty": row.get("qty_long"), "entry": row.get("long_put_entry"), "ltp": row.get("long_put_ltp")})
                                legs_live.append({"leg": "Short Put", "strike": row.get("short_put"), "qty": row.get("qty_short"), "entry": row.get("short_put_entry"), "ltp": row.get("short_put_ltp")})
                                st.markdown("##### Legs at cursor")
                                st.table(legs_live)
                        else:
                            st.info("No playback file found (batman_backtest_playback_*.csv). Run backtest to generate.")
                except Exception as exc:
                    st.warning(f"Could not load backtest file: {exc}")
            else:
                st.info("Run a backtest to see results here.")

    elif current_strategy_file == "weekly_theta_strangle.py":
        with st.expander(f"Run Backtest ({strategy_label})", expanded=False):
            btn_run, btn_reset = st.columns(2)
            with btn_run:
                if st.button("Run Weekly Strangle Backtest"):
                    try:
                        import subprocess
                        out = subprocess.run(
                            [
                                PYTHON,
                                str(ROOT / "data_engine/market_ai/scripts/run_weekly_theta_backtest.py"),
                                "--input",
                                str(ROOT / "reports/intraday_from_rolling_latest.csv"),
                                "--output-dir",
                                str(ROOT / "reports/weekly_theta_backtest"),
                            ],
                            cwd=str(ROOT),
                            capture_output=True,
                            text=True,
                            timeout=600,
                        )
                        st.code(out.stdout + "\n" + out.stderr, language="log")
                        st.session_state["wt_bt_show"] = True
                        st.success("Weekly strangle backtest complete.")
                    except Exception as exc:
                        st.error(f"Backtest failed: {exc}")
            with btn_reset:
                if st.button("Reset Backtest View", key="wbt_reset"):
                    for k in list(st.session_state.keys()):
                        if k.startswith("wt_bt_"):
                            st.session_state.pop(k, None)
                    st.session_state["wt_bt_show"] = False
                    st.success("Backtest view reset.")

            show_latest = st.session_state.get("wt_bt_show", True)
            trades_path = ROOT / "reports/weekly_theta_backtest/weekly_theta_trades.csv"
            summary_path = ROOT / "reports/weekly_theta_backtest/weekly_theta_summary.json"
            daily_path = ROOT / "reports/weekly_theta_backtest/weekly_theta_daily_log.csv"
            if show_latest and trades_path.exists():
                st.markdown(f"Latest backtest trades: `{trades_path.name}`")
                try:
                    import pandas as pd
                    df_trades = pd.read_csv(trades_path)
                    st.dataframe(df_trades, use_container_width=True)
                except Exception as exc:
                    st.warning(f"Could not load trades file: {exc}")
            if summary_path.exists() and show_latest:
                try:
                    import json as _json
                    summary_data = _json.loads(summary_path.read_text())
                    st.json(summary_data)
                except Exception as exc:
                    st.warning(f"Could not load summary: {exc}")
            if daily_path.exists() and show_latest:
                try:
                    import pandas as pd
                    df_daily = pd.read_csv(daily_path)
                    st.markdown("#### Daily log")
                    st.dataframe(df_daily, use_container_width=True)
                except Exception as exc:
                    st.warning(f"Could not load daily log: {exc}")
            if not trades_path.exists() and not summary_path.exists():
                st.info("Run a backtest to see results here.")

    else:
        st.info(f"Backtest is not available for {strategy_label}.")

    total_orders = int(summary.get("total_orders", 0))
    executed_orders = int(summary.get("executed_orders", 0))
    warn_orders = int(summary.get("warn_only_orders", 0))
    credit = float(summary.get("credit_value", 0.0))
    debit = float(summary.get("debit_value", 0.0))
    net = float(summary.get("net_value", 0.0))

    c1, c2, c3 = st.columns(3)
    c1.metric("Orders Logged", total_orders, warn_orders, help="Total blotter entries (delta shows warn-only count)")
    c2.metric("Executed Orders", executed_orders)
    net_delta = credit if net >= 0 else -debit
    c3.metric("Net Credit", f"₹{net:,.2f}", f"₹{net_delta:,.2f}")

    if df is not None and not df.empty:
        st.markdown("#### Order Blotter")
        show_df = df.copy()
        if "timestamp" in show_df.columns:
            show_df["timestamp"] = pd.to_datetime(show_df["timestamp"], errors="coerce")
        for col in ("warn_only", "executed"):
            if col in show_df.columns:
                show_df[col] = pd.to_numeric(show_df[col], errors="coerce").astype('Int64')
        st.dataframe(show_df.fillna(""), width="stretch")

        executed_df = show_df[show_df.get("executed", 0) == 1].copy()
        if not executed_df.empty and "timestamp" in executed_df.columns:
            executed_df.sort_values("timestamp", inplace=True)
            executed_df["signed_notional"] = pd.to_numeric(executed_df.get("price"), errors="coerce").fillna(0.0) \
                * pd.to_numeric(executed_df.get("quantity"), errors="coerce").fillna(0.0) \
                * executed_df["side"].map({"SELL": 1.0, "BUY": -1.0}).fillna(0.0)
            executed_df["cumulative"] = executed_df["signed_notional"].cumsum()

            st.markdown("#### Cumulative Executed Credit")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=executed_df["timestamp"],
                y=executed_df["cumulative"],
                mode="lines+markers",
                name="Net Credit",
            ))
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=40))
        st.plotly_chart(fig, width="stretch")

    if summary:
        st.markdown("#### Latest Summary")
        st.json(summary)


def _weekly_plan_tab() -> None:
    _ensure_weekly_settings_state()
    st.markdown("## Weekly Plan & Intents")
    c1, c2 = st.columns([1, 1])
    script_path = globals().get("WEEKLY_PLAN_SCRIPT", Path("scripts/weekly_plan_cron.sh"))
    with c1:
        if st.button("Run weekly plan script now"):
            if script_path.exists():
                with st.spinner("Running weekly_plan_cron.sh ..."):
                    dir_flag = "1" if st.session_state.get("weekly_directional_enabled", True) else "0"
                    cmd = [
                        "bash",
                        str(script_path),
                        str(st.session_state.get("weekly_min_prev_range", 0.003)),
                        str(st.session_state.get("weekly_pnl_target", 6000.0)),
                        str(st.session_state.get("weekly_pnl_stop", 4000.0)),
                        dir_flag,
                        str(st.session_state.get("weekly_max_gap_pct", 0.01)),
                    ]
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                    )
                if result.returncode == 0:
                    st.success("Plan regenerated successfully.")
                else:
                    st.error("Script failed.")
                    st.code(result.stderr or result.stdout or "No output")
            else:
                st.error(f"Script not found: {script_path}")
    with c2:
        st.caption(f"Script path: `{script_path}`")
        st.caption("Cron entry already calls this every Monday 08:45 local.")

    with st.expander("Weekly Plan Settings", expanded=False):
        col1, col2 = st.columns(2)
        min_prev = float(st.session_state.get("weekly_min_prev_range", 0.003))
        pnl_target = float(st.session_state.get("weekly_pnl_target", 6000.0))
        pnl_stop = float(st.session_state.get("weekly_pnl_stop", 4000.0))
        max_gap = float(st.session_state.get("weekly_max_gap_pct", 0.01))
        directional_enabled = bool(st.session_state.get("weekly_directional_enabled", True))
        with col1:
            st.number_input(
                "Min prev-range pct",
                min_value=0.001,
                max_value=0.05,
                value=min_prev,
                step=0.001,
                key="weekly_min_prev_range",
            )
            st.number_input(
                "Profit target (₹)",
                min_value=1000.0,
                max_value=15000.0,
                value=pnl_target,
                step=500.0,
                key="weekly_pnl_target",
            )
            st.number_input(
                "Max gap % (skip if open > %)",
                min_value=0.001,
                max_value=0.05,
                value=max_gap,
                step=0.001,
                format="%.3f",
                key="weekly_max_gap_pct",
            )
        with col2:
            st.number_input(
                "Stop loss (₹)",
                min_value=1000.0,
                max_value=10000.0,
                value=pnl_stop,
                step=500.0,
                key="weekly_pnl_stop",
            )
            st.toggle(
                "Directional spreads enabled",
                value=directional_enabled,
                key="weekly_directional_enabled",
                help="Allow bull put / bear call spreads when trend bias is strong.",
            )
        if st.button("💾 Save Weekly Settings"):
            payload = {
                "min_prev_range": float(st.session_state["weekly_min_prev_range"]),
                "pnl_target": float(st.session_state["weekly_pnl_target"]),
                "pnl_stop": float(st.session_state["weekly_pnl_stop"]),
                "max_gap_pct": float(st.session_state["weekly_max_gap_pct"]),
                "directional_enabled": bool(st.session_state["weekly_directional_enabled"]),
            }
            try:
                _save_weekly_settings(payload)
                st.success(f"Settings saved to {WEEKLY_SETTINGS_FILE}")
            except Exception as exc:
                st.error(f"Failed to save settings: {exc}")
        st.caption(f"Settings file: `{WEEKLY_SETTINGS_FILE}`")

    st.markdown("### Option Chain Snapshots")
    option_script = globals().get("OPTION_CHAIN_SCRIPT")
    if option_script and option_script.exists():
        if st.button("Fetch live option chain snapshot", key="option_chain_snapshot"):
            ss = st.session_state
            if not ss.get("creds_verified") and not _verify_and_remember_creds():
                st.error("Please verify DHAN credentials first.")
            else:
                env = os.environ.copy()
                env["DHAN_CLIENT_ID"] = str(ss.get("client_id", env.get("DHAN_CLIENT_ID", "")))
                env["DHAN_ACCESS_TOKEN"] = str(ss.get("access_token", env.get("DHAN_ACCESS_TOKEN", "")))
                if not env["DHAN_CLIENT_ID"] or not env["DHAN_ACCESS_TOKEN"]:
                    st.error("Missing DHAN credentials.")
                else:
                    with st.spinner("Collecting option chain snapshot..."):
                        result = subprocess.run(
                            ["python3", str(option_script)],
                            capture_output=True,
                            text=True,
                            env=env,
                        )
                    if result.returncode == 0:
                        st.success("Snapshot saved under state/option_chain_history.")
                        if result.stdout:
                            st.code(result.stdout.strip()[-1000:])
                    else:
                        st.error("Option-chain collector failed.")
                        if result.stderr:
                            st.code(result.stderr.strip()[-2000:])
    else:
        st.info("Option-chain collector script not found.")

    fallback_plan_dir = Path(st.session_state.get("cwd_override") or ".") / "state"
    plan_path = globals().get("WEEKLY_PLAN_FILE", fallback_plan_dir / "weekly_plan.json")
    if plan_path.exists():
        try:
            plan = json.loads(plan_path.read_text())
        except Exception as exc:
            st.error(f"Failed to load plan: {exc}")
            plan = None
        if plan:
            st.markdown("### Plan Summary")
            summary = plan.get("summary") or {}
            cols = st.columns(3)
            cols[0].metric("Entries", summary.get("entries", 0))
            cols[1].metric("Closures", summary.get("closes", 0))
            cols[2].metric("Total Realized", f"₹{summary.get('total_realized', 0):,.0f}")
            st.markdown("#### Config Snapshot")
            st.json(plan.get("config", {}))
            latest = plan.get("latest_signal")
            if latest:
                st.markdown("#### Latest Signal")
                df = pd.DataFrame([latest])
                if "expiry" in df.columns:
                    st.caption(f"Expiry: {df.loc[0, 'expiry']}")
                st.dataframe(df, width="stretch")
    else:
        st.info(f"No plan file yet. Expected at `{plan_path}`.")

    intents = _load_recent_intents(limit=10)
    st.markdown("### Recent Weekly Intents")
    if intents:
        df = pd.DataFrame(intents)
        st.dataframe(df, width="stretch")
    else:
        st.caption("No intents logged yet.")

# =============================================================================
# Sidebar
# =============================================================================
def _sidebar() -> None:
    st.sidebar.markdown("### Broker Credentials")
    st.sidebar.text_input("Dhan Client ID", key="client_id", on_change=_mark_creds_dirty)
    st.sidebar.text_input("Dhan Access Token", key="access_token", type="password", on_change=_mark_creds_dirty)

    cred_status = "✅ Verified" if st.session_state.get("creds_verified") else "⚪️ Not verified"
    st.sidebar.caption(f"Credentials: {cred_status}")

    st.sidebar.markdown("### Agent Control")
    agent_running = st.session_state.get("agent_running", False)
    st.sidebar.radio(
        "Trade Mode",
        options=["paper", "live"],
        key="trade_mode",
        disabled=agent_running,
        help="Stop the agent before switching modes." if agent_running else None,
    )
    strategy_options = [
        ("Monthly Strangle w/ Hedge", "monthly_strangle_with_weekly_hedge.py"),
        ("Weekly Theta Strangle (alpha live)", "weekly_theta_strangle.py"),
        ("Batman Monthly (with hedges)", "batman"),
    ]
    # restore last strategy if present
    last_strategy_file = None
    if LAST_STRATEGY_FILE.exists():
        try:
            last_strategy_file = json.loads(LAST_STRATEGY_FILE.read_text()).get("strategy_file")
        except Exception:
            last_strategy_file = None
    if last_strategy_file and "strategy_choice" not in st.session_state:
        for opt in strategy_options:
            if opt[1] == last_strategy_file:
                st.session_state["strategy_choice"] = opt
                st.session_state["strategy_file"] = opt[1]
                break

    st.sidebar.selectbox(
        "Strategy",
        options=strategy_options,
        format_func=lambda x: x[0],
        key="strategy_choice",
        disabled=agent_running,
        help="Pick strategy before starting agent.",
    )
    # store selected file in session for env export
    choice = st.session_state.get("strategy_choice")
    if choice:
        st.session_state["strategy_file"] = choice[1]
        try:
            LAST_STRATEGY_FILE.write_text(json.dumps({"strategy_file": choice[1]}, indent=2))
        except Exception:
            pass
    if st.session_state.get("strategy_file") == "weekly_theta_strangle.py":
        st.sidebar.selectbox(
            "Weekly order type",
            options=["MARKET", "LIMIT"],
            key="weekly_order_type",
            disabled=agent_running,
        )
        st.sidebar.slider(
            "Weekly slippage (LIMIT, %)",
            min_value=0.1,
            max_value=1.0,
            step=0.05,
            value=0.2,
            key="weekly_slip_pct",
            help="Limit orders are placed at LTP*(1 - slip%).",
            disabled=agent_running or st.session_state.get("weekly_order_type", "MARKET") != "LIMIT",
        )
        st.sidebar.checkbox(
            "Weekly: enable hedge on drawdown",
            value=False,
            key="weekly_hedge_enabled",
            disabled=agent_running,
            help="Buys cheap wings when drawdown crosses the configured trigger.",
        )

    c1, c2 = st.sidebar.columns(2)
    if c1.button("▶️ Start Agent"):
        if _verify_and_remember_creds():
            start_agent()
            st.rerun()
    if c2.button("⏹ Stop Agent"):
        stop_agent()
        st.rerun()

    agent_mode = st.session_state.get("agent_started_mode")
    if agent_running:
        st.sidebar.caption(f"Agent: 🟢 Running ({agent_mode or st.session_state.get('trade_mode','live')})")
    else:
        st.sidebar.caption("Agent: 🔴 Stopped")

    st.sidebar.markdown("### Refresh")
    st.sidebar.slider("Refresh interval (sec)", 2, 30, key="refresh_sec", value=5)

# =============================================================================
# Main
# =============================================================================
def _rehydrate_dw_if_needed() -> None:
    """
    Make sure st.session_state['dw'] and ['creds_verified'] are live.
    This is resilient to cache clears / reruns.
    """
    ss = st.session_state
    if not ss.get("creds_verified"):
        return
    # Already valid?
    if ss.get("dw") is not None and ss.get("creds_verified") is True:
        return

    cid = (ss.get("client_id") or "").strip()
    tok = (ss.get("access_token") or "").strip()
    if not cid or not tok:
        return  # nothing to build from

    # Try constructing a fresh DhanWrapper
    dw = _new_dw()
    if not dw:
        ss["creds_verified"] = False
        ss["dw"] = None
        return

    ss["dw"] = dw
    ss["creds_verified"] = True
    _post_dw_setup(dw)

def main() -> None:
    st.set_page_config(page_title="Algo Agent Dashboard", layout="wide")
    _init_state()
    _sync_agent_state_from_pidfile()
    _sidebar()
    ss = st.session_state

    if ss.get("agent_running") and not ss.get("creds_verified"):
        _verify_and_remember_creds()

    _rehydrate_dw_if_needed()

    tab_labels = ["Trade", "Positions", "Strategy Monitor", "Observability", "Weekly Plan"]
    show_paper = ss.get("trade_mode") == "paper"
    if show_paper:
        tab_labels.append("Paper P&L")
    tab_labels.extend(["Agent Logs", "Settings"])

    tabs = st.tabs(tab_labels)

    idx = 0
    with tabs[idx]:
        st.markdown("## Funds Overview")
        _funds_cards(st.session_state.get("funds", {}))
        st.divider()
        _nifty_tile()
    idx += 1

    with tabs[idx]:
        _positions_tab(st.session_state.get("dw"))
    idx += 1

    with tabs[idx]:
        _strategy_monitor_tab()
    idx += 1

    with tabs[idx]:
        _observability_tab()
    idx += 1

    with tabs[idx]:
        _weekly_plan_tab()
    idx += 1

    if show_paper:
        with tabs[idx]:
            _paper_pnl_tab()
        idx += 1

    with tabs[idx]:
        _agent_logs_tab()
    idx += 1

    with tabs[idx]:
        _settings_tab()

    # gentle pause so the UI cadence feels steady with refresh slider
    time.sleep(max(0.1, float(st.session_state.get("refresh_sec", 5))))

def _load_recent_intents(limit: int = 10) -> List[Dict[str, Any]]:
    if not ORDER_INTENT_FILE.exists():
        return []
    try:
        lines = ORDER_INTENT_FILE.read_text().strip().splitlines()
    except Exception:
        return []
    entries: List[Dict[str, Any]] = []
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        entries.append(payload)
        if len(entries) >= limit:
            break
    return entries

if __name__ == "__main__":
    main()
