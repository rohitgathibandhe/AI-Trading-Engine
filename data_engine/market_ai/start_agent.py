#!/usr/bin/env python3
"""
Refactored agent entrypoint.

Uses the new StrategySelector framework (regime-aware) instead of the legacy
weekly_theta_strangle loop. Placement always goes to whichever mode (live vs paper)
is selected via credentials; there is no warn-only mode.
"""
from __future__ import annotations

import atexit
import json
import csv
import logging
import os
import sys
import time
from datetime import datetime, date as dt_date, time as dtime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # Python 3.9+
    from zoneinfo import ZoneInfo  # type: ignore
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

from logging.handlers import RotatingFileHandler

# ── Path/setup ────────────────────────────────────────────────────────────────
THIS_FILE = Path(__file__).resolve()
ENGINE_DIR = THIS_FILE.parent
DATA_ENGINE_DIR = ENGINE_DIR.parent
PROJECT_ROOT = DATA_ENGINE_DIR.parent
STATE_DIR = ENGINE_DIR / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
AGENT_LOG = STATE_DIR / "agent.log"
PID_FILE = STATE_DIR / "agent.pid"
FEATURE_LOG_PATH = STATE_DIR / "feature_history.csv"
INTEL_LOG_PATH = STATE_DIR / "intel_log.csv"
TRADE_BLOTTER_PATH = STATE_DIR / "trade_blotter.csv"
TRADE_BLOTTER_SUMMARY = STATE_DIR / "trade_blotter_summary.json"
ENTRY_STATUS_FILE = STATE_DIR / "entry_criteria_status.json"
STRATEGY_STATE_FILE = STATE_DIR / "strategy_state.json"
LIVE_GATE_STATUS_FILE = STATE_DIR / "live_gate_status.json"
LIVE_GATE_SESSIONS_FILE = STATE_DIR / "live_gate_sessions.jsonl"
POSITION_RECONCILE_STATUS_FILE = STATE_DIR / "position_reconcile_status.json"
EXECUTION_JOURNAL_FILE = STATE_DIR / "execution_journal.jsonl"
EXECUTION_RECOVERY_STATUS_FILE = STATE_DIR / "execution_recovery_status.json"
AGENT_HEARTBEAT_FILE = STATE_DIR / "agent_heartbeat.json"
AGENT_ALERTS_FILE = STATE_DIR / "agent_alerts.jsonl"
TELEGRAM_ALERT_STATUS_FILE = STATE_DIR / "telegram_alert_status.json"
BATMAN_BKM_AI_STATUS_FILE = STATE_DIR / "batman_bkm_ai_status.json"
BATMAN_BKM_AI_EVENTS_FILE = STATE_DIR / "batman_bkm_ai_events.jsonl"
BATMAN_BKM_AI_PROTECT_STATUS_FILE = STATE_DIR / "batman_bkm_ai_protect_status.json"
BATMAN_BKM_AI_PROTECT_CLEAR_REQUEST_FILE = STATE_DIR / "batman_bkm_ai_protect_clear_request.json"
INTRADAY_AI_ADVISOR_STATUS_FILE = STATE_DIR / "intraday_ai_advisor_status.json"
BLOTTER_FIELDS = [
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
    "delta",
    "expiry",
    "strike",
    "tag",
    "notes",
]
FEATURE_FIELDS = [
    "timestamp",
    "strategy",
    "expiry",
    "exchange_seg",
    "spot",
    "ce_strike",
    "pe_strike",
    "ce_ltp",
    "pe_ltp",
    "ce_delta",
    "pe_delta",
    "ce_entry",
    "pe_entry",
    "net_credit",
    "warn_only",
    "trade_mode",
    "context",
    "block_reason",
    "net_delta",
    "total_notional",
    "positions_summary",
]
_LAST_ENV_LOG: Optional[datetime] = None
DEFAULT_SETTINGS: Dict[str, Any] = {
    "lot_size": 65,
    "nifty_lot_size": 65,
    "nifty_expiry_weekday": "Tuesday",
    "expiry_shift_if_holiday": True,
    "holiday_list": [],
    "smart_selector_enabled": True,
    "max_daily_trades": 5,
    "max_daily_loss_pct": 0.03,
    "partial_target_pct": 0.65,
    "per_leg_sl_mult": 1.6,
    "per_leg_tp_mult": 0.5,
    "vix_adaptive_low": 12.0,
    "vix_adaptive_high": 20.0,
    "strangle_delta_low": 0.15,
    "strangle_delta_high": 0.15,
    "strangle_offset_low": 150.0,
    "strangle_offset_high": 150.0,
    "spread_short_delta_low": 0.25,
    "spread_short_delta_high": 0.25,
    # Batman V2 (paper) defaults
    "batman_v2_target_delta": 0.22,
    "batman_v2_delta_low": 0.20,
    "batman_v2_delta_high": 0.25,
    "batman_v2_net_credit_required": True,
    "batman_v2_max_entry_time": "10:30",
    "batman_v2_min_vix": 12.0,
    "batman_v2_max_gap_pct": 1.0,
    "batman_v2_monthly_band_low": 0.2,
    "batman_v2_monthly_band_high": 0.8,
    "batman_v2_lots": 1,
    "batman_v2_lot_size": 65,
    "batman_v2_capital": 500000.0,
    "batman_v2_max_loss_pct": -0.03,
    "batman_v2_max_both_delta": 0.45,
    "batman_v2_dte_exit": 3,
    "batman_v2_be_buffer": 50.0,
    "batman_v2_mtm_naked_loss_x": 1.8,
    "batman_v2_roll_recovery_x": 0.5,
    "batman_v2_roll_hold_dte": 3,
    # Batman BKM monthly defaults
    "batman_bkm_base_distance": 400,
    "batman_bkm_inner_step": 200,
    "batman_bkm_outer_step": 800,
    "batman_bkm_strike_rounding": 50,
    "batman_bkm_lot_multiplier": 1,
    "batman_bkm_max_credit_pct": 6.0,
    "batman_bkm_credit_step": 100,
    "batman_bkm_max_widen_iterations": 10,
    "batman_bkm_balance_tolerance": 5000.0,
    "batman_bkm_max_hedge_lots": 6,
    "batman_bkm_tp_pct": 0.02,
    "batman_bkm_sl_pct": 0.025,
    "batman_bkm_entry_time": "15:16",
    "batman_bkm_exit_time": "15:10",
    "batman_bkm_enable_balance": True,
    "batman_bkm_estimated_margin": 1_000_000.0,
    # For paper runs, allow immediate entry (bypass schedule) if needed
    "batman_bkm_force_entry": False,
    # Batman BKM live catch-up / entry planning safeguards
    "batman_bkm_catchup_not_before": "09:30",
    "batman_bkm_pretrade_snapshot_count": 3,
    "batman_bkm_pretrade_snapshot_gap_sec": 2.0,
    "batman_bkm_pretrade_max_spot_span_points": 120.0,
    "batman_bkm_pretrade_max_credit_span_pct": 0.40,
    "batman_bkm_pretrade_require_signature_stable": True,
    "batman_bkm_pretrade_enforce_delta_limit": False,
    "batman_bkm_pretrade_max_net_delta_abs": 5000.0,
    "batman_bkm_monitor_log_interval_sec": 20.0,
    "batman_bkm_market_closed_park_enabled": True,
    "batman_bkm_park_log_interval_sec": 300.0,
    # Batman BKM AI trade management advisor (monitor/recommend only by default)
    "batman_bkm_ai_enabled": True,
    "batman_bkm_ai_mode": "ADVISOR",
    "batman_bkm_ai_sample_window_sec": 600.0,
    "batman_bkm_ai_open_grace_sec": 180.0,
    "batman_bkm_ai_near_short_buffer_points": 200.0,
    "batman_bkm_ai_velocity_warn_pts_per_min": 90.0,
    "batman_bkm_ai_warn_score": 30.0,
    "batman_bkm_ai_reduce_score": 55.0,
    "batman_bkm_ai_flatten_score": 80.0,
    "batman_bkm_ai_loss_ratio_warn": 0.50,
    "batman_bkm_ai_loss_ratio_critical": 0.90,
    "batman_bkm_ai_drawdown_ratio_warn": 0.60,
    "batman_bkm_ai_drawdown_ratio_critical": 0.90,
    "batman_bkm_ai_event_emit_min_interval_sec": 30.0,
    "batman_bkm_ai_context_refresh_sec": 60.0,
    "batman_bkm_ai_near_atm_oi_band_points": 500.0,
    "batman_bkm_ai_pcr_bullish_threshold": 1.15,
    "batman_bkm_ai_pcr_bearish_threshold": 0.85,
    "batman_bkm_ai_trend_bias_threshold": 0.35,
    "batman_bkm_ai_context_risk_score_cap": 20.0,
    "batman_bkm_ai_protective_lock_min_action": "REDUCE_RISK",
    "batman_bkm_ai_protective_watch_escalates_alert": True,
    "batman_bkm_ai_protect_auto_unlock_enabled": True,
    "batman_bkm_ai_protect_auto_unlock_stable_sec": 1200.0,
    "batman_bkm_ai_protect_auto_unlock_max_score": 18.0,
    "batman_bkm_ai_protect_auto_unlock_require_action": "HOLD",
    "batman_bkm_ai_protect_auto_unlock_market_hours_only": True,
    "batman_bkm_ai_protect_auto_unlock_require_clean_system": True,
    # Intraday option-selling advisor (Phase A: advisory-only)
    "intraday_ai_enabled": True,
    "intraday_ai_refresh_sec": 60.0,
    "intraday_ai_market_open_time": "09:15",
    "intraday_ai_entry_not_before": "09:25",
    "intraday_ai_last_new_entry_time": "14:20",
    "intraday_ai_max_hold_till": "15:05",
    "intraday_ai_allow_parallel_with_bkm": False,
    "intraday_ai_no_trade_conflict_threshold": 60.0,
    "intraday_ai_directional_max_conflict": 50.0,
    "intraday_ai_range_max_conflict": 35.0,
    "intraday_ai_directional_min_trend_confidence": 0.62,
    "intraday_ai_range_max_trend_confidence": 0.58,
    "intraday_ai_min_sr_distance_points": 160.0,
    "intraday_ai_sr_safety_buffer_points": 80.0,
    "intraday_ai_fallback_otm_buffer_points": 220.0,
    "intraday_ai_spread_width_points_low_vol": 150,
    "intraday_ai_spread_width_points_normal_vol": 200,
    "intraday_ai_spread_width_points_high_vol": 250,
    "intraday_ai_ic_target_capture_pct": 0.30,
    "intraday_ai_spread_target_capture_pct": 0.40,
    "intraday_ai_ic_stop_credit_multiple": 1.8,
    "intraday_ai_spread_stop_credit_multiple": 1.6,
    "intraday_ai_min_credit_per_set_rs": 500.0,
    # Batman BKM live rollout gate
    "live_gate_enabled": True,
    "live_probation_sessions": 10,
    "live_probation_min_pass": 8,
    "live_stage1_lot_multiplier": 1,
    "live_stage2_lot_multiplier": 2,
    "live_daily_loss_cap_abs": 5000.0,
    "live_data_fail_zero_spot_count": 3,
    "live_data_fail_zero_spot_window_sec": 120,
    "live_data_fail_conn_count": 3,
    "live_data_fail_conn_window_sec": 600,
    "live_data_fail_action": "AUTO_FLATTEN_LOCK_DAY",
    "live_allow_normal_carry": True,
    "live_probation_cum_mtm_floor": -15000.0,
    "live_reconcile_enabled": True,
    "live_reconcile_grace_sec": 30,
    "live_reconcile_mismatch_confirm_count": 2,
    "live_reconcile_hard_lock": True,
    "live_exec_recovery_enabled": True,
    "live_exec_recovery_hard_lock": True,
    "live_exec_recovery_lookback_days": 45,
    "live_order_verify_enabled": True,
    "live_order_fill_wait_sec": 20,
    "live_order_fill_poll_sec": 2.0,
    "live_order_retry_count": 2,
    "live_order_settle_delay_sec": 1.0,
    "live_order_verify_positions": True,
    "ops_heartbeat_interval_sec": 10.0,
    "ops_watchdog_stale_after_sec": 45.0,
    "ops_alert_dedupe_sec": 60.0,
    "telegram_alerts_enabled": False,
    "telegram_alert_min_severity": "CRITICAL",
    "telegram_alert_live_only": True,
    "telegram_alert_poll_interval_sec": 5.0,
    "telegram_alert_timeout_sec": 10.0,
    "telegram_alert_max_batch": 5,
    "telegram_alert_max_message_chars": 3000,
    "telegram_alert_failure_backoff_sec": 30.0,
    "telegram_alert_disable_link_preview": True,
    "telegram_trade_summary_enabled": True,
    "telegram_trade_summary_interval_sec": 900.0,
    "telegram_trade_summary_market_hours_only": True,
    "telegram_market_close_summary_enabled": True,
    "telegram_market_close_summary_not_before": "15:32",
    "telegram_ai_lock_change_enabled": True,
    "telegram_intraday_signal_enabled": True,
    "telegram_intraday_signal_market_hours_only": True,
    "gap_entry_threshold": 0.004,
    "iv_floor_percentile": 0.2,
    "short_lots": 1,
    "hedge_lots_live": 1.0,
    "hedge_lots_paper": 0.33,
    "cycle_day_min": 1,
    "cycle_day_max": 7,
    "allow_early_next_cycle": True,
    "early_next_cycle_days": [5, 7],
    "monthly_filters": {
        "use_adx": True,
        "adx_length": 14,
        "adx_max": 25.0,
        "use_gap": True,
        "max_gap_pct": 0.8,
        "use_range_band": True,
        "range_band_min": 0.3,
        "range_band_max": 0.7,
        "use_vix": False,
        "min_vix": 12.0,
    },
}
CREDS_FILE = STATE_DIR / "creds.json"
INDEX_SECURITY_ID = int(os.getenv("MARKET_AI_INDEX_SECURITY_ID", "13"))
INDEX_EXCHANGE_SEG = os.getenv("MARKET_AI_INDEX_SEGMENT", "IDX_I")
INDEX_INSTRUMENT = os.getenv("MARKET_AI_INDEX_INSTRUMENT", "INDEX")
INTRADAY_INTERVAL_MIN = int(os.getenv("MARKET_AI_INTRADAY_INTERVAL", "5"))

_INTRADAY_CACHE: Dict[Tuple[str, int], Dict[str, Any]] = {}

for p in (ENGINE_DIR, DATA_ENGINE_DIR, PROJECT_ROOT, ENGINE_DIR / "strategies"):
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

from market_ai.dhan_wrapper import DhanWrapper  # noqa: E402
from market_ai.modules.data_fetch.dhan_scrip_cache import (  # noqa: E402
    resolve_option_security_id,
    refresh_scrip_master,
)
from market_ai.strategies import (  # noqa: E402
    StrategySelector,
    MarketSnapshot,
    OptionLeg,
    RiskConfig,
    StrategyType,
    OptionType,
    LegSide,
    TrendContext,
    TrendSide,
    ORBState,
    TradeAction,
)
# Monthly strangle manager (rule-based)
from market_ai.strategies.monthly_strangle_manager import (
    MonthlyStrangleConfig,
    manage_basket as manage_monthly_basket,
    propose_entry as propose_monthly_entry,
    _in_entry_window,
    _within_cycle_day,
)
from market_ai.signals.monthly_signals import MonthlyFiltersConfig
from market_ai.strategies.trend_detector import detect_trend_from_open
from market_ai.strategies.orb_detector import ORBConfig, compute_orb_levels, detect_orb_breakout
from market_ai.strategies.sr_levels import compute_sr_levels
from market_ai.engine.feature_extractor import FeatureExtractor
from market_ai.engine.regime_scorer import RegimeScorer
from market_ai.engine.policy_engine import PolicyEngine
from market_ai.engine.learning_manager import LearningManager
from market_ai.modules.strategies.batman_bkm_monthly import BatmanBKMConfig, BatmanBKMStrategy
from market_ai.modules.agents.live_gate import LiveGate, LiveGateConfig
from market_ai.modules.agents.position_reconciler import PositionReconciler, PositionReconcilerConfig
from market_ai.modules.agents.live_order_executor import LiveOrderExecutor, LiveOrderExecutorConfig
from market_ai.modules.agents.execution_recovery_guard import (
    ExecutionJournal,
    ExecutionRecoveryGuard,
    ExecutionRecoveryConfig,
)
from market_ai.modules.agents.ops_monitor import (
    AgentHeartbeat,
    HeartbeatConfig,
    AlertJournal,
    AlertConfig,
)
from market_ai.modules.agents.telegram_alerts import (
    TelegramAlertForwarder,
    TelegramAlertConfig,
)
from market_ai.modules.agents.batman_bkm_ai_manager import (
    BatmanBKMAIManager,
    BatmanBKMAIConfig,
)
from market_ai.modules.agents.intraday_option_selling_advisor import (
    IntradayOptionSellingAdvisor,
    IntradayOptionSellingAdvisorConfig,
)

# ── Logging ──────────────────────────────────────────────────────────────────
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
handler = RotatingFileHandler(AGENT_LOG, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
handler.setFormatter(formatter)
console = logging.StreamHandler(sys.stdout)
console.setFormatter(formatter)
root = logging.getLogger()
root.setLevel(logging.INFO)
root.handlers = [handler, console]
log = logging.getLogger("start_agent")
LAST_STRATEGY_FILE = STATE_DIR / "last_strategy.json"

# Which strategy file was chosen in the UI (if any)
SELECTED_STRATEGY_FILE = None
if LAST_STRATEGY_FILE.exists():
    try:
        SELECTED_STRATEGY_FILE = json.loads(LAST_STRATEGY_FILE.read_text()).get("strategy_file")
    except Exception:
        SELECTED_STRATEGY_FILE = None
if SELECTED_STRATEGY_FILE == "batman_v2_paper":
    # Migrate legacy selection to the new monthly Batman (BKM) strategy
    SELECTED_STRATEGY_FILE = "batman_bkm_monthly"
    try:
        LAST_STRATEGY_FILE.write_text(json.dumps({"strategy_file": SELECTED_STRATEGY_FILE}, indent=2))
    except Exception:
        pass

# Cache for daily candles to compute higher-timeframe filters
_DAILY_CACHE: Dict[str, List[Dict[str, Any]]] = {}
_MONTHLY_FILTER_DEFAULTS = DEFAULT_SETTINGS["monthly_filters"]


def _fetch_daily_candles(dw: DhanWrapper, days: int = 40) -> List[Dict[str, Any]]:
    today = dt_date.today()
    start = (today - timedelta(days=days)).isoformat()
    end = today.isoformat()
    key = f"{start}:{end}"
    cached = _DAILY_CACHE.get(key)
    if cached:
        return cached
    candles = dw.get_daily_candles(
        security_id=INDEX_SECURITY_ID,
        exchange_segment=INDEX_EXCHANGE_SEG,
        instrument_type=INDEX_INSTRUMENT,
        from_date=start,
        to_date=end,
    )
    _DAILY_CACHE[key] = candles
    return candles


def _compute_adx(candles: List[Dict[str, Any]], period: int = 14) -> Optional[float]:
    """
    Minimal ADX implementation on daily candles (dicts with high/low/close).
    Returns None if insufficient data.
    """
    if len(candles) < period + 1:
        return None
    trs = []
    plus_dm = []
    minus_dm = []
    for i in range(1, len(candles)):
        h = candles[i]["high"]
        l = candles[i]["low"]
        c_prev = candles[i - 1]["close"]
        tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
        up_move = h - candles[i - 1]["high"]
        down_move = candles[i - 1]["low"] - l
        trs.append(tr)
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0)
    def _ewm(vals):
        alpha = 1 / period
        sm = vals[0]
        out = []
        for v in vals:
            sm = sm + alpha * (v - sm)
            out.append(sm)
        return out
    trn = _ewm(trs)
    pdmn = _ewm(plus_dm)
    mdmn = _ewm(minus_dm)
    dx = []
    for t, p, m in zip(trn, pdmn, mdmn):
        if t == 0:
            dx.append(0.0)
            continue
        pdi = 100 * (p / t)
        mdi = 100 * (m / t)
        denom = pdi + mdi
        dx.append(0.0 if denom == 0 else 100 * abs(pdi - mdi) / denom)
    adx = _ewm(dx)[-1] if dx else None
    return adx


def _compute_monthly_filters(candles: List[Dict[str, Any]], spot: float) -> Dict[str, float]:
    """
    Compute ADX, max_body_pct (last 3 days), gap_pct, monthly_range_frac.
    """
    res = {"adx": None, "max_body_pct": None, "gap_pct": None, "monthly_range_frac": None}
    if not candles:
        return res
    # ADX
    res["adx"] = _compute_adx(candles, period=14)
    # Gap pct from last two daily candles
    if len(candles) >= 2:
        prev_close = candles[-2]["close"]
        today_open = candles[-1]["open"]
        if prev_close:
            res["gap_pct"] = (today_open - prev_close) / prev_close * 100.0
    # Max body pct last 3 days
    bodies = []
    for c in candles[-3:]:
        body = abs(c["close"] - c["open"])
        ref = c["open"] if c["open"] else 1.0
        bodies.append(body / ref * 100.0)
    if bodies:
        res["max_body_pct"] = max(bodies)
    # Monthly range fraction over last 20 days
    window = candles[-20:] if len(candles) >= 20 else candles
    highs = [c["high"] for c in window]
    lows = [c["low"] for c in window]
    hi = max(highs) if highs else spot
    lo = min(lows) if lows else spot
    rng = hi - lo if hi != lo else 1.0
    res["monthly_range_frac"] = (spot - lo) / rng
    return res


def _build_monthly_filters_config(settings: Dict[str, Any]) -> MonthlyFiltersConfig:
    data = settings.get("monthly_filters", {}) if settings else {}
    merged = {**_MONTHLY_FILTER_DEFAULTS, **(data or {})}
    return MonthlyFiltersConfig.from_dict(merged)


def _paper_positions_from_blotter() -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(TRADE_BLOTTER_PATH)
    except Exception:
        return None
    if df.empty:
        return None
    df = df[df.get("trade_mode") == "paper"].copy()
    if df.empty:
        return None
    # signed qty
    def _signed(row):
        try:
            qty = int(row.get("quantity") or 0)
        except Exception:
            qty = 0
        side = str(row.get("side") or "").upper()
        return qty if side == "SELL" else -qty
    df["signed_qty"] = df.apply(_signed, axis=1)
    grouped = df.groupby(["strike"], as_index=False).agg(
        {
            "signed_qty": "sum",
            "price": "mean",
            "delta": "mean",
            "expiry": "last",
        }
    )
    grouped.rename(columns={"signed_qty": "net_qty"}, inplace=True)
    grouped["product"] = "PAPER"
    return grouped


def load_strategy_state() -> Dict[str, Any]:
    if not STRATEGY_STATE_FILE.exists():
        return {}
    try:
        data = json.loads(STRATEGY_STATE_FILE.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        log.exception("Failed to load strategy state")
        return {}


def save_strategy_state(state: Dict[str, Any]) -> None:
    try:
        STRATEGY_STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception:
        log.exception("Failed to save strategy state")


def is_monthly_strangle_open(expiry: str) -> bool:
    state = load_strategy_state()
    return (
        state.get("MONTHLY_STRANGLE", {})
        .get(expiry, {})
        .get("status")
        == "OPEN"
    )


def _build_monthly_basket_id(expiry: str) -> str:
    expiry_token = expiry.replace("-", "")
    timestamp = _ist_now().strftime("%H%M%S")
    return f"MS_{expiry_token}_{timestamp}"


def _mark_monthly_strangle_open(expiry: str, basket_id: str) -> None:
    state = load_strategy_state()
    bucket = state.setdefault("MONTHLY_STRANGLE", {})
    bucket[expiry] = {
        "status": "OPEN",
        "opened_at": _now_iso(),
        "basket_id": basket_id,
    }
    save_strategy_state(state)


def _mark_monthly_strangle_closed(expiry: str) -> None:
    state = load_strategy_state()
    bucket = state.setdefault("MONTHLY_STRANGLE", {})
    entry = bucket.setdefault(expiry, {})
    entry["status"] = "CLOSED"
    entry["closed_at"] = _now_iso()
    save_strategy_state(state)


def _is_batman_blocked(expiry: str) -> bool:
    state = load_strategy_state()
    entry = state.get("BATMAN_V2", {}).get(expiry, {})
    return entry.get("status") in {"OPEN", "CLOSED"}


def _mark_batman_open(expiry: str) -> None:
    state = load_strategy_state()
    bucket = state.setdefault("BATMAN_V2", {})
    bucket[expiry] = {"status": "OPEN", "opened_at": _now_iso()}
    save_strategy_state(state)


def _mark_batman_closed(expiry: str, reason: str = "") -> None:
    state = load_strategy_state()
    bucket = state.setdefault("BATMAN_V2", {})
    entry = bucket.setdefault(expiry, {})
    entry["status"] = "CLOSED"
    entry["closed_at"] = _now_iso()
    if reason:
        entry["reason"] = reason
    save_strategy_state(state)


def _clear_batman_expiry(expiry: str) -> None:
    state = load_strategy_state()
    bucket = state.get("BATMAN_V2", {})
    if expiry in bucket:
        bucket.pop(expiry)
        state["BATMAN_V2"] = bucket
        save_strategy_state(state)


def _is_bkm_blocked(expiry: str) -> bool:
    state = load_strategy_state()
    return state.get("BATMAN_BKM", {}).get(expiry, {}).get("status") in {"OPEN", "CLOSED"}


def _mark_bkm_open(expiry: str, meta: Optional[Dict[str, Any]] = None) -> None:
    state = load_strategy_state()
    bucket = state.setdefault("BATMAN_BKM", {})
    bucket[expiry] = {"status": "OPEN", "opened_at": _now_iso()}
    if meta:
        bucket[expiry].update(meta)
    save_strategy_state(state)


def _clear_bkm_expiry(expiry: str) -> None:
    state = load_strategy_state()
    bucket = state.get("BATMAN_BKM", {})
    if expiry in bucket:
        bucket.pop(expiry, None)
        state["BATMAN_BKM"] = bucket
        save_strategy_state(state)


def _mark_bkm_closed(expiry: str, reason: str = "", pnl: Optional[float] = None) -> None:
    state = load_strategy_state()
    bucket = state.setdefault("BATMAN_BKM", {})
    entry = bucket.setdefault(expiry, {})
    entry["status"] = "CLOSED"
    entry["closed_at"] = _now_iso()
    if reason:
        entry["reason"] = reason
    if pnl is not None:
        entry["pnl"] = pnl
    save_strategy_state(state)


def _rebuild_bkm_basket_from_blotter(expiry: datetime.date, trade_mode: str, cfg: "BatmanBKMConfig") -> Optional["BatmanBKMBasket"]:
    """
    On restart, reconstruct the BKM basket legs from the blotter so MTM can resume.
    """
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return None
    if not TRADE_BLOTTER_PATH.exists():
        return None
    try:
        df = pd.read_csv(TRADE_BLOTTER_PATH)
    except Exception:
        return None
    if df.empty:
        return None
    try:
        df = df[df.get("trade_mode").str.lower() == trade_mode]
        df = df[df.get("expiry") == expiry.isoformat()]
        df = df.sort_values("timestamp")
    except Exception:
        return None
    if df.empty:
        return None
    legs = {}
    for _, row in df.iterrows():
        key = (str(row.get("security_id") or ""), str(row.get("strike") or ""))
        side = str(row.get("side") or "").upper()
        qty = int(row.get("quantity") or 0)
        price = float(row.get("price") or 0.0)
        notes = str(row.get("notes") or "").upper()
        leg = legs.setdefault(
            key,
            {
                "option_type": None,
                "side": side,
                "strike": float(row.get("strike") or 0.0),
                "qty": 0,
                "entry": None,
                "ltp": None,
                "security_id": str(row.get("security_id") or ""),
            },
        )
        if notes == "OPEN" and leg["entry"] is None:
            leg["entry"] = price
            leg["qty"] = qty
            # infer option type from SecID heuristic if present; otherwise leave None
        if notes == "MTM":
            leg["ltp"] = price
            leg["qty"] = qty
            leg["side"] = side
    built_legs = []
    from market_ai.modules.strategies.batman_bkm_monthly import Leg as BKMLeg, BatmanBKMBasket  # type: ignore
    for leg in legs.values():
        if leg["entry"] is None:
            continue
        built_legs.append(
            BKMLeg(
                option_type=leg.get("option_type") or "",  # left blank if unknown
                side=leg.get("side") or "SELL",
                strike=leg.get("strike") or 0.0,
                qty=int(leg.get("qty") or 0),
                entry=float(leg.get("entry") or 0.0),
                ltp=leg.get("ltp"),
                security_id=leg.get("security_id"),
                expiry=expiry.isoformat(),
            )
        )
    if not built_legs:
        return None
    net_credit = sum((1 if l.side == "SELL" else -1) * l.entry * l.qty for l in built_legs)
    margin = float(cfg.estimated_margin)
    credit_pct = (net_credit / margin) * 100 if margin else 0.0
    basket = BatmanBKMBasket(
        expiry=expiry,
        legs=built_legs,
        net_credit=net_credit,
        margin_required=margin,
        credit_pct=credit_pct,
        entry_ts=_ist_now(),
        hedge_qty_call=2 * cfg.lot_multiplier,
        hedge_qty_put=2 * cfg.lot_multiplier,
    )
    return basket


def _load_saved_creds() -> Dict[str, str]:
    if not CREDS_FILE.exists():
        return {}
    try:
        data = json.loads(CREDS_FILE.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_entry_status(payload: Dict[str, Any]) -> None:
    """Persist entry criteria status for the UI."""
    try:
        ENTRY_STATUS_FILE.write_text(json.dumps(payload, indent=2, default=str))
    except Exception:
        log.exception("Failed to write entry status")


def _ensure_dhan_credentials() -> None:
    saved = _load_saved_creds()
    cid_file = str(saved.get("client_id") or "").strip()
    tok_file = str(saved.get("access_token") or "").strip()
    cid_env = os.getenv("DHAN_CLIENT_ID", "").strip()
    tok_env = os.getenv("DHAN_ACCESS_TOKEN", "").strip()
    # Always prefer the saved creds if available (overwrites stale env)
    if cid_file:
        os.environ["DHAN_CLIENT_ID"] = cid_file
        if cid_file != cid_env:
            log.info("DHAN_CLIENT_ID loaded from %s (overwrote env)", CREDS_FILE)
    if tok_file:
        os.environ["DHAN_ACCESS_TOKEN"] = tok_file
        if tok_file != tok_env:
            log.info("DHAN_ACCESS_TOKEN loaded from %s (overwrote env)", CREDS_FILE)
    cid_final = os.getenv("DHAN_CLIENT_ID", "").strip()
    tok_final = os.getenv("DHAN_ACCESS_TOKEN", "").strip()
    if not cid_final or not tok_final:
        log.warning("DHAN credentials not found in env or %s; agent will fail to authenticate.", CREDS_FILE)


def _startup_preflight(*, strategy_file: Optional[str], trade_mode_env: Optional[str]) -> str:
    mode_raw = (trade_mode_env or "").strip().lower()
    if mode_raw not in {"paper", "live"}:
        raise RuntimeError("TRADE_MODE must be explicitly set to 'paper' or 'live'")
    if strategy_file != "batman_bkm_monthly":
        raise RuntimeError(
            f"Live gate rollout requires strategy_file='batman_bkm_monthly', got: {strategy_file!r}"
        )
    if not CREDS_FILE.exists():
        raise RuntimeError(f"Credentials file missing: {CREDS_FILE}")
    creds = _load_saved_creds()
    client_id = str(creds.get("client_id") or "").strip()
    access_token = str(creds.get("access_token") or "").strip()
    if not client_id or not access_token:
        raise RuntimeError(f"Credentials incomplete in {CREDS_FILE}; client_id/access_token required")
    return mode_raw


def _load_agent_settings() -> Dict[str, Any]:
    path = os.getenv("AGENT_SETTINGS_JSON")
    if not path:
        default_path = STATE_DIR / "agent_settings.json"
        if default_path.exists():
            path = str(default_path)
    if not path:
        return dict(DEFAULT_SETTINGS)
    try:
        data = json.loads(Path(path).read_text())
        if not isinstance(data, dict):
            raise TypeError("settings json is not a dict")
        merged = dict(DEFAULT_SETTINGS)
        merged.update(data)
        # keep lot_size and nifty_lot_size in sync if only one is provided
        if "nifty_lot_size" not in merged:
            merged["nifty_lot_size"] = merged.get("lot_size", DEFAULT_SETTINGS["lot_size"])
        if "lot_size" not in merged:
            merged["lot_size"] = merged.get("nifty_lot_size", DEFAULT_SETTINGS["nifty_lot_size"])
        # ensure monthly_filters present and merged
        mf_default = DEFAULT_SETTINGS.get("monthly_filters", {})
        mf_data = data.get("monthly_filters", {}) if isinstance(data, dict) else {}
        mf_merged = {**mf_default, **(mf_data or {})}
        merged["monthly_filters"] = mf_merged
        return merged
    except Exception as exc:
        log.warning("Failed to load agent settings from %s (%s); falling back to defaults", path, exc)
        return dict(DEFAULT_SETTINGS)


def _coerce_option_type_from_bkm(leg: Any) -> OptionType:
    raw = str(getattr(leg, "option_type", "") or "").upper()
    if raw in {"PE", "PUT", "P"}:
        return OptionType.PUT
    return OptionType.CALL


def _coerce_expiry_date(raw_expiry: Any) -> dt_date:
    if isinstance(raw_expiry, dt_date):
        return raw_expiry
    try:
        return datetime.fromisoformat(str(raw_expiry)).date()
    except Exception:
        return datetime.now().date()


def _bkm_leg_to_option_leg(leg: Any) -> OptionLeg:
    return OptionLeg(
        symbol="NIFTY",
        expiry=_coerce_expiry_date(getattr(leg, "expiry", None)),
        strike=float(getattr(leg, "strike", 0.0) or 0.0),
        option_type=_coerce_option_type_from_bkm(leg),
        side=LegSide.SELL if str(getattr(leg, "side", "")).upper() == "SELL" else LegSide.BUY,
        quantity=int(getattr(leg, "qty", 0) or 0),
        entry_price=float(getattr(leg, "entry", 0.0) or 0.0),
        security_id=str(getattr(leg, "security_id", "") or ""),
        current_ltp=getattr(leg, "ltp", None),
        strategy_type=StrategyType.NONE,
    )


def _bkm_leg_match_key(leg: OptionLeg) -> Tuple[str, float, str]:
    expiry = leg.expiry.isoformat() if hasattr(leg.expiry, "isoformat") else str(leg.expiry)
    strike = float(leg.strike or 0.0)
    opt = leg.option_type.value if hasattr(leg.option_type, "value") else str(leg.option_type)
    return (expiry, strike, str(opt).upper())


def _bkm_option_leg_to_journal_dict(leg: OptionLeg) -> Dict[str, Any]:
    opt = leg.option_type.value if hasattr(leg.option_type, "value") else str(leg.option_type)
    side = leg.side.value if hasattr(leg.side, "value") else str(leg.side)
    return {
        "expiry": leg.expiry.isoformat() if hasattr(leg.expiry, "isoformat") else str(leg.expiry),
        "strike": float(leg.strike or 0.0),
        "option_type": str(opt).upper(),
        "side": str(side).upper(),
        "quantity": int(leg.quantity or 0),
        "entry_price": float(getattr(leg, "entry_price", 0.0) or 0.0),
        "security_id": str(getattr(leg, "security_id", "") or ""),
    }


def _bkm_journal_event(
    *,
    journal: Optional[ExecutionJournal],
    event_type: str,
    op_id: Optional[str],
    expiry: str,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    if not journal:
        return
    row: Dict[str, Any] = {
        "strategy": "BATMAN_BKM",
        "expiry": expiry,
    }
    if op_id:
        row["op_id"] = op_id
    if payload:
        row.update(payload)
    journal.record(event_type, row, when=_ist_now())


def _build_bkm_basket_from_journal_snapshot(snapshot: Dict[str, Any], cfg: "BatmanBKMConfig") -> Optional["BatmanBKMBasket"]:
    if not isinstance(snapshot, dict):
        return None
    expiry_raw = snapshot.get("expiry")
    if not expiry_raw:
        return None
    try:
        expiry = datetime.fromisoformat(str(expiry_raw)).date()
    except Exception:
        return None
    legs_payload = snapshot.get("legs")
    if not isinstance(legs_payload, list) or not legs_payload:
        return None
    from market_ai.modules.strategies.batman_bkm_monthly import Leg as BKMLeg, BatmanBKMBasket  # type: ignore

    built_legs = []
    for item in legs_payload:
        if not isinstance(item, dict):
            continue
        opt = str(item.get("option_type") or "").upper()
        if opt == "CALL":
            opt = "CE"
        elif opt == "PUT":
            opt = "PE"
        side = str(item.get("side") or "SELL").upper()
        try:
            strike = float(item.get("strike") or 0.0)
            qty = int(float(item.get("quantity") or item.get("qty") or 0))
            entry_price = float(item.get("entry_price") or item.get("entry") or 0.0)
        except Exception:
            continue
        built_legs.append(
            BKMLeg(
                option_type=opt or "CE",
                side=side if side in {"BUY", "SELL"} else "SELL",
                strike=strike,
                qty=qty,
                entry=entry_price,
                ltp=item.get("ltp"),
                security_id=str(item.get("security_id") or ""),
                expiry=expiry.isoformat(),
            )
        )
    if not built_legs:
        return None
    meta = snapshot.get("meta") if isinstance(snapshot.get("meta"), dict) else {}
    net_credit = meta.get("net_credit")
    margin_required = meta.get("margin_required")
    credit_pct = meta.get("credit_pct")
    try:
        net_credit_f = float(net_credit) if net_credit is not None else sum(
            (1 if str(l.side).upper() == "SELL" else -1) * float(l.entry) * int(l.qty) for l in built_legs
        )
    except Exception:
        net_credit_f = 0.0
    try:
        margin_f = float(margin_required) if margin_required is not None else float(cfg.estimated_margin)
    except Exception:
        margin_f = float(cfg.estimated_margin)
    try:
        credit_pct_f = float(credit_pct) if credit_pct is not None else ((net_credit_f / margin_f) * 100.0 if margin_f else 0.0)
    except Exception:
        credit_pct_f = 0.0
    return BatmanBKMBasket(
        expiry=expiry,
        legs=built_legs,
        net_credit=net_credit_f,
        margin_required=margin_f,
        credit_pct=credit_pct_f,
        entry_ts=_ist_now(),
        hedge_qty_call=2 * cfg.lot_multiplier,
        hedge_qty_put=2 * cfg.lot_multiplier,
    )


def _rollback_bkm_partial_open(
    *,
    dw: DhanWrapper,
    planned_legs: List[OptionLeg],
    live_order_executor: Optional[LiveOrderExecutor],
) -> Dict[str, Any]:
    if not planned_legs:
        return {"ok": True, "submitted_close_legs": 0, "errors": []}
    try:
        broker_positions_raw = dw.get_positions_raw()
        broker_legs = _map_positions(broker_positions_raw)
    except Exception as exc:
        log.exception("[BatmanBKM] rollback failed to fetch broker positions")
        return {"ok": False, "submitted_close_legs": 0, "errors": [f"positions_fetch_failed:{exc}"]}

    target_keys = {_bkm_leg_match_key(leg) for leg in planned_legs}
    to_close = [leg for leg in broker_legs if _bkm_leg_match_key(leg) in target_keys]
    if not to_close:
        return {"ok": True, "submitted_close_legs": 0, "errors": []}

    errors: List[Dict[str, Any]] = []
    submitted_close_legs = 0
    for broker_leg in to_close:
        close_res = _place_leg_order(
            dw,
            broker_leg,
            close=True,
            trade_mode="live",
            live_order_executor=live_order_executor,
        )
        if bool(close_res.get("ok")):
            submitted_close_legs += 1
        else:
            errors.append(
                {
                    "strike": broker_leg.strike,
                    "opt": broker_leg.option_type.value if hasattr(broker_leg.option_type, "value") else str(broker_leg.option_type),
                    "qty": broker_leg.quantity,
                    "error": close_res,
                }
            )
    return {"ok": len(errors) == 0, "submitted_close_legs": submitted_close_legs, "errors": errors}


def _execute_bkm_open_live(
    *,
    dw: DhanWrapper,
    basket: Any,
    live_order_executor: Optional[LiveOrderExecutor],
    execution_journal: Optional[ExecutionJournal] = None,
) -> Dict[str, Any]:
    option_legs: List[OptionLeg] = []
    for raw_leg in getattr(basket, "legs", []) or []:
        try:
            option_legs.append(_bkm_leg_to_option_leg(raw_leg))
        except Exception:
            continue
    if not option_legs:
        return {"ok": False, "opened_legs": 0, "planned_legs": 0, "error": "no_option_legs"}
    expiry_str = basket.expiry.isoformat() if hasattr(basket, "expiry") else ""
    op_id = f"BKMOPEN-{int(time.time() * 1000)}"
    _bkm_journal_event(
        journal=execution_journal,
        event_type="BKM_OPEN_BEGIN",
        op_id=op_id,
        expiry=expiry_str,
        payload={
            "legs": [_bkm_option_leg_to_journal_dict(leg) for leg in option_legs],
            "meta": {
                "net_credit": float(getattr(basket, "net_credit", 0.0) or 0.0),
                "margin_required": float(getattr(basket, "margin_required", 0.0) or 0.0),
                "credit_pct": float(getattr(basket, "credit_pct", 0.0) or 0.0),
            },
        },
    )

    opened_legs = 0
    errors: List[Dict[str, Any]] = []
    for leg in option_legs:
        open_res = _place_leg_order(
            dw,
            leg,
            close=False,
            trade_mode="live",
            live_order_executor=live_order_executor,
        )
        if bool(open_res.get("ok")):
            opened_legs += 1
            continue
        errors.append(
            {
                "strike": leg.strike,
                "opt": leg.option_type.value if hasattr(leg.option_type, "value") else str(leg.option_type),
                "qty": leg.quantity,
                "error": open_res,
            }
        )
        break

    rollback: Dict[str, Any] = {"ok": True, "submitted_close_legs": 0, "errors": []}
    if errors:
        rollback = _rollback_bkm_partial_open(
            dw=dw,
            planned_legs=option_legs,
            live_order_executor=live_order_executor,
        )
    out = {
        "ok": len(errors) == 0,
        "opened_legs": opened_legs,
        "planned_legs": len(option_legs),
        "errors": errors,
        "rollback": rollback,
        "op_id": op_id,
    }
    if out["ok"]:
        _bkm_journal_event(
            journal=execution_journal,
            event_type="BKM_OPEN_SUCCESS",
            op_id=op_id,
            expiry=expiry_str,
            payload={
                "legs": [_bkm_option_leg_to_journal_dict(leg) for leg in option_legs],
                "meta": {
                    "net_credit": float(getattr(basket, "net_credit", 0.0) or 0.0),
                    "margin_required": float(getattr(basket, "margin_required", 0.0) or 0.0),
                    "credit_pct": float(getattr(basket, "credit_pct", 0.0) or 0.0),
                },
                "details": {"opened_legs": opened_legs, "planned_legs": len(option_legs)},
            },
        )
    else:
        _bkm_journal_event(
            journal=execution_journal,
            event_type="BKM_OPEN_FAIL",
            op_id=op_id,
            expiry=expiry_str,
            payload={"details": out},
        )
    return out


def _flatten_bkm_basket(
    *,
    dw: DhanWrapper,
    bkm_strategy: BatmanBKMStrategy,
    trade_mode: str,
    reason: str,
    live_order_executor: Optional[LiveOrderExecutor] = None,
    execution_journal: Optional[ExecutionJournal] = None,
) -> Dict[str, Any]:
    basket = bkm_strategy.basket
    if not basket:
        return {"ok": True, "closed_legs": 0, "planned_legs": 0, "errors": []}
    expiry_str = basket.expiry.isoformat() if hasattr(basket, "expiry") else ""
    op_id = f"BKMCLOSE-{int(time.time() * 1000)}"
    option_legs: List[OptionLeg] = []
    for raw_leg in basket.legs:
        try:
            option_legs.append(_bkm_leg_to_option_leg(raw_leg))
        except Exception:
            continue
    _bkm_journal_event(
        journal=execution_journal,
        event_type="BKM_CLOSE_BEGIN",
        op_id=op_id,
        expiry=expiry_str,
        payload={
            "reason": reason,
            "legs": [_bkm_option_leg_to_journal_dict(leg) for leg in option_legs],
        },
    )
    errors: List[Dict[str, Any]] = []
    closed_ok = 0
    for leg in option_legs:
        close_res = _place_leg_order(
            dw,
            leg,
            close=True,
            trade_mode=trade_mode,
            live_order_executor=live_order_executor if trade_mode == "live" else None,
        )
        if bool(close_res.get("ok")):
            closed_ok += 1
            continue
        errors.append(
            {
                "strike": leg.strike,
                "opt": leg.option_type.value if hasattr(leg.option_type, "value") else str(leg.option_type),
                "qty": leg.quantity,
                "error": close_res,
            }
        )
    if errors:
        log.error(
            "[BatmanBKM] close verification failed reason=%s closed_ok=%s planned=%s errors=%s",
            reason,
            closed_ok,
            len(option_legs),
            errors,
        )
        out = {"ok": False, "closed_legs": closed_ok, "planned_legs": len(option_legs), "errors": errors, "op_id": op_id}
        _bkm_journal_event(
            journal=execution_journal,
            event_type="BKM_CLOSE_FAIL",
            op_id=op_id,
            expiry=expiry_str,
            payload={"reason": reason, "details": out},
        )
        return out
    try:
        _log_batman_blotter(trade_mode, basket.legs, "CLOSE")
    except Exception:
        pass
    _mark_bkm_closed(basket.expiry.isoformat(), reason, basket.mtm())
    _bkm_journal_event(
        journal=execution_journal,
        event_type="BKM_CLOSE_SUCCESS",
        op_id=op_id,
        expiry=expiry_str,
        payload={
            "reason": reason,
            "legs": [_bkm_option_leg_to_journal_dict(leg) for leg in option_legs],
            "details": {"closed_legs": len(option_legs), "planned_legs": len(option_legs)},
        },
    )
    bkm_strategy.basket = None
    return {"ok": True, "closed_legs": len(option_legs), "planned_legs": len(option_legs), "errors": [], "op_id": op_id}


def _apply_live_gate_failsafe(
    *,
    live_gate: LiveGate,
    reason: str,
    dw: DhanWrapper,
    bkm_strategy: Optional[BatmanBKMStrategy],
    trade_mode: str,
    live_order_executor: Optional[LiveOrderExecutor] = None,
    execution_journal: Optional[ExecutionJournal] = None,
) -> int:
    live_gate.trigger_failsafe(reason)
    if not bkm_strategy or not bkm_strategy.basket:
        return 0
    flatten_res = _flatten_bkm_basket(
        dw=dw,
        bkm_strategy=bkm_strategy,
        trade_mode=trade_mode,
        reason="DATA_FAILSAFE_LOCK",
        live_order_executor=live_order_executor,
        execution_journal=execution_journal,
    )
    return int(flatten_res.get("closed_legs", 0))


def _build_risk_config(settings: Dict[str, Any]) -> RiskConfig:
    partial = float(settings.get("partial_target_pct", DEFAULT_SETTINGS["partial_target_pct"]))
    partial = max(0.0, min(0.95, partial))
    per_leg_sl = float(settings.get("per_leg_sl_mult", DEFAULT_SETTINGS["per_leg_sl_mult"]))
    per_leg_tp = float(settings.get("per_leg_tp_mult", DEFAULT_SETTINGS["per_leg_tp_mult"]))
    per_leg_sl = max(0.0, per_leg_sl)
    per_leg_tp = max(0.0, per_leg_tp)
    vix_low = float(settings.get("vix_adaptive_low", DEFAULT_SETTINGS["vix_adaptive_low"]))
    vix_high = float(settings.get("vix_adaptive_high", DEFAULT_SETTINGS["vix_adaptive_high"]))
    strangle_delta_low = float(settings.get("strangle_delta_low", DEFAULT_SETTINGS["strangle_delta_low"]))
    strangle_delta_high = float(settings.get("strangle_delta_high", DEFAULT_SETTINGS["strangle_delta_high"]))
    strangle_offset_low = float(settings.get("strangle_offset_low", DEFAULT_SETTINGS["strangle_offset_low"]))
    strangle_offset_high = float(settings.get("strangle_offset_high", DEFAULT_SETTINGS["strangle_offset_high"]))
    spread_delta_low = float(settings.get("spread_short_delta_low", DEFAULT_SETTINGS["spread_short_delta_low"]))
    spread_delta_high = float(settings.get("spread_short_delta_high", DEFAULT_SETTINGS["spread_short_delta_high"]))
    return RiskConfig(
        max_intraday_loss=float(settings.get("max_intraday_loss", -3000)),
        intraday_target=float(settings.get("intraday_target", 4000)),
        allow_carry_forward=bool(settings.get("allow_carry_forward", False)),
        max_carry_days=int(settings.get("max_carry_days", 0)),
        vix_carry_threshold=float(settings.get("vix_carry_threshold", 12.0)),
        last_entry_time=dtime(14, 45),
        force_exit_time=dtime(15, 15),
        max_daily_loss_pct=float(settings.get("max_daily_loss_pct", DEFAULT_SETTINGS["max_daily_loss_pct"])),
        max_daily_trades=int(settings.get("max_daily_trades", DEFAULT_SETTINGS["max_daily_trades"])),
        per_leg_sl_mult=per_leg_sl,
        per_leg_tp_mult=per_leg_tp,
        partial_target_pct=partial,
        vix_adaptive_low=vix_low,
        vix_adaptive_high=vix_high,
        strangle_delta_low=strangle_delta_low,
        strangle_delta_high=strangle_delta_high,
        strangle_offset_low=strangle_offset_low,
        strangle_offset_high=strangle_offset_high,
        spread_short_delta_low=spread_delta_low,
        spread_short_delta_high=spread_delta_high,
    )


def _normalize_leg_side(side) -> LegSide:
    if isinstance(side, LegSide):
        return side
    if isinstance(side, str):
        return LegSide.BUY if side.upper().startswith("B") else LegSide.SELL
    return LegSide.BUY


def _ensure_leg_security_id(leg: OptionLeg, symbol: str, expiry) -> Optional[int]:
    if leg.security_id:
        try:
            return int(float(leg.security_id))
        except Exception:
            return None
    resolved = _resolve_security_id_cached(symbol, expiry.isoformat(), leg.strike, leg.option_type.value)
    if resolved:
        try:
            return int(float(resolved))
        except Exception:
            return None
    return None


def _write_pid_file() -> None:
    data: Dict[str, Any] = {}
    if PID_FILE.exists():
        try:
            data = json.loads(PID_FILE.read_text()) or {}
        except Exception:
            data = {}
    data.update(
        {
            "pid": os.getpid(),
            "trade_mode": os.getenv("TRADE_MODE", "live"),
            "started_at": datetime.now().isoformat(),
        }
    )
    try:
        PID_FILE.write_text(json.dumps(data, indent=2))
    except Exception:
        log.warning("Failed to write pid file %s", PID_FILE)


def _cleanup_pid_file() -> None:
    try:
        if PID_FILE.exists():
            data = json.loads(PID_FILE.read_text())
            if data.get("pid") == os.getpid():
                PID_FILE.unlink()
    except Exception:
        pass


atexit.register(_cleanup_pid_file)


def _ensure_csv_headers(path: Path, headers: List[str]) -> None:
    if path.exists():
        return
    try:
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=headers)
            writer.writeheader()
    except Exception as exc:
        log.warning("Failed to initialize CSV %s: %s", path, exc)


def _append_csv_row(path: Path, headers: List[str], row: Dict[str, Any]) -> None:
    _ensure_csv_headers(path, headers)
    try:
        with path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=headers)
            writer.writerow(row)
    except Exception as exc:
        log.warning("Failed to append row to %s: %s", path, exc)


def _update_blotter_summary(entry: Dict[str, Any]) -> None:
    summary: Dict[str, Any] = {"warn_only_orders": 0, "executed_orders": 0}
    if TRADE_BLOTTER_SUMMARY.exists():
        try:
            summary = json.loads(TRADE_BLOTTER_SUMMARY.read_text()) or summary
        except Exception:
            summary = {"warn_only_orders": 0, "executed_orders": 0}
    summary["executed_orders"] = int(summary.get("executed_orders", 0)) + int(entry.get("executed", 0))
    summary["warn_only_orders"] = 0
    summary["last_order_ts"] = entry.get("timestamp")
    try:
        TRADE_BLOTTER_SUMMARY.write_text(json.dumps(summary, indent=2))
    except Exception as exc:
        log.warning("Failed to update blotter summary: %s", exc)


def _log_trade_event(*, trade_mode: str, side: str, leg: OptionLeg, order_type: str, notes: str = "") -> None:
    # In paper mode, mark events as non-executed to avoid inflating executed count.
    executed_flag = 0 if trade_mode == "paper" else 1
    entry = {
        "timestamp": datetime.now().isoformat(),
        "trade_mode": trade_mode,
        "warn_only": 0,
        "executed": executed_flag,
        "side": side,
        "order_type": order_type,
        "exchange_seg": "NSE_FNO",
        "product_type": "MARGIN",
        "security_id": _ensure_leg_security_id(leg, leg.symbol, leg.expiry) or "",
        "quantity": leg.quantity,
        "price": float(getattr(leg, "entry_price", 0.0) or leg.current_ltp or 0.0),
        "delta": getattr(leg, "delta", None),
        "expiry": getattr(leg, "expiry", "") if leg and getattr(leg, "expiry", None) is not None else "",
        "strike": leg.strike,
        "tag": getattr(leg, "strategy_type", StrategyType.NONE).name,
        "notes": notes,
    }
    _append_csv_row(TRADE_BLOTTER_PATH, BLOTTER_FIELDS, entry)
    _update_blotter_summary(entry)


def _log_batman_blotter(trade_mode: str, legs: list, action: str, use_last: bool = False) -> None:
    """
    Log Batman V2 legs into the blotter so the UI can display paper positions.
    Action "OPEN" logs the leg direction; action "CLOSE" logs the opposite side.
    Action "MTM" logs current LTP (last_price) for paper P&L visibility.
    """
    executed_flag = 0 if trade_mode == "paper" else 1
    now_ts = datetime.now().isoformat()
    for leg in legs or []:
        try:
            # Support both legacy BatmanV2 legs and new BKM legs
            is_long = bool(getattr(leg, "is_long", False))
            side = getattr(leg, "side", None)
            side = side if side else ("BUY" if is_long else "SELL")
            if action == "CLOSE":
                side = "SELL" if side == "BUY" else "BUY"
            price_val = getattr(leg, "entry_price", None)
            if price_val is None:
                price_val = getattr(leg, "entry", 0.0)
            if action in {"CLOSE", "MTM"}:
                price_val = getattr(leg, "last_price", None) or getattr(leg, "ltp", None) or price_val
            tag = "BATMAN_BKM" if hasattr(leg, "qty") else "BATMAN_V2"
            entry = {
                "timestamp": now_ts,
                "trade_mode": trade_mode,
                "warn_only": 0,
                "executed": executed_flag,
                "side": side,
                "order_type": "MARKET",
                "exchange_seg": "NSE_FNO",
                "product_type": "MARGIN",
                "security_id": getattr(leg, "instrument_id", None)
                or getattr(leg, "security_id", "")
                or "",
                "quantity": getattr(leg, "quantity", None) or getattr(leg, "qty", 0),
                "price": float(price_val or 0.0),
                "delta": getattr(leg, "delta", None),
                "expiry": getattr(leg, "expiry", "") or "",
                "strike": getattr(leg, "strike", ""),
                "tag": tag,
                "notes": action,
            }
            _append_csv_row(TRADE_BLOTTER_PATH, BLOTTER_FIELDS, entry)
            _update_blotter_summary(entry)
        except Exception:
            continue


def _log_feature_event(row: Dict[str, Any]) -> None:
    _append_csv_row(FEATURE_LOG_PATH, FEATURE_FIELDS, row)


def _log_environment_state(market: MarketSnapshot, trend_ctx: TrendContext, trade_mode: str) -> None:
    global _LAST_ENV_LOG
    now = datetime.now()
    if _LAST_ENV_LOG and (now - _LAST_ENV_LOG).total_seconds() < 60:
        return
    _LAST_ENV_LOG = now
    context = {
        "market_state": {
            "trend": trend_ctx.trend_side.name.lower(),
            "bull_score": trend_ctx.bull_score,
            "bear_score": trend_ctx.bear_score,
            "confidence": trend_ctx.confidence,
            "spot": market.spot,
            "vix": market.india_vix,
            "as_of": market.now.isoformat(),
        },
        "strategy_candidates": [],
    }
    row = {
        "timestamp": now.isoformat(),
        "strategy": "environment",
        "expiry": "",
        "exchange_seg": "NSE_FNO",
        "spot": market.spot,
        "ce_strike": "",
        "pe_strike": "",
        "ce_ltp": "",
        "pe_ltp": "",
        "ce_delta": "",
        "pe_delta": "",
        "ce_entry": "",
        "pe_entry": "",
        "net_credit": "",
        "warn_only": 0,
        "trade_mode": trade_mode,
        "context": json.dumps(context),
        "block_reason": "",
        "net_delta": "",
        "total_notional": "",
        "positions_summary": "",
    }
    _log_feature_event(row)


def _log_strategy_event(
    *,
    strategy: StrategyType,
    action: str,
    reason: str,
    market: MarketSnapshot,
    trade_mode: str,
    legs: List[OptionLeg],
    basket_mtm: float | None = None,
    basket_peak: float | None = None,
    trail_floor: float | None = None,
) -> None:
    ce_leg = next((leg for leg in legs if leg.side == LegSide.SELL and leg.option_type == OptionType.CALL), None)
    pe_leg = next((leg for leg in legs if leg.side == LegSide.SELL and leg.option_type == OptionType.PUT), None)
    context = {"action": action, "reason": reason}
    if basket_mtm is not None:
        context["basket_mtm"] = basket_mtm
    if basket_peak is not None:
        context["basket_peak"] = basket_peak
    if trail_floor is not None:
        context["trail_floor"] = trail_floor
    row = {
        "timestamp": datetime.now().isoformat(),
        "strategy": strategy.name.lower(),
        "expiry": legs[0].expiry.isoformat() if legs else "",
        "exchange_seg": "NSE_FNO",
        "spot": market.spot,
        "ce_strike": ce_leg.strike if ce_leg else "",
        "pe_strike": pe_leg.strike if pe_leg else "",
        "ce_ltp": ce_leg.current_ltp if ce_leg else "",
        "pe_ltp": pe_leg.current_ltp if pe_leg else "",
        "ce_delta": getattr(ce_leg, "delta", ""),
        "pe_delta": getattr(pe_leg, "delta", ""),
        "ce_entry": ce_leg.entry_price if ce_leg else "",
        "pe_entry": pe_leg.entry_price if pe_leg else "",
        "net_credit": (
            (ce_leg.entry_price if ce_leg else 0.0) + (pe_leg.entry_price if pe_leg else 0.0)
            if action.startswith("OPEN")
            else ""
        ),
        "warn_only": 0,
        "trade_mode": trade_mode,
        "context": json.dumps(context),
        "block_reason": "",
        "net_delta": "",
        "total_notional": "",
        "positions_summary": "",
    }
    _log_feature_event(row)


_SECURITY_ID_CACHE: Dict[tuple[str, str, float, str], Optional[str]] = {}
_SECURITY_ID_MISS_LOGGED: set[tuple[str, str, float, str]] = set()
_SCRIP_MASTER_WARMED = False


def _warm_scrip_master(force: bool = False) -> None:
    global _SCRIP_MASTER_WARMED
    if _SCRIP_MASTER_WARMED and not force:
        return
    try:
        refresh_scrip_master(force=force)
        _SCRIP_MASTER_WARMED = True
    except Exception as exc:  # pragma: no cover - best effort
        log.warning("Failed to refresh Dhan scrip master (force=%s): %s", force, exc)


def _resolve_security_id_cached(symbol: str, expiry: str, strike: float, option_type: str) -> Optional[str]:
    strike_norm = float(strike)
    key = (symbol.upper(), expiry, strike_norm, option_type.upper())
    if key in _SECURITY_ID_CACHE:
        return _SECURITY_ID_CACHE[key]
    sec_id: Optional[str] = None
    try:
        sec_val = resolve_option_security_id(symbol, expiry, strike_norm, option_type)
        if sec_val:
            sec_id = str(sec_val)
        elif not _SCRIP_MASTER_WARMED:
            _warm_scrip_master(force=True)
            sec_val = resolve_option_security_id(symbol, expiry, strike_norm, option_type)
            if sec_val:
                sec_id = str(sec_val)
    except Exception as exc:  # pragma: no cover - defensive log
        log.warning(
            "SecurityId lookup failed symbol=%s expiry=%s strike=%s opt=%s err=%s",
            symbol,
            expiry,
            strike_norm,
            option_type,
            exc,
        )
    if not sec_id and key not in _SECURITY_ID_MISS_LOGGED:
        _SECURITY_ID_MISS_LOGGED.add(key)
        log.warning(
            "Missing security_id for symbol=%s expiry=%s strike=%s opt=%s",
            symbol,
            expiry,
            strike_norm,
            option_type,
        )
    _SECURITY_ID_CACHE[key] = sec_id
    return sec_id

# ── Helpers ─────────────────────────────────────────────────────────────────
def _as_bool(val: Optional[str], default: bool) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def _ist_now() -> datetime:
    tz = ZoneInfo("Asia/Kolkata") if ZoneInfo else None
    return datetime.now(tz) if tz else datetime.now()


def _now_iso() -> str:
    return _ist_now().isoformat(timespec="seconds")


def _is_india_market_open(now: Optional[datetime] = None) -> bool:
    current = now or _ist_now()
    if current.weekday() >= 5:
        return False
    start = dtime(9, 15)
    end = dtime(15, 30)
    return start <= current.time() <= end


def _default_bkm_ai_protect_status(now: Optional[datetime] = None) -> Dict[str, Any]:
    ts = now or _ist_now()
    return {
        "status": "UNLOCKED",
        "active": False,
        "protection_enabled": True,
        "session_date": ts.date().isoformat(),
        "reason": None,
        "source_action": None,
        "score": None,
        "expiry": None,
        "updated_at": ts.isoformat(timespec="seconds"),
        "last_transition_at": ts.isoformat(timespec="seconds"),
    }


def _load_bkm_ai_protect_status(now: Optional[datetime] = None) -> Dict[str, Any]:
    default = _default_bkm_ai_protect_status(now)
    try:
        if not BATMAN_BKM_AI_PROTECT_STATUS_FILE.exists():
            return default
        payload = json.loads(BATMAN_BKM_AI_PROTECT_STATUS_FILE.read_text())
        if not isinstance(payload, dict):
            return default
        out = dict(default)
        out.update(payload)
        out["active"] = bool(out.get("active", False))
        out["protection_enabled"] = bool(out.get("protection_enabled", True))
        out["status"] = "LOCKED" if out["active"] else "UNLOCKED"
        return out
    except Exception:
        return default


def _save_bkm_ai_protect_status(payload: Dict[str, Any]) -> None:
    try:
        BATMAN_BKM_AI_PROTECT_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        BATMAN_BKM_AI_PROTECT_STATUS_FILE.write_text(json.dumps(payload, indent=2, default=str))
    except Exception:
        log.exception("[BatmanBKM-AI] failed to persist protect status")


def _set_bkm_ai_protect_lock(
    *,
    active: bool,
    now: Optional[datetime] = None,
    reason: Optional[str] = None,
    source_action: Optional[str] = None,
    score: Optional[float] = None,
    expiry: Optional[str] = None,
    protection_enabled: Optional[bool] = None,
) -> Dict[str, Any]:
    ts = now or _ist_now()
    existing = _load_bkm_ai_protect_status(ts)
    session_date = ts.date().isoformat()
    changed = bool(existing.get("active")) != bool(active) or str(existing.get("session_date") or "") != session_date
    payload = {
        "status": "LOCKED" if active else "UNLOCKED",
        "active": bool(active),
        "protection_enabled": bool(existing.get("protection_enabled", True) if protection_enabled is None else protection_enabled),
        "session_date": session_date,
        "reason": reason if active else None,
        "source_action": source_action if active else None,
        "score": (None if score is None else round(float(score), 2)),
        "expiry": expiry if active else None,
        "updated_at": ts.isoformat(timespec="seconds"),
        "last_transition_at": (ts.isoformat(timespec="seconds") if changed else existing.get("last_transition_at") or ts.isoformat(timespec="seconds")),
    }
    _save_bkm_ai_protect_status(payload)
    return payload


def _set_bkm_ai_protect_enabled(
    *,
    enabled: bool,
    now: Optional[datetime] = None,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    ts = now or _ist_now()
    existing = _load_bkm_ai_protect_status(ts)
    # Disabling protection also clears any active entry lock immediately for the session.
    active = bool(existing.get("active", False)) if enabled else False
    payload = _set_bkm_ai_protect_lock(
        active=active,
        now=ts,
        reason=(existing.get("reason") if active else None),
        source_action=(existing.get("source_action") if active else None),
        score=existing.get("score"),
        expiry=existing.get("expiry"),
        protection_enabled=bool(enabled),
    )
    payload["protection_note"] = str(note or "")
    _save_bkm_ai_protect_status(payload)
    return payload


def _restore_bkm_ai_protect_lock_for_session(now: Optional[datetime] = None) -> Dict[str, Any]:
    ts = now or _ist_now()
    payload = _load_bkm_ai_protect_status(ts)
    if str(payload.get("session_date") or "") != ts.date().isoformat():
        # Session rollover: automatically clear previous-day protective lock.
        payload = _set_bkm_ai_protect_lock(active=False, now=ts, protection_enabled=True)
    return payload


def _consume_bkm_ai_protect_clear_request() -> Optional[Dict[str, Any]]:
    try:
        if not BATMAN_BKM_AI_PROTECT_CLEAR_REQUEST_FILE.exists():
            return None
        raw = BATMAN_BKM_AI_PROTECT_CLEAR_REQUEST_FILE.read_text()
        payload = json.loads(raw) if raw else {}
        if not isinstance(payload, dict):
            payload = {}
    except Exception:
        payload = {}
    try:
        BATMAN_BKM_AI_PROTECT_CLEAR_REQUEST_FILE.unlink(missing_ok=True)  # type: ignore[arg-type]
    except TypeError:
        try:
            if BATMAN_BKM_AI_PROTECT_CLEAR_REQUEST_FILE.exists():
                BATMAN_BKM_AI_PROTECT_CLEAR_REQUEST_FILE.unlink()
        except Exception:
            pass
    except Exception:
        pass
    return payload or {}


def _looks_like_chain_dict(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    for val in obj.values():
        if isinstance(val, dict):
            lower_keys = {str(k).lower() for k in val.keys()}
            if {"ce", "pe"} & lower_keys:
                return True
    return False


def _rows_to_chain(rows: Any) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        strike = (
            row.get("strike")
            or row.get("strikePrice")
            or row.get("strikeprice")
            or row.get("StrikePrice")
        )
        if strike is None:
            continue
        try:
            strike_val = float(strike)
        except Exception:
            continue
        strike_key = str(int(strike_val)) if abs(strike_val - int(strike_val)) < 1e-6 else str(strike_val)
        ce = (
            row.get("ce")
            or row.get("CE")
            or row.get("call")
            or row.get("Call")
            or row.get("callOption")
        )
        pe = (
            row.get("pe")
            or row.get("PE")
            or row.get("put")
            or row.get("Put")
            or row.get("putOption")
        )
        entry: Dict[str, Any] = {}
        if ce:
            entry["ce"] = ce
        if pe:
            entry["pe"] = pe
        if entry:
            out[strike_key] = entry
    return out


def _coerce_chain_dict(chain_raw: Any) -> Dict[str, Dict[str, Any]]:
    """
    DHAN responses often nest the actual strike→legs dictionary under several keys.
    Walk those wrappers and return the first dict that looks like an option chain.
    """
    queue = [chain_raw]
    seen: set[int] = set()
    while queue:
        cur = queue.pop(0)
        if isinstance(cur, dict):
            oid = id(cur)
            if oid in seen:
                continue
            seen.add(oid)
            if _looks_like_chain_dict(cur):
                return cur  # already strike→legs
            for key in ("data", "Data", "DATA", "oc", "option_chain", "optionchain", "records", "chain"):
                if key in cur:
                    queue.append(cur[key])
            continue
        if isinstance(cur, list):
            converted = _rows_to_chain(cur)
            if converted:
                return converted
    return {}


def _map_positions(rows: list) -> List[OptionLeg]:
    legs: List[OptionLeg] = []
    for r in rows or []:
        try:
            exch = str(r.get("exchangeSegment") or r.get("exchange_seg") or "").upper()
            if not exch.startswith("NSE_FNO"):
                continue
            symbol = r.get("tradingSymbol") or r.get("symbol") or ""
            if "NIFTY" not in symbol.upper():
                continue
            net = float(r.get("netQty") or r.get("netqty") or 0)
            if net == 0:
                continue
            qty = abs(int(net))
            side = LegSide.SELL if net < 0 else LegSide.BUY
            symbol = r.get("tradingSymbol") or r.get("symbol") or ""

            def _parse_expiry_from_symbol(sym: str) -> Optional[dt_date]:
                if not sym:
                    return None
                sym = sym.upper().replace("-", " ")
                parts = sym.split()
                months = {
                    "JAN": 1,
                    "FEB": 2,
                    "MAR": 3,
                    "APR": 4,
                    "MAY": 5,
                    "JUN": 6,
                    "JUL": 7,
                    "AUG": 8,
                    "SEP": 9,
                    "OCT": 10,
                    "NOV": 11,
                    "DEC": 12,
                }
                for idx in range(len(parts) - 2):
                    if parts[idx].isdigit() and parts[idx + 1] in months:
                        day = int(parts[idx])
                        month = months[parts[idx + 1]]
                        year = datetime.now().year
                        try:
                            return dt_date(year, month, day)
                        except Exception:
                            continue
                return None

            expiry = (
                r.get("expiryDate")
                or r.get("expiry")
                or r.get("drvExpiryDate")
                or r.get("drvExpiry")
                or r.get("expirydate")
                or _parse_expiry_from_symbol(symbol)
            )
            expiry_date = datetime.fromisoformat(str(expiry)).date() if expiry else datetime.now().date()

            def _parse_strike_from_symbol(sym: str) -> Optional[float]:
                if not sym:
                    return None
                import re

                matches = re.findall(r"(\d{4,6}(?:\.\d+)?)", sym)
                if not matches:
                    return None
                try:
                    return float(matches[-1])
                except Exception:
                    return None

            strike = (
                r.get("strikePrice")
                or r.get("strike")
                or r.get("drvStrikePrice")
                or r.get("drvStrike")
                or _parse_strike_from_symbol(symbol)
                or 0
            )
            opt_type_raw = (
                r.get("optionType")
                or r.get("option_type")
                or r.get("drvOptionType")
                or r.get("drv_option_type")
                or symbol
                or "C"
            )

            def _coerce_option_type(val: str) -> OptionType:
                val = (val or "").upper()
                if "P" in val:
                    return OptionType.PUT
                return OptionType.CALL

            option_type = _coerce_option_type(opt_type_raw)

            legs.append(
                OptionLeg(
                    symbol="NIFTY",
                    expiry=expiry_date,
                    strike=float(strike),
                    option_type=option_type,
                    side=side,
                    quantity=qty,
                    entry_price=float(r.get("avgPrice") or r.get("avg_price") or 0.0),
                    security_id=str(r.get("securityId") or r.get("security_id") or ""),
                    strategy_type=StrategyType.NONE,
                )
            )
        except Exception:
            continue
    return legs


def _map_chain(chain_raw: Any, expiry, symbol: str, spot: float = 0.0) -> List[dict]:
    """
    Normalize a raw DHAN option chain response into a list of rows with:
      expiry, option_type ("CE"/"PE"), strike, ltp, delta, security_id, spot,
      oi, oi_change, volume.

    Supports both legacy and v2 /v2/optionchain shapes by delegating to
    _coerce_chain_dict, which walks through 'data'/'oc'/etc.
    """
    rows: List[dict] = []
    chain_dict = _coerce_chain_dict(chain_raw)
    if not isinstance(chain_dict, dict):
        return rows

    expiry_str = expiry.isoformat() if hasattr(expiry, "isoformat") else str(expiry)

    def _as_float(*values: Any) -> Optional[float]:
        for v in values:
            if v in (None, "", "-", "--"):
                continue
            try:
                return float(v)
            except Exception:
                continue
        return None

    for strike, legs in chain_dict.items():
        if not isinstance(legs, dict):
            continue
        try:
            strike_f = float(strike)
        except Exception:
            continue

        for opt_name, opt_data in (legs or {}).items():
            if not isinstance(opt_data, dict):
                continue

            opt_name_u = str(opt_name).upper()
            option_type = "CE" if opt_name_u.startswith("C") else "PE"

            # Dhan v2 keeps greeks under "greeks"/"Greeks"
            greeks = opt_data.get("greeks") or opt_data.get("Greeks") or {}

            sec_id = opt_data.get("securityId") or opt_data.get("security_id")
            if not sec_id:
                resolved = _resolve_security_id_cached(symbol, expiry_str, strike_f, option_type)
                if resolved:
                    sec_id = resolved

            ltp = (
                opt_data.get("last_price")
                or opt_data.get("LastPrice")
                or opt_data.get("ltp")
                or opt_data.get("LTP")
                or opt_data.get("close")
            )
            oi = _as_float(
                opt_data.get("oi"),
                opt_data.get("open_interest"),
                opt_data.get("openInterest"),
                opt_data.get("OI"),
            )
            oi_change = _as_float(
                opt_data.get("oi_change"),
                opt_data.get("oiChange"),
                opt_data.get("open_interest_change"),
                opt_data.get("changeInOpenInterest"),
                opt_data.get("changeinOpenInterest"),
                opt_data.get("change_in_oi"),
            )
            volume = _as_float(
                opt_data.get("volume"),
                opt_data.get("Volume"),
                opt_data.get("tradeVolume"),
                opt_data.get("totalTradedVolume"),
            )

            rows.append(
                {
                    "expiry": expiry,
                    "option_type": option_type,
                    "strike": strike_f,
                    "ltp": ltp,
                    "delta": greeks.get("delta") or opt_data.get("delta"),
                    "spot": spot,
                    "security_id": sec_id,
                    "oi": oi,
                    "oi_change": oi_change,
                    "volume": volume,
                }
            )
    return rows


def _update_leg_ltps_from_chain(legs: List[OptionLeg], chain_rows: List[dict]) -> None:
    lookup: Dict[Tuple[str, float], float] = {}
    for row in chain_rows or []:
        try:
            opt = str(row.get("option_type") or "").upper()
            strike = float(row.get("strike") or 0.0)
            ltp = row.get("ltp")
            if not opt or strike == 0 or ltp is None:
                continue
            lookup[(opt, strike)] = float(ltp)
        except Exception:
            continue
    for leg in legs or []:
        try:
            opt_name = leg.option_type.value if hasattr(leg.option_type, "value") else str(leg.option_type)
            opt_name = opt_name.upper()
            if "CALL" in opt_name and opt_name != "CE":
                opt_name = "CE"
            elif "PUT" in opt_name and opt_name != "PE":
                opt_name = "PE"
            strike = float(leg.strike)
            ltp = lookup.get((opt_name, strike))
            if ltp is not None:
                leg.current_ltp = ltp
        except Exception:
            continue


def _classify_trend_from_candles(candles: List[dict], *, interval_min: int) -> Dict[str, Any]:
    rows = [c for c in (candles or []) if c]
    rows = sorted(rows, key=lambda c: c.get("timestamp") or datetime.min)
    closes = [float(c.get("close") or 0.0) for c in rows if c.get("close") not in (None, "")]
    highs = [float(c.get("high") or c.get("close") or 0.0) for c in rows if c.get("high") is not None or c.get("close") is not None]
    lows = [float(c.get("low") or c.get("close") or 0.0) for c in rows if c.get("low") is not None or c.get("close") is not None]
    vols = [float(c.get("volume") or 0.0) for c in rows if c.get("volume") not in (None, "")]
    if len(closes) < 2:
        return {
            "interval_min": interval_min,
            "trend": "UNKNOWN",
            "pattern": "UNKNOWN",
            "bars": len(closes),
            "change_points": None,
            "change_pct": None,
            "range_points": None,
            "close_position_in_range": None,
            "dir_score": 0.0,
            "support": None,
            "resistance": None,
            "distance_to_support": None,
            "distance_to_resistance": None,
            "atr_like_points": None,
            "atr_like_pct": None,
            "volatility_regime": "UNKNOWN",
            "breakout_dir": "NONE",
            "breakout_confirmed": False,
        }

    lookback = min(len(closes), 12 if interval_min <= 5 else (10 if interval_min <= 15 else 8))
    window = closes[-lookback:]
    window_vols = vols[-lookback:] if vols else []
    first_close = float(window[0])
    last_close = float(window[-1])
    change_points = last_close - first_close
    change_pct = (change_points / max(1.0, abs(first_close))) * 100.0
    window_high = max(highs[-lookback:]) if highs else max(window)
    window_low = min(lows[-lookback:]) if lows else min(window)
    range_points = max(0.0, float(window_high - window_low))
    close_pos = None
    if range_points > 0:
        close_pos = (last_close - window_low) / range_points

    # Simple support/resistance and range-position features.
    support = round(float(window_low), 2)
    resistance = round(float(window_high), 2)
    dist_support = max(0.0, float(last_close - window_low))
    dist_resistance = max(0.0, float(window_high - last_close))

    # ATR-like volatility proxy from average bar range in the lookback window.
    bar_ranges: List[float] = []
    for candle in rows[-lookback:]:
        try:
            h = float(candle.get("high") or candle.get("close") or 0.0)
            l = float(candle.get("low") or candle.get("close") or 0.0)
            bar_ranges.append(max(0.0, h - l))
        except Exception:
            continue
    atr_like_points = (sum(bar_ranges) / len(bar_ranges)) if bar_ranges else 0.0
    atr_like_pct = ((atr_like_points / max(1.0, abs(last_close))) * 100.0) if last_close else 0.0
    if atr_like_pct >= (0.32 if interval_min <= 5 else (0.45 if interval_min <= 15 else 0.65)):
        vol_regime = "HIGH"
    elif atr_like_pct <= (0.14 if interval_min <= 5 else (0.20 if interval_min <= 15 else 0.30)):
        vol_regime = "LOW"
    else:
        vol_regime = "NORMAL"

    # Breakout confirmation vs prior window extremes, optionally confirmed by volume expansion.
    breakout_dir = "NONE"
    breakout_confirmed = False
    if len(window) >= 3:
        prior_high = max(window[:-1])
        prior_low = min(window[:-1])
        vol_avg = (sum(window_vols[:-1]) / len(window_vols[:-1])) if len(window_vols) >= 2 else 0.0
        vol_last = float(window_vols[-1]) if window_vols else 0.0
        vol_confirm = True if vol_avg <= 0 else (vol_last >= (1.15 * vol_avg))
        if last_close > prior_high:
            breakout_dir = "UP"
            breakout_confirmed = bool(vol_confirm)
        elif last_close < prior_low:
            breakout_dir = "DOWN"
            breakout_confirmed = bool(vol_confirm)

    # Practical thresholds tuned by timeframe for simple trend/pattern labeling.
    trend_threshold_pct = 0.10 if interval_min <= 5 else (0.18 if interval_min <= 15 else 0.28)
    trend = "RANGE"
    if change_pct >= trend_threshold_pct:
        trend = "UP"
    elif change_pct <= -trend_threshold_pct:
        trend = "DOWN"

    pattern = "RANGE"
    if trend == "UP":
        pattern = "UPTREND" if (close_pos is not None and close_pos >= 0.65) else "UP_BIAS"
    elif trend == "DOWN":
        pattern = "DOWNTREND" if (close_pos is not None and close_pos <= 0.35) else "DOWN_BIAS"

    dir_score = 1.0 if trend == "UP" else (-1.0 if trend == "DOWN" else 0.0)
    if trend == "RANGE" and close_pos is not None:
        # Preserve a small directional bias inside range.
        dir_score = round((close_pos - 0.5) * 0.6, 3)

    return {
        "interval_min": interval_min,
        "trend": trend,
        "pattern": pattern,
        "bars": len(window),
        "change_points": round(float(change_points), 2),
        "change_pct": round(float(change_pct), 3),
        "range_points": round(float(range_points), 2),
        "close_position_in_range": None if close_pos is None else round(float(close_pos), 3),
        "dir_score": round(float(dir_score), 3),
        "close": round(float(last_close), 2),
        "support": support,
        "resistance": resistance,
        "distance_to_support": round(float(dist_support), 2),
        "distance_to_resistance": round(float(dist_resistance), 2),
        "atr_like_points": round(float(atr_like_points), 2),
        "atr_like_pct": round(float(atr_like_pct), 3),
        "volatility_regime": vol_regime,
        "breakout_dir": breakout_dir,
        "breakout_confirmed": bool(breakout_confirmed),
    }


def _build_bkm_mtf_trend_context(dw: DhanWrapper, *, trade_day: dt_date) -> Dict[str, Any]:
    per_tf: Dict[str, Dict[str, Any]] = {}
    weighted_sum = 0.0
    weighted_denom = 0.0
    errors: List[str] = []
    high_vol_votes = 0
    low_vol_votes = 0
    breakout_votes: List[str] = []
    for interval, weight in ((5, 1.0), (15, 2.0), (60, 3.0)):
        try:
            candles = _cached_intraday_for_day(dw, trade_day, interval)
        except Exception as exc:
            per_tf[str(interval)] = {
                "interval_min": interval,
                "trend": "UNKNOWN",
                "pattern": "UNKNOWN",
                "error": str(exc),
            }
            errors.append(f"{interval}m:{exc}")
            continue
        snap = _classify_trend_from_candles(candles, interval_min=interval)
        per_tf[str(interval)] = snap
        try:
            weighted_sum += float(snap.get("dir_score") or 0.0) * weight
            weighted_denom += weight
        except Exception:
            continue
        if str(snap.get("volatility_regime") or "").upper() == "HIGH":
            high_vol_votes += 1
        elif str(snap.get("volatility_regime") or "").upper() == "LOW":
            low_vol_votes += 1
        if bool(snap.get("breakout_confirmed")):
            breakout_votes.append(str(snap.get("breakout_dir") or "NONE").upper())
    bias_score = (weighted_sum / weighted_denom) if weighted_denom > 0 else 0.0
    bias = "NEUTRAL"
    if bias_score >= 0.35:
        bias = "BULLISH"
    elif bias_score <= -0.35:
        bias = "BEARISH"
    volatility_regime = "NORMAL"
    if high_vol_votes >= 2:
        volatility_regime = "HIGH"
    elif low_vol_votes >= 2:
        volatility_regime = "LOW"
    breakout_confirmation = "NONE"
    if breakout_votes:
        up = sum(1 for b in breakout_votes if b == "UP")
        down = sum(1 for b in breakout_votes if b == "DOWN")
        if up > down and up >= 1:
            breakout_confirmation = "UP_CONFIRMED"
        elif down > up and down >= 1:
            breakout_confirmation = "DOWN_CONFIRMED"
    return {
        "timeframes": per_tf,
        "bias_score": round(float(bias_score), 3),
        "bias": bias,
        "volatility_regime": volatility_regime,
        "breakout_confirmation": breakout_confirmation,
        "errors": errors,
    }


def _build_bkm_structure_confidence_context(
    *,
    spot: float,
    trend_ctx: Optional[Dict[str, Any]],
    oc_ctx: Optional[Dict[str, Any]],
    daily_candles: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    trend_ctx = trend_ctx if isinstance(trend_ctx, dict) else {}
    oc_ctx = oc_ctx if isinstance(oc_ctx, dict) else {}
    tf_map = trend_ctx.get("timeframes") if isinstance(trend_ctx.get("timeframes"), dict) else {}
    spot_f = float(spot or 0.0)

    def _f(val: Any) -> Optional[float]:
        if val in (None, "", "-", "--"):
            return None
        try:
            return float(val)
        except Exception:
            return None

    def _pick_nearest_support(cands: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        below = [c for c in cands if _f(c.get("level")) is not None and float(c["level"]) <= spot_f]
        pool = below if below else [c for c in cands if _f(c.get("level")) is not None]
        pool = sorted(pool, key=lambda c: abs(spot_f - float(c["level"])))
        first = pool[0] if pool else None
        second = pool[1] if len(pool) > 1 else None
        return first, second

    def _pick_nearest_resistance(cands: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        above = [c for c in cands if _f(c.get("level")) is not None and float(c["level"]) >= spot_f]
        pool = above if above else [c for c in cands if _f(c.get("level")) is not None]
        pool = sorted(pool, key=lambda c: abs(float(c["level"]) - spot_f))
        first = pool[0] if pool else None
        second = pool[1] if len(pool) > 1 else None
        return first, second

    def _level_payload(item: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        lvl = _f((item or {}).get("level"))
        dist = None
        if lvl is not None and spot_f > 0:
            dist = abs(lvl - spot_f)
        return {
            "level": None if lvl is None else round(lvl, 2),
            "source": (item or {}).get("source"),
            "strength": (item or {}).get("strength"),
            "distance_from_spot": None if dist is None else round(float(dist), 2),
        }

    intraday_support_cands: List[Dict[str, Any]] = []
    intraday_resistance_cands: List[Dict[str, Any]] = []
    for tf_key, tf_weight in (("5", 1.0), ("15", 1.5), ("60", 2.0)):
        tf = tf_map.get(tf_key) if isinstance(tf_map.get(tf_key), dict) else {}
        sup = _f(tf.get("support"))
        res = _f(tf.get("resistance"))
        if sup is not None:
            intraday_support_cands.append({"level": sup, "source": f"{tf_key}m_price", "strength": tf_weight})
        if res is not None:
            intraday_resistance_cands.append({"level": res, "source": f"{tf_key}m_price", "strength": tf_weight})

    put_wall_below = oc_ctx.get("put_wall_below") if isinstance(oc_ctx.get("put_wall_below"), dict) else {}
    call_wall_above = oc_ctx.get("call_wall_above") if isinstance(oc_ctx.get("call_wall_above"), dict) else {}
    put_wall = oc_ctx.get("put_wall") if isinstance(oc_ctx.get("put_wall"), dict) else {}
    call_wall = oc_ctx.get("call_wall") if isinstance(oc_ctx.get("call_wall"), dict) else {}

    for wall, src, strength in (
        (put_wall_below, "oi_put_wall_below", 2.5),
        (put_wall, "oi_put_wall", 1.5),
    ):
        lvl = _f(wall.get("strike"))
        if lvl is not None:
            intraday_support_cands.append({"level": lvl, "source": src, "strength": strength})
    for wall, src, strength in (
        (call_wall_above, "oi_call_wall_above", 2.5),
        (call_wall, "oi_call_wall", 1.5),
    ):
        lvl = _f(wall.get("strike"))
        if lvl is not None:
            intraday_resistance_cands.append({"level": lvl, "source": src, "strength": strength})

    intraday_support_1, intraday_support_2 = _pick_nearest_support(intraday_support_cands)
    intraday_res_1, intraday_res_2 = _pick_nearest_resistance(intraday_resistance_cands)

    # Weekly structure from recent daily candles (trading days).
    dc = [c for c in (daily_candles or []) if isinstance(c, dict)]
    daily_rows: List[Dict[str, Any]] = []
    for c in dc:
        try:
            daily_rows.append(
                {
                    "open": float(c.get("open") or 0.0),
                    "high": float(c.get("high") or c.get("open") or 0.0),
                    "low": float(c.get("low") or c.get("open") or 0.0),
                    "close": float(c.get("close") or c.get("open") or 0.0),
                }
            )
        except Exception:
            continue
    weekly_window = daily_rows[-5:] if len(daily_rows) >= 5 else daily_rows
    weekly_support = min((r["low"] for r in weekly_window), default=None)
    weekly_resistance = max((r["high"] for r in weekly_window), default=None)
    weekly_mid = None
    if weekly_support is not None and weekly_resistance is not None:
        weekly_mid = (float(weekly_support) + float(weekly_resistance)) / 2.0
    weekly_close = weekly_window[-1]["close"] if weekly_window else None
    weekly_bias = "UNKNOWN"
    if weekly_window and len(weekly_window) >= 2:
        first_close = float(weekly_window[0]["close"])
        last_close = float(weekly_window[-1]["close"])
        move = last_close - first_close
        move_pct = (move / max(1.0, abs(first_close))) * 100.0
        if move_pct >= 0.35:
            weekly_bias = "BULLISH"
        elif move_pct <= -0.35:
            weekly_bias = "BEARISH"
        else:
            weekly_bias = "RANGE"

    # Directional signal alignment / conflict using trend + PCR + OI build + breakout.
    def _vote_from_label(label: str, bullish_tokens: Tuple[str, ...], bearish_tokens: Tuple[str, ...]) -> int:
        text = str(label or "").upper()
        if any(tok in text for tok in bullish_tokens):
            return 1
        if any(tok in text for tok in bearish_tokens):
            return -1
        return 0

    trend_vote = _vote_from_label(trend_ctx.get("bias"), ("BULL",), ("BEAR",))
    pcr_vote = _vote_from_label(oc_ctx.get("pcr_bias"), ("BULL",), ("BEAR",))
    oi_build = oc_ctx.get("oi_build") if isinstance(oc_ctx.get("oi_build"), dict) else {}
    oi_build_vote = _vote_from_label(oi_build.get("bias"), ("BULL", "SUPPORT"), ("BEAR", "RESISTANCE"))
    breakout_vote = _vote_from_label(trend_ctx.get("breakout_confirmation"), ("UP", "BULL"), ("DOWN", "BEAR"))
    weekly_vote = _vote_from_label(weekly_bias, ("BULL",), ("BEAR",))

    votes = {
        "trend": trend_vote,
        "pcr": pcr_vote,
        "oi_build": oi_build_vote,
        "breakout": breakout_vote,
        "weekly": weekly_vote,
    }
    non_zero_votes = [v for v in votes.values() if v != 0]
    bullish_votes = sum(1 for v in non_zero_votes if v > 0)
    bearish_votes = sum(1 for v in non_zero_votes if v < 0)
    known_votes = len(non_zero_votes)
    dominant_votes = max(bullish_votes, bearish_votes) if known_votes else 0
    conflict_votes = max(0, known_votes - dominant_votes)
    conflict_ratio = (conflict_votes / known_votes) if known_votes else 0.0
    trend_bias_score = abs(float(_f(trend_ctx.get("bias_score")) or 0.0))
    base_conf = (dominant_votes / known_votes) if known_votes else 0.0
    trend_confidence = (0.55 * base_conf) + (0.25 * min(1.0, trend_bias_score)) + (0.20 * (1.0 - conflict_ratio))

    pcr_total = _f(oc_ctx.get("pcr_total"))
    pcr_near = _f(oc_ctx.get("pcr_near_atm"))
    pcr_unbalanced = False
    pcr_unbalanced_side = "NEUTRAL"
    pcr_extreme = None
    for val in [pcr_near, pcr_total]:
        if val is None:
            continue
        if pcr_extreme is None or abs(val - 1.0) > abs(pcr_extreme - 1.0):
            pcr_extreme = val
    if pcr_extreme is not None:
        if pcr_extreme >= 1.30:
            pcr_unbalanced = True
            pcr_unbalanced_side = "BULLISH"
            trend_confidence += 0.05 if bullish_votes >= bearish_votes else -0.05
        elif pcr_extreme <= 0.70:
            pcr_unbalanced = True
            pcr_unbalanced_side = "BEARISH"
            trend_confidence += 0.05 if bearish_votes >= bullish_votes else -0.05

    # OI/price S-R alignment improves confidence slightly when walls align with price levels.
    sr_alignment_hits = 0
    for a, b in (
        (intraday_support_1, {"level": _f(put_wall_below.get("strike"))}),
        (intraday_res_1, {"level": _f(call_wall_above.get("strike"))}),
    ):
        la = _f((a or {}).get("level"))
        lb = _f((b or {}).get("level"))
        if la is not None and lb is not None and abs(la - lb) <= 100.0:
            sr_alignment_hits += 1
    if sr_alignment_hits:
        trend_confidence += 0.03 * sr_alignment_hits

    vol_regime = str(trend_ctx.get("volatility_regime") or "NORMAL").upper()
    if vol_regime == "HIGH" and conflict_ratio > 0.25:
        trend_confidence -= 0.06
    trend_confidence = max(0.05, min(0.98, trend_confidence))

    signal_conflict_score = max(0.0, min(100.0, (conflict_ratio * 100.0)))
    if known_votes >= 3 and bullish_votes > 0 and bearish_votes > 0:
        signal_conflict_score = min(100.0, signal_conflict_score + 10.0)
    signal_conflict_score = round(signal_conflict_score, 1)

    dominant_bias = "NEUTRAL"
    if bullish_votes > bearish_votes and bullish_votes >= 2:
        dominant_bias = "BULLISH"
    elif bearish_votes > bullish_votes and bearish_votes >= 2:
        dominant_bias = "BEARISH"

    return {
        "intraday_support": _level_payload(intraday_support_1),
        "intraday_support_secondary": _level_payload(intraday_support_2),
        "intraday_resistance": _level_payload(intraday_res_1),
        "intraday_resistance_secondary": _level_payload(intraday_res_2),
        "weekly_support": None if weekly_support is None else round(float(weekly_support), 2),
        "weekly_resistance": None if weekly_resistance is None else round(float(weekly_resistance), 2),
        "weekly_mid": None if weekly_mid is None else round(float(weekly_mid), 2),
        "weekly_close": None if weekly_close is None else round(float(weekly_close), 2),
        "weekly_bias": weekly_bias,
        "weekly_window_days": len(weekly_window),
        "trend_confidence": round(float(trend_confidence), 3),
        "signal_conflict_score": signal_conflict_score,
        "dominant_signal_bias": dominant_bias,
        "votes": votes,
        "bullish_votes": bullish_votes,
        "bearish_votes": bearish_votes,
        "known_votes": known_votes,
        "conflict_votes": conflict_votes,
        "pcr_unbalanced": bool(pcr_unbalanced),
        "pcr_unbalanced_side": pcr_unbalanced_side,
        "pcr_extreme": None if pcr_extreme is None else round(float(pcr_extreme), 3),
        "sr_alignment_hits": sr_alignment_hits,
        "volatility_regime": vol_regime,
    }


def _summarize_bkm_option_chain_context(
    chain_rows: List[dict],
    *,
    spot: float,
    near_atm_band_points: float = 500.0,
    prev_oi_by_key: Optional[Dict[Tuple[str, float], float]] = None,
) -> Tuple[Dict[str, Any], Dict[Tuple[str, float], float]]:
    rows = [r for r in (chain_rows or []) if isinstance(r, dict)]
    ce_rows: List[dict] = []
    pe_rows: List[dict] = []
    current_oi_map: Dict[Tuple[str, float], float] = {}

    def _fv(val: Any) -> Optional[float]:
        if val in (None, "", "-", "--"):
            return None
        try:
            return float(val)
        except Exception:
            return None

    for r in rows:
        opt = str(r.get("option_type") or "").upper()
        strike = _fv(r.get("strike"))
        oi = _fv(r.get("oi"))
        if opt not in {"CE", "PE"} or strike is None:
            continue
        row = dict(r)
        row["strike"] = strike
        row["oi"] = oi
        row["oi_change"] = _fv(r.get("oi_change"))
        if oi is not None:
            current_oi_map[(opt, float(strike))] = float(oi)
        if opt == "CE":
            ce_rows.append(row)
        else:
            pe_rows.append(row)

    def _sum_oi(items: List[dict]) -> float:
        return sum(float(x.get("oi") or 0.0) for x in items if x.get("oi") is not None)

    total_call_oi = _sum_oi(ce_rows)
    total_put_oi = _sum_oi(pe_rows)
    pcr_total = (total_put_oi / total_call_oi) if total_call_oi > 0 else None

    band = max(100.0, float(near_atm_band_points))
    near_ce = [r for r in ce_rows if abs(float(r.get("strike") or 0.0) - float(spot or 0.0)) <= band]
    near_pe = [r for r in pe_rows if abs(float(r.get("strike") or 0.0) - float(spot or 0.0)) <= band]
    near_call_oi = _sum_oi(near_ce)
    near_put_oi = _sum_oi(near_pe)
    pcr_near = (near_put_oi / near_call_oi) if near_call_oi > 0 else None

    def _max_oi_row(items: List[dict]) -> Optional[dict]:
        with_oi = [r for r in items if r.get("oi") is not None]
        if not with_oi:
            return None
        try:
            return max(with_oi, key=lambda x: float(x.get("oi") or 0.0))
        except Exception:
            return None

    call_wall = _max_oi_row(ce_rows)
    put_wall = _max_oi_row(pe_rows)
    call_wall_above = _max_oi_row([r for r in ce_rows if float(r.get("strike") or 0.0) >= float(spot or 0.0)])
    put_wall_below = _max_oi_row([r for r in pe_rows if float(r.get("strike") or 0.0) <= float(spot or 0.0)])

    # OI build-up: prefer explicit oi_change from chain; else derive from previous snapshot.
    near_call_oi_delta = 0.0
    near_put_oi_delta = 0.0
    oi_delta_points = 0
    for r in near_ce + near_pe:
        opt = str(r.get("option_type") or "").upper()
        strike = float(r.get("strike") or 0.0)
        oi_change = _fv(r.get("oi_change"))
        if oi_change is None and prev_oi_by_key:
            cur_oi = _fv(r.get("oi"))
            prev_oi = prev_oi_by_key.get((opt, strike))
            if cur_oi is not None and prev_oi is not None:
                oi_change = float(cur_oi) - float(prev_oi)
        if oi_change is None:
            continue
        oi_delta_points += 1
        if opt == "CE":
            near_call_oi_delta += float(oi_change)
        elif opt == "PE":
            near_put_oi_delta += float(oi_change)

    pcr_oi_change_near = None
    if abs(near_call_oi_delta) > 0:
        pcr_oi_change_near = near_put_oi_delta / near_call_oi_delta

    oi_build_bias = "UNKNOWN"
    if oi_delta_points > 0:
        if near_put_oi_delta - near_call_oi_delta > 0:
            oi_build_bias = "BULLISH_SUPPORT"
        elif near_call_oi_delta - near_put_oi_delta > 0:
            oi_build_bias = "BEARISH_RESISTANCE"
        else:
            oi_build_bias = "NEUTRAL"

    pcr_bias = "NEUTRAL"
    if pcr_near is not None:
        if pcr_near >= 1.15:
            pcr_bias = "BULLISH"
        elif pcr_near <= 0.85:
            pcr_bias = "BEARISH"

    def _wall_payload(row: Optional[dict], *, side: str) -> Dict[str, Any]:
        strike = _fv((row or {}).get("strike"))
        oi = _fv((row or {}).get("oi"))
        dist = None
        if strike is not None:
            if side == "CALL":
                dist = float(strike) - float(spot or 0.0)
            else:
                dist = float(spot or 0.0) - float(strike)
        return {
            "strike": strike,
            "oi": oi,
            "distance_from_spot": None if dist is None else round(float(dist), 2),
        }

    return (
        {
            "rows_count": len(rows),
            "oi_rows_count": len([r for r in ce_rows + pe_rows if r.get("oi") is not None]),
            "spot": round(float(spot or 0.0), 2),
            "near_atm_band_points": float(band),
            "total_call_oi": round(float(total_call_oi), 2),
            "total_put_oi": round(float(total_put_oi), 2),
            "pcr_total": None if pcr_total is None else round(float(pcr_total), 3),
            "near_call_oi": round(float(near_call_oi), 2),
            "near_put_oi": round(float(near_put_oi), 2),
            "pcr_near_atm": None if pcr_near is None else round(float(pcr_near), 3),
            "pcr_bias": pcr_bias,
            "near_rows_count": len(near_ce) + len(near_pe),
            "call_wall": _wall_payload(call_wall, side="CALL"),
            "put_wall": _wall_payload(put_wall, side="PUT"),
            "call_wall_above": _wall_payload(call_wall_above, side="CALL"),
            "put_wall_below": _wall_payload(put_wall_below, side="PUT"),
            "oi_build": {
                "available": oi_delta_points > 0,
                "points": int(oi_delta_points),
                "near_call_oi_change": round(float(near_call_oi_delta), 2) if oi_delta_points else None,
                "near_put_oi_change": round(float(near_put_oi_delta), 2) if oi_delta_points else None,
                "pcr_oi_change_near": None if pcr_oi_change_near is None else round(float(pcr_oi_change_near), 3),
                "bias": oi_build_bias,
            },
        },
        current_oi_map,
    )


def _compute_mtm(legs: List[OptionLeg]) -> float:
    mtm = 0.0
    for leg in legs:
        if leg.current_ltp is None:
            continue
        side = _normalize_leg_side(leg.side)
        pnl = (leg.entry_price - leg.current_ltp) if side == LegSide.SELL else (leg.current_ltp - leg.entry_price)
        mtm += pnl * leg.quantity
    return mtm


def _fetch_expiry(dw: DhanWrapper):
    return _fetch_expiry_with_settings(dw, DEFAULT_SETTINGS)


def _weekday_to_int(value) -> Optional[int]:
    if isinstance(value, int):
        return value if 0 <= value <= 6 else None
    if isinstance(value, str):
        name = value.strip().lower()
        mapping = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }
        if name in mapping:
            return mapping[name]
        try:
            num = int(name)
            if 0 <= num <= 6:
                return num
        except Exception:
            return None
    return None


def _parse_holiday_list(settings: Dict[str, Any]) -> set:
    holidays_raw = settings.get("holiday_list") or []
    # support comma separated string for backward compatibility
    if isinstance(holidays_raw, str):
        holidays_raw = [h.strip() for h in holidays_raw.replace(",", "\n").splitlines() if h.strip()]
    out = set()
    for item in holidays_raw:
        try:
            clean = str(item).split("T", 1)[0].split(" ", 1)[0]
            out.add(datetime.fromisoformat(clean).date())
        except Exception:
            continue
    return out


def _derive_shifted_holidays(target_weekday: Optional[int], expiries: List[dt_date]) -> set:
    """
    If the exchange has shifted expiry (e.g., no Tuesday but a Wednesday expiry),
    infer the missing target weekday immediately before the shifted expiry as a holiday.
    """
    if target_weekday is None or not expiries:
        return set()
    today = datetime.now().date()
    future = [e for e in expiries if e >= today]
    if not future:
        return set()
    future.sort()
    first_exp = future[0]
    if first_exp.weekday() == target_weekday:
        return set()
    # compute the most recent target_weekday before the shifted expiry
    delta_days = (first_exp.weekday() - target_weekday) % 7
    inferred_holiday = first_exp - timedelta(days=delta_days or 7)
    if inferred_holiday >= today - timedelta(days=7):
        return {inferred_holiday}
    return set()


def _fetch_expiry_with_settings(dw: DhanWrapper, settings: Dict[str, Any]):
    try:
        expiries = dw.get_optionchain_expirylist("IDX_I", 13)
    except AttributeError:
        try:
            expiries = dw.get_expiry_list(13, "IDX_I")
        except Exception:
            expiries = []
    except Exception:
        expiries = []

    # Normalize into a list of ISO strings if we received a dict
    expiry_list: List[str] = []
    if isinstance(expiries, list):
        expiry_list = [str(e) for e in expiries]
    elif isinstance(expiries, dict):
        for key in ("Expiry", "expiry", "expiries", "data"):
            val = expiries.get(key)
            if isinstance(val, list):
                expiry_list = [str(e) for e in val]
                break

    today = datetime.now().date()
    parsed: List[dt_date] = []
    for raw in expiry_list:
        clean = raw.strip()
        if "T" in clean:
            clean = clean.split("T", 1)[0]
        if " " in clean:
            clean = clean.split(" ", 1)[0]
        try:
            parsed.append(datetime.fromisoformat(clean).date())
        except Exception:
            continue
    parsed.sort()
    if not parsed:
        return today

    target_weekday = _weekday_to_int(settings.get("nifty_expiry_weekday", "Tuesday"))
    holidays = _parse_holiday_list(settings)
    shift_if_holiday = bool(settings.get("expiry_shift_if_holiday", True))
    holidays |= _derive_shifted_holidays(target_weekday, parsed)

    def is_valid(cand: dt_date) -> bool:
        if cand < today:
            return False
        if target_weekday is not None and cand.weekday() != target_weekday:
            return False
        if shift_if_holiday and cand in holidays:
            return False
        return True

    filtered = [c for c in parsed if is_valid(c)]
    if filtered:
        return filtered[0]

    # fallback: ignore weekday filter but still respect holiday shift
    filtered_no_weekday = [c for c in parsed if c >= today and (not shift_if_holiday or c not in holidays)]
    if filtered_no_weekday:
        return filtered_no_weekday[0]

    # ultimate fallback: last known expiry
    return parsed[-1]


def _fetch_monthly_expiry(dw: DhanWrapper) -> Optional[dt_date]:
    """
    Pick the last available expiry in the nearest future month (approx monthly).
    """
    try:
        expiries = dw.get_optionchain_expirylist("IDX_I", 13)
    except AttributeError:
        try:
            expiries = dw.get_expiry_list(13, "IDX_I")
        except Exception:
            expiries = []
    except Exception:
        expiries = []

    expiry_list: List[str] = []
    if isinstance(expiries, list):
        expiry_list = [str(e) for e in expiries]
    elif isinstance(expiries, dict):
        for key in ("Expiry", "expiry", "expiries", "data"):
            val = expiries.get(key)
            if isinstance(val, list):
                expiry_list = [str(e) for e in val]
                break
    parsed: List[dt_date] = []
    for raw in expiry_list:
        clean = raw.split("T", 1)[0].split(" ", 1)[0].strip()
        try:
            parsed.append(datetime.fromisoformat(clean).date())
        except Exception:
            continue
    if not parsed:
        return None
    today = datetime.now().date()
    future = sorted(d for d in parsed if d >= today)
    if not future:
        return max(parsed)
    first_month = (future[0].year, future[0].month)
    month_candidates = [d for d in future if (d.year, d.month) == first_month]
    if month_candidates:
        return max(month_candidates)
    return future[-1]


def _fetch_bkm_monthly_roll_plan(dw: DhanWrapper) -> Tuple[Optional[dt_date], Optional[dt_date]]:
    """
    Batman BKM monthly roll plan:
    - anchor expiry = nearest monthly expiry (current monthly cycle)
    - target expiry = next monthly expiry after anchor (series to trade)

    Example on Feb monthly cycle:
      anchor = 2026-02-24
      target = 2026-03-31
      planned entry day = last Friday before anchor (2026-02-20)
    """
    try:
        expiries = dw.get_optionchain_expirylist("IDX_I", 13)
    except AttributeError:
        try:
            expiries = dw.get_expiry_list(13, "IDX_I")
        except Exception:
            expiries = []
    except Exception:
        expiries = []

    expiry_list: List[str] = []
    if isinstance(expiries, list):
        expiry_list = [str(e) for e in expiries]
    elif isinstance(expiries, dict):
        for key in ("Expiry", "expiry", "expiries", "data"):
            val = expiries.get(key)
            if isinstance(val, list):
                expiry_list = [str(e) for e in val]
                break

    parsed: List[dt_date] = []
    for raw in expiry_list:
        clean = str(raw).split("T", 1)[0].split(" ", 1)[0].strip()
        try:
            parsed.append(datetime.fromisoformat(clean).date())
        except Exception:
            continue
    if not parsed:
        return None, None

    today = datetime.now().date()
    future = sorted(set(d for d in parsed if d >= today))
    if not future:
        future = sorted(set(parsed))
        if not future:
            return None, None

    month_last: List[dt_date] = []
    bucket: Dict[Tuple[int, int], List[dt_date]] = {}
    for d in future:
        bucket.setdefault((d.year, d.month), []).append(d)
    for ym in sorted(bucket.keys()):
        month_last.append(max(bucket[ym]))

    if not month_last:
        return None, None
    anchor = month_last[0]
    target = month_last[1] if len(month_last) > 1 else anchor
    return anchor, target


def _last_friday_before(expiry: dt_date) -> dt_date:
    probe = expiry
    while probe.weekday() != 4:
        probe -= timedelta(days=1)
    return probe


def _place_leg_order(
    dw: DhanWrapper,
    leg: OptionLeg,
    *,
    close: bool,
    trade_mode: str,
    live_order_executor: Optional[LiveOrderExecutor] = None,
) -> Dict[str, Any]:
    side = _normalize_leg_side(leg.side)
    if close:
        order_side = LegSide.BUY if side == LegSide.SELL else LegSide.SELL
    else:
        order_side = side
    if trade_mode == "paper":
        # Paper: do not hit broker; log a synthetic blotter row once.
        _log_trade_event(trade_mode=trade_mode, side=order_side.value, leg=leg, order_type="MARKET", notes="paper")
        return {"ok": True, "verified": True, "paper": True, "order_side": order_side.value}
    security_id = _ensure_leg_security_id(leg, leg.symbol, leg.expiry)
    if not security_id:
        log.error("Skipping leg order due to missing security_id (strike=%s opt=%s)", leg.strike, leg.option_type)
        return {"ok": False, "verified": False, "error": "missing_security_id", "order_side": order_side.value}
    log.info(
        "Placing %s order sec_id=%s strike=%s opt=%s qty=%s expiry=%s",
        order_side.value,
        security_id,
        leg.strike,
        leg.option_type.value if hasattr(leg.option_type, "value") else str(leg.option_type),
        leg.quantity,
        leg.expiry,
    )
    if trade_mode == "live" and live_order_executor:
        exec_result = live_order_executor.place_and_verify(
            dw=dw,
            side=order_side.value,
            exchange_seg="NSE_FNO",
            security_id=int(security_id),
            quantity=int(leg.quantity),
            product_type="MARGIN",
            order_type="MARKET",
            client_order_prefix="BKMCL" if close else "BKMOP",
        )
        if not bool(exec_result.get("ok")):
            log.error(
                "[ExecGuard] order verify failed side=%s sec_id=%s qty=%s remaining=%s err=%s",
                order_side.value,
                security_id,
                leg.quantity,
                exec_result.get("remaining_qty"),
                exec_result.get("error"),
            )
            return {
                "ok": False,
                "verified": bool(exec_result.get("verified")),
                "order_side": order_side.value,
                "security_id": int(security_id),
                "result": exec_result,
            }
        _log_trade_event(
            trade_mode=trade_mode,
            side=order_side.value,
            leg=leg,
            order_type="MARKET",
            notes="verified" if bool(exec_result.get("verified")) else "",
        )
        return {
            "ok": True,
            "verified": bool(exec_result.get("verified")),
            "order_side": order_side.value,
            "security_id": int(security_id),
            "result": exec_result,
        }

    dw.place_order(
        side=order_side.value,
        exchange_seg="NSE_FNO",
        security_id=security_id,
        quantity=leg.quantity,
        product_type="MARGIN",
        order_type="MARKET",
    )
    _log_trade_event(trade_mode=trade_mode, side=order_side.value, leg=leg, order_type="MARKET")
    return {"ok": True, "verified": False, "order_side": order_side.value, "security_id": int(security_id)}


def _cached_intraday_for_day(dw: DhanWrapper, trade_day: dt_date, interval: int = INTRADAY_INTERVAL_MIN) -> List[dict]:
    key = (trade_day.isoformat(), interval)
    now = datetime.now()
    entry = _INTRADAY_CACHE.get(key)
    ttl = 60 if trade_day == now.date() else 3600
    if entry:
        age = (now - entry["ts"]).total_seconds()
        if age < ttl:
            return entry["data"]
    candles = dw.get_intraday_candles(
        INDEX_SECURITY_ID,
        INDEX_EXCHANGE_SEG,
        INDEX_INSTRUMENT,
        interval=interval,
        from_date=trade_day.isoformat(),
        to_date=trade_day.isoformat(),
    )
    if candles:
        _INTRADAY_CACHE[key] = {"ts": now, "data": candles}
    elif trade_day < now.date():
        # cache empty for past non-trading days to avoid repeated calls
        _INTRADAY_CACHE[key] = {"ts": now, "data": []}
    return candles


def _simplify_candles(raw: List[dict], limit: int = 2) -> List[dict]:
    simplified: List[dict] = []
    for candle in raw[:limit]:
        simplified.append(
            {
                "open": float(candle.get("open") or 0.0),
                "high": float(candle.get("high") or candle.get("open") or 0.0),
                "low": float(candle.get("low") or candle.get("open") or 0.0),
                "close": float(candle.get("close") or candle.get("open") or 0.0),
                "volume": float(candle.get("volume") or 0.0),
            }
        )
    return simplified


def _calc_vwap_from_candles(candles: List[dict], fallback: float) -> float:
    total_volume = 0.0
    pv_sum = 0.0
    for candle in candles:
        vol = float(candle.get("volume") or 0.0)
        if vol <= 0:
            continue
        typical = (float(candle.get("open") or 0.0) + float(candle.get("high") or 0.0) + float(candle.get("low") or 0.0) + float(candle.get("close") or 0.0)) / 4.0
        total_volume += vol
        pv_sum += typical * vol
    if total_volume > 0:
        return pv_sum / total_volume
    closes = [float(candle.get("close") or 0.0) for candle in candles if candle.get("close") is not None]
    return closes[-1] if closes else fallback


def _average_volume(candles: List[dict]) -> float:
    vols = [float(candle.get("volume") or 0.0) for candle in candles if candle.get("volume") not in (None, "")]
    if not vols:
        return 0.0
    return sum(vols) / len(vols)


def _prev_day_levels(candles: List[dict], fallback: float) -> Tuple[float, float, float]:
    highs = [float(candle.get("high") or 0.0) for candle in candles if candle.get("high") is not None]
    lows = [float(candle.get("low") or 0.0) for candle in candles if candle.get("low") is not None]
    close = fallback
    for candle in reversed(candles):
        if candle.get("close") is not None:
            close = float(candle.get("close") or 0.0)
            break
    high = max(highs) if highs else fallback
    low = min(lows) if lows else fallback
    return high, low, close


def build_market_snapshot(dw: DhanWrapper, *, avg_volume_hint: float = 0.0) -> Tuple[MarketSnapshot, float, float, float]:
    now = datetime.now()
    spot = float(dw.get_ltp_once(INDEX_EXCHANGE_SEG, INDEX_SECURITY_ID) or 0.0)
    trade_day = now.date()
    todays_candles = _cached_intraday_for_day(dw, trade_day, INTRADAY_INTERVAL_MIN)
    if not spot and todays_candles:
        spot = float(todays_candles[-1].get("close") or 0.0)

    prev_candles: List[dict] = []
    probe = trade_day - timedelta(days=1)
    for _ in range(5):
        if probe < dt_date(2000, 1, 1):
            break
        prev = _cached_intraday_for_day(dw, probe, INTRADAY_INTERVAL_MIN)
        if prev:
            prev_candles = prev
            break
        probe -= timedelta(days=1)

    yesterday_high, yesterday_low, yesterday_close = _prev_day_levels(prev_candles, spot or 0.0)
    avg_volume = _average_volume(prev_candles)
    if avg_volume <= 0 and avg_volume_hint:
        avg_volume = float(avg_volume_hint)
    vwap = _calc_vwap_from_candles(todays_candles, spot or 0.0)
    pivot = (yesterday_high + yesterday_low + yesterday_close) / 3.0 if prev_candles else (spot or 0.0)

    snapshot = MarketSnapshot(
        symbol="NIFTY",
        spot=spot,
        candles_5m=_simplify_candles(todays_candles, limit=2),
        yesterday_high=yesterday_high,
        yesterday_low=yesterday_low,
        yesterday_close=yesterday_close,
        india_vix=0.0,
        now=now,
    )
    return snapshot, float(avg_volume), float(vwap), float(pivot)


def main() -> None:
    poll_sec = float(os.getenv("POLL_SEC", "10"))
    _warm_scrip_master(force=False)
    _ensure_dhan_credentials()
    settings = _load_agent_settings()
    selected_strategy_file = SELECTED_STRATEGY_FILE
    trade_mode = _startup_preflight(
        strategy_file=selected_strategy_file,
        trade_mode_env=os.getenv("TRADE_MODE"),
    )
    agent_heartbeat = AgentHeartbeat(
        path=AGENT_HEARTBEAT_FILE,
        config=HeartbeatConfig.from_settings(settings),
        logger=log,
    )
    alert_journal = AlertJournal(
        path=AGENT_ALERTS_FILE,
        config=AlertConfig.from_settings(settings),
        logger=log,
    )
    telegram_forwarder = TelegramAlertForwarder(
        config=TelegramAlertConfig.from_settings(settings),
        alerts_path=AGENT_ALERTS_FILE,
        status_path=TELEGRAM_ALERT_STATUS_FILE,
        logger=log,
    )

    def _ops_alert(
        severity: str,
        code: str,
        message: str,
        *,
        details: Optional[Dict[str, Any]] = None,
        dedupe_key: Optional[str] = None,
        force: bool = False,
    ) -> None:
        try:
            payload_details: Dict[str, Any] = dict(details or {})
            code_u = str(code or "").upper()
            if code_u == "LIVEGATE_FAILSAFE_TRIGGERED":
                payload_details.setdefault("agent_action", "Safety mode ON. Agent locked new entries for today and tried to close the trade if positions were open.")
                payload_details.setdefault("trade_impact", "Current trade may remain open only if close could not be executed.")
                payload_details.setdefault("what_you_should_do", "Check broker positions when possible. If trade is still open, keep watching Telegram alerts.")
                payload_details.setdefault("plain_reason", "Market data/broker feed looked unstable repeatedly.")
            elif code_u == "POSITION_RECONCILE_MISMATCH_LOCKED":
                payload_details.setdefault("agent_action", "Agent locked new entries to avoid taking wrong action on a mismatched trade.")
                payload_details.setdefault("trade_impact", "Monitoring continues, but sync needs review.")
                payload_details.setdefault("what_you_should_do", "Check the app/broker positions when possible.")
                payload_details.setdefault("plain_reason", "Agent and broker positions did not match.")
            elif code_u == "ORDER_EXECUTION_OPEN_FAIL":
                payload_details.setdefault("agent_action", "Agent stopped new entries and attempted rollback of the partial open.")
                payload_details.setdefault("trade_impact", "Trade may be partially opened if rollback did not fully close.")
                payload_details.setdefault("what_you_should_do", "Check broker positions as soon as possible.")
                payload_details.setdefault("plain_reason", "Some order legs failed while opening the Batman trade.")
            elif code_u == "ORDER_EXECUTION_CLOSE_FAIL":
                payload_details.setdefault("agent_action", "Agent tried to close the trade but close execution failed; entries were locked for safety.")
                payload_details.setdefault("trade_impact", "Trade may still be open.")
                payload_details.setdefault("what_you_should_do", "Check broker positions and app as soon as possible.")
                payload_details.setdefault("plain_reason", "Broker rejected/failed one or more close orders.")
            elif code_u.startswith("BKM_AI_"):
                action_name = code_u.replace("BKM_AI_", "")
                payload_details.setdefault("agent_action", "AI warning only. No automatic close was done by AI.")
                payload_details.setdefault("trade_impact", "Trade remains open unless a hard safety rule or strategy exit closes it.")
                payload_details.setdefault("what_you_should_do", "Check the app when possible, especially if repeated warnings continue.")
                payload_details.setdefault("plain_reason", f"AI manager changed recommendation to {action_name}.")
            alert_journal.emit(
                severity=severity,
                code=code,
                message=message,
                details=payload_details,
                source="start_agent",
                when=_ist_now(),
                dedupe_key=dedupe_key,
                force=force,
            )
        except Exception:
            pass

    def _heartbeat(
        *,
        force: bool = False,
        status: str = "RUNNING",
        phase: str = "loop",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "status": status,
            "phase": phase,
            "pid": os.getpid(),
            "trade_mode": trade_mode,
            "strategy_file": selected_strategy_file,
        }
        if extra:
            payload.update(extra)
        try:
            agent_heartbeat.beat(payload, when=_ist_now(), force=force)
        except Exception:
            pass

    def _telegram_pump(*, force: bool = False) -> None:
        try:
            out = telegram_forwarder.process_pending(
                creds=_load_saved_creds(),
                trade_mode=trade_mode,
                strategy_file=selected_strategy_file,
                force=force,
                when=_ist_now(),
            )
            if out.get("reason") in {"SEND_FAILED"}:
                log.warning("[TelegramAlerts] send failed: %s", out.get("error"))
        except Exception as exc:
            # Never let notifier failures affect trading loop.
            log.warning("[TelegramAlerts] pump error: %s", exc)

    def _telegram_trade_summary(
        *,
        now: datetime,
        basket_expiry: str,
        spot: float,
        pnl: float,
        tp: float,
        sl: float,
        day_mode_value: str,
        ai_eval: Optional[Dict[str, Any]] = None,
    ) -> None:
        nonlocal bkm_last_telegram_summary_at
        if trade_mode != "live":
            return
        if not bool(settings.get("telegram_trade_summary_enabled", True)):
            return
        if bool(settings.get("telegram_trade_summary_market_hours_only", True)) and not _is_india_market_open(now):
            return
        interval_sec = max(60.0, float(settings.get("telegram_trade_summary_interval_sec", 900.0)))
        if (
            bkm_last_telegram_summary_at is not None
            and (now - bkm_last_telegram_summary_at).total_seconds() < interval_sec
        ):
            return

        ai_snap = ai_eval if isinstance(ai_eval, dict) and ai_eval else (bkm_ai_manager.snapshot() if bkm_ai_manager else {})
        ai_action = str((ai_snap or {}).get("action") or "HOLD").upper()
        ai_sev = str((ai_snap or {}).get("severity") or "INFO").upper()
        ai_reasons = list((ai_snap or {}).get("reasons") or [])
        ai_reason = str(ai_reasons[0]) if ai_reasons else "NORMAL_MONITOR"

        gate_snap = live_gate.snapshot() if live_gate else {}
        rec_snap = position_reconciler.snapshot() if position_reconciler else {}
        exr_snap = execution_recovery_guard.snapshot() if execution_recovery_guard else {}
        gate_status = str((gate_snap or {}).get("status") or "NA").upper()
        gate_locked_for = (gate_snap or {}).get("locked_for_date")
        rec_status = str((rec_snap or {}).get("status") or "NA").upper()
        exr_status = str((exr_snap or {}).get("status") or "NA").upper()
        rec_match = (((rec_snap or {}).get("last_diff_summary") or {}).get("match"))

        level = "RELAX"
        reason_text = "Trade looks stable and the agent is watching it."
        next_step = "No action needed now."
        if exr_status != "OK" or rec_status != "OK" or rec_match is False:
            level = "WORRY"
            reason_text = "Agent found a trade sync problem."
            next_step = "Please check the app when possible."
        elif ai_action == "FLATTEN_RECOMMEND" or ai_sev == "CRITICAL":
            level = "WORRY"
            reason_text = "Risk is high right now."
            next_step = "Please check the app soon."
        elif ai_action in {"WATCH_TIGHT", "REDUCE_RISK"} or ai_sev == "WARN" or str(day_mode_value).upper() == "LOCKED_RED":
            level = "WATCH"
            reason_text = "Risk is rising, but monitoring is active."
            next_step = "Check when you get a chance."

        headline = {
            "RELAX": "Relax: your trade looks okay.",
            "WATCH": "Watch closely: risk is increasing.",
            "WORRY": "Worry: please check your trade.",
        }.get(level, "Trade update")

        if rec_status == "OK" and exr_status == "OK" and rec_match is True:
            sync_line = "Trade sync with broker: OK."
        else:
            sync_line = f"Trade sync: reconcile={rec_status}, recovery={exr_status}."
        entry_line = "No new trades today (safety lock is on)." if (gate_status == "LOCKED" and gate_locked_for) else "New trades are allowed."
        if exr_status != "OK" or rec_status != "OK" or rec_match is False:
            agent_action_line = "Agent action: safety lock is active due to sync issue."
        elif ai_action == "FLATTEN_RECOMMEND":
            agent_action_line = "Agent action: no auto-close by AI (AI is advisory only)."
        elif str(day_mode_value).upper() == "LOCKED_RED" or (gate_status == "LOCKED" and gate_locked_for):
            agent_action_line = "Agent action: monitoring continues; new entries are blocked."
        else:
            agent_action_line = "Agent action: monitoring only (no trade change made)."

        def _fmt_rs(value: Any) -> str:
            try:
                amt = float(value or 0.0)
                sign = "+" if amt >= 0 else "-"
                return f"{sign}Rs {abs(amt):,.0f}"
            except Exception:
                return "NA"

        text = (
            f"{headline}\n"
            f"Agent is actively monitoring your Batman trade.\n"
            f"Current P&L: {_fmt_rs(pnl)}\n"
            f"NIFTY spot: {float(spot or 0.0):,.0f}\n"
            f"AI view: {ai_action} ({ai_reason})\n"
            f"{agent_action_line}\n"
            f"{sync_line}\n"
            f"{entry_line}\n"
            f"Why: {reason_text}\n"
            f"What you should do: {next_step}\n"
            f"Time: {now.strftime('%I:%M %p').lstrip('0')}"
        )
        out = telegram_forwarder.send_direct_message(
            creds=_load_saved_creds(),
            text=text,
            code=f"TELEGRAM_TRADE_SUMMARY_{level}",
            trade_mode=trade_mode,
            when=now,
        )
        if out.get("ok"):
            bkm_last_telegram_summary_at = now
        elif out.get("reason") == "SEND_FAILED":
            log.warning("[TelegramSummary] send failed: %s", out.get("error"))

    def _telegram_ai_lock_change(
        *,
        now: datetime,
        active: bool,
        source: str,
        ai_eval: Optional[Dict[str, Any]] = None,
        basket_expiry: Optional[str] = None,
        note: Optional[str] = None,
        stable_for_sec: Optional[float] = None,
    ) -> None:
        if trade_mode != "live":
            return
        if not bool(settings.get("telegram_ai_lock_change_enabled", True)):
            return
        ai_snap = ai_eval if isinstance(ai_eval, dict) and ai_eval else (bkm_ai_manager.snapshot() if bkm_ai_manager else {})
        ai_action = str((ai_snap or {}).get("action") or "HOLD").upper()
        ai_score = float((ai_snap or {}).get("score") or 0.0)
        ai_reasons = list((ai_snap or {}).get("reasons") or [])
        ai_reason = str(ai_reasons[0]) if ai_reasons else "NORMAL_MONITOR"
        expiry_txt = basket_expiry or str((ai_snap or {}).get("basket_expiry") or "NA")
        if active:
            headline = "Watch closely: agent paused new entries."
            action_line = "Agent action: blocked new entries only. Current trade monitoring continues."
            why_line = f"Why: AI protective mode saw higher risk ({ai_action}, score {ai_score:.1f}, {ai_reason})."
            next_step = "No urgent action needed. Review only if you want to override."
        else:
            stable_txt = ""
            if stable_for_sec is not None and stable_for_sec > 0:
                stable_min = max(1, int(round(float(stable_for_sec) / 60.0)))
                stable_txt = f" after stable conditions for ~{stable_min} min"
            headline = "Relax: agent re-enabled new entries."
            action_line = "Agent action: AI protective entry lock removed. Monitoring continues."
            why_line = f"Why: risk stayed controlled{stable_txt}."
            if source == "manual_override":
                why_line = "Why: you manually cleared the AI protective lock."
            elif note:
                why_line = f"Why: {note}"
            next_step = "No action needed now."
        text = (
            f"{headline}\n"
            f"{action_line}\n"
            f"{why_line}\n"
            f"Current AI view: {ai_action} (score {ai_score:.1f})\n"
            f"Trade expiry: {expiry_txt}\n"
            f"Source: {source}\n"
            f"What you should do: {next_step}\n"
            f"Time: {now.strftime('%I:%M %p').lstrip('0')}"
        )
        out = telegram_forwarder.send_direct_message(
            creds=_load_saved_creds(),
            text=text,
            code=f"TELEGRAM_BKM_AI_LOCK_{'ON' if active else 'OFF'}",
            trade_mode=trade_mode,
            when=now,
        )
        if out.get("reason") == "SEND_FAILED":
            log.warning("[TelegramAILock] send failed: %s", out.get("error"))

    def _telegram_intraday_signal(
        *,
        now: datetime,
        signal_payload: Dict[str, Any],
    ) -> None:
        if trade_mode != "live":
            return
        if not bool(settings.get("telegram_intraday_signal_enabled", True)):
            return
        if bool(settings.get("telegram_intraday_signal_market_hours_only", True)) and not _is_india_market_open(now):
            return
        signal = str(signal_payload.get("signal") or "NO_TRADE").upper()
        rec = signal_payload.get("recommendation") if isinstance(signal_payload.get("recommendation"), dict) else {}
        strategy = str(signal_payload.get("strategy") or rec.get("strategy_label") or "NA")
        bias = str(signal_payload.get("market_bias") or "NEUTRAL")
        conflict = float(signal_payload.get("signal_conflict_score") or 0.0)
        trend_conf = float(signal_payload.get("trend_confidence") or 0.0)
        headline = str(rec.get("headline") or rec.get("signal_text") or "Intraday advisor update")
        what_to_enter = str(rec.get("what_to_enter") or "No trade now.")
        sl_text = str((rec.get("sl") or {}).get("text") or rec.get("sl_text") or "SL not applicable")
        tp_text = str((rec.get("tp") or {}).get("text") or rec.get("tp_text") or "TP not applicable")
        hold_text = str((rec.get("hold") or {}).get("text") or rec.get("hold_text") or "Hold guidance not available")
        why = rec.get("why") if isinstance(rec.get("why"), list) else []
        why_line = str(why[0]) if why else ", ".join([str(x) for x in (signal_payload.get("reasons") or [])[:2]]) or "No specific reason."
        if signal == "ENTER_NOW":
            action_line = "Action: You can take this setup if it matches your risk size."
        elif signal == "WAIT":
            action_line = "Action: Wait. Do not force an entry yet."
        else:
            action_line = "Action: No trade. Stay disciplined and avoid overtrading."
        text = (
            f"Intraday AI signal: {signal}\n"
            f"{headline}\n"
            f"Setup: {strategy} | Bias: {bias}\n"
            f"What to enter: {what_to_enter}\n"
            f"{sl_text}\n"
            f"{tp_text}\n"
            f"{hold_text}\n"
            f"Why: {why_line}\n"
            f"{action_line}\n"
            f"Trend confidence: {int(round(trend_conf * 100.0))}% | Conflict: {int(round(conflict))}%\n"
            f"Time: {now.strftime('%I:%M %p').lstrip('0')}"
        )
        out = telegram_forwarder.send_direct_message(
            creds=_load_saved_creds(),
            text=text,
            code=f"TELEGRAM_INTRADAY_SIGNAL_{signal}",
            trade_mode=trade_mode,
            when=now,
        )
        if out.get("reason") == "SEND_FAILED":
            log.warning("[TelegramIntradaySignal] send failed: %s", out.get("error"))

    def _telegram_market_close_summary(
        *,
        now: datetime,
        basket_expiry: str,
        spot: float,
        pnl: float,
        tp: float,
        sl: float,
        day_mode_value: str,
        ai_eval: Optional[Dict[str, Any]] = None,
    ) -> None:
        nonlocal bkm_last_market_close_summary_date
        if trade_mode != "live":
            return
        if not bool(settings.get("telegram_market_close_summary_enabled", True)):
            return
        if now.weekday() >= 5:
            return
        if _is_india_market_open(now):
            return
        try:
            hh, mm = str(settings.get("telegram_market_close_summary_not_before", "15:32")).split(":")
            not_before = dtime(int(hh), int(mm))
        except Exception:
            not_before = dtime(15, 32)
        if now.time() < not_before:
            return
        if bkm_last_market_close_summary_date == now.date().isoformat():
            return

        ai_snap = ai_eval if isinstance(ai_eval, dict) and ai_eval else (bkm_ai_manager.snapshot() if bkm_ai_manager else {})
        ai_action = str((ai_snap or {}).get("action") or "HOLD").upper()
        ai_reasons = list((ai_snap or {}).get("reasons") or [])
        ai_reason = str(ai_reasons[0]) if ai_reasons else "NORMAL_MONITOR"
        rec_snap = position_reconciler.snapshot() if position_reconciler else {}
        exr_snap = execution_recovery_guard.snapshot() if execution_recovery_guard else {}
        gate_snap = live_gate.snapshot() if live_gate else {}
        rec_status = str((rec_snap or {}).get("status") or "NA").upper()
        exr_status = str((exr_snap or {}).get("status") or "NA").upper()
        rec_match = (((rec_snap or {}).get("last_diff_summary") or {}).get("match"))
        gate_status = str((gate_snap or {}).get("status") or "NA").upper()

        level = "RELAX"
        if exr_status != "OK" or rec_status != "OK" or rec_match is False:
            level = "WORRY"
        elif ai_action in {"WATCH_TIGHT", "REDUCE_RISK", "FLATTEN_RECOMMEND"}:
            level = "WATCH" if ai_action != "FLATTEN_RECOMMEND" else "WORRY"
        elif str(day_mode_value).upper() == "LOCKED_RED":
            level = "WATCH"

        headline = {
            "RELAX": "Market close update: relax.",
            "WATCH": "Market close update: watch this trade.",
            "WORRY": "Market close update: please review.",
        }.get(level, "Market close update")

        if exr_status == "OK" and rec_status == "OK" and rec_match is True:
            sync_line = "Trade sync with broker: OK."
        else:
            sync_line = f"Trade sync: reconcile={rec_status}, recovery={exr_status}."
        if gate_status == "LOCKED":
            entry_line = "Tomorrow new entries may stay blocked until you reset the safety lock."
        else:
            entry_line = "No entry lock is active right now."

        if level == "RELAX":
            action_line = "Agent action: monitoring continues overnight."
            next_step = "You can relax. Just watch Telegram alerts."
        elif level == "WATCH":
            action_line = "Agent action: monitoring continues; no forced action taken."
            next_step = "Check the app when you have time."
        else:
            action_line = "Agent action: safety condition detected. Please review app/broker."
            next_step = "Check the app before next session."

        def _fmt_rs(value: Any) -> str:
            try:
                amt = float(value or 0.0)
                sign = "+" if amt >= 0 else "-"
                return f"{sign}Rs {abs(amt):,.0f}"
            except Exception:
                return "NA"

        text = (
            f"{headline}\n"
            f"Trade is still being monitored.\n"
            f"Current P&L: {_fmt_rs(pnl)}\n"
            f"NIFTY spot (last): {float(spot or 0.0):,.0f}\n"
            f"AI view: {ai_action} ({ai_reason})\n"
            f"{action_line}\n"
            f"{sync_line}\n"
            f"{entry_line}\n"
            f"What you should do: {next_step}\n"
            f"Time: {now.strftime('%I:%M %p').lstrip('0')}"
        )
        out = telegram_forwarder.send_direct_message(
            creds=_load_saved_creds(),
            text=text,
            code=f"TELEGRAM_MARKET_CLOSE_SUMMARY_{level}",
            trade_mode=trade_mode,
            when=now,
        )
        if out.get("ok"):
            bkm_last_market_close_summary_date = now.date().isoformat()
        elif out.get("reason") == "SEND_FAILED":
            log.warning("[TelegramCloseSummary] send failed: %s", out.get("error"))

    # Enforce rollout policy for live mode.
    if trade_mode == "live":
        settings["max_intraday_loss"] = -abs(float(settings.get("live_daily_loss_cap_abs", 5000.0)))
        settings["batman_bkm_force_entry"] = False
        settings["allow_carry_forward"] = bool(settings.get("live_allow_normal_carry", True))

    live_gate: Optional[LiveGate] = None
    if trade_mode == "live" and bool(settings.get("live_gate_enabled", True)):
        live_gate = LiveGate(
            config=LiveGateConfig.from_settings(settings),
            status_path=LIVE_GATE_STATUS_FILE,
            sessions_path=LIVE_GATE_SESSIONS_FILE,
            logger=log,
        )
    position_reconciler: Optional[PositionReconciler] = None
    if trade_mode == "live" and bool(settings.get("live_reconcile_enabled", True)):
        position_reconciler = PositionReconciler(
            config=PositionReconcilerConfig.from_settings(settings),
            status_path=POSITION_RECONCILE_STATUS_FILE,
            logger=log,
        )
    live_order_executor: Optional[LiveOrderExecutor] = None
    if trade_mode == "live" and selected_strategy_file == "batman_bkm_monthly":
        live_order_executor = LiveOrderExecutor(
            config=LiveOrderExecutorConfig.from_settings(settings),
            logger=log,
        )
    execution_journal: Optional[ExecutionJournal] = None
    execution_recovery_guard: Optional[ExecutionRecoveryGuard] = None
    if trade_mode == "live" and selected_strategy_file == "batman_bkm_monthly":
        execution_journal = ExecutionJournal(journal_path=EXECUTION_JOURNAL_FILE, logger=log)
        if bool(settings.get("live_exec_recovery_enabled", True)):
            execution_recovery_guard = ExecutionRecoveryGuard(
                config=ExecutionRecoveryConfig.from_settings(settings),
                status_path=EXECUTION_RECOVERY_STATUS_FILE,
                logger=log,
            )
    bkm_ai_manager: Optional[BatmanBKMAIManager] = None
    if selected_strategy_file == "batman_bkm_monthly" and bool(settings.get("batman_bkm_ai_enabled", True)):
        bkm_ai_manager = BatmanBKMAIManager(
            config=BatmanBKMAIConfig.from_settings(settings),
            status_path=BATMAN_BKM_AI_STATUS_FILE,
            events_path=BATMAN_BKM_AI_EVENTS_FILE,
            logger=log,
        )
    intraday_ai_advisor: Optional[IntradayOptionSellingAdvisor] = None
    if selected_strategy_file == "batman_bkm_monthly" and bool(settings.get("intraday_ai_enabled", True)):
        intraday_ai_advisor = IntradayOptionSellingAdvisor(
            config=IntradayOptionSellingAdvisorConfig.from_settings(settings),
            status_path=INTRADAY_AI_ADVISOR_STATUS_FILE,
            logger=log,
        )

    dw = DhanWrapper(logger=logging.getLogger("dhan_wrapper"))
    startup_resume_bkm_snapshot: Optional[Dict[str, Any]] = None
    if trade_mode == "live" and selected_strategy_file == "batman_bkm_monthly" and execution_recovery_guard and execution_journal:
        startup_now = _ist_now()
        try:
            broker_positions_raw = dw.get_positions_raw()
            broker_legs_startup = _map_positions(broker_positions_raw)
        except Exception as exc:
            recovery_snap = execution_recovery_guard.lock(
                "EXEC_RECOVERY_STARTUP_BROKER_FETCH_FAIL",
                details={"error": str(exc)},
                when=startup_now,
            )
            log.error("[ExecRecovery] startup broker positions fetch failed; locking entries: %s", recovery_snap)
            _ops_alert(
                "CRITICAL",
                "EXEC_RECOVERY_STARTUP_BROKER_FETCH_FAIL",
                "Startup broker position fetch failed; live entries locked.",
                details={"error": str(exc)},
                force=True,
            )
            broker_legs_startup = []
        else:
            local_bkm_state = load_strategy_state().get("BATMAN_BKM", {})
            journal_summary = execution_journal.analyze_bkm(
                lookback_days=int(settings.get("live_exec_recovery_lookback_days", 45)),
                when=startup_now,
            )
            recovery_out = execution_recovery_guard.evaluate_startup(
                local_bkm_state=local_bkm_state if isinstance(local_bkm_state, dict) else {},
                broker_legs=broker_legs_startup,
                journal_summary=journal_summary,
                when=startup_now,
            )
            if recovery_out.get("ok"):
                active_baskets = recovery_out.get("active_baskets") or {}
                if isinstance(active_baskets, dict) and len(active_baskets) == 1:
                    startup_resume_bkm_snapshot = next(iter(active_baskets.values()))
                log.info(
                    "[ExecRecovery] startup check ok reason=%s active_baskets=%s",
                    recovery_out.get("reason"),
                    list(active_baskets.keys()) if isinstance(active_baskets, dict) else [],
                )
            else:
                log.error(
                    "[ExecRecovery] startup check locked reason=%s details=%s",
                    recovery_out.get("reason"),
                    recovery_out.get("details"),
                )
                _ops_alert(
                    "CRITICAL",
                    str(recovery_out.get("reason") or "EXEC_RECOVERY_STARTUP_LOCK"),
                    "Execution recovery startup validation locked live entries.",
                    details={"details": recovery_out.get("details")},
                    force=True,
                )
                if live_gate:
                    live_gate.mark_loop_error("EXEC_RECOVERY_STARTUP_LOCK", when=startup_now)
    _write_pid_file()
    _heartbeat(force=True, status="STARTING", phase="startup_post_pid")
    _telegram_pump(force=True)
    lot_size = int(settings.get("lot_size", DEFAULT_SETTINGS["lot_size"]))
    selector = StrategySelector(symbol="NIFTY", lot_size=lot_size)
    risk = _build_risk_config(settings)
    smart_enabled = bool(settings.get("smart_selector_enabled", True))
    avg_volume_hint = float(settings.get("avg_5m_volume", 0.0))
    _ensure_csv_headers(TRADE_BLOTTER_PATH, BLOTTER_FIELDS)
    _ensure_csv_headers(INTEL_LOG_PATH, ["timestamp", "prob_bull", "prob_bear", "prob_sideways", "regime", "strategy", "size_multiplier", "notes", "gap_pct", "vix", "vwap_distance"])
    _ensure_csv_headers(FEATURE_LOG_PATH, FEATURE_FIELDS)
    if live_gate:
        log.info(
            "[LiveGate] enabled status=%s stage=%s lot=%s",
            live_gate.snapshot().get("status"),
            live_gate.snapshot().get("stage"),
            live_gate.get_lot_multiplier(),
        )
    if position_reconciler:
        log.info(
            "[Reconcile] enabled status=%s hard_lock=%s grace_sec=%s confirm=%s",
            position_reconciler.snapshot().get("status"),
            position_reconciler.snapshot().get("hard_lock"),
            int(settings.get("live_reconcile_grace_sec", 30)),
            int(settings.get("live_reconcile_mismatch_confirm_count", 2)),
        )
    if live_order_executor:
        log.info(
            "[ExecGuard] enabled=%s fill_wait=%ss poll=%ss retries=%s verify_positions=%s",
            bool(settings.get("live_order_verify_enabled", True)),
            int(settings.get("live_order_fill_wait_sec", 20)),
            float(settings.get("live_order_fill_poll_sec", 2.0)),
            int(settings.get("live_order_retry_count", 2)),
            bool(settings.get("live_order_verify_positions", True)),
        )
    if execution_recovery_guard:
        log.info(
            "[ExecRecovery] enabled status=%s hard_lock=%s lookback_days=%s",
            execution_recovery_guard.snapshot().get("status"),
            execution_recovery_guard.snapshot().get("hard_lock"),
            int(settings.get("live_exec_recovery_lookback_days", 45)),
        )
    if bkm_ai_manager:
        ai_snap = bkm_ai_manager.snapshot()
        log.info(
            "[BatmanBKM-AI] enabled=%s mode=%s status=%s warn=%.1f reduce=%.1f flatten=%.1f",
            bool(settings.get("batman_bkm_ai_enabled", True)),
            str(settings.get("batman_bkm_ai_mode", "ADVISOR")).upper(),
            ai_snap.get("status"),
            float(settings.get("batman_bkm_ai_warn_score", 30.0)),
            float(settings.get("batman_bkm_ai_reduce_score", 55.0)),
            float(settings.get("batman_bkm_ai_flatten_score", 80.0)),
        )
    log.info(
        "[OpsMonitor] heartbeat_interval=%ss watchdog_stale_after=%ss alert_dedupe=%ss",
        float(settings.get("ops_heartbeat_interval_sec", 10.0)),
        float(settings.get("ops_watchdog_stale_after_sec", 45.0)),
        float(settings.get("ops_alert_dedupe_sec", 60.0)),
    )
    log.info(
        "[TelegramAlerts] enabled=%s min_severity=%s live_only=%s",
        bool(settings.get("telegram_alerts_enabled", False)),
        str(settings.get("telegram_alert_min_severity", "CRITICAL")).upper(),
        bool(settings.get("telegram_alert_live_only", True)),
    )
    log.info("Regime-aware agent started (mode=%s)", trade_mode)
    log.info("Selected strategy file=%s", selected_strategy_file)
    daily_trades = 0
    current_day = _ist_now().date()
    day_entries = 0
    day_legs = 0
    startup_exec_locked = bool(execution_recovery_guard and execution_recovery_guard.should_block_entries(_ist_now()))
    day_mode = "LOCKED_RED" if startup_exec_locked else "NORMAL"
    _heartbeat(
        force=True,
        status="RUNNING",
        phase="startup_ready",
        extra={
            "day_mode": day_mode,
            "live_gate_status": live_gate.snapshot().get("status") if live_gate else None,
            "reconcile_status": position_reconciler.snapshot().get("status") if position_reconciler else None,
            "execution_recovery_status": execution_recovery_guard.snapshot().get("status") if execution_recovery_guard else None,
            "bkm_ai_status": bkm_ai_manager.snapshot().get("status") if bkm_ai_manager else None,
            "bkm_ai_action": bkm_ai_manager.snapshot().get("action") if bkm_ai_manager else None,
        },
    )
    _telegram_pump(force=True)
    orb_config = ORBConfig()
    orb_state = ORBState()
    basket_peak_mtm = 0.0
    trail_active = False
    trail_floor = -1000.0
    day_pnl = 0.0
    feat_extractor = FeatureExtractor()
    regime_scorer = RegimeScorer()
    learning = LearningManager()
    policy = PolicyEngine(learning_manager=learning)
    batman_v2_notice_logged = False
    from market_ai.strategies import BatmanStrategy, BatmanConfig
    batman_v2: Optional[BatmanStrategy] = None
    bkm_strategy: Optional[BatmanBKMStrategy] = None
    loop_seq = 0
    bkm_last_monitor_log_at: Optional[datetime] = None
    bkm_last_park_log_at: Optional[datetime] = None
    bkm_last_telegram_summary_at: Optional[datetime] = None
    bkm_last_market_close_summary_date: Optional[str] = None
    bkm_ai_last_context_at: Optional[datetime] = None
    bkm_ai_context_cache: Optional[Dict[str, Any]] = None
    bkm_ai_prev_oi_map: Dict[Tuple[str, float], float] = {}
    intraday_ai_last_eval_at: Optional[datetime] = None
    intraday_ai_prev_oi_map: Dict[Tuple[str, float], float] = {}
    intraday_ai_last_expiry: Optional[str] = None
    bkm_ai_entry_lock_active: bool = False
    bkm_ai_entry_lock_reason: Optional[str] = None
    bkm_ai_last_entry_lock_state: Optional[bool] = None
    bkm_ai_unlock_stable_since: Optional[datetime] = None
    bkm_ai_protection_enabled: bool = True
    bkm_ai_protect_session_date: str = _ist_now().date().isoformat()
    if bkm_ai_manager and str(getattr(bkm_ai_manager.config, "mode", "ADVISOR")).upper() in {"AUTO_PROTECT", "PROTECTIVE"}:
        try:
            restored_protect = _restore_bkm_ai_protect_lock_for_session(_ist_now())
            bkm_ai_protect_session_date = str(restored_protect.get("session_date") or bkm_ai_protect_session_date)
            bkm_ai_protection_enabled = bool(restored_protect.get("protection_enabled", True))
            if bool(restored_protect.get("active", False)):
                if bkm_ai_protection_enabled:
                    bkm_ai_entry_lock_active = True
                    bkm_ai_entry_lock_reason = str(restored_protect.get("reason") or "AI_PROTECTIVE_ENTRY_LOCK")
                    bkm_ai_last_entry_lock_state = True
                    log.warning(
                        "[BatmanBKM-AI] restored protective entry lock for session=%s reason=%s source_action=%s",
                        restored_protect.get("session_date"),
                        restored_protect.get("reason"),
                        restored_protect.get("source_action"),
                    )
                else:
                    log.info("[BatmanBKM-AI] ignoring restored lock because protection is disabled for session")
        except Exception:
            log.exception("[BatmanBKM-AI] failed to restore protective entry lock state")

    while True:
        try:
            loop_seq += 1
            now = datetime.now()
            now_ist = _ist_now()
            if now_ist.date().isoformat() != bkm_ai_protect_session_date:
                bkm_ai_protect_session_date = now_ist.date().isoformat()
                if bkm_ai_entry_lock_active:
                    log.info("[BatmanBKM-AI] clearing protective entry lock on new session")
                bkm_ai_entry_lock_active = False
                bkm_ai_entry_lock_reason = None
                bkm_ai_last_entry_lock_state = False
                bkm_ai_unlock_stable_since = None
                bkm_ai_protection_enabled = True
                try:
                    _set_bkm_ai_protect_lock(active=False, now=now_ist, protection_enabled=True)
                except Exception:
                    log.exception("[BatmanBKM-AI] failed to clear protective lock on session rollover")
            protect_clear_req = _consume_bkm_ai_protect_clear_request()
            if protect_clear_req is not None:
                req_source = str(protect_clear_req.get("source") or "manual_ui")
                req_note = str(protect_clear_req.get("note") or "")
                req_op = str(protect_clear_req.get("operation") or "clear_lock").strip().lower()
                if req_op == "set_protection_enabled":
                    requested_enabled = bool(protect_clear_req.get("protection_enabled", True))
                    had_lock = bool(bkm_ai_entry_lock_active)
                    prev_enabled = bool(bkm_ai_protection_enabled)
                    bkm_ai_protection_enabled = requested_enabled
                    if not requested_enabled:
                        bkm_ai_entry_lock_active = False
                        bkm_ai_entry_lock_reason = None
                        bkm_ai_last_entry_lock_state = False
                        bkm_ai_unlock_stable_since = None
                    try:
                        _set_bkm_ai_protect_lock(
                            active=bool(bkm_ai_entry_lock_active and bkm_ai_protection_enabled),
                            now=now_ist,
                            protection_enabled=bkm_ai_protection_enabled,
                        )
                    except Exception:
                        log.exception("[BatmanBKM-AI] failed to persist protection toggle request")
                    log.warning(
                        "[BatmanBKM-AI] protection toggled enabled=%s prev_enabled=%s source=%s had_lock=%s note=%s",
                        bkm_ai_protection_enabled,
                        prev_enabled,
                        req_source,
                        had_lock,
                        req_note,
                    )
                    if trade_mode == "live" and prev_enabled != bkm_ai_protection_enabled:
                        code = "BKM_AI_PROTECTION_DISABLED_TODAY" if not bkm_ai_protection_enabled else "BKM_AI_PROTECTION_ENABLED"
                        msg = (
                            "AI entry protection disabled for this session; monitoring continues."
                            if not bkm_ai_protection_enabled
                            else "AI entry protection enabled; agent can apply protective entry locks again."
                        )
                        _ops_alert(
                            "INFO",
                            code,
                            msg,
                            details={
                                "source": req_source,
                                "had_lock": had_lock,
                                "note": req_note,
                                "protection_enabled": bkm_ai_protection_enabled,
                            },
                            dedupe_key=code,
                        )
                else:
                    had_lock = bool(bkm_ai_entry_lock_active)
                    bkm_ai_entry_lock_active = False
                    bkm_ai_entry_lock_reason = None
                    bkm_ai_last_entry_lock_state = False
                    bkm_ai_unlock_stable_since = None
                    try:
                        _set_bkm_ai_protect_lock(active=False, now=now_ist, protection_enabled=bkm_ai_protection_enabled)
                    except Exception:
                        log.exception("[BatmanBKM-AI] failed to persist clear after manual override")
                    log.warning(
                        "[BatmanBKM-AI] protective entry lock cleared by override source=%s had_lock=%s note=%s",
                        req_source,
                        had_lock,
                        req_note,
                    )
                    if trade_mode == "live":
                        _ops_alert(
                            "INFO",
                            "BKM_AI_PROTECT_ENTRY_LOCK_CLEARED",
                            "AI protective entry lock was manually cleared.",
                            details={"source": req_source, "had_lock": had_lock, "note": req_note},
                            dedupe_key="BKM_AI_PROTECT_ENTRY_LOCK_CLEARED",
                        )
                        try:
                            _telegram_ai_lock_change(
                                now=now_ist,
                                active=False,
                                source="manual_override",
                                note=req_note or "manual override",
                            )
                        except Exception:
                            log.exception("[TelegramAILock] manual clear send failed")
            if live_gate:
                live_gate.on_tick(now_ist)
            if position_reconciler:
                position_reconciler.on_tick(now_ist)
            if execution_recovery_guard:
                execution_recovery_guard.on_tick(now_ist)
            _telegram_pump()
            _heartbeat(
                phase="loop_start",
                extra={
                    "loop_seq": loop_seq,
                    "now_ist": now_ist.isoformat(timespec="seconds"),
                    "day_mode": day_mode,
                    "live_gate_status": live_gate.snapshot().get("status") if live_gate else None,
                    "reconcile_status": position_reconciler.snapshot().get("status") if position_reconciler else None,
                    "execution_recovery_status": execution_recovery_guard.snapshot().get("status") if execution_recovery_guard else None,
                    "bkm_ai_entry_lock_active": bool(bkm_ai_entry_lock_active),
                    "bkm_ai_entry_lock_reason": bkm_ai_entry_lock_reason,
                    "bkm_ai_protection_enabled": bool(bkm_ai_protection_enabled),
                },
            )
            log.info("[Loop] tick start mode=%s strategy=%s", trade_mode, selected_strategy_file)
            today = now_ist.date()
            if today != current_day:
                current_day = today
                daily_trades = 0
                day_entries = 0
                day_legs = 0
                day_mode = "NORMAL"
                orb_state = ORBState()
                basket_peak_mtm = 0.0
                trail_active = False
                trail_floor = -1000.0
                day_pnl = 0.0
            # Skip processing on weekends when the market is closed.
            if now_ist.weekday() >= 5:
                log.info("Weekend detected; market is closed. Sleeping for %ss.", poll_sec)
                _heartbeat(phase="weekend_sleep", extra={"loop_seq": loop_seq, "day_mode": day_mode})
                time.sleep(poll_sec)
                continue
            try:
                market, avg_volume, vwap, pivot = build_market_snapshot(dw, avg_volume_hint=avg_volume_hint)
            except Exception:
                if trade_mode == "live" and selected_strategy_file == "batman_bkm_monthly" and live_gate:
                    fail_when = _ist_now()
                    if _is_india_market_open(fail_when):
                        triggered, fail_reason = live_gate.register_connection_failure(when=fail_when)
                        if triggered and not live_gate.should_block_entries(fail_when):
                            closed_legs = _apply_live_gate_failsafe(
                                live_gate=live_gate,
                                reason=fail_reason,
                                dw=dw,
                                bkm_strategy=bkm_strategy,
                                trade_mode=trade_mode,
                                live_order_executor=live_order_executor,
                                execution_journal=execution_journal,
                            )
                            if closed_legs > 0 and position_reconciler:
                                position_reconciler.defer_checks(
                                    seconds=int(settings.get("live_reconcile_grace_sec", 30)),
                                    reason="DATA_FAILSAFE_CLOSE_SUBMITTED",
                                    when=_ist_now(),
                                )
                            day_mode = "LOCKED_RED"
                            log.error(
                                "[LiveGate] FAILSAFE_TRIGGERED reason=%s source=market_snapshot closed_legs=%s",
                                fail_reason,
                                closed_legs,
                            )
                            _ops_alert(
                                "CRITICAL",
                                "LIVEGATE_FAILSAFE_TRIGGERED",
                                "LiveGate fail-safe triggered from market snapshot failure.",
                                details={"reason": fail_reason, "source": "market_snapshot", "closed_legs": closed_legs},
                                dedupe_key="LIVEGATE_FAILSAFE_TRIGGERED",
                            )
                log.exception("[Loop] market snapshot fetch failed")
                _ops_alert(
                    "ERROR",
                    "MARKET_SNAPSHOT_FETCH_FAIL",
                    "Market snapshot fetch failed in agent loop.",
                    details={"loop_seq": loop_seq, "strategy_file": selected_strategy_file},
                    dedupe_key="MARKET_SNAPSHOT_FETCH_FAIL",
                )
                _heartbeat(
                    force=True,
                    status="DEGRADED",
                    phase="market_snapshot_error",
                    extra={"loop_seq": loop_seq, "day_mode": day_mode},
                )
                time.sleep(poll_sec)
                continue
            log.info("[Loop] snapshot spot=%.2f", market.spot)
            _heartbeat(
                phase="market_snapshot_ok",
                extra={"loop_seq": loop_seq, "spot": float(market.spot or 0.0), "day_mode": day_mode},
            )
            # ── Batman BKM monthly branch ───────────────────────────────────
            if selected_strategy_file == "batman_bkm_monthly":
                live_bkm_gate_enabled = trade_mode == "live" and live_gate is not None
                live_bkm_reconcile_enabled = trade_mode == "live" and position_reconciler is not None

                def _defer_reconcile_checks(reason: str) -> None:
                    if not live_bkm_reconcile_enabled or not position_reconciler:
                        return
                    position_reconciler.defer_checks(
                        seconds=int(settings.get("live_reconcile_grace_sec", 30)),
                        reason=reason,
                        when=_ist_now(),
                    )

                def _trigger_bkm_conn_failure(source: str) -> None:
                    nonlocal day_mode
                    if not live_bkm_gate_enabled or not live_gate:
                        return
                    fail_when = _ist_now()
                    if not _is_india_market_open(fail_when):
                        return
                    triggered, fail_reason = live_gate.register_connection_failure(when=fail_when)
                    log.warning(
                        "[LiveGate] conn_failure source=%s triggered=%s reason=%s",
                        source,
                        triggered,
                        fail_reason or "",
                    )
                    if triggered and not live_gate.should_block_entries(fail_when):
                        closed_legs = _apply_live_gate_failsafe(
                            live_gate=live_gate,
                            reason=fail_reason,
                            dw=dw,
                            bkm_strategy=bkm_strategy,
                            trade_mode=trade_mode,
                            live_order_executor=live_order_executor,
                            execution_journal=execution_journal,
                        )
                        if closed_legs > 0:
                            _defer_reconcile_checks("DATA_FAILSAFE_CLOSE_SUBMITTED")
                        day_mode = "LOCKED_RED"
                        log.error(
                            "[LiveGate] FAILSAFE_TRIGGERED reason=%s source=%s closed_legs=%s",
                            fail_reason,
                            source,
                            closed_legs,
                        )
                        _ops_alert(
                            "CRITICAL",
                            "LIVEGATE_FAILSAFE_TRIGGERED",
                            "LiveGate fail-safe triggered from connection failure.",
                            details={"reason": fail_reason, "source": source, "closed_legs": closed_legs},
                            dedupe_key="LIVEGATE_FAILSAFE_TRIGGERED",
                        )

                def _fetch_bkm_chain(target_expiry: dt_date, *, spot_hint: Optional[float] = None) -> Optional[List[dict]]:
                    expiry_iso = target_expiry.isoformat()
                    try:
                        chain_raw = dw.get_option_chain(INDEX_SECURITY_ID, INDEX_EXCHANGE_SEG, expiry_iso)
                    except Exception:
                        log.exception("[BatmanBKM] failed to fetch option chain expiry=%s", expiry_iso)
                        _trigger_bkm_conn_failure("option_chain_exception")
                        return None
                    chain_spot = float(market.spot if spot_hint is None else (spot_hint or 0.0))
                    chain_rows = _map_chain(chain_raw, target_expiry, symbol="NIFTY", spot=chain_spot)
                    if not chain_rows:
                        log.warning("[BatmanBKM] unusable option chain payload expiry=%s", expiry_iso)
                        _trigger_bkm_conn_failure("option_chain_unusable")
                        return None
                    return chain_rows

                if live_bkm_gate_enabled and live_gate:
                    spot_when = _ist_now()
                    if _is_india_market_open(spot_when):
                        spot_triggered, spot_reason = live_gate.register_spot(market.spot, when=spot_when)
                        if spot_triggered and not live_gate.should_block_entries(spot_when):
                            closed_legs = _apply_live_gate_failsafe(
                                live_gate=live_gate,
                                reason=spot_reason,
                                dw=dw,
                                bkm_strategy=bkm_strategy,
                                trade_mode=trade_mode,
                                live_order_executor=live_order_executor,
                                execution_journal=execution_journal,
                            )
                            if closed_legs > 0:
                                _defer_reconcile_checks("DATA_FAILSAFE_CLOSE_SUBMITTED")
                            day_mode = "LOCKED_RED"
                            log.error(
                                "[LiveGate] FAILSAFE_TRIGGERED reason=%s source=spot_zero spot=%s closed_legs=%s",
                                spot_reason,
                                market.spot,
                                closed_legs,
                            )
                            _ops_alert(
                                "CRITICAL",
                                "LIVEGATE_FAILSAFE_TRIGGERED",
                                "LiveGate fail-safe triggered from zero/invalid spot feed.",
                                details={"reason": spot_reason, "source": "spot_zero", "spot": market.spot, "closed_legs": closed_legs},
                                dedupe_key="LIVEGATE_FAILSAFE_TRIGGERED",
                            )

                active_lot_multiplier = int(settings.get("batman_bkm_lot_multiplier", 1))
                if live_bkm_gate_enabled and live_gate:
                    active_lot_multiplier = live_gate.get_lot_multiplier()

                if bkm_strategy is None:
                    def _parse_time(val: str, fallback: dtime) -> dtime:
                        try:
                            hh, mm = str(val).split(":")
                            return dtime(int(hh), int(mm))
                        except Exception:
                            return fallback

                    cfg = BatmanBKMConfig(
                        base_distance_points=int(settings.get("batman_bkm_base_distance", 400)),
                        inner_step_points=int(settings.get("batman_bkm_inner_step", 200)),
                        outer_step_points=int(settings.get("batman_bkm_outer_step", 800)),
                        strike_rounding=int(settings.get("batman_bkm_strike_rounding", 50)),
                        lot_size=int(settings.get("batman_v2_lot_size", settings.get("lot_size", 65))),
                        lot_multiplier=active_lot_multiplier,
                        max_credit_pct=float(settings.get("batman_bkm_max_credit_pct", 6.0)),
                        credit_step_points=int(settings.get("batman_bkm_credit_step", 100)),
                        max_widen_iterations=int(settings.get("batman_bkm_max_widen_iterations", 10)),
                        balance_tolerance=float(settings.get("batman_bkm_balance_tolerance", 5000.0)),
                        max_hedge_lots=int(settings.get("batman_bkm_max_hedge_lots", 6)),
                        tp_pct=float(settings.get("batman_bkm_tp_pct", 0.02)),
                        sl_pct=float(settings.get("batman_bkm_sl_pct", 0.025)),
                        entry_time=_parse_time(settings.get("batman_bkm_entry_time", "15:16"), dtime(15, 16)),
                        exit_time=_parse_time(settings.get("batman_bkm_exit_time", "15:10"), dtime(15, 10)),
                        payoff_range=int(settings.get("batman_bkm_payoff_range", 2500)),
                        payoff_step=int(settings.get("batman_bkm_payoff_step", 50)),
                        enable_balance=bool(settings.get("batman_bkm_enable_balance", True)),
                        estimated_margin=float(settings.get("batman_bkm_estimated_margin", 1_000_000.0)),
                    )
                    bkm_strategy = BatmanBKMStrategy(cfg)
                elif bkm_strategy.basket is None and bkm_strategy.cfg.lot_multiplier != active_lot_multiplier:
                    log.info(
                        "[LiveGate] updating Batman BKM lot multiplier %s -> %s",
                        bkm_strategy.cfg.lot_multiplier,
                        active_lot_multiplier,
                    )
                    bkm_strategy.cfg.lot_multiplier = active_lot_multiplier
                # Try to restore basket from execution journal (live) or blotter (paper)
                if (
                    bkm_strategy.basket is None
                    and trade_mode == "live"
                    and startup_resume_bkm_snapshot
                    and not (execution_recovery_guard and execution_recovery_guard.should_block_entries(now_ist))
                ):
                    restored = _build_bkm_basket_from_journal_snapshot(startup_resume_bkm_snapshot, bkm_strategy.cfg)
                    if restored:
                        bkm_strategy.basket = restored
                        log.info(
                            "[ExecRecovery] restored Batman BKM basket from execution journal expiry=%s",
                            restored.expiry.isoformat(),
                        )
                    startup_resume_bkm_snapshot = None
                if bkm_strategy.basket is None and trade_mode == "paper":
                    state = load_strategy_state().get("BATMAN_BKM", {})
                    open_expiries = []
                    for k, v in state.items():
                        try:
                            if v.get("status") == "OPEN":
                                open_expiries.append(datetime.fromisoformat(k).date())
                        except Exception:
                            continue
                    if open_expiries:
                        target_exp = max(open_expiries)
                        restored = _rebuild_bkm_basket_from_blotter(target_exp, trade_mode, bkm_strategy.cfg)
                        if restored:
                            bkm_strategy.basket = restored
                            log.info("Restored Batman BKM basket from blotter for expiry=%s", target_exp.isoformat())
                # If we already have a basket, stick to its expiry for MTM/exit updates
                roll_anchor_expiry: Optional[dt_date] = None
                if bkm_strategy.basket:
                    expiry = bkm_strategy.basket.expiry
                    roll_anchor_expiry = expiry
                else:
                    roll_anchor_expiry, roll_target_expiry = _fetch_bkm_monthly_roll_plan(dw)
                    expiry = roll_target_expiry or roll_anchor_expiry or today
                expiry_str = expiry.isoformat()
                if intraday_ai_advisor:
                    try:
                        refresh_sec = max(15.0, float(getattr(intraday_ai_advisor.config, "refresh_sec", 60.0)))
                        if (
                            intraday_ai_last_eval_at is None
                            or (now_ist - intraday_ai_last_eval_at).total_seconds() >= refresh_sec
                        ):
                            intraday_chain: List[dict] = []
                            intraday_ctx: Dict[str, Any] = {}
                            advisor_expiry = None
                            if _is_india_market_open(now_ist):
                                try:
                                    advisor_expiry = _fetch_expiry_with_settings(dw, settings)
                                    intraday_ai_last_expiry = advisor_expiry.isoformat()
                                except Exception:
                                    log.exception("[IntradayAI] expiry fetch failed")
                                    advisor_expiry = None
                                if advisor_expiry is not None:
                                    try:
                                        intraday_chain_raw = dw.get_option_chain(
                                            INDEX_SECURITY_ID,
                                            INDEX_EXCHANGE_SEG,
                                            advisor_expiry.isoformat(),
                                        )
                                        intraday_chain = _map_chain(
                                            intraday_chain_raw,
                                            advisor_expiry,
                                            symbol="NIFTY",
                                            spot=float(market.spot or 0.0),
                                        )
                                        if not intraday_chain:
                                            log.warning(
                                                "[IntradayAI] unusable option chain payload expiry=%s (advisor only; no fail-safe impact)",
                                                advisor_expiry.isoformat(),
                                            )
                                    except Exception:
                                        log.exception(
                                            "[IntradayAI] option chain fetch failed expiry=%s (advisor only; no fail-safe impact)",
                                            advisor_expiry.isoformat(),
                                        )
                                        intraday_chain = []
                                if intraday_chain:
                                    oc_ctx, next_intraday_oi = _summarize_bkm_option_chain_context(
                                        intraday_chain,
                                        spot=float(market.spot or 0.0),
                                        near_atm_band_points=float(settings.get("batman_bkm_ai_near_atm_oi_band_points", 500.0)),
                                        prev_oi_by_key=intraday_ai_prev_oi_map,
                                    )
                                    if next_intraday_oi:
                                        intraday_ai_prev_oi_map = next_intraday_oi
                                    trend_ctx = _build_bkm_mtf_trend_context(dw, trade_day=now_ist.date())
                                    try:
                                        daily_for_intraday = _fetch_daily_candles(dw, days=50)
                                    except Exception:
                                        daily_for_intraday = []
                                    try:
                                        structure_ctx = _build_bkm_structure_confidence_context(
                                            spot=float(market.spot or 0.0),
                                            trend_ctx=trend_ctx,
                                            oc_ctx=oc_ctx,
                                            daily_candles=daily_for_intraday,
                                        )
                                    except Exception:
                                        log.exception("[IntradayAI] structure context build failed")
                                        structure_ctx = {}
                                    intraday_ctx = {
                                        "computed_at": now_ist.isoformat(timespec="seconds"),
                                        "option_chain": oc_ctx,
                                        "trend": trend_ctx,
                                        "structure": structure_ctx,
                                    }
                            if not intraday_ctx:
                                # Preserve explainability when fresh data is temporarily unavailable.
                                snap = intraday_ai_advisor.snapshot()
                                prev_ctx = snap.get("market_context")
                                intraday_ctx = prev_ctx if isinstance(prev_ctx, dict) else {}
                            intraday_expiry_str = (
                                intraday_ai_last_expiry
                                or (advisor_expiry.isoformat() if advisor_expiry is not None else None)
                                or expiry_str
                            )
                            intraday_out = intraday_ai_advisor.update(
                                now=now_ist,
                                expiry=str(intraday_expiry_str),
                                spot=float(market.spot or 0.0),
                                chain_rows=intraday_chain,
                                context=intraday_ctx,
                                has_open_bkm=bool(bkm_strategy and bkm_strategy.basket),
                            )
                            intraday_ai_last_eval_at = now_ist
                            if bool(intraday_out.get("signal_changed")):
                                log.info(
                                    "[IntradayAI] signal=%s strategy=%s bias=%s trend_conf=%.2f conflict=%.1f expiry=%s",
                                    intraday_out.get("signal"),
                                    intraday_out.get("strategy"),
                                    intraday_out.get("market_bias"),
                                    float(intraday_out.get("trend_confidence") or 0.0),
                                    float(intraday_out.get("signal_conflict_score") or 0.0),
                                    intraday_out.get("expiry"),
                                )
                                try:
                                    _telegram_intraday_signal(now=now_ist, signal_payload=intraday_out)
                                except Exception:
                                    log.exception("[TelegramIntradaySignal] send failed")
                    except Exception:
                        log.exception("[IntradayAI] advisor refresh failed")
                if live_bkm_reconcile_enabled and position_reconciler:
                    try:
                        broker_positions_raw = dw.get_positions_raw()
                        broker_legs_all = _map_positions(broker_positions_raw)
                    except Exception:
                        log.exception("[Reconcile] broker positions fetch failed")
                        _trigger_bkm_conn_failure("positions_fetch_exception")
                        time.sleep(poll_sec)
                        continue
                    expected_bkm_legs: List[OptionLeg] = []
                    if bkm_strategy and bkm_strategy.basket:
                        for raw_leg in bkm_strategy.basket.legs:
                            try:
                                expected_bkm_legs.append(_bkm_leg_to_option_leg(raw_leg))
                            except Exception:
                                continue
                        broker_legs = [leg for leg in broker_legs_all if getattr(leg, "expiry", None) == expiry]
                    else:
                        broker_legs = broker_legs_all
                    reconcile_out = position_reconciler.evaluate(
                        expected_legs=expected_bkm_legs,
                        broker_legs=broker_legs,
                        when=now_ist,
                    )
                    if reconcile_out.get("skipped"):
                        log.info("[Reconcile] skipped: %s", reconcile_out.get("reason"))
                    elif not reconcile_out.get("ok"):
                        if reconcile_out.get("locked"):
                            day_mode = "LOCKED_RED"
                            if live_bkm_gate_enabled and live_gate:
                                live_gate.mark_loop_error("POSITION_RECONCILE_MISMATCH", when=_ist_now())
                            log.error(
                                "[Reconcile] MISMATCH_LOCKED streak=%s diff=%s",
                                reconcile_out.get("mismatch_streak"),
                                reconcile_out.get("diff"),
                            )
                            _ops_alert(
                                "CRITICAL",
                                "POSITION_RECONCILE_MISMATCH_LOCKED",
                                "Broker/local position reconciliation locked trading.",
                                details={
                                    "mismatch_streak": reconcile_out.get("mismatch_streak"),
                                    "diff": reconcile_out.get("diff"),
                                },
                                dedupe_key="POSITION_RECONCILE_MISMATCH_LOCKED",
                            )
                            time.sleep(poll_sec)
                            continue
                        log.warning(
                            "[Reconcile] mismatch_detected streak=%s diff=%s",
                            reconcile_out.get("mismatch_streak"),
                            reconcile_out.get("diff"),
                        )
                entries_locked = day_mode == "LOCKED_RED" or (
                    live_bkm_gate_enabled and live_gate and live_gate.should_block_entries(now_ist)
                ) or (
                    live_bkm_reconcile_enabled and position_reconciler and position_reconciler.should_block_entries(now_ist)
                ) or (
                    trade_mode == "live" and execution_recovery_guard and execution_recovery_guard.should_block_entries(now_ist)
                )
                if bkm_ai_manager and (not bkm_strategy or bkm_strategy.basket is None):
                    ai_mode_name = str(getattr(bkm_ai_manager.config, "mode", "ADVISOR") or "ADVISOR").upper()
                    protect_mode = ai_mode_name in {"AUTO_PROTECT", "PROTECTIVE"} and bool(bkm_ai_protection_enabled)
                    if not protect_mode:
                        if bkm_ai_entry_lock_active:
                            log.info("[BatmanBKM-AI] protective entry lock cleared (mode=%s)", ai_mode_name)
                        bkm_ai_entry_lock_active = False
                        bkm_ai_entry_lock_reason = None
                        bkm_ai_last_entry_lock_state = False
                        bkm_ai_unlock_stable_since = None
                        try:
                            _set_bkm_ai_protect_lock(
                                active=False,
                                now=now_ist,
                                protection_enabled=bkm_ai_protection_enabled,
                            )
                        except Exception:
                            log.exception("[BatmanBKM-AI] failed to clear protective lock while idle")
                    elif bkm_ai_entry_lock_active:
                        # Keep session-scoped protective lock even if no basket is currently open.
                        bkm_ai_last_entry_lock_state = True
                        bkm_ai_unlock_stable_since = None
                    try:
                        bkm_ai_manager.reset_idle(when=now_ist, reason="NO_OPEN_BKM_BASKET")
                    except Exception:
                        log.exception("[BatmanBKM-AI] reset idle failed")
                ai_protect_entries_locked = bool(trade_mode == "live" and bkm_ai_entry_lock_active)
                entries_locked = entries_locked or ai_protect_entries_locked
                if bkm_strategy.basket is None and _is_bkm_blocked(expiry_str):
                    blotter_empty = (not TRADE_BLOTTER_PATH.exists()) or TRADE_BLOTTER_PATH.stat().st_size == 0
                    if trade_mode == "paper" and blotter_empty:
                        log.warning("[BatmanBKM] clearing stale state for expiry=%s (no blotter rows)", expiry_str)
                        _clear_bkm_expiry(expiry_str)
                entry_anchor_expiry = roll_anchor_expiry or expiry
                entry_day = _last_friday_before(entry_anchor_expiry)
                # In paper mode, always force entry unless explicitly overridden elsewhere.
                force_entry = True if trade_mode == "paper" else bool(settings.get("batman_bkm_force_entry", False))
                # Prevent multiple entries per expiry
                if bkm_strategy.basket is None and entries_locked:
                    lock_payload = live_gate.snapshot() if (live_bkm_gate_enabled and live_gate) else {}
                    reconcile_payload = position_reconciler.snapshot() if (live_bkm_reconcile_enabled and position_reconciler) else {}
                    exec_recovery_payload = execution_recovery_guard.snapshot() if (trade_mode == "live" and execution_recovery_guard) else {}
                    log.warning(
                        "[BatmanBKM] entry blocked day_mode=%s live_gate_status=%s live_gate_locked_for=%s reconcile_status=%s reconcile_hard_lock=%s exec_recovery_status=%s exec_recovery_hard_lock=%s ai_entry_lock=%s ai_entry_lock_reason=%s",
                        day_mode,
                        lock_payload.get("status"),
                        lock_payload.get("locked_for_date"),
                        reconcile_payload.get("status"),
                        reconcile_payload.get("hard_lock"),
                        exec_recovery_payload.get("status"),
                        exec_recovery_payload.get("hard_lock"),
                        ai_protect_entries_locked,
                        bkm_ai_entry_lock_reason,
                    )
                elif bkm_strategy.basket is None and not _is_bkm_blocked(expiry_str):
                    log.info(
                        "[BatmanBKM] loop spot=%.2f expiry=%s roll_anchor=%s force_entry=%s now=%s entry_day=%s entry_time=%s",
                        market.spot,
                        expiry_str,
                        entry_anchor_expiry.isoformat() if entry_anchor_expiry else None,
                        force_entry,
                        now_ist.isoformat(),
                        entry_day,
                        bkm_strategy.cfg.entry_time,
                    )
                    market_open_now = _is_india_market_open(now_ist)
                    scheduled_window = (
                        now_ist.date() == entry_day
                        and now_ist.time() >= bkm_strategy.cfg.entry_time
                        and market_open_now
                    )
                    catchup_window = (
                        now_ist.date() > entry_day
                        and now_ist.date() <= entry_anchor_expiry
                        and market_open_now
                    )
                    in_window = scheduled_window or catchup_window
                    if force_entry or in_window:
                        catchup_not_before = _parse_time(
                            settings.get("batman_bkm_catchup_not_before", "09:30"),
                            dtime(9, 30),
                        )
                        if force_entry:
                            log.info("[BatmanBKM] force_entry enabled (paper=%s)", trade_mode)
                        elif not market_open_now:
                            log.info("[BatmanBKM] market closed; skipping entry check")
                            time.sleep(poll_sec)
                            continue
                        elif catchup_window and now_ist.time() < catchup_not_before:
                            log.info(
                                "[BatmanBKM] catch-up cooldown active target_expiry=%s now=%s not_before=%s",
                                expiry_str,
                                now_ist.time().isoformat(timespec="seconds"),
                                catchup_not_before.isoformat(timespec="minutes"),
                            )
                            time.sleep(poll_sec)
                            continue
                        elif catchup_window:
                            log.info(
                                "[BatmanBKM] catch-up entry window active target_expiry=%s anchor_expiry=%s planned_entry_day=%s",
                                expiry_str,
                                entry_anchor_expiry.isoformat() if entry_anchor_expiry else None,
                                entry_day.isoformat(),
                            )
                        chain: Optional[List[dict]] = None
                        entry_spot = float(market.spot or 0.0)
                        planner_snapshots = 1
                        planner_summary: Dict[str, Any] = {}
                        if trade_mode == "live":
                            planner_snapshots = max(1, int(settings.get("batman_bkm_pretrade_snapshot_count", 3)))
                            planner_gap_sec = max(0.0, float(settings.get("batman_bkm_pretrade_snapshot_gap_sec", 2.0)))
                            planner_max_spot_span = max(0.0, float(settings.get("batman_bkm_pretrade_max_spot_span_points", 120.0)))
                            planner_max_credit_span = max(0.0, float(settings.get("batman_bkm_pretrade_max_credit_span_pct", 0.40)))
                            planner_require_sig = bool(settings.get("batman_bkm_pretrade_require_signature_stable", True))
                            planner_enforce_delta = bool(settings.get("batman_bkm_pretrade_enforce_delta_limit", False))
                            planner_delta_limit = abs(float(settings.get("batman_bkm_pretrade_max_net_delta_abs", 5000.0)))
                            snap_spots: List[float] = []
                            snap_credit_pcts: List[float] = []
                            snap_signatures: List[Any] = []
                            snap_net_deltas: List[float] = []
                            snap_reasons: List[str] = []
                            planner_block_reason: Optional[str] = None
                            latest_plan_chain: Optional[List[dict]] = None
                            latest_plan_spot = entry_spot
                            latest_plan_basket = None
                            for snap_idx in range(planner_snapshots):
                                if snap_idx == 0:
                                    snap_market = market
                                else:
                                    if planner_gap_sec > 0:
                                        time.sleep(planner_gap_sec)
                                    try:
                                        snap_market, _, _, _ = build_market_snapshot(dw, avg_volume_hint=avg_volume_hint)
                                    except Exception:
                                        log.exception(
                                            "[BatmanBKM] pretrade snapshot fetch failed i=%s/%s",
                                            snap_idx + 1,
                                            planner_snapshots,
                                        )
                                        _trigger_bkm_conn_failure("pretrade_market_snapshot_exception")
                                        planner_block_reason = "PRETRADE_MARKET_SNAPSHOT_FAIL"
                                        break
                                snap_spot = float(getattr(snap_market, "spot", 0.0) or 0.0)
                                if snap_spot <= 0:
                                    planner_block_reason = "PRETRADE_SPOT_INVALID"
                                    break
                                snap_chain = _fetch_bkm_chain(expiry, spot_hint=snap_spot)
                                if not snap_chain:
                                    planner_block_reason = "PRETRADE_BAD_QUOTES"
                                    break
                                probe = BatmanBKMStrategy(bkm_strategy.cfg)
                                plan_basket, plan_reason = probe.maybe_enter(snap_spot, snap_chain, expiry)
                                snap_reasons.append(plan_reason)
                                if not plan_basket or plan_reason != "ENTER":
                                    planner_block_reason = f"PRETRADE_{plan_reason}"
                                    break
                                try:
                                    plan_sig = tuple(sorted(
                                        (
                                            str(getattr(l, "option_type", "")),
                                            str(getattr(l, "side", "")),
                                            float(getattr(l, "strike", 0.0) or 0.0),
                                            int(getattr(l, "qty", 0) or 0),
                                        )
                                        for l in plan_basket.legs
                                    ))
                                except Exception:
                                    plan_sig = ()
                                delta_lookup: Dict[Tuple[str, float], float] = {}
                                for row in snap_chain:
                                    try:
                                        opt = str(row.get("option_type") or "").upper()
                                        strike = float(row.get("strike") or 0.0)
                                        delta_val = row.get("delta")
                                        if not opt or strike == 0 or delta_val in (None, ""):
                                            continue
                                        delta_lookup[(opt, strike)] = float(delta_val)
                                    except Exception:
                                        continue
                                net_delta_acc = 0.0
                                missing_delta = False
                                for l in plan_basket.legs:
                                    try:
                                        opt = str(getattr(l, "option_type", "")).upper()
                                        side = str(getattr(l, "side", "")).upper()
                                        strike = float(getattr(l, "strike", 0.0) or 0.0)
                                        qty = int(getattr(l, "qty", 0) or 0)
                                        dlt = delta_lookup.get((opt, strike))
                                        if dlt is None:
                                            missing_delta = True
                                            continue
                                        sign = 1.0 if side == "BUY" else -1.0
                                        net_delta_acc += sign * dlt * qty
                                    except Exception:
                                        missing_delta = True
                                net_delta_val = None if missing_delta else net_delta_acc

                                max_up = None
                                max_down = None
                                loss_skew_abs = None
                                try:
                                    ce_buys = [float(l.strike) for l in plan_basket.legs if str(l.option_type).upper() == "CE" and str(l.side).upper() == "BUY"]
                                    pe_buys = [float(l.strike) for l in plan_basket.legs if str(l.option_type).upper() == "PE" and str(l.side).upper() == "BUY"]
                                    if ce_buys and pe_buys:
                                        atm_guess = (min(ce_buys) + max(pe_buys)) / 2.0
                                        max_up, max_down = probe._max_losses(plan_basket.legs, atm_guess)  # type: ignore[attr-defined]
                                        loss_skew_abs = abs(abs(max_up) - abs(max_down))
                                except Exception:
                                    pass

                                snap_spots.append(snap_spot)
                                snap_credit_pcts.append(float(plan_basket.credit_pct))
                                snap_signatures.append(plan_sig)
                                if net_delta_val is not None:
                                    snap_net_deltas.append(float(net_delta_val))
                                latest_plan_chain = snap_chain
                                latest_plan_spot = snap_spot
                                latest_plan_basket = plan_basket
                                planner_summary = {
                                    "credit_pct": float(plan_basket.credit_pct),
                                    "net_credit": float(plan_basket.net_credit),
                                    "net_delta": net_delta_val,
                                    "max_up_loss": max_up,
                                    "max_down_loss": max_down,
                                    "loss_skew_abs": loss_skew_abs,
                                    "snapshots": planner_snapshots,
                                }

                            if planner_block_reason is None:
                                spot_span = (max(snap_spots) - min(snap_spots)) if snap_spots else 0.0
                                credit_span = (max(snap_credit_pcts) - min(snap_credit_pcts)) if snap_credit_pcts else 0.0
                                sig_stable = (len({sig for sig in snap_signatures}) <= 1) if snap_signatures else False
                                planner_summary.update(
                                    {
                                        "spot_span": float(spot_span),
                                        "credit_span_pct": float(credit_span),
                                        "signature_stable": bool(sig_stable),
                                        "spot_first": float(snap_spots[0]) if snap_spots else None,
                                        "spot_last": float(snap_spots[-1]) if snap_spots else None,
                                    }
                                )
                                if planner_require_sig and not sig_stable:
                                    planner_block_reason = "PRETRADE_SIGNATURE_DRIFT"
                                elif spot_span > planner_max_spot_span:
                                    planner_block_reason = "PRETRADE_SPOT_UNSTABLE"
                                elif credit_span > planner_max_credit_span:
                                    planner_block_reason = "PRETRADE_CREDIT_UNSTABLE"
                                elif (
                                    planner_enforce_delta
                                    and planner_summary.get("net_delta") is not None
                                    and abs(float(planner_summary.get("net_delta") or 0.0)) > planner_delta_limit
                                ):
                                    planner_block_reason = "PRETRADE_DELTA_SKEW"
                                log.info(
                                    "[BatmanBKM] pretrade plan expiry=%s anchor=%s snapshots=%s spot_span=%.2f credit_span_pct=%.3f sig_stable=%s net_delta=%s loss_skew_abs=%s net_credit=%.2f credit_pct=%.3f",
                                    expiry_str,
                                    entry_anchor_expiry.isoformat() if entry_anchor_expiry else None,
                                    planner_snapshots,
                                    float(planner_summary.get('spot_span') or 0.0),
                                    float(planner_summary.get('credit_span_pct') or 0.0),
                                    planner_summary.get("signature_stable"),
                                    "NA" if planner_summary.get("net_delta") is None else f"{float(planner_summary.get('net_delta')):.2f}",
                                    "NA" if planner_summary.get("loss_skew_abs") is None else f"{float(planner_summary.get('loss_skew_abs')):.2f}",
                                    float(planner_summary.get("net_credit") or 0.0),
                                    float(planner_summary.get("credit_pct") or 0.0),
                                )
                            if planner_block_reason:
                                log.warning(
                                    "[BatmanBKM] pretrade blocked reason=%s expiry=%s anchor=%s snapshots=%s spots=%s credit_pcts=%s reasons=%s",
                                    planner_block_reason,
                                    expiry_str,
                                    entry_anchor_expiry.isoformat() if entry_anchor_expiry else None,
                                    planner_snapshots,
                                    [round(v, 2) for v in snap_spots],
                                    [round(v, 3) for v in snap_credit_pcts],
                                    snap_reasons,
                                )
                                _heartbeat(
                                    phase="bkm_pretrade_blocked",
                                    extra={
                                        "day_mode": day_mode,
                                        "bkm_pretrade_block_reason": planner_block_reason,
                                        "bkm_pretrade_expiry": expiry_str,
                                    },
                                )
                                time.sleep(poll_sec)
                                continue
                            chain = latest_plan_chain
                            entry_spot = latest_plan_spot
                            if chain is None or latest_plan_basket is None:
                                time.sleep(poll_sec)
                                continue
                        else:
                            chain = _fetch_bkm_chain(expiry)
                            if not chain:
                                time.sleep(poll_sec)
                                continue
                        basket, reason = bkm_strategy.maybe_enter(entry_spot, chain, expiry)
                        log.info("[BatmanBKM] attempt reason=%s expiry=%s", reason, expiry_str)
                        if basket and reason == "ENTER":
                            if trade_mode == "live":
                                open_exec = _execute_bkm_open_live(
                                    dw=dw,
                                    basket=basket,
                                    live_order_executor=live_order_executor,
                                    execution_journal=execution_journal,
                                )
                                if not bool(open_exec.get("ok")):
                                    day_mode = "LOCKED_RED"
                                    if live_bkm_gate_enabled and live_gate:
                                        live_gate.mark_loop_error("ORDER_EXECUTION_OPEN_FAIL", when=_ist_now())
                                    if execution_recovery_guard:
                                        execution_recovery_guard.lock(
                                            "ORDER_EXECUTION_OPEN_FAIL",
                                            details={"open_exec": open_exec},
                                            when=_ist_now(),
                                        )
                                    rollback = open_exec.get("rollback") or {}
                                    rollback_closed = int(rollback.get("submitted_close_legs", 0) or 0)
                                    if rollback_closed > 0:
                                        _defer_reconcile_checks("BKM_OPEN_ROLLBACK_SUBMITTED")
                                    log.error(
                                        "[BatmanBKM] live open failed opened=%s planned=%s rollback_closed=%s errors=%s rollback_errors=%s",
                                        open_exec.get("opened_legs"),
                                        open_exec.get("planned_legs"),
                                        rollback_closed,
                                        open_exec.get("errors"),
                                        rollback.get("errors"),
                                    )
                                    _ops_alert(
                                        "CRITICAL",
                                        "ORDER_EXECUTION_OPEN_FAIL",
                                        "Batman BKM live OPEN execution failed; rollback attempted and entries locked.",
                                        details={
                                            "opened_legs": open_exec.get("opened_legs"),
                                            "planned_legs": open_exec.get("planned_legs"),
                                            "rollback_closed": rollback_closed,
                                            "errors": open_exec.get("errors"),
                                            "rollback_errors": rollback.get("errors"),
                                        },
                                        dedupe_key="ORDER_EXECUTION_OPEN_FAIL",
                                    )
                                    time.sleep(poll_sec)
                                    continue
                                _defer_reconcile_checks("BKM_OPEN_SUBMITTED")
                            _mark_bkm_open(expiry_str, {"net_credit": basket.net_credit, "credit_pct": basket.credit_pct})
                            try:
                                _log_batman_blotter(trade_mode, basket.legs, "OPEN")
                            except Exception:
                                pass
                            if live_bkm_gate_enabled and live_gate:
                                live_gate.note_live_event(mtm=basket.mtm(), when=_ist_now())
                elif bkm_strategy.basket:
                    basket_ref = bkm_strategy.basket
                    if bool(settings.get("batman_bkm_market_closed_park_enabled", True)) and not _is_india_market_open(now_ist):
                        park_log_every = max(30.0, float(settings.get("batman_bkm_park_log_interval_sec", 300.0)))
                        try:
                            park_pnl = float(basket_ref.mtm() or 0.0) if basket_ref else 0.0
                        except Exception:
                            park_pnl = 0.0
                        tp_val = (bkm_strategy.cfg.tp_pct * float(getattr(basket_ref, "margin_required", 0.0) or 0.0)) if basket_ref else 0.0
                        sl_val = -(bkm_strategy.cfg.sl_pct * float(getattr(basket_ref, "margin_required", 0.0) or 0.0)) if basket_ref else 0.0
                        if (
                            bkm_last_park_log_at is None
                            or (now_ist - bkm_last_park_log_at).total_seconds() >= park_log_every
                        ):
                            bkm_last_park_log_at = now_ist
                            log.info(
                                "[BatmanBKM] park mode market_closed expiry=%s spot=%.2f last_pnl=%.2f tp=%.2f sl=%.2f day_mode=%s",
                                basket_ref.expiry.isoformat() if basket_ref else expiry_str,
                                float(market.spot or 0.0),
                                float(park_pnl),
                                float(tp_val),
                                float(sl_val),
                                day_mode,
                            )
                            _heartbeat(
                                phase="bkm_market_closed_park",
                                extra={
                                    "day_mode": day_mode,
                                    "bkm_open_expiry": basket_ref.expiry.isoformat() if basket_ref else expiry_str,
                                    "bkm_pnl": float(park_pnl),
                                    "bkm_tp": float(tp_val),
                                    "bkm_sl": float(sl_val),
                                    "bkm_quotes_stale": True,
                                    "market_open": False,
                                    "bkm_ai_action": bkm_ai_manager.snapshot().get("action") if bkm_ai_manager else None,
                                    "bkm_ai_score": float(bkm_ai_manager.snapshot().get("score") or 0.0) if bkm_ai_manager else None,
                                    "bkm_ai_severity": bkm_ai_manager.snapshot().get("severity") if bkm_ai_manager else None,
                                    "bkm_ai_entry_lock_active": bool(bkm_ai_entry_lock_active),
                                    "bkm_ai_entry_lock_reason": bkm_ai_entry_lock_reason,
                                },
                            )
                        try:
                            _telegram_market_close_summary(
                                now=now_ist,
                                basket_expiry=basket_ref.expiry.isoformat() if basket_ref else expiry_str,
                                spot=float(market.spot or 0.0),
                                pnl=float(park_pnl),
                                tp=float(tp_val),
                                sl=float(sl_val),
                                day_mode_value=day_mode,
                                ai_eval=(bkm_ai_manager.snapshot() if bkm_ai_manager else None),
                            )
                        except Exception:
                            log.exception("[TelegramCloseSummary] send failed")
                        time.sleep(poll_sec)
                        continue
                    bkm_last_park_log_at = None
                    chain = _fetch_bkm_chain(expiry)
                    if not chain:
                        time.sleep(poll_sec)
                        continue
                    pnl = bkm_strategy.update_mtm(chain) or 0.0
                    if live_bkm_gate_enabled and live_gate:
                        live_gate.note_live_event(mtm=pnl, when=_ist_now())
                    ai_eval: Optional[Dict[str, Any]] = None
                    bkm_ai_context: Optional[Dict[str, Any]] = None
                    basket_ref = bkm_strategy.basket
                    tp_val = (bkm_strategy.cfg.tp_pct * float(getattr(basket_ref, "margin_required", 0.0) or 0.0)) if basket_ref else 0.0
                    sl_val = -(bkm_strategy.cfg.sl_pct * float(getattr(basket_ref, "margin_required", 0.0) or 0.0)) if basket_ref else 0.0
                    if bkm_ai_manager and basket_ref:
                        try:
                            ctx_refresh_sec = max(5.0, float(settings.get("batman_bkm_ai_context_refresh_sec", 60.0)))
                            need_ctx_refresh = (
                                bkm_ai_context_cache is None
                                or bkm_ai_last_context_at is None
                                or (now_ist - bkm_ai_last_context_at).total_seconds() >= ctx_refresh_sec
                            )
                            if need_ctx_refresh:
                                chain_ctx, next_oi_map = _summarize_bkm_option_chain_context(
                                    chain,
                                    spot=float(market.spot or 0.0),
                                    near_atm_band_points=float(settings.get("batman_bkm_ai_near_atm_oi_band_points", 500.0)),
                                    prev_oi_by_key=bkm_ai_prev_oi_map,
                                )
                                if next_oi_map:
                                    bkm_ai_prev_oi_map = next_oi_map
                                trend_ctx = _build_bkm_mtf_trend_context(dw, trade_day=now_ist.date())
                                structure_ctx: Dict[str, Any] = {}
                                try:
                                    daily_candles_ctx = _fetch_daily_candles(dw, days=50)
                                except Exception:
                                    daily_candles_ctx = []
                                try:
                                    structure_ctx = _build_bkm_structure_confidence_context(
                                        spot=float(market.spot or 0.0),
                                        trend_ctx=trend_ctx,
                                        oc_ctx=chain_ctx,
                                        daily_candles=daily_candles_ctx,
                                    )
                                except Exception:
                                    log.exception("[BatmanBKM-AI] structure context build failed")
                                    structure_ctx = {}
                                bkm_ai_context_cache = {
                                    "computed_at": now_ist.isoformat(timespec="seconds"),
                                    "option_chain": chain_ctx,
                                    "trend": trend_ctx,
                                    "structure": structure_ctx,
                                }
                                bkm_ai_last_context_at = now_ist
                            bkm_ai_context = dict(bkm_ai_context_cache or {})
                        except Exception:
                            log.exception("[BatmanBKM-AI] market context build failed")
                            bkm_ai_context = bkm_ai_context_cache
                    if bkm_ai_manager and basket_ref:
                        try:
                            ai_eval = bkm_ai_manager.update_open(
                                basket=basket_ref,
                                spot=float(market.spot or 0.0),
                                pnl=float(pnl or 0.0),
                                tp=float(tp_val),
                                sl=float(sl_val),
                                day_mode=day_mode,
                                context=bkm_ai_context,
                                when=now_ist,
                            )
                            ai_plan = ai_eval.get("plan") if isinstance(ai_eval.get("plan"), dict) else {}
                            ai_mode_name = str(getattr(bkm_ai_manager.config, "mode", "ADVISOR") or "ADVISOR").upper()
                            ai_protect_mode = ai_mode_name in {"AUTO_PROTECT", "PROTECTIVE"} and bool(bkm_ai_protection_enabled)
                            if not bool(bkm_ai_protection_enabled):
                                ai_plan["entry_lock_active"] = False
                                ai_plan["entry_lock_reason"] = None
                                ai_plan["entry_protection_enabled"] = False
                                ai_plan["next_course"] = (
                                    str(ai_plan.get("next_course") or "Hold and monitor. No action needed.")
                                    + " (AI entry protection disabled for today.)"
                                )
                            computed_ai_entry_lock_active = bool(ai_plan.get("entry_lock_active", False))
                            computed_ai_entry_lock_reason = ai_plan.get("entry_lock_reason") if computed_ai_entry_lock_active else None
                            prev_ai_entry_lock_active = bool(bkm_ai_entry_lock_active)
                            prev_ai_entry_lock_reason = bkm_ai_entry_lock_reason
                            ai_lock_auto_unlocked = False
                            ai_lock_auto_unlock_stable_for_sec: Optional[float] = None
                            next_ai_entry_lock_active = computed_ai_entry_lock_active if ai_protect_mode else False
                            next_ai_entry_lock_reason = computed_ai_entry_lock_reason if ai_protect_mode else None

                            # In protective mode, hold the AI entry lock until risk stays stable for a configured window.
                            if ai_protect_mode and prev_ai_entry_lock_active:
                                if computed_ai_entry_lock_active:
                                    bkm_ai_unlock_stable_since = None
                                elif bool(settings.get("batman_bkm_ai_protect_auto_unlock_enabled", True)):
                                    stable_sec_required = max(
                                        60.0,
                                        float(settings.get("batman_bkm_ai_protect_auto_unlock_stable_sec", 1200.0)),
                                    )
                                    max_unlock_score = float(settings.get("batman_bkm_ai_protect_auto_unlock_max_score", 18.0))
                                    req_action_raw = str(
                                        settings.get("batman_bkm_ai_protect_auto_unlock_require_action", "HOLD")
                                    ).strip()
                                    req_actions = {a.strip().upper() for a in req_action_raw.split(",") if a.strip()}
                                    current_action = str(ai_eval.get("action") or "HOLD").upper()
                                    current_score = float(ai_eval.get("score") or 0.0)
                                    action_ok = (not req_actions) or ("ANY" in req_actions) or (current_action in req_actions)
                                    score_ok = current_score <= max_unlock_score
                                    market_hours_ok = True
                                    if bool(settings.get("batman_bkm_ai_protect_auto_unlock_market_hours_only", True)):
                                        market_hours_ok = _is_india_market_open(now_ist)
                                    clean_system_ok = True
                                    if bool(settings.get("batman_bkm_ai_protect_auto_unlock_require_clean_system", True)):
                                        clean_system_ok = (
                                            day_mode != "LOCKED_RED"
                                            and not (live_bkm_gate_enabled and live_gate and live_gate.should_block_entries(now_ist))
                                            and not (live_bkm_reconcile_enabled and position_reconciler and position_reconciler.should_block_entries(now_ist))
                                            and not (trade_mode == "live" and execution_recovery_guard and execution_recovery_guard.should_block_entries(now_ist))
                                        )
                                    unlock_candidate = (
                                        (not computed_ai_entry_lock_active)
                                        and action_ok
                                        and score_ok
                                        and market_hours_ok
                                        and clean_system_ok
                                    )
                                    if unlock_candidate:
                                        if bkm_ai_unlock_stable_since is None:
                                            bkm_ai_unlock_stable_since = now_ist
                                        stable_elapsed = max(
                                            0.0,
                                            (now_ist - bkm_ai_unlock_stable_since).total_seconds(),
                                        )
                                        ai_lock_auto_unlock_stable_for_sec = stable_elapsed
                                        if stable_elapsed < stable_sec_required:
                                            next_ai_entry_lock_active = True
                                            next_ai_entry_lock_reason = (
                                                prev_ai_entry_lock_reason or "AI_PROTECTIVE_ENTRY_LOCK"
                                            )
                                            ai_plan["entry_lock_active"] = True
                                            ai_plan["entry_lock_reason"] = next_ai_entry_lock_reason
                                            ai_plan["entry_lock_auto_unlock_pending"] = True
                                            ai_plan["entry_lock_auto_unlock_remaining_sec"] = round(
                                                max(0.0, stable_sec_required - stable_elapsed),
                                                1,
                                            )
                                        else:
                                            next_ai_entry_lock_active = False
                                            next_ai_entry_lock_reason = None
                                            ai_lock_auto_unlocked = True
                                            bkm_ai_unlock_stable_since = None
                                            ai_plan["entry_lock_active"] = False
                                            ai_plan["entry_lock_reason"] = None
                                            ai_plan["entry_lock_auto_unlocked"] = True
                                            ai_plan["entry_lock_auto_unlock_stable_for_sec"] = round(stable_elapsed, 1)
                                    else:
                                        bkm_ai_unlock_stable_since = None
                                        next_ai_entry_lock_active = True
                                        next_ai_entry_lock_reason = (
                                            prev_ai_entry_lock_reason or "AI_PROTECTIVE_ENTRY_LOCK"
                                        )
                                        ai_plan["entry_lock_active"] = True
                                        ai_plan["entry_lock_reason"] = next_ai_entry_lock_reason
                                        ai_plan["entry_lock_auto_unlock_pending"] = False
                                else:
                                    bkm_ai_unlock_stable_since = None
                            elif not next_ai_entry_lock_active:
                                bkm_ai_unlock_stable_since = None

                            bkm_ai_entry_lock_active = next_ai_entry_lock_active
                            bkm_ai_entry_lock_reason = next_ai_entry_lock_reason
                            if bkm_ai_last_entry_lock_state is None:
                                bkm_ai_last_entry_lock_state = bkm_ai_entry_lock_active
                            elif bkm_ai_last_entry_lock_state != bkm_ai_entry_lock_active:
                                bkm_ai_last_entry_lock_state = bkm_ai_entry_lock_active
                                try:
                                    _set_bkm_ai_protect_lock(
                                        active=bkm_ai_entry_lock_active,
                                        now=now_ist,
                                        reason=bkm_ai_entry_lock_reason,
                                        source_action=str(ai_eval.get("action") or "HOLD"),
                                        score=float(ai_eval.get("score") or 0.0),
                                        expiry=(basket_ref.expiry.isoformat() if basket_ref else expiry_str),
                                        protection_enabled=bkm_ai_protection_enabled,
                                    )
                                except Exception:
                                    log.exception("[BatmanBKM-AI] failed to persist protective entry lock state")
                                if trade_mode == "live":
                                    if bkm_ai_entry_lock_active:
                                        _ops_alert(
                                            "WARN",
                                            "BKM_AI_PROTECT_ENTRY_LOCK_ON",
                                            "AI protective mode blocked new entries; monitoring current trade.",
                                            details={
                                                "action": ai_eval.get("action"),
                                                "reason": bkm_ai_entry_lock_reason,
                                                "score": ai_eval.get("score"),
                                                "expiry": basket_ref.expiry.isoformat() if basket_ref else expiry_str,
                                            },
                                            dedupe_key="BKM_AI_PROTECT_ENTRY_LOCK_ON",
                                        )
                                        try:
                                            _telegram_ai_lock_change(
                                                now=now_ist,
                                                active=True,
                                                source="ai_protect",
                                                ai_eval=ai_eval,
                                                basket_expiry=(basket_ref.expiry.isoformat() if basket_ref else expiry_str),
                                            )
                                        except Exception:
                                            log.exception("[TelegramAILock] lock-on send failed")
                                    else:
                                        log.info("[BatmanBKM-AI] protective entry lock cleared")
                                        if ai_lock_auto_unlocked:
                                            _ops_alert(
                                                "INFO",
                                                "BKM_AI_PROTECT_ENTRY_LOCK_AUTO_UNLOCKED",
                                                "AI protective entry lock auto-cleared after stable conditions.",
                                                details={
                                                    "action": ai_eval.get("action"),
                                                    "score": ai_eval.get("score"),
                                                    "stable_for_sec": ai_lock_auto_unlock_stable_for_sec,
                                                    "expiry": basket_ref.expiry.isoformat() if basket_ref else expiry_str,
                                                },
                                                dedupe_key="BKM_AI_PROTECT_ENTRY_LOCK_AUTO_UNLOCKED",
                                            )
                                            try:
                                                _telegram_ai_lock_change(
                                                    now=now_ist,
                                                    active=False,
                                                    source="auto_unlock",
                                                    ai_eval=ai_eval,
                                                    basket_expiry=(basket_ref.expiry.isoformat() if basket_ref else expiry_str),
                                                    stable_for_sec=ai_lock_auto_unlock_stable_for_sec,
                                                )
                                            except Exception:
                                                log.exception("[TelegramAILock] auto-unlock send failed")
                            elif (
                                bkm_ai_entry_lock_active
                                and bool(bkm_ai_protection_enabled)
                                and str(getattr(bkm_ai_manager.config, "mode", "ADVISOR")).upper() in {"AUTO_PROTECT", "PROTECTIVE"}
                            ):
                                # Refresh persisted metadata while lock remains active (low-frequency writes avoided by monitor cadence).
                                try:
                                    _set_bkm_ai_protect_lock(
                                        active=True,
                                        now=now_ist,
                                        reason=bkm_ai_entry_lock_reason,
                                        source_action=str(ai_eval.get("action") or "HOLD"),
                                        score=float(ai_eval.get("score") or 0.0),
                                        expiry=(basket_ref.expiry.isoformat() if basket_ref else expiry_str),
                                        protection_enabled=bkm_ai_protection_enabled,
                                    )
                                except Exception:
                                    pass
                            if ai_eval.get("action_changed"):
                                alert_sev = str(ai_eval.get("alert_severity_override") or ai_eval.get("severity") or "INFO").upper()
                                log_fn = log.warning if alert_sev in {"WARN", "CRITICAL"} else log.info
                                log_fn(
                                    "[BatmanBKM-AI] action=%s severity=%s score=%.1f confidence=%.2f reasons=%s spot=%.2f pnl=%.2f",
                                    ai_eval.get("action"),
                                    ai_eval.get("severity"),
                                    float(ai_eval.get("score") or 0.0),
                                    float(ai_eval.get("confidence") or 0.0),
                                    ",".join(list(ai_eval.get("reasons") or [])),
                                    float(market.spot or 0.0),
                                    float(pnl or 0.0),
                                )
                                sev = alert_sev
                                if trade_mode == "live" and sev in {"WARN", "CRITICAL"}:
                                    _ops_alert(
                                        sev,
                                        f"BKM_AI_{str(ai_eval.get('action') or 'HOLD')}",
                                        "Batman BKM AI manager recommendation changed.",
                                        details={
                                            "action": ai_eval.get("action"),
                                            "recommended_action": (ai_plan.get("recommended_action") if isinstance(ai_plan, dict) else None),
                                            "entry_lock_active": (ai_plan.get("entry_lock_active") if isinstance(ai_plan, dict) else None),
                                            "next_course": (ai_plan.get("next_course") if isinstance(ai_plan, dict) else None),
                                            "score": ai_eval.get("score"),
                                            "confidence": ai_eval.get("confidence"),
                                            "reasons": ai_eval.get("reasons"),
                                            "spot": float(market.spot or 0.0),
                                            "pnl": float(pnl or 0.0),
                                            "expiry": basket_ref.expiry.isoformat() if basket_ref else expiry_str,
                                        },
                                        dedupe_key=f"BKM_AI_{str(ai_eval.get('action') or 'HOLD')}",
                                    )
                        except Exception:
                            log.exception("[BatmanBKM-AI] evaluation failed")
                    try:
                        _telegram_trade_summary(
                            now=now_ist,
                            basket_expiry=basket_ref.expiry.isoformat() if basket_ref else expiry_str,
                            spot=float(market.spot or 0.0),
                            pnl=float(pnl or 0.0),
                            tp=float(tp_val),
                            sl=float(sl_val),
                            day_mode_value=day_mode,
                            ai_eval=ai_eval,
                        )
                    except Exception:
                        log.exception("[TelegramSummary] periodic trade summary failed")
                    monitor_every = max(5.0, float(settings.get("batman_bkm_monitor_log_interval_sec", 20.0)))
                    if (
                        bkm_last_monitor_log_at is None
                        or (now_ist - bkm_last_monitor_log_at).total_seconds() >= monitor_every
                    ):
                        bkm_last_monitor_log_at = now_ist
                        log.info(
                            "[BatmanBKM] monitor expiry=%s spot=%.2f pnl=%.2f tp=%.2f sl=%.2f net_credit=%.2f credit_pct=%.3f day_mode=%s",
                            basket_ref.expiry.isoformat() if basket_ref else expiry_str,
                            float(market.spot or 0.0),
                            float(pnl),
                            float(tp_val),
                            float(sl_val),
                            float(getattr(basket_ref, "net_credit", 0.0) or 0.0) if basket_ref else 0.0,
                            float(getattr(basket_ref, "credit_pct", 0.0) or 0.0) if basket_ref else 0.0,
                            day_mode,
                        )
                        _heartbeat(
                            phase="bkm_monitor",
                            extra={
                                "day_mode": day_mode,
                                "bkm_open_expiry": basket_ref.expiry.isoformat() if basket_ref else expiry_str,
                                "bkm_pnl": float(pnl),
                                "bkm_tp": float(tp_val),
                                "bkm_sl": float(sl_val),
                                "bkm_ai_action": ai_eval.get("action") if ai_eval else (bkm_ai_manager.snapshot().get("action") if bkm_ai_manager else None),
                                "bkm_ai_score": float(ai_eval.get("score") or 0.0) if ai_eval else (float(bkm_ai_manager.snapshot().get("score") or 0.0) if bkm_ai_manager else None),
                                "bkm_ai_severity": ai_eval.get("severity") if ai_eval else (bkm_ai_manager.snapshot().get("severity") if bkm_ai_manager else None),
                                "bkm_ai_entry_lock_active": bool(bkm_ai_entry_lock_active),
                                "bkm_ai_entry_lock_reason": bkm_ai_entry_lock_reason,
                                "bkm_ai_protection_enabled": bool(bkm_ai_protection_enabled),
                            },
                        )
                    try:
                        _log_batman_blotter(trade_mode, bkm_strategy.basket.legs, "MTM")
                    except Exception:
                        pass
                    if day_mode != "LOCKED_RED" and pnl <= risk.daily_max_loss:
                        day_mode = "LOCKED_RED"
                        if live_bkm_gate_enabled and live_gate:
                            live_gate.mark_daily_lock("DAILY_LOCK_RED", when=_ist_now())
                        flatten_res = _flatten_bkm_basket(
                            dw=dw,
                            bkm_strategy=bkm_strategy,
                            trade_mode=trade_mode,
                            reason="DAILY_LOCK_RED",
                            live_order_executor=live_order_executor if trade_mode == "live" else None,
                            execution_journal=execution_journal if trade_mode == "live" else None,
                        )
                        closed_legs = int(flatten_res.get("closed_legs", 0))
                        if trade_mode == "live" and not bool(flatten_res.get("ok")):
                            if live_bkm_gate_enabled and live_gate:
                                live_gate.mark_loop_error("ORDER_EXECUTION_CLOSE_FAIL", when=_ist_now())
                            if execution_recovery_guard:
                                execution_recovery_guard.lock(
                                    "ORDER_EXECUTION_CLOSE_FAIL",
                                    details={"flatten_res": flatten_res, "reason": "DAILY_LOCK_RED"},
                                    when=_ist_now(),
                                )
                            log.error("[BatmanBKM] daily-lock close failed: %s", flatten_res)
                            _ops_alert(
                                "CRITICAL",
                                "ORDER_EXECUTION_CLOSE_FAIL",
                                "Batman BKM daily-lock close execution failed; entries locked.",
                                details={"reason": "DAILY_LOCK_RED", "flatten_res": flatten_res},
                                dedupe_key="ORDER_EXECUTION_CLOSE_FAIL",
                            )
                        if closed_legs > 0:
                            _defer_reconcile_checks("BKM_CLOSE_SUBMITTED_DAILY_LOCK")
                        log.warning(
                            "[BatmanBKM] daily loss cap triggered pnl=%.2f cap=%.2f closed_legs=%s",
                            pnl,
                            risk.daily_max_loss,
                            closed_legs,
                        )
                        time.sleep(poll_sec)
                        continue
                    decision = bkm_strategy.maybe_exit(pnl, _ist_now())
                    if decision:
                        flatten_res = _flatten_bkm_basket(
                            dw=dw,
                            bkm_strategy=bkm_strategy,
                            trade_mode=trade_mode,
                            reason=decision,
                            live_order_executor=live_order_executor if trade_mode == "live" else None,
                            execution_journal=execution_journal if trade_mode == "live" else None,
                        )
                        closed_legs = int(flatten_res.get("closed_legs", 0))
                        if trade_mode == "live" and not bool(flatten_res.get("ok")):
                            day_mode = "LOCKED_RED"
                            if live_bkm_gate_enabled and live_gate:
                                live_gate.mark_loop_error("ORDER_EXECUTION_CLOSE_FAIL", when=_ist_now())
                            if execution_recovery_guard:
                                execution_recovery_guard.lock(
                                    "ORDER_EXECUTION_CLOSE_FAIL",
                                    details={"flatten_res": flatten_res, "reason": decision},
                                    when=_ist_now(),
                                )
                            if closed_legs > 0:
                                _defer_reconcile_checks("BKM_CLOSE_PARTIAL_SUBMITTED_EXIT")
                            log.error("[BatmanBKM] exit close failed; locking day: %s", flatten_res)
                            _ops_alert(
                                "CRITICAL",
                                "ORDER_EXECUTION_CLOSE_FAIL",
                                "Batman BKM exit close execution failed; entries locked.",
                                details={"reason": decision, "flatten_res": flatten_res},
                                dedupe_key="ORDER_EXECUTION_CLOSE_FAIL",
                            )
                            time.sleep(poll_sec)
                            continue
                        if closed_legs > 0:
                            _defer_reconcile_checks("BKM_CLOSE_SUBMITTED_EXIT")
                        log.info("Batman BKM exited: %s pnl=%.2f closed_legs=%s", decision, pnl, closed_legs)
                time.sleep(poll_sec)
                continue
            if selected_strategy_file == "batman_v2_paper":
                if batman_v2 is None:
                    bat_cfg = BatmanConfig(
                        target_delta=float(settings.get("batman_v2_target_delta", 0.22)),
                        # Widen delta band in paper to ensure a pick
                        delta_band=(0.05, 0.95),
                        # Relax gates in paper mode
                        net_credit_required=False,
                        # Allow entry through end of session in paper mode
                        max_entry_time="23:59",
                        # Disable VIX gate in paper mode to allow entry
                        min_vix=0.0,
                        max_gap_pct=10.0,
                        monthly_center_band=(
                            0.0,
                            1.0,
                        ),
                        lots=int(settings.get("batman_v2_lots", 1)),
                        lot_size=int(settings.get("batman_v2_lot_size", settings.get("lot_size", 75))),
                        capital=float(settings.get("batman_v2_capital", 500000.0)),
                        tp_pct_deployed=float(settings.get("batman_v2_tp_pct_deployed", 0.025)),
                        sl_pct_deployed=float(settings.get("batman_v2_sl_pct_deployed", 0.02)),
                        time_exit_days=int(settings.get("batman_v2_time_exit_days", 2)),
                        # Allow repeated paper entries so UI blotter reflects trades
                        one_trade_per_expiry=False if trade_mode == "paper" else bool(settings.get("batman_v2_one_trade_per_expiry", True)),
                    )
                    batman_v2 = BatmanStrategy(bat_cfg)
                    log.info("Batman V2 (paper) VIX gate disabled and entry window open till 23:59")
                # Build a minimal feature set
                daily_candles = _fetch_daily_candles(dw, days=40)
                spot = float(market.spot or 0.0)
                if spot <= 0:
                    log.info("Batman V2 (paper) entry blocked: spot unavailable")
                    time.sleep(poll_sec)
                    continue
                feats = _compute_monthly_filters(daily_candles, spot)
                vix_val = market.india_vix if market.india_vix not in (None, 0.0) else None
                gap_pct = feats.get("gap_pct", 0.0)
                monthly_pos = feats.get("monthly_range_frac")
                try:
                    if monthly_pos is not None:
                        monthly_pos = max(0.0, min(1.0, float(monthly_pos)))
                except Exception:
                    monthly_pos = None
                log.info("[BatmanV2] spot=%.2f vix=%s monthly_pos=%s", spot, vix_val, monthly_pos)
                expiry = _fetch_monthly_expiry(dw) or _fetch_expiry_with_settings(dw, settings)
                chain_raw = dw.get_option_chain(INDEX_SECURITY_ID, INDEX_EXCHANGE_SEG, expiry.isoformat())
                chain = _map_chain(chain_raw, expiry, symbol="NIFTY", spot=spot)
                # Entry if none
                if batman_v2.state is None or not batman_v2.state.is_active():
                    entered = None
                    expiry_str = expiry.isoformat()
                    # Clear prior state to allow immediate redeploy in paper
                    if trade_mode == "paper":
                        _clear_batman_expiry(expiry_str)
                    vix_gate_enabled = batman_v2.cfg.min_vix is not None and batman_v2.cfg.min_vix > 0
                    if vix_gate_enabled and vix_val is None:
                        log.info("Batman V2 (paper) entry blocked: VIX_MISSING (min_vix=%.2f)", batman_v2.cfg.min_vix)
                    else:
                        entered = batman_v2.maybe_enter(
                            as_of=now,
                            spot=spot,
                            expiry=expiry,
                            vix=vix_val,
                            gap_pct=gap_pct,
                            monthly_range_pos=monthly_pos,
                            chain=chain,
                        )
                    if entered:
                        if batman_v2.cfg.one_trade_per_expiry:
                            batman_v2.entered_expiries.add(expiry)
                        _mark_batman_open(expiry.isoformat())
                        try:
                            _log_batman_blotter(trade_mode, batman_v2.state.all_legs(), "OPEN")  # type: ignore[arg-type]
                        except Exception:
                            pass
                        log.info("Batman V2 (paper) entered: expiry=%s", expiry)
                else:
                    # update tick
                    decision = batman_v2.on_tick(
                        as_of=now,
                        spot=spot,
                        vix=vix_val,
                        adx=feats.get("adx", 0.0),
                        chain=chain,
                    )
                    if trade_mode == "paper" and batman_v2.state and batman_v2.state.is_active():
                        try:
                            _log_batman_blotter(trade_mode, batman_v2.state.all_legs(), "MTM", use_last=True)  # type: ignore[arg-type]
                        except Exception:
                            pass
                    if batman_v2.state and not batman_v2.state.is_active():
                        try:
                            _log_batman_blotter(trade_mode, batman_v2.state.all_legs(), "CLOSE")  # type: ignore[arg-type]
                        except Exception:
                            pass
                        _mark_batman_closed(expiry.isoformat(), decision or "EXIT")
                        log.info("Batman V2 (paper) exited.")
                time.sleep(poll_sec)
                continue
            market_open = datetime.combine(market.now.date(), dtime(9, 15))
            if orb_config.enabled and orb_state.orb_levels is None:
                levels = compute_orb_levels(market.candles_5m, market_open, orb_config)
                if levels:
                    orb_state.orb_levels = levels
            if orb_state.orb_levels:
                breakout = detect_orb_breakout(market, orb_state.orb_levels, orb_config)
                orb_state.breakout_signal = breakout if breakout.active else None
            sr_levels = compute_sr_levels(market, orb_state.orb_levels)
            expiry = _fetch_expiry_with_settings(dw, settings)
            if trade_mode == "paper":
                positions_df = _paper_positions_from_blotter()
                legs: List[OptionLeg] = []
            else:
                positions_raw = dw.get_positions_raw()
                legs = _map_positions(positions_raw)
                positions_df = None  # live positions handled via legs
            if legs and trade_mode != "paper":
                log.info(
                    "[Positions detail] %s",
                    [
                        (
                            leg.side.value if hasattr(leg.side, "value") else str(leg.side),
                            leg.option_type.value if hasattr(leg.option_type, "value") else str(leg.option_type),
                            leg.strike,
                            leg.expiry,
                            leg.quantity,
                        )
                        for leg in legs
                    ],
                )
            expiry_str = expiry.isoformat() if hasattr(expiry, "isoformat") else str(expiry)
            log.info("[OC request] underlying=%s expiry=%s trade_mode=%s", INDEX_SECURITY_ID, expiry_str, trade_mode)
            chain_raw = dw.get_option_chain(INDEX_SECURITY_ID, INDEX_EXCHANGE_SEG, expiry_str)
            chain = _map_chain(chain_raw, expiry, symbol="NIFTY", spot=market.spot)
            _update_leg_ltps_from_chain(legs, chain)
            basket_mtm = _compute_mtm(legs)
            day_pnl = basket_mtm  # single-basket approximation for open PnL; realized adds into MTM over time
            if not legs:
                basket_peak_mtm = 0.0
                trail_active = False
                trail_floor = -1000.0
            # Daily hard stop: lock the day if PnL breaches daily_max_loss
            forced_decision: Optional[TradeAction] = None
            if day_mode != "LOCKED_RED" and day_pnl <= risk.daily_max_loss:
                forced_decision = TradeAction(
                    action_type="CLOSE_ALL",
                    strategy_type=StrategyType.NONE,
                    legs_to_open=[],
                    legs_to_close=legs,
                    reason="DAILY_LOCK_RED",
                )
                day_mode = "LOCKED_RED"
            if day_mode != "LOCKED_RED" and basket_mtm <= risk.daily_max_loss:
                day_mode = "LOCKED_RED"
            elif day_mode == "NORMAL" and basket_mtm >= risk.daily_target:
                day_mode = "LOCKED_GREEN"
            if legs:
                if basket_mtm > basket_peak_mtm:
                    basket_peak_mtm = basket_mtm
                # Hard per-basket SL always enforced
                if basket_mtm <= -1000.0 and forced_decision is None:
                    forced_decision = TradeAction(
                        action_type="CLOSE_ALL",
                        strategy_type=StrategyType.NONE,
                        legs_to_open=[],
                        legs_to_close=legs,
                        reason="BASKET_SL_HIT",
                    )
                if not trail_active and forced_decision is None:
                    if basket_mtm >= 2500.0:
                        trail_active = True
                        trail_floor = max(1500.0, basket_mtm - 1000.0)
                elif trail_active and forced_decision is None:
                    desired_floor = max(1500.0, basket_peak_mtm * 0.6)
                    if desired_floor > trail_floor:
                        trail_floor = desired_floor
                    if basket_mtm <= trail_floor:
                        forced_decision = TradeAction(
                            action_type="CLOSE_ALL",
                            strategy_type=StrategyType.NONE,
                            legs_to_open=[],
                            legs_to_close=legs,
                            reason="BASKET_TRAIL_HIT",
                        )

            # Branch: Monthly Strangle w/ Hedge uses rule-based manager
            if selected_strategy_file == "monthly_strangle_with_weekly_hedge.py":
                cfg = MonthlyStrangleConfig()
                filters_cfg = _build_monthly_filters_config(settings)
                daily_candles = _fetch_daily_candles(dw, days=40)
                daily_feats = _compute_monthly_filters(daily_candles, market.spot)
                daily_count = len(daily_candles)
                adx_val = daily_feats["adx"] if daily_count >= 15 else None
                max_body_pct = daily_feats["max_body_pct"] if daily_count >= 3 else None
                gap_pct_daily = daily_feats["gap_pct"] if daily_count >= 2 else None
                monthly_range_frac = daily_feats["monthly_range_frac"] if daily_count >= 1 else None
                vix_val = market.india_vix if market.india_vix not in (None, 0.0) else None

                # Persist entry criteria status for the UI
                try:
                    expiry_date = expiry if isinstance(expiry, dt_date) else expiry.date()
                except Exception:
                    expiry_date = None
                entry_cfg = cfg.entry
                entry_cfg.cycle_day_min = int(settings.get("cycle_day_min", entry_cfg.cycle_day_min))
                entry_cfg.cycle_day_max = int(settings.get("cycle_day_max", entry_cfg.cycle_day_max))
                entry_cfg.allow_early_next_cycle = bool(settings.get("allow_early_next_cycle", entry_cfg.allow_early_next_cycle))
                enc_days = settings.get("early_next_cycle_days", entry_cfg.early_next_cycle_days)
                try:
                    entry_cfg.early_next_cycle_days = (int(enc_days[0]), int(enc_days[1]))
                except Exception:
                    entry_cfg.early_next_cycle_days = entry_cfg.early_next_cycle_days
                # Sync thresholds from settings-driven filters
                entry_cfg.adx_max = float(filters_cfg.adx_max)
                entry_cfg.max_gap_pct = float(filters_cfg.max_gap_pct)
                entry_cfg.monthly_range_band = (float(filters_cfg.range_band_min), float(filters_cfg.range_band_max))
                entry_cfg.vix_min = float(filters_cfg.min_vix)
                use_adx = bool(filters_cfg.use_adx)
                use_range = bool(filters_cfg.use_range_band)
                use_vix = bool(filters_cfg.use_vix)
                use_gap = bool(filters_cfg.use_gap)

                entry_window_ok = _in_entry_window(now, entry_cfg, trade_mode)
                cycle_day_ok = _within_cycle_day(now.date(), expiry_date, entry_cfg) if expiry_date else False
                body_ok = max_body_pct is not None and max_body_pct <= entry_cfg.max_body_pct
                gap_ok = True if not use_gap else (gap_pct_daily is not None and abs(gap_pct_daily) <= filters_cfg.max_gap_pct)
                range_ok = True if not use_range else (
                    monthly_range_frac is not None
                    and filters_cfg.range_band_min <= monthly_range_frac <= filters_cfg.range_band_max
                )
                vix_ok = True if not use_vix else (vix_val is not None and vix_val >= filters_cfg.min_vix)
                adx_ok = True if not use_adx else (adx_val is not None and adx_val < filters_cfg.adx_max)
                dte = (expiry_date - now.date()).days if expiry_date else None
                status_payload = {
                    "timestamp": now.isoformat(timespec="seconds"),
                    "mode": trade_mode,
                    "criteria": {
                        "entry_window": entry_window_ok,
                        "cycle_day": cycle_day_ok,
                        "adx_ok": adx_ok,
                        "body_ok": body_ok,
                        "gap_ok": gap_ok,
                        "range_ok": range_ok,
                        "vix_ok": vix_ok,
                    },
                    "values": {
                        "adx": adx_val,
                        "max_body_pct": max_body_pct,
                        "gap_pct": gap_pct_daily,
                        "monthly_range_frac": monthly_range_frac,
                        "vix": vix_val,
                        "expiry": expiry_date.isoformat() if expiry_date else None,
                        "now_time": now.strftime("%H:%M:%S"),
                        "day_of_month": now.day,
                        "dte": dte,
                        "candles_daily_count": daily_count,
                    },
                    "thresholds": {
                        "use_adx": use_adx,
                        "use_range": use_range,
                        "use_vix": use_vix,
                        "use_gap": use_gap,
                        "adx_max": filters_cfg.adx_max,
                        "adx_length": filters_cfg.adx_length,
                        "gap_max_pct": filters_cfg.max_gap_pct,
                        "range_band": [filters_cfg.range_band_min, filters_cfg.range_band_max],
                        "vix_min": filters_cfg.min_vix,
                        "window_start": entry_cfg.entry_window_start.strftime("%H:%M"),
                        "window_end": entry_cfg.entry_window_end.strftime("%H:%M"),
                        "cycle_day_min": entry_cfg.cycle_day_min,
                        "cycle_day_max": entry_cfg.cycle_day_max,
                        "early_next_cycle_days": list(entry_cfg.early_next_cycle_days),
                    },
                }
                _write_entry_status(status_payload)
                if legs:
                    margin_est = sum(
                        abs((leg.entry_price or leg.current_ltp or 0.0) * leg.quantity)
                        for leg in legs
                        if leg.side == LegSide.SELL
                    ) or 1.0
                    # Map leg deltas if available from chain
                    delta_map = {}
                    for row in chain:
                        key = (row.get("strike"), row.get("option_type"))
                        delta_map[key] = row.get("delta")
                    leg_deltas = []
                    for leg in legs:
                        key = (leg.strike, "CE" if leg.option_type == OptionType.CALL else "PE")
                        leg_deltas.append(delta_map.get(key, 0.0))
                    # Replace None with 0.0 to avoid abs(None) crashes
                    leg_deltas = [d if d is not None else 0.0 for d in leg_deltas]
                    decision = manage_monthly_basket(
                        now=now,
                        expiry=expiry,
                        margin_deployed=margin_est,
                        basket_mtm=basket_mtm,
                        legs=legs,
                        attempt_count=0,
                        cfg=cfg,
                        leg_deltas=leg_deltas,
                        adx=adx_val,
                    )
                else:
                    # Strategy-level gate: only one active basket per expiry.
                    if is_monthly_strangle_open(expiry_str):
                        log.info(
                            "[MonthlyStrangle] HOLD: basket already OPEN for expiry %s",
                            expiry_str,
                        )
                        decision = TradeAction(
                            action_type="HOLD",
                            strategy_type=StrategyType.STRANGLE,
                            legs_to_open=[],
                            legs_to_close=[],
                            reason="MONTHLY_BASKET_ALREADY_OPEN",
                        )
                    else:
                        lot_qty = int(settings.get("lot_size", 75))
                        short_qty = int(lot_qty * float(settings.get("short_lots", 1)))
                        hedge_mult = float(settings.get("hedge_lots_live" if trade_mode == "live" else "hedge_lots_paper", 1.0))
                        hedge_qty = int(lot_qty * hedge_mult)
                        decision = propose_monthly_entry(
                            now=now,
                            spot=market.spot,
                            adx=adx_val,
                            max_body_pct=max_body_pct,
                            gap_pct=gap_pct_daily,
                            monthly_range_frac=monthly_range_frac,
                            vix=market.india_vix,
                            vix_rising=True,
                            option_chain=chain,
                            expiry=expiry,
                            short_qty=short_qty,
                            hedge_qty=hedge_qty,
                            cfg=cfg,
                            trade_mode=trade_mode,
                        )
                # Paper-mode margin gate: assume virtual capital 1,000,000
                if decision.action_type == "OPEN" and trade_mode == "paper" and decision.legs_to_open:
                    required_margin = 0.0
                    for leg in decision.legs_to_open:
                        if leg.side == LegSide.SELL:
                            px = (leg.entry_price or leg.current_ltp or 0.0)
                            required_margin += abs(px * leg.quantity)
                    if required_margin > 1_000_000:
                        log.info("Paper: skipping entry due to virtual margin check (need %.0f, cap 1,000,000)", required_margin)
                        decision = TradeAction("HOLD", decision.strategy_type, [], [], "Paper margin insufficient")
                _log_strategy_event(
                    strategy=decision.strategy_type,
                    action=decision.action_type,
                    reason=decision.reason,
                    market=market,
                    legs=legs,
                    trade_mode=trade_mode,
                    basket_mtm=basket_mtm,
                    basket_peak=basket_peak_mtm,
                    trail_floor=trail_floor,
                )
                if decision.action_type != "HOLD" and decision.legs_to_open:
                    for leg in decision.legs_to_open:
                        _log_trade_event(trade_mode=trade_mode, side=leg.side.value, leg=leg, order_type="MARKET")
                if decision.action_type in ("CLOSE_ALL", "CLOSE_LEGS") and decision.legs_to_close:
                    for leg in decision.legs_to_close:
                        _log_trade_event(trade_mode=trade_mode, side=leg.side.value, leg=leg, order_type="MARKET", notes="close")
                if decision.strategy_type == StrategyType.STRANGLE and decision.action_type == "OPEN" and decision.legs_to_open:
                    if not is_monthly_strangle_open(expiry_str):
                        basket_id = _build_monthly_basket_id(expiry_str)
                        _mark_monthly_strangle_open(expiry_str, basket_id)
                if decision.strategy_type == StrategyType.STRANGLE and decision.action_type in ("CLOSE_ALL", "CLOSE_LEGS") and decision.legs_to_close:
                    _mark_monthly_strangle_closed(expiry_str)
                continue
            if smart_enabled:
                trend_ctx = detect_trend_from_open(market, avg_volume, vwap=vwap, pivot=pivot)
            else:
                trend_ctx = TrendContext(TrendSide.SIDEWAYS, 0, 0, 0)
            _log_environment_state(market, trend_ctx, trade_mode)

            if forced_decision and forced_decision.reason == "BASKET_TRAIL_HIT" and basket_mtm >= risk.daily_target:
                day_mode = "LOCKED_GREEN"

            # Intelligence layer: features -> scores -> policy
            features = feat_extractor.compute(market, vwap, pivot)
            scores = regime_scorer.score(features)
            pol = policy.decide(scores.as_dict())
            # Override trend context with policy regime when trading
            if not pol.should_trade:
                decision = forced_decision or TradeAction("HOLD", StrategyType.NONE, [], [], pol.notes)
            else:
                if pol.regime == TrendSide.BULL:
                    trend_ctx = TrendContext(TrendSide.BULL, int(scores.prob_bull * 10), int(scores.prob_bear * 10), int(pol.confidence * 10))
                elif pol.regime == TrendSide.BEAR:
                    trend_ctx = TrendContext(TrendSide.BEAR, int(scores.prob_bull * 10), int(scores.prob_bear * 10), int(pol.confidence * 10))
                elif pol.regime == TrendSide.SIDEWAYS:
                    trend_ctx = TrendContext(TrendSide.SIDEWAYS, int(scores.prob_bull * 10), int(scores.prob_bear * 10), int(pol.confidence * 10))
                decision = (
                    forced_decision
                    if forced_decision
                    else selector.decide(
                        market=market,
                        option_chain=chain,
                        expiry=expiry,
                        risk=risk,
                        current_positions=legs,
                        basket_mtm=basket_mtm,
                        trend_ctx=trend_ctx,
                        daily_trades=daily_trades,
                        day_mode=day_mode,
                        entries_used=day_entries,
                        legs_used=day_legs,
                        orb_state=orb_state,
                        sr_levels=sr_levels,
                        reentry_buffer=orb_config.reentry_buffer_points,
                    )
                )
            # Log intelligence snapshot
            try:
                with INTEL_LOG_PATH.open("a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            _ist_now().isoformat(),
                            scores.prob_bull,
                            scores.prob_bear,
                            scores.prob_sideways,
                            pol.regime.name if hasattr(pol, "regime") else "NA",
                            pol.strategy.name if hasattr(pol, "strategy") else "NA",
                            pol.size_multiplier if hasattr(pol, "size_multiplier") else 0.0,
                            pol.notes if hasattr(pol, "notes") else "",
                            features.gap_pct,
                            features.vix_value,
                            features.vwap_distance,
                        ]
                    )
            except Exception:
                pass
            log.info(
                "Decision=%s strategy=%s reason=%s (trend=%s bull=%s bear=%s)",
                decision.action_type,
                decision.strategy_type.value,
                decision.reason,
                trend_ctx.trend_side.value,
                trend_ctx.bull_score,
                trend_ctx.bear_score,
            )
            if decision.action_type != "HOLD" and decision.strategy_type != StrategyType.NONE:
                legs_to_log = decision.legs_to_open if decision.legs_to_open else decision.legs_to_close or legs
                _log_strategy_event(
                    strategy=decision.strategy_type,
                    action=decision.action_type,
                    reason=decision.reason,
                    market=market,
                    trade_mode=trade_mode,
                    legs=legs_to_log,
                )
            if decision.action_type.startswith("OPEN"):
                if _is_india_market_open():
                    for leg in decision.legs_to_open:
                        leg.opened_at = _ist_now()
                        _place_leg_order(dw, leg, close=False, trade_mode=trade_mode)
                        day_legs += 1
                    daily_trades += 1
                    day_entries += 1
                else:
                    log.warning("Market closed; skipping OPEN action (%s)", decision.action_type)
            elif decision.action_type.startswith("CLOSE"):
                target_legs = decision.legs_to_close or legs
                for leg in target_legs:
                    _place_leg_order(dw, leg, close=True, trade_mode=trade_mode)
                    day_legs += 1
        except Exception as exc:
            if trade_mode == "live" and selected_strategy_file == "batman_bkm_monthly" and live_gate:
                live_gate.mark_loop_error("AGENT_LOOP_ERROR", when=_ist_now())
            _ops_alert(
                "CRITICAL",
                "AGENT_LOOP_ERROR",
                "Unhandled agent loop exception.",
                details={"error": str(exc), "loop_seq": loop_seq, "strategy_file": selected_strategy_file},
                dedupe_key="AGENT_LOOP_ERROR",
            )
            _heartbeat(
                force=True,
                status="ERROR",
                phase="loop_exception",
                extra={"loop_seq": loop_seq, "error": str(exc), "day_mode": day_mode},
            )
            log.exception("Agent loop error: %s", exc)
        time.sleep(poll_sec)


if __name__ == "__main__":
    main()
