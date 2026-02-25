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
      expiry, option_type ("CE"/"PE"), strike, ltp, delta, security_id, spot.

    Supports both legacy and v2 /v2/optionchain shapes by delegating to
    _coerce_chain_dict, which walks through 'data'/'oc'/etc.
    """
    rows: List[dict] = []
    chain_dict = _coerce_chain_dict(chain_raw)
    if not isinstance(chain_dict, dict):
        return rows

    expiry_str = expiry.isoformat() if hasattr(expiry, "isoformat") else str(expiry)

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

            rows.append(
                {
                    "expiry": expiry,
                    "option_type": option_type,
                    "strike": strike_f,
                    "ltp": ltp,
                    "delta": greeks.get("delta") or opt_data.get("delta"),
                    "spot": spot,
                    "security_id": sec_id,
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
            alert_journal.emit(
                severity=severity,
                code=code,
                message=message,
                details=details,
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

    while True:
        try:
            loop_seq += 1
            now = datetime.now()
            now_ist = _ist_now()
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
                        "[BatmanBKM] entry blocked day_mode=%s live_gate_status=%s live_gate_locked_for=%s reconcile_status=%s reconcile_hard_lock=%s exec_recovery_status=%s exec_recovery_hard_lock=%s",
                        day_mode,
                        lock_payload.get("status"),
                        lock_payload.get("locked_for_date"),
                        reconcile_payload.get("status"),
                        reconcile_payload.get("hard_lock"),
                        exec_recovery_payload.get("status"),
                        exec_recovery_payload.get("hard_lock"),
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
                    chain = _fetch_bkm_chain(expiry)
                    if not chain:
                        time.sleep(poll_sec)
                        continue
                    pnl = bkm_strategy.update_mtm(chain) or 0.0
                    if live_bkm_gate_enabled and live_gate:
                        live_gate.note_live_event(mtm=pnl, when=_ist_now())
                    monitor_every = max(5.0, float(settings.get("batman_bkm_monitor_log_interval_sec", 20.0)))
                    if (
                        bkm_last_monitor_log_at is None
                        or (now_ist - bkm_last_monitor_log_at).total_seconds() >= monitor_every
                    ):
                        bkm_last_monitor_log_at = now_ist
                        basket_ref = bkm_strategy.basket
                        tp_val = (bkm_strategy.cfg.tp_pct * float(getattr(basket_ref, "margin_required", 0.0) or 0.0)) if basket_ref else 0.0
                        sl_val = -(bkm_strategy.cfg.sl_pct * float(getattr(basket_ref, "margin_required", 0.0) or 0.0)) if basket_ref else 0.0
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
