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


def _load_agent_settings() -> Dict[str, Any]:
    path = os.getenv("AGENT_SETTINGS_JSON")
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
                price_val = getattr(leg, "last_price", None) or price_val
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
                "tag": "BATMAN_BKM" if getattr(leg, "option_type", None) else "BATMAN_V2",
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


def _last_friday_before(expiry: dt_date) -> dt_date:
    probe = expiry
    while probe.weekday() != 4:
        probe -= timedelta(days=1)
    return probe


def _place_leg_order(dw: DhanWrapper, leg: OptionLeg, *, close: bool, trade_mode: str) -> None:
    side = _normalize_leg_side(leg.side)
    if close:
        order_side = LegSide.BUY if side == LegSide.SELL else LegSide.SELL
    else:
        order_side = side
    if trade_mode == "paper":
        # Paper: do not hit broker; log a synthetic blotter row once.
        _log_trade_event(trade_mode=trade_mode, side=order_side.value, leg=leg, order_type="MARKET", notes="paper")
        return
    security_id = _ensure_leg_security_id(leg, leg.symbol, leg.expiry)
    if not security_id:
        log.error("Skipping leg order due to missing security_id (strike=%s opt=%s)", leg.strike, leg.option_type)
        return
    log.info(
        "Placing %s order sec_id=%s strike=%s opt=%s qty=%s expiry=%s",
        order_side.value,
        security_id,
        leg.strike,
        leg.option_type.value if hasattr(leg.option_type, "value") else str(leg.option_type),
        leg.quantity,
        leg.expiry,
    )
    dw.place_order(
        side=order_side.value,
        exchange_seg="NSE_FNO",
        security_id=security_id,
        quantity=leg.quantity,
        product_type="MARGIN",
        order_type="MARKET",
    )
    _log_trade_event(trade_mode=trade_mode, side=order_side.value, leg=leg, order_type="MARKET")


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
    dw = DhanWrapper(logger=logging.getLogger("dhan_wrapper"))
    _write_pid_file()
    lot_size = int(settings.get("lot_size", DEFAULT_SETTINGS["lot_size"]))
    selector = StrategySelector(symbol="NIFTY", lot_size=lot_size)
    risk = _build_risk_config(settings)
    smart_enabled = bool(settings.get("smart_selector_enabled", True))
    avg_volume_hint = float(settings.get("avg_5m_volume", 0.0))
    trade_mode = os.getenv("TRADE_MODE", "live")
    _ensure_csv_headers(TRADE_BLOTTER_PATH, BLOTTER_FIELDS)
    _ensure_csv_headers(INTEL_LOG_PATH, ["timestamp", "prob_bull", "prob_bear", "prob_sideways", "regime", "strategy", "size_multiplier", "notes", "gap_pct", "vix", "vwap_distance"])
    _ensure_csv_headers(FEATURE_LOG_PATH, FEATURE_FIELDS)
    log.info("Regime-aware agent started (mode=%s)", trade_mode)
    log.info("Selected strategy file=%s", SELECTED_STRATEGY_FILE)
    daily_trades = 0
    current_day = datetime.now().date()
    day_entries = 0
    day_legs = 0
    day_mode = "NORMAL"
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
    # Batman BKM (monthly) strategy
    bkm_strategy: Optional[BatmanBKMStrategy] = None

    while True:
        try:
            now = datetime.now()
            today = datetime.now().date()
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
            if now.weekday() >= 5:
                log.info("Weekend detected; market is closed. Sleeping for %ss.", poll_sec)
                time.sleep(poll_sec)
                continue
            market, avg_volume, vwap, pivot = build_market_snapshot(dw, avg_volume_hint=avg_volume_hint)
            # ── Batman BKM monthly branch ───────────────────────────────────
            if SELECTED_STRATEGY_FILE == "batman_bkm_monthly":
                if bkm_strategy is None:
                    def _parse_time(val: str, fallback: time) -> time:
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
                        lot_multiplier=int(settings.get("batman_bkm_lot_multiplier", 1)),
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
                # If we already have a basket, stick to its expiry for MTM/exit updates
                if bkm_strategy.basket:
                    expiry = bkm_strategy.basket.expiry
                else:
                    expiry = _fetch_monthly_expiry(dw) or today
                expiry_str = expiry.isoformat()
                entry_day = _last_friday_before(expiry)
                now_ist = _ist_now()
                # In paper mode, always force entry unless explicitly overridden elsewhere.
                force_entry = True if trade_mode == "paper" else bool(settings.get("batman_bkm_force_entry", False))
                # Prevent multiple entries per expiry
                if bkm_strategy.basket is None and not _is_bkm_blocked(expiry_str):
                    log.info(
                        "[BatmanBKM] loop spot=%.2f expiry=%s force_entry=%s now=%s entry_day=%s entry_time=%s",
                        market.spot,
                        expiry_str,
                        force_entry,
                        now_ist.isoformat(),
                        entry_day,
                        bkm_strategy.cfg.entry_time,
                    )
                    in_window = (
                        now_ist.date() == entry_day
                        and now_ist.time() >= bkm_strategy.cfg.entry_time
                        and _is_india_market_open(now_ist)
                    )
                    if force_entry or in_window:
                        if force_entry:
                            log.info("[BatmanBKM] force_entry enabled (paper=%s)", trade_mode)
                        elif not _is_india_market_open(now_ist):
                            log.info("[BatmanBKM] market closed; skipping entry check")
                            time.sleep(poll_sec)
                            continue
                        try:
                            chain_raw = dw.get_option_chain(INDEX_SECURITY_ID, INDEX_EXCHANGE_SEG, expiry.isoformat())
                            chain = _map_chain(chain_raw, expiry, symbol="NIFTY", spot=market.spot)
                        except Exception:
                            log.exception("[BatmanBKM] failed to fetch option chain")
                            time.sleep(poll_sec)
                            continue
                        basket, reason = bkm_strategy.maybe_enter(market.spot, chain, expiry)
                        log.info("[BatmanBKM] attempt reason=%s expiry=%s", reason, expiry_str)
                        if basket and reason == "ENTER":
                            _mark_bkm_open(expiry_str, {"net_credit": basket.net_credit, "credit_pct": basket.credit_pct})
                            try:
                                _log_batman_blotter(trade_mode, basket.legs, "OPEN")
                            except Exception:
                                pass
                elif bkm_strategy.basket:
                    chain_raw = dw.get_option_chain(INDEX_SECURITY_ID, INDEX_EXCHANGE_SEG, expiry.isoformat())
                    chain = _map_chain(chain_raw, expiry, symbol="NIFTY", spot=market.spot)
                    pnl = bkm_strategy.update_mtm(chain) or 0.0
                    try:
                        _log_batman_blotter(trade_mode, bkm_strategy.basket.legs, "MTM")
                    except Exception:
                        pass
                    decision = bkm_strategy.maybe_exit(pnl, _ist_now())
                    if decision:
                        try:
                            _log_batman_blotter(trade_mode, bkm_strategy.basket.legs, "CLOSE")
                        except Exception:
                            pass
                        _mark_bkm_closed(expiry_str, decision, pnl)
                        log.info("Batman BKM exited: %s pnl=%.2f", decision, pnl)
                        bkm_strategy.basket = None
                time.sleep(poll_sec)
                continue
            if SELECTED_STRATEGY_FILE == "batman_v2_paper":
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
            if SELECTED_STRATEGY_FILE == "monthly_strangle_with_weekly_hedge.py":
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
            log.exception("Agent loop error: %s", exc)
        time.sleep(poll_sec)

if __name__ == "__main__":
    main()
