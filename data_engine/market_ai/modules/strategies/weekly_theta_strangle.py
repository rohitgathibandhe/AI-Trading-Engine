"""
Weekly Theta Harvesting Strategy
================================

Simplified prototype focused on selling ATM strangles early in the week and
managing them through weekly expiry with MTM targets/stops.

This reuses the same intraday dataset produced for the intraday engine
(`reports/intraday_from_rolling_*.csv`). We collapse each trade date down to
its first/last ticks to approximate entry (open) and daily MTM (close).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pickle
import pandas as pd
import json
import logging
import numpy as np
import uuid
import os

LOG = logging.getLogger(__name__)

__all__ = [
    "WeeklyConfig",
    "WeeklyTargets",
    "EntryRules",
    "run_backtest",
    "WeeklyThetaStrangle",
    "LiveConfig",
    "run_live",
]


@dataclass
class EntryRules:
    min_premium_pct: float = 0.016
    min_iv_rank: float = 0.05
    max_iv_rank: float = 1.0
    max_prev_range_pct: float = 0.035
    min_prev_range_pct: float = 0.003
    demand_prev_break: bool = True
    min_days_to_target_expiry: int = 0
    max_gap_pct: float = 0.01  # 1% gap guard


@dataclass
class WeeklyTargets:
    pnl_target: float = 6_000.0
    pnl_stop: float = 4_000.0
    trailing_lock_pct: float = 0.25
    hard_exit_day: int = 3  # Thursday (0=Mon)
    max_hold_days: int = 5


@dataclass
class WeeklyConfig:
    lot_size: int = 50
    qty: int = 2
    entry_day: int = 0  # Monday
    exit_day: int = 3   # Thursday
    entry_rules: EntryRules = field(default_factory=EntryRules)
    targets: WeeklyTargets = field(default_factory=WeeklyTargets)
    timezone: str = "Asia/Kolkata"
    structure: str = "STRANGLE"  # default fallback
    wing_offset: int = 2
    hybrid_enabled: bool = False
    trend_range_threshold: float = 0.02
    condor_range_threshold: float = 0.01
    oi_distance_pct: float = 0.01
    event_calendar_path: Optional[str] = "data_engine/market_ai/state/events.json"
    skip_event_severities: Tuple[str, ...] = ("high",)
    expiry_offset_weeks: int = 0
    ml_exit_model_path: Optional[str] = None
    ml_exit_min_prob: float = 0.6
    directional_enabled: bool = True
    directional_bias_threshold: float = 0.02


# ---------------------- Live (placeholder, conservative) ----------------------
@dataclass
class LiveConfig:
    lot_size: int = 50
    qty: int = 2
    entry_day: int = 0  # Monday
    entry_hour: int = 9
    entry_minute: int = 30
    warn_only: bool = True  # paper by default; set False to send live orders
    underlying_symbol: str = "NIFTY"
    underlying_seg: str = "NSE_FNO"
    underlying_id: int = 13  # NIFTY security id in Dhan
    product_type: str = "MIS"
    sl_pct: float = 0.02
    tp_pct: float = 0.025
    max_hold_days: int = 5
    hard_exit_hour: int = 15
    hard_exit_minute: int = 1
    notes: str = "Weekly live alpha."
    poll_seconds: int = 30
    event_block_hours: int = 6  # block entry if high-importance event within +/- window
    order_type: str = os.environ.get("WEEKLY_ORDER_TYPE", "MARKET")  # MARKET or LIMIT
    slippage_pct: float = float(os.environ.get("WEEKLY_SLIPPAGE_PCT") or 0.002)  # used if order_type=LIMIT
    fill_wait_seconds: int = 10
    fill_poll_seconds: int = 1
    max_requote: int = 2  # attempts to re-place remaining qty if fill not met
    hedge_enabled: bool = bool(int(os.environ.get("WEEKLY_HEDGE_ENABLED", "0")))
    hedge_trigger_pct: float = 0.01  # hedge when MTM drawdown beyond this
    hedge_distance: int = 200  # points away from ATM for hedge wings
    hedge_cost_cap: float = 500.0  # max premium spend per hedge leg


def run_live(dw, cfg: LiveConfig) -> None:
    """
    Alpha live loop: when flat and entry window opens on entry_day, enters ATM short strangle for nearest Tuesday expiry,
    sets MTM TP/SL, and exits on targets or max_hold/hard exit. Uses Dhan scrip master to resolve security IDs.
    """
    LOG.info("[weekly_live] starting with cfg=%s", cfg)
    import time
    from market_ai.modules.data_fetch.dhan_scrip_cache import resolve_option_security_id, refresh_scrip_master

    intents_path = Path(__file__).resolve().parents[2] / "state" / "weekly_order_intents.jsonl"
    intents_path.parent.mkdir(parents=True, exist_ok=True)
    status_path = Path(__file__).resolve().parents[2] / "state" / "weekly_live_status.json"
    status_path = Path(__file__).resolve().parents[2] / "state" / "weekly_live_status.json"

    def _append_intent(payload: dict) -> None:
        try:
            with intents_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, default=str) + "\n")
        except Exception as exc:
            LOG.warning("[weekly_live] failed to log intent: %s", exc)

    open_position: Optional[Dict[str, Any]] = None

    while True:
        now = datetime.now()
        weekday = now.weekday()

        try:
            positions = dw.get_positions_live() if hasattr(dw, "get_positions_live") else []
        except Exception as exc:
            LOG.warning("[weekly_live] positions fetch failed: %s", exc)
            positions = []
        open_deriv = [
            p for p in positions if isinstance(p, dict) and str(p.get("exchangeSegment", "")).upper().startswith("NSE_FNO") and float(p.get("netQty", 0)) != 0
        ]

        if open_position:
            # monitor MTM TP/SL and exit conditions
            ce_ltp = pe_ltp = None
            ltps = _fetch_ltps(dw, cfg.underlying_seg, [open_position["ce_sec_id"], open_position["pe_sec_id"]])
            ce_ltp = ltps.get(open_position["ce_sec_id"])
            pe_ltp = ltps.get(open_position["pe_sec_id"])
            pnl, pnl_pct = _compute_mtm(open_position, ce_ltp, pe_ltp)
            open_position["mtm"] = pnl
            open_position["mtm_pct"] = pnl_pct
            if pnl_pct is not None:
                LOG.debug("[weekly_live] MTM pnl=%.2f pnl%%=%.2f", pnl or 0.0, pnl_pct * 100)
            age_days = (now.date() - open_position["entry_date"].date()).days
            # exit at/after expiry date end-of-day
            expiry_date = open_position.get("expiry_date")
            if expiry_date and now.date() >= expiry_date and now.hour >= cfg.hard_exit_hour:
                LOG.info("[weekly_live] Expiry reached, closing")
                _close_weekly(dw, open_position, cfg)
                open_position = None
            if pnl_pct is not None and pnl_pct >= cfg.tp_pct:
                LOG.info("[weekly_live] TP hit %.2f%%, closing", pnl_pct * 100)
                _close_weekly(dw, open_position, cfg)
                open_position = None
            elif pnl_pct is not None and pnl_pct <= -cfg.sl_pct:
                LOG.info("[weekly_live] SL hit %.2f%%, closing", pnl_pct * 100)
                _close_weekly(dw, open_position, cfg)
                open_position = None
            elif age_days >= cfg.max_hold_days or (now.hour >= cfg.hard_exit_hour and now.minute >= cfg.hard_exit_minute):
                LOG.info("[weekly_live] Time exit (age=%s days), closing", age_days)
                _close_weekly(dw, open_position, cfg)
                open_position = None
            elif cfg.hedge_enabled and (pnl_pct or 0) <= -cfg.hedge_trigger_pct and not (open_position.get("hedged")):
                if _maybe_hedge(dw, cfg, open_position):
                    open_position["hedged"] = True
                    _write_status(status_path, open_position)
            _write_status(status_path, open_position)
            time.sleep(cfg.poll_seconds)
            continue

        # skip entry if any other FNO positions exist
        if open_deriv:
            LOG.debug("[weekly_live] other positions open (%d), skipping entry", len(open_deriv))
            time.sleep(cfg.poll_seconds * 2)
            continue
        if weekday != cfg.entry_day:
            time.sleep(cfg.poll_seconds * 4)
            continue
        if now.hour < cfg.entry_hour or (now.hour == cfg.entry_hour and now.minute < cfg.entry_minute):
            time.sleep(cfg.poll_seconds)
            continue
        # block entry if high importance event within window
        if _events_block(cfg.event_block_hours):
            LOG.info("[weekly_live] High-importance event within %sh, skipping entry", cfg.event_block_hours)
            time.sleep(cfg.poll_seconds * 4)
            continue

        # Entry: fetch spot via option chain ATM
        try:
            expiry = _next_tuesday_expiry(now)
            oc = dw.get_option_chain(cfg.underlying_id, cfg.underlying_seg, expiry)
            spot = _extract_spot_from_oc(oc)
        except Exception:
            spot = None
        if spot is None:
            LOG.warning("[weekly_live] no spot from option chain; skipping entry this cycle")
            time.sleep(cfg.poll_seconds * 2)
            continue
        # skip if holiday on expiry or trade day based on events
        if _holiday_block(now, expiry):
            LOG.info("[weekly_live] Holiday block for trade/expiry date; skipping entry")
            time.sleep(cfg.poll_seconds * 4)
            continue
        atm = _round_to_int(spot, 50)
        strikes = {"CE": atm, "PE": atm}

        def _resolve(sec_type: str) -> Optional[int]:
            try:
                refresh_scrip_master(force=False)
                return resolve_option_security_id(cfg.underlying_symbol, expiry, strikes[sec_type], sec_type)
            except Exception:
                return None

        ce_sec = _resolve("CE")
        pe_sec = _resolve("PE")
        if not ce_sec or not pe_sec:
            LOG.warning("[weekly_live] could not resolve sec ids for strikes %s", strikes)
            time.sleep(cfg.poll_seconds * 2)
            continue

        intent = {
            "timestamp": now.isoformat(timespec="seconds"),
            "action": "ENTER_WEEKLY_STRANGLE",
            "expiry": expiry,
            "spot": spot,
            "atm": atm,
            "ce_sec_id": ce_sec,
            "pe_sec_id": pe_sec,
            "lot_size": cfg.lot_size,
            "qty": cfg.qty,
            "warn_only": cfg.warn_only,
            "note": cfg.notes,
        }
        _append_intent(intent)
        if cfg.warn_only:
            LOG.info("[weekly_live] (warn_only) logged intent: %s", intent)
            _write_status(status_path, {"intent": intent})
            time.sleep(cfg.poll_seconds * 10)
            continue

        # place orders
        try:
            ce_qty = cfg.qty * cfg.lot_size
            pe_qty = cfg.qty * cfg.lot_size
            ltps_for_limits = _fetch_ltps(dw, cfg.underlying_seg, [ce_sec, pe_sec])
            ce_price = ltps_for_limits.get(ce_sec)
            pe_price = ltps_for_limits.get(pe_sec)
            order_type = cfg.order_type or "MARKET"
            ce_limit = pe_limit = None
            if order_type.upper() == "LIMIT":
                if ce_price:
                    ce_limit = ce_price * (1.0 - cfg.slippage_pct)
                if pe_price:
                    pe_limit = pe_price * (1.0 - cfg.slippage_pct)
            ce_order_id = _place_with_retries(dw, cfg, ce_sec, ce_qty, order_type, ce_limit)
            pe_order_id = _place_with_retries(dw, cfg, pe_sec, pe_qty, order_type, pe_limit)
            ltps_entry = _fetch_ltps(dw, cfg.underlying_seg, [ce_sec, pe_sec], retries=3)
            entry_ce = ltps_entry.get(ce_sec)
            entry_pe = ltps_entry.get(pe_sec)
            open_position = {
                "expiry": expiry,
                "atm": atm,
                "ce_sec_id": ce_sec,
                "pe_sec_id": pe_sec,
                "underlying_symbol": cfg.underlying_symbol,
                "qty": cfg.qty,
                "lot_size": cfg.lot_size,
                "ce_strike": strikes["CE"],
                "pe_strike": strikes["PE"],
                "entry_ce": float(entry_ce) if entry_ce is not None else None,
                "entry_pe": float(entry_pe) if entry_pe is not None else None,
                "entry_date": now,
                "underlying_id": cfg.underlying_id,
                "expiry_date": datetime.fromisoformat(expiry).date(),
                "order_type": order_type,
                "ce_limit": ce_limit,
                "pe_limit": pe_limit,
                "ce_order_id": ce_order_id,
                "pe_order_id": pe_order_id,
                "hedged": False,
            }
            LOG.info("[weekly_live] Entered weekly strangle CE %s PE %s", ce_sec, pe_sec)
            _write_status(status_path, open_position)
        except Exception as exc:
            LOG.warning("[weekly_live] order placement failed: %s", exc)
            open_position = None
        time.sleep(cfg.poll_seconds * 10)


def _round_to_int(x: float, step: int = 50) -> int:
    return int(round(x / step) * step)


def _next_tuesday_expiry(now: datetime) -> str:
    from datetime import timedelta
    d = now.date()
    holidays = _holiday_dates()
    max_ahead = 10
    ahead = 0
    while True:
        if d.weekday() == 1 and d not in holidays:
            return d.isoformat()
        d += timedelta(days=1)
        ahead += 1
        if ahead > max_ahead:
            return d.isoformat()


def _extract_spot_from_oc(oc: Dict[str, Any]) -> Optional[float]:
    if isinstance(oc, dict):
        data = oc.get("data") or oc
        spot = data.get("last_price") or data.get("spot") or data.get("underlyingValue")
        if spot:
            try:
                return float(spot)
            except Exception:
                return None
    return None


def _close_weekly(dw, pos: Dict[str, Any], cfg: LiveConfig) -> None:
    for key, side in (("ce_sec_id", "BUY"), ("pe_sec_id", "BUY")):
        sec = pos.get(key)
        if not sec:
            continue
        try:
            qty = pos.get("qty", 0) * pos.get("lot_size", 0)
            if qty <= 0:
                continue
            dw.place_order(
                side=side,
                exchange_seg=cfg.underlying_seg,
                security_id=sec,
                quantity=qty,
                product_type=cfg.product_type,
                order_type="MARKET",
            )
        except Exception as exc:
            LOG.warning("[weekly_live] close leg failed %s: %s", key, exc)


def _compute_mtm(pos: Dict[str, Any], ce_ltp: Optional[float], pe_ltp: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    entry_ce = pos.get("entry_ce")
    entry_pe = pos.get("entry_pe")
    if entry_ce is None or entry_pe is None or ce_ltp is None or pe_ltp is None:
        # if entry missing but current LTPs exist, set entry to current as fallback
        if entry_ce is None and ce_ltp is not None:
            pos["entry_ce"] = ce_ltp
            entry_ce = ce_ltp
        if entry_pe is None and pe_ltp is not None:
            pos["entry_pe"] = pe_ltp
            entry_pe = pe_ltp
        if entry_ce is None or entry_pe is None or ce_ltp is None or pe_ltp is None:
            return pos.get("mtm"), pos.get("mtm_pct")
    qty = pos.get("qty", 0) * pos.get("lot_size", 0)
    entry_credit = (entry_ce + entry_pe) * qty
    current_debit = (ce_ltp + pe_ltp) * qty
    pnl = entry_credit - current_debit
    pnl_pct = pnl / entry_credit if entry_credit else None
    return pnl, pnl_pct


def _events_block(window_hours: int) -> bool:
    """
    Look at events.jsonl and block entry if any importance>=3 event within +/- window_hours.
    """
    path = Path(__file__).resolve().parents[2] / "state" / "events.jsonl"
    if not path.exists():
        return False
    try:
        now = datetime.now()
        window = timedelta(hours=window_hours)
        for line in path.read_text().splitlines():
            try:
                ev = json.loads(line)
                ts = ev.get("timestamp")
                if not ts:
                    continue
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if abs((dt - now).total_seconds()) <= window.total_seconds() and int(ev.get("importance", 1)) >= 3:
                    return True
            except Exception:
                continue
    except Exception:
        return False
    return False


def _holiday_block(now: datetime, expiry_str: str) -> bool:
    """Block if a holiday event matches trade day or expiry day."""
    target_dates = {now.date(), datetime.fromisoformat(expiry_str).date()}
    dates = _holiday_dates()
    return bool(target_dates & dates)


def _holiday_dates() -> set:
    dates = set()
    path = Path(__file__).resolve().parents[2] / "state" / "events.jsonl"
    if not path.exists():
        return dates
    try:
        for line in path.read_text().splitlines():
            try:
                ev = json.loads(line)
                if ev.get("event_type") != "holiday":
                    continue
                ts = ev.get("timestamp")
                if not ts:
                    continue
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).date()
                dates.add(dt)
            except Exception:
                continue
    except Exception:
        return dates
    return dates


def _write_status(path: Path, pos: Optional[Dict[str, Any]]) -> None:
    try:
        payload = pos or {}
        payload["timestamp"] = datetime.now().isoformat(timespec="seconds")
        path.write_text(json.dumps(payload, default=str, indent=2))
    except Exception:
        LOG.debug("[weekly_live] failed to write status")


def _fetch_ltps(dw, exchange_seg: str, sec_ids: list[int], retries: int = 2, delay: float = 1.0) -> Dict[int, Optional[float]]:
    out: Dict[int, Optional[float]] = {}
    for attempt in range(retries):
        try:
            pairs = [(exchange_seg, sid) for sid in sec_ids]
            ltps = dw.get_ltp_bulk(pairs) if hasattr(dw, "get_ltp_bulk") else {}
            for sid in sec_ids:
                out[sid] = ltps.get((exchange_seg, sid))
            if all(v is not None for v in out.values()):
                return out
        except Exception:
            pass
        time.sleep(delay)
    return out


def _ensure_fill(dw, cfg: LiveConfig, sec_id: int, target_qty: int, order_id: Optional[str] = None) -> bool:
    """
    Poll positions to confirm net qty for sec_id reaches target within the fill window.
    """
    deadline = datetime.now().timestamp() + cfg.fill_wait_seconds
    # First try order status if available
    if order_id and hasattr(dw, "order_status"):
        try:
            status = dw.order_status(order_id)
            if _order_filled(status):
                return True
        except Exception:
            pass
    while datetime.now().timestamp() < deadline:
        try:
            rows = dw.get_positions_live()
        except Exception:
            time.sleep(cfg.fill_poll_seconds)
            continue
        net = 0
        for r in rows:
            try:
                if int(r.get("security_id") or r.get("securityId") or 0) == int(sec_id):
                    net = int(float(r.get("netQty") or r.get("netqty") or 0))
                    break
            except Exception:
                continue
        if net == target_qty:
            return True
        time.sleep(cfg.fill_poll_seconds)
    LOG.warning("[weekly_live] Fill not confirmed for %s target %s", sec_id, target_qty)
    return False


def _order_filled(status_resp: Dict[str, Any]) -> bool:
    st = str(status_resp.get("orderStatus") or status_resp.get("status") or "").lower()
    if st in {"completed", "executed", "filled", "traded", "complete"}:
        return True
    filled = status_resp.get("filledQuantity") or status_resp.get("filledQty")
    remaining = status_resp.get("pendingQuantity") or status_resp.get("remainingQuantity")
    try:
        filled = int(float(filled or 0))
        remaining = int(float(remaining or 0))
        if filled > 0 and remaining == 0:
            return True
    except Exception:
        pass
    return False


def _maybe_hedge(dw, cfg: LiveConfig, pos: Dict[str, Any]) -> bool:
    """
    Buy cheap wings to limit risk if drawdown crosses threshold.
    """
    expiry = pos.get("expiry")
    if not expiry:
        return False
    hedges = []
    for opt_type, strike in (("CALL", pos.get("ce_strike", pos.get("atm", 0)) + cfg.hedge_distance), ("PUT", pos.get("pe_strike", pos.get("atm", 0)) - cfg.hedge_distance)):
        try:
            if strike is None or strike <= 0:
                continue
            sec_id = resolve_option_security_id(pos.get("underlying_symbol", "NIFTY") or cfg.underlying_symbol, expiry, strike, opt_type)
        except Exception:
            sec_id = None
        if not sec_id:
            continue
        try:
            ltp = dw.get_ltp_once(cfg.underlying_seg, sec_id) if hasattr(dw, "get_ltp_once") else None
        except Exception:
            ltp = None
        if ltp is None or ltp * cfg.lot_size > cfg.hedge_cost_cap:
            continue
        hedges.append((sec_id, ltp))
    placed = False
    for sec_id, ltp in hedges:
        try:
            price = ltp * (1.0 + cfg.slippage_pct)
            dw.place_order(
                side="BUY",
                exchange_seg=cfg.underlying_seg,
                security_id=sec_id,
                quantity=cfg.lot_size,
                product_type=cfg.product_type,
                order_type="LIMIT",
                price=price,
                client_order_id=f"HEDGE-{uuid.uuid4().hex[:6]}",
            )
            placed = True
        except Exception as exc:
            LOG.warning("[weekly_live] hedge placement failed for %s: %s", sec_id, exc)
    return placed


def _place_with_retries(dw, cfg: LiveConfig, sec_id: int, qty: int, order_type: str, limit_price: Optional[float]) -> Optional[str]:
    """Place order and, if fill not reached, retry a limited number of times with adjusted limit."""
    side = "SELL"
    remaining = qty
    client_id = f"WEEKLY-{uuid.uuid4().hex[:8]}"
    last_order_id: Optional[str] = None
    for attempt in range(cfg.max_requote + 1):
        try:
            resp = dw.place_order(
                side=side,
                exchange_seg=cfg.underlying_seg,
                security_id=sec_id,
                quantity=remaining,
                product_type=cfg.product_type,
                order_type=order_type,
                price=limit_price if order_type.upper() == "LIMIT" else None,
                client_order_id=client_id,
            )
            if isinstance(resp, dict):
                last_order_id = resp.get("orderId") or resp.get("order_id") or last_order_id
        except Exception as exc:
            LOG.warning("[weekly_live] order placement failed (attempt %d) sec=%s: %s", attempt + 1, sec_id, exc)
        time.sleep(2)
        if _ensure_fill(dw, cfg, sec_id, -remaining, last_order_id):
            return last_order_id
        # fetch positions to see if filled
        try:
            rows = dw.get_positions_live()
            net = 0
            for r in rows:
                if int(r.get("security_id") or r.get("securityId") or 0) == int(sec_id):
                    net = int(float(r.get("netQty") or 0))
                    break
            if net == -qty:
                return
            # calculate remaining based on filled qty
            filled = max(net + qty, 0)  # net is negative for sells
            remaining = max(qty - filled, 0)
            if remaining <= 0:
                return
        except Exception:
            pass
        # adjust limit slightly if not filled
        if order_type.upper() == "LIMIT" and limit_price:
            limit_price = limit_price * (1.0 + cfg.slippage_pct * 0.5)
    LOG.warning("[weekly_live] Could not confirm fill for %s after retries", sec_id)
    return last_order_id


@dataclass
class WeeklyPosition:
    entry_date: datetime
    entry_call: float
    entry_put: float
    qty: int
    lot_size: int
    structure: str = "STRANGLE"
    long_call_entry: float = 0.0
    long_put_entry: float = 0.0
    short_call_strike: Optional[float] = None
    short_put_strike: Optional[float] = None
    long_call_strike: Optional[float] = None
    long_put_strike: Optional[float] = None
    best_pnl: float = 0.0
    target_expiry: Optional[pd.Timestamp] = None

    def entry_credit(self) -> float:
        credit = (self.entry_call + self.entry_put) - (self.long_call_entry + self.long_put_entry)
        return credit * self.qty * self.lot_size

    def current_pnl(
        self,
        call_ltp: float,
        put_ltp: float,
        long_call_ltp: Optional[float] = None,
        long_put_ltp: Optional[float] = None,
    ) -> float:
        long_call_ltp = long_call_ltp or 0.0
        long_put_ltp = long_put_ltp or 0.0
        debit = (call_ltp + put_ltp - long_call_ltp - long_put_ltp) * self.qty * self.lot_size
        return self.entry_credit() - debit


class WeeklyThetaStrangle:
    def __init__(self, cfg: WeeklyConfig):
        self.cfg = cfg
        self.position: Optional[WeeklyPosition] = None
        self.week_id: Optional[Tuple[int, int]] = None
        self.trades: List[Dict[str, Any]] = []
        self._event_calendar = _load_event_calendar(cfg.event_calendar_path)
        self.daily_snapshots: List[Dict[str, Any]] = []
        self._last_close_reason: Optional[str] = None
        self._last_close_time: Optional[pd.Timestamp] = None
        self._last_close_pnl: Optional[float] = None
        self._last_closed_position: Optional[WeeklyPosition] = None
        self._ml_exit_helper: Optional[MLExitHelper] = None
        self._last_ml_prob: Optional[float] = None
        self._last_ml_decision: Optional[str] = None
        model_path = cfg.ml_exit_model_path
        if model_path:
            try:
                self._ml_exit_helper = MLExitHelper(Path(model_path), cfg.ml_exit_min_prob)
                LOG.info("Loaded ML exit model from %s (threshold=%.2f)", model_path, self._ml_exit_helper.threshold)
            except Exception as exc:
                LOG.warning("Could not load ML exit model %s: %s", model_path, exc)

    def on_new_day(self, day_rows: pd.DataFrame) -> None:
        if day_rows.empty:
            return
        day_rows = day_rows.sort_values("timestamp")
        morning, close = day_rows.iloc[0], day_rows.iloc[-1]
        current_date = pd.to_datetime(morning["timestamp"]).date()
        weekday = current_date.weekday()
        current_week = (current_date.isocalendar().year, current_date.isocalendar().week)
        entry_attempted = False
        entry_outcome = None
        entry_block_reason = None

        if self.position is None and weekday <= self.cfg.exit_day:
            entry_attempted = True
            allowed, block_reason = self._entry_allowed(day_rows)
            if allowed:
                self.week_id = current_week
                self._open_position(morning)
                entry_outcome = "OPENED"
            else:
                entry_block_reason = block_reason or "blocked"

        if self.position:
            if current_week != self.week_id:
                # Safety: week rolled but we still have a position -> force close at prev close price.
                self._close_position(close, reason="week_roll")
                return
            pnl = self.position.current_pnl(close["atm_call_ltp"], close["atm_put_ltp"])
            self.position.best_pnl = max(self.position.best_pnl, pnl)
            exit_reason = self._check_exit(weekday, pnl, morning, close)
            if exit_reason:
                self._close_position(close, reason=exit_reason)
        self._log_daily_snapshot(morning, close, weekday, entry_attempted, entry_outcome, entry_block_reason)

    def _open_position(self, row: pd.Series) -> None:
        structure = self._determine_structure(row)
        legs = self._resolve_entry_legs(row, structure)
        if legs is None:
            return
        expiry, days_to = self._target_expiry_info(row)
        if expiry is None:
            return
        pos = WeeklyPosition(
            entry_date=pd.to_datetime(row["timestamp"]),
            entry_call=legs["short_call"],
            entry_put=legs["short_put"],
            qty=self.cfg.qty,
            lot_size=self.cfg.lot_size,
            structure=legs["structure"],
            long_call_entry=legs.get("long_call", 0.0),
            long_put_entry=legs.get("long_put", 0.0),
            short_call_strike=legs.get("short_call_strike"),
            short_put_strike=legs.get("short_put_strike"),
            long_call_strike=legs.get("long_call_strike"),
            long_put_strike=legs.get("long_put_strike"),
            target_expiry=pd.to_datetime(expiry),
        )
        self.position = pos
        self.trades.append(
            {
                "timestamp": pos.entry_date,
                "action": "OPEN",
                "call_ltp": pos.entry_call,
                "put_ltp": pos.entry_put,
                "long_call_ltp": pos.long_call_entry or None,
                "long_put_ltp": pos.long_put_entry or None,
                "call_strike": pos.short_call_strike,
                "put_strike": pos.short_put_strike,
                "long_call_strike": pos.long_call_strike,
                "long_put_strike": pos.long_put_strike,
                "expiry": expiry.isoformat() if hasattr(expiry, "isoformat") else expiry,
                "target_days": days_to,
                "expiry_mode": "NEXT_WEEK" if self.cfg.expiry_offset_weeks else "FRONT_WEEK",
                "structure": pos.structure,
                "notes": "weekly_entry",
            }
        )

    def _close_position(self, row: pd.Series, reason: str) -> None:
        if self.position is None:
            return
        current_pos = self.position
        current_legs = self._resolve_close_legs(row, current_pos.structure)
        pnl = self.position.current_pnl(
            current_legs["short_call"],
            current_legs["short_put"],
            current_legs.get("long_call"),
            current_legs.get("long_put"),
        )
        self.trades.append(
            {
                "timestamp": pd.to_datetime(row["timestamp"]),
                "action": "CLOSE",
                "call_ltp": current_legs["short_call"],
                "put_ltp": current_legs["short_put"],
                "long_call_ltp": current_legs.get("long_call"),
                "long_put_ltp": current_legs.get("long_put"),
                "realized": pnl,
                "reason": reason,
            }
        )
        self._last_close_reason = reason
        self._last_close_time = pd.to_datetime(row["timestamp"])
        self._last_close_pnl = pnl
        self._last_closed_position = current_pos
        self.position = None
        self.week_id = None

    def _entry_allowed(self, day_rows: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        rules = self.cfg.entry_rules
        day_rows = day_rows.sort_values("timestamp")
        row = day_rows.iloc[0]
        premium_pct = row.get("combined_premium_pct")
        if pd.isna(premium_pct) or premium_pct < rules.min_premium_pct:
            return False, "premium_lt_min"
        iv_rank = row.get("iv_rank")
        if not pd.isna(iv_rank) and not (rules.min_iv_rank <= iv_rank <= rules.max_iv_rank):
            return False, "iv_rank_band"
        prev_range = row.get("prev_day_range")
        spot = row.get("spot")
        if not pd.isna(prev_range) and spot:
            prev_range_pct = abs(prev_range) / max(spot, 1.0)
            if prev_range_pct < rules.min_prev_range_pct or prev_range_pct > rules.max_prev_range_pct:
                return False, "prev_range_band"
        if rules.demand_prev_break:
            prev_high = row.get("prev_day_high")
            prev_low = row.get("prev_day_low")
            if not pd.isna(prev_high) and not pd.isna(prev_low):
                buffer = spot * 0.0008 if spot else 0.0
                day_high = day_rows["spot"].max()
                day_low = day_rows["spot"].min()
                if not ((day_high >= (prev_high + buffer)) or (day_low <= (prev_low - buffer))):
                    return False, "no_prev_break"
        prev_close = row.get("prev_day_close")
        if not pd.isna(prev_close) and spot:
            gap_pct = abs(spot - prev_close) / max(prev_close, 1.0)
            if gap_pct > getattr(rules, "max_gap_pct", 0.01):
                return False, "gap_guard"
        expiry, days_to = self._target_expiry_info(row)
        min_days = getattr(rules, "min_days_to_target_expiry", 0)
        if expiry is None:
            return False, "missing_expiry"
        if min_days and (days_to is None or days_to < min_days):
            return False, "days_to_expiry"
        if not self._event_allowed(pd.to_datetime(row["timestamp"]).date()):
            return False, "event_block"
        return True, None

    def _determine_structure(self, row: pd.Series) -> str:
        if not self.cfg.hybrid_enabled:
            return self.cfg.structure.upper()
        spot = row.get("spot")
        prev_range = row.get("prev_day_range")
        prev_range_pct = abs(prev_range) / max(spot, 1.0) if spot and prev_range else 0.0
        bias = row.get("spot_trend_20") or 0.0
        if self.cfg.directional_enabled and prev_range_pct >= self.cfg.directional_bias_threshold and bias:
            if bias >= 0:
                return "BULL_PUT_SPREAD"
            return "BEAR_CALL_SPREAD"
        if prev_range_pct >= self.cfg.trend_range_threshold:
            return "STRANGLE"
        if prev_range_pct <= self.cfg.condor_range_threshold and self._condor_viable(row):
            return "IRON_CONDOR"
        return self.cfg.structure.upper()

    def _condor_viable(self, row: pd.Series) -> bool:
        spot = row.get("spot")
        call_strike = row.get("call_oi_max_strike")
        put_strike = row.get("put_oi_max_strike")
        if pd.isna(spot) or pd.isna(call_strike) or pd.isna(put_strike):
            return False
        distance_call = abs(call_strike - spot) / max(spot, 1.0)
        distance_put = abs(spot - put_strike) / max(spot, 1.0)
        min_distance = max(0.005, self.cfg.oi_distance_pct)
        return distance_call >= min_distance and distance_put >= min_distance

    def _resolve_entry_legs(self, row: pd.Series, structure: str) -> Optional[Dict[str, float]]:
        structure = structure.upper()
        short_call = row.get("atm_call_ltp")
        short_put = row.get("atm_put_ltp")
        if pd.isna(short_call) or pd.isna(short_put):
            return None
        legs = {
            "short_call": float(short_call),
            "short_put": float(short_put),
            "short_call_strike": float(row.get("atm_call_strike", 0.0)) if not pd.isna(row.get("atm_call_strike")) else None,
            "short_put_strike": float(row.get("atm_put_strike", 0.0)) if not pd.isna(row.get("atm_put_strike")) else None,
            "structure": structure,
        }
        if structure == "BULL_PUT_SPREAD":
            legs["short_call"] = 0.0
            legs["short_call_strike"] = None
            long_put = row.get(_selector_field("put", -2, "ltp"))
            long_put_strike = row.get(_selector_field("put", -2, "strike"))
            if pd.isna(long_put) or pd.isna(long_put_strike):
                legs["structure"] = "STRANGLE"
                return legs
            legs["long_put"] = float(long_put)
            legs["long_put_strike"] = float(long_put_strike)
            legs["structure"] = "BULL_PUT_SPREAD"
            return legs
        if structure == "BEAR_CALL_SPREAD":
            legs["short_put"] = 0.0
            legs["short_put_strike"] = None
            long_call = row.get(_selector_field("call", 2, "ltp"))
            long_call_strike = row.get(_selector_field("call", 2, "strike"))
            if pd.isna(long_call) or pd.isna(long_call_strike):
                legs["structure"] = "STRANGLE"
                return legs
            legs["long_call"] = float(long_call)
            legs["long_call_strike"] = float(long_call_strike)
            legs["structure"] = "BEAR_CALL_SPREAD"
            return legs
        if structure != "IRON_CONDOR":
            legs["structure"] = "STRANGLE"
            return legs
        offset = max(1, int(self.cfg.wing_offset))
        long_call = row.get(_selector_field("call", offset, "ltp"))
        long_put = row.get(_selector_field("put", -offset, "ltp"))
        if pd.isna(long_call) or pd.isna(long_put):
            legs["structure"] = "STRANGLE"
            return legs
        legs["long_call"] = float(long_call)
        legs["long_put"] = float(long_put)
        legs["long_call_strike"] = float(row.get(_selector_field("call", offset, "strike"), 0.0)) if not pd.isna(row.get(_selector_field("call", offset, "strike"))) else None
        legs["long_put_strike"] = float(row.get(_selector_field("put", -offset, "strike"), 0.0)) if not pd.isna(row.get(_selector_field("put", -offset, "strike"))) else None
        return legs

    def _resolve_close_legs(self, row: pd.Series, structure: str) -> Dict[str, float]:
        legs = {
            "short_call": float(row.get("atm_call_ltp", 0.0)),
            "short_put": float(row.get("atm_put_ltp", 0.0)),
        }
        if structure == "BULL_PUT_SPREAD":
            long_put = row.get(_selector_field("put", -2, "ltp"))
            legs["long_put"] = float(long_put) if not pd.isna(long_put) else 0.0
            return legs
        if structure == "BEAR_CALL_SPREAD":
            long_call = row.get(_selector_field("call", 2, "ltp"))
            legs["long_call"] = float(long_call) if not pd.isna(long_call) else 0.0
            return legs
        if structure != "IRON_CONDOR":
            return legs
        offset = max(1, int(self.cfg.wing_offset))
        long_call = row.get(_selector_field("call", offset, "ltp"))
        long_put = row.get(_selector_field("put", -offset, "ltp"))
        legs["long_call"] = float(long_call) if not pd.isna(long_call) else 0.0
        legs["long_put"] = float(long_put) if not pd.isna(long_put) else 0.0
        return legs

    def _target_expiry_info(self, row: pd.Series) -> Tuple[Optional[pd.Timestamp], Optional[float]]:
        base_expiry = pd.to_datetime(row.get("expiryDate"), errors="coerce") if "expiryDate" in row else None
        base_days = row.get("days_to_expiry")
        offset = max(0, int(self.cfg.expiry_offset_weeks))
        target_expiry = base_expiry
        target_days = float(base_days) if base_days is not None and not pd.isna(base_days) else None
        if offset > 0:
            if offset == 1 and "next_week_expiry" in row:
                next_expiry = pd.to_datetime(row.get("next_week_expiry"), errors="coerce")
                if next_expiry is not None and not pd.isna(next_expiry):
                    target_expiry = next_expiry
                next_days = row.get("days_to_next_expiry")
                if next_days is not None and not pd.isna(next_days):
                    target_days = float(next_days)
            if (target_expiry is None or pd.isna(target_expiry)) and base_expiry is not None and not pd.isna(base_expiry):
                target_expiry = base_expiry + pd.to_timedelta(7 * offset, unit="D")
            if (target_days is None) and base_days is not None and not pd.isna(base_days):
                target_days = float(base_days) + 7 * offset
        if target_expiry is not None and pd.isna(target_expiry):
            target_expiry = None
        return target_expiry, target_days

    def _events_for(self, current_date: datetime.date) -> List[Dict[str, Any]]:
        if not self._event_calendar:
            return []
        return self._event_calendar.get(current_date.isoformat(), [])

    def _event_allowed(self, current_date: datetime.date) -> bool:
        entries = self._events_for(current_date)
        skip_set = {s.lower() for s in self.cfg.skip_event_severities}
        for entry in entries:
            sev = str(entry.get("severity", "")).lower()
            if sev in skip_set:
                return False
        return True

    def _log_daily_snapshot(
        self,
        morning: pd.Series,
        close: pd.Series,
        weekday: int,
        entry_attempted: bool,
        entry_outcome: Optional[str],
        entry_block_reason: Optional[str],
    ) -> None:
        current_date = pd.to_datetime(morning["timestamp"]).date()
        events = self._events_for(current_date)
        events_json = json.dumps(events, default=str)
        event_tags = ",".join(
            sorted(
                {
                    str(evt.get("tag") or evt.get("event") or evt.get("title") or "").strip()
                    for evt in events
                    if evt
                }
            )
        ).strip(",")
        active_position = self.position
        last_closed_applicable = (
            self._last_closed_position if self._last_close_time and self._last_close_time.date() == current_date else None
        )
        pnl = None
        best_pnl = None
        target_expiry = None
        target_days_remaining = None
        entry_date = None
        structure = None
        if active_position:
            pnl = active_position.current_pnl(close["atm_call_ltp"], close["atm_put_ltp"])
            best_pnl = active_position.best_pnl
            target_expiry = active_position.target_expiry
            if target_expiry is not None and not pd.isna(target_expiry):
                target_days_remaining = (target_expiry.date() - current_date).days
            entry_date = active_position.entry_date
            structure = active_position.structure
        elif last_closed_applicable:
            target_expiry = last_closed_applicable.target_expiry
            if target_expiry is not None and not pd.isna(target_expiry):
                target_days_remaining = (target_expiry.date() - current_date).days
            entry_date = last_closed_applicable.entry_date
            structure = last_closed_applicable.structure
        snapshot = {
            "date": current_date.isoformat(),
            "weekday": weekday,
            "entry_attempted": entry_attempted,
            "entry_outcome": entry_outcome,
            "entry_block_reason": entry_block_reason,
            "spot_open": morning.get("spot"),
            "spot_close": close.get("spot"),
            "combined_premium_pct": morning.get("combined_premium_pct"),
            "iv_rank": morning.get("iv_rank"),
            "iv_skew": morning.get("iv_skew"),
            "oi_skew": morning.get("oi_skew"),
            "volume_skew": morning.get("volume_skew"),
            "call_oi_max_strike": morning.get("call_oi_max_strike"),
            "put_oi_max_strike": morning.get("put_oi_max_strike"),
            "spot_return_1": close.get("spot_return_1"),
            "spot_return_3": close.get("spot_return_3"),
            "spot_return_5": close.get("spot_return_5"),
            "spot_trend_20": close.get("spot_trend_20"),
            "spot_trend_50": close.get("spot_trend_50"),
            "spot_trend_100": close.get("spot_trend_100"),
            "spot_volatility": close.get("spot_volatility"),
            "spot_intraday_range_pct": close.get("spot_intraday_range_pct"),
            "call_oi_change": close.get("call_oi_change"),
            "put_oi_change": close.get("put_oi_change"),
            "events": events_json,
            "event_count": len(events),
            "event_tags": event_tags,
            "has_position": active_position is not None,
            "structure": structure,
            "entry_date": entry_date.isoformat() if entry_date is not None else None,
            "target_expiry": target_expiry.isoformat() if target_expiry is not None and not pd.isna(target_expiry) else None,
            "target_days_remaining": target_days_remaining,
            "current_pnl": pnl,
            "best_pnl": best_pnl,
            "close_reason": self._last_close_reason if last_closed_applicable else None,
            "close_pnl": self._last_close_pnl if last_closed_applicable else None,
            "ml_exit_prob": self._last_ml_prob,
            "ml_exit_decision": self._last_ml_decision,
        }
        self.daily_snapshots.append(snapshot)
        if last_closed_applicable:
            self._last_closed_position = None
            self._last_close_reason = None
            self._last_close_pnl = None
            self._last_close_time = None
        self._last_ml_prob = None
        self._last_ml_decision = None

    def _check_exit(self, weekday: int, pnl: float, morning_row: pd.Series, close_row: pd.Series) -> Optional[str]:
        targets = self.cfg.targets
        ml_reason = self._ml_exit_decision(weekday, pnl, morning_row, close_row)
        if ml_reason:
            return ml_reason
        if pnl >= targets.pnl_target:
            return "target_hit"
        if pnl <= -targets.pnl_stop:
            return "weekly_stop"
        if self.position:
            lock_floor = self.position.best_pnl * (1 - targets.trailing_lock_pct)
            if lock_floor and pnl <= lock_floor:
                return "trailing_lock"
        days_held = (weekday - self.cfg.entry_day) % 7
        if days_held >= targets.max_hold_days - 1:
            return "max_hold_days"
        if weekday >= targets.hard_exit_day:
            return "hard_exit"
        return None

    def _ml_exit_decision(self, weekday: int, pnl: float, morning_row: pd.Series, close_row: pd.Series) -> Optional[str]:
        self._last_ml_prob = None
        self._last_ml_decision = None
        helper = self._ml_exit_helper
        if not helper or self.position is None:
            return None
        features = self._ml_feature_vector(weekday, pnl, morning_row, close_row)
        if features is None:
            return None
        prob = helper.predict_prob(features)
        self._last_ml_prob = prob
        if prob >= helper.threshold:
            self._last_ml_decision = "EXIT"
            return "ml_exit"
        self._last_ml_decision = "HOLD"
        return None

    def _ml_feature_vector(self, weekday: int, pnl: float, morning_row: pd.Series, close_row: pd.Series) -> Optional[Dict[str, float]]:
        if self.position is None:
            return None
        current_date = pd.to_datetime(close_row["timestamp"]).date()
        entry_date = self.position.entry_date.date()
        target_expiry = self.position.target_expiry.date() if self.position.target_expiry is not None else None
        days_since_entry = (current_date - entry_date).days
        target_days_remaining = (target_expiry - current_date).days if target_expiry else None
        events = self._events_for(current_date)
        event_count = len(events)
        has_event_risk = 1 if event_count > 0 else 0
        drawdown_from_best = (self.position.best_pnl - pnl) if self.position.best_pnl is not None else None
        return {
            "weekday": weekday,
            "spot_open": float(morning_row.get("spot")) if not pd.isna(morning_row.get("spot")) else 0.0,
            "spot_close": float(close_row.get("spot")) if not pd.isna(close_row.get("spot")) else 0.0,
            "combined_premium_pct": float(morning_row.get("combined_premium_pct") or 0.0),
            "iv_rank": float(morning_row.get("iv_rank") or 0.0),
            "iv_skew": float(morning_row.get("iv_skew") or 0.0),
            "oi_skew": float(close_row.get("oi_skew") or 0.0),
            "volume_skew": float(close_row.get("volume_skew") or 0.0),
            "call_oi_max_strike": float(close_row.get("call_oi_max_strike") or 0.0),
            "put_oi_max_strike": float(close_row.get("put_oi_max_strike") or 0.0),
            "spot_return_1": float(close_row.get("spot_return_1") or 0.0),
            "spot_return_3": float(close_row.get("spot_return_3") or 0.0),
            "spot_return_5": float(close_row.get("spot_return_5") or 0.0),
            "spot_trend_20": float(close_row.get("spot_trend_20") or 0.0),
            "spot_trend_50": float(close_row.get("spot_trend_50") or 0.0),
            "spot_trend_100": float(close_row.get("spot_trend_100") or 0.0),
            "spot_volatility": float(close_row.get("spot_volatility") or 0.0),
            "spot_intraday_range_pct": float(close_row.get("spot_intraday_range_pct") or 0.0),
            "call_oi_change": float(close_row.get("call_oi_change") or 0.0),
            "put_oi_change": float(close_row.get("put_oi_change") or 0.0),
            "event_count": float(event_count),
            "has_event_risk": float(has_event_risk),
            "days_since_entry": float(days_since_entry if days_since_entry is not None else 0.0),
            "target_days_remaining": float(target_days_remaining if target_days_remaining is not None else 0.0),
            "current_pnl": float(pnl),
            "best_pnl": float(self.position.best_pnl or 0.0),
            "drawdown_from_best": float(drawdown_from_best or 0.0),
        }


def _coerce_config(cfg_input: Optional[Any]) -> WeeklyConfig:
    if cfg_input is None:
        return WeeklyConfig()
    if isinstance(cfg_input, WeeklyConfig):
        return cfg_input
    data = dict(cfg_input)
    if isinstance(data.get("entry_rules"), dict):
        data["entry_rules"] = EntryRules(**data["entry_rules"])
    if isinstance(data.get("targets"), dict):
        data["targets"] = WeeklyTargets(**data["targets"])
    return WeeklyConfig(**data)


def _selector_field(option: str, offset: int, metric: str) -> str:
    option = option.lower()
    if offset == 0:
        prefix = f"atm_{option}"
    elif offset > 0:
        prefix = f"{option}_atm_plus{offset}"
    else:
        prefix = f"{option}_atm_minus{abs(offset)}"
    return f"{prefix}_{metric}"


def _load_event_calendar(path_str: Optional[str]) -> Dict[str, List[Dict[str, Any]]]:
    if not path_str:
        return {}
    path = Path(path_str)
    if not path.exists():
        return {}
    try:
        import json

        data = json.loads(path.read_text())
    except Exception:
        return {}
    if isinstance(data, dict):
        return {k: (v if isinstance(v, list) else [v]) for k, v in data.items()}
    events: Dict[str, List[Dict[str, Any]]] = {}
    if isinstance(data, list):
        for entry in data:
            if not isinstance(entry, dict):
                continue
            date_key = entry.get("date")
            if not date_key:
                continue
            events.setdefault(date_key, []).append(entry)
    return events


def run_backtest(df: pd.DataFrame, cfg_dict: Optional[Any] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    cfg = _coerce_config(cfg_dict)
    strat = WeeklyThetaStrangle(cfg)
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for col in ("expiryDate", "next_week_expiry"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ("days_to_expiry", "days_to_next_expiry"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "source_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["source_date"]).dt.date
    else:
        df["trade_date"] = df["timestamp"].dt.date
    last_rows = None
    for date, day_rows in df.groupby("trade_date"):
        strat.on_new_day(day_rows)
        last_rows = day_rows.sort_values("timestamp")
    if strat.position and last_rows is not None:
        strat._close_position(last_rows.iloc[-1], reason="dataset_end")
    trades_df = pd.DataFrame(strat.trades)
    daily_df = pd.DataFrame(strat.daily_snapshots)
    total_realized = float(trades_df.loc[trades_df["action"] == "CLOSE", "realized"].sum()) if not trades_df.empty else 0.0
    summary = {
        "entries": int((trades_df["action"] == "OPEN").sum()) if not trades_df.empty else 0,
        "closes": int((trades_df["action"] == "CLOSE").sum()) if not trades_df.empty else 0,
        "total_realized": total_realized,
        "avg_per_trade": float(total_realized / max(1, (trades_df["action"] == "CLOSE").sum())) if not trades_df.empty else 0.0,
    }
    return trades_df, daily_df, summary
ML_FEATURES = [
    "weekday",
    "spot_open",
    "spot_close",
    "combined_premium_pct",
    "iv_rank",
    "iv_skew",
    "oi_skew",
    "volume_skew",
    "call_oi_max_strike",
    "put_oi_max_strike",
    "spot_return_1",
    "spot_return_3",
    "spot_return_5",
    "spot_trend_20",
    "spot_trend_50",
    "spot_trend_100",
    "spot_volatility",
    "spot_intraday_range_pct",
    "call_oi_change",
    "put_oi_change",
    "event_count",
    "has_event_risk",
    "days_since_entry",
    "target_days_remaining",
    "current_pnl",
    "best_pnl",
    "drawdown_from_best",
]


class MLExitHelper:
    def __init__(self, model_path: Path, fallback_threshold: float):
        with Path(model_path).open("rb") as fh:
            payload = pickle.load(fh)
        self.model = payload.get("estimator")
        self.feature_names = payload.get("feature_names", ML_FEATURES)
        self.threshold = float(payload.get("prob_threshold", fallback_threshold))
        if self.model is None:
            raise RuntimeError(f"Invalid model payload at {model_path}")

    def predict_prob(self, features: Dict[str, float]) -> float:
        data = {name: [float(features.get(name, 0.0))] for name in self.feature_names}
        X = pd.DataFrame(data)
        return float(self.model.predict_proba(X)[0][1])
