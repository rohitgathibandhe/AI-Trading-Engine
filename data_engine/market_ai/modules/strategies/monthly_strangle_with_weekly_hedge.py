from __future__ import annotations

# --- Imports (deduped and sorted) ---
import csv
import logging
import math
import time
import json
from dataclasses import dataclass, asdict, field
from datetime import date, timedelta, datetime
from typing import Any, Dict, Optional, Tuple, List, cast, TYPE_CHECKING

import pandas as pd
from pathlib import Path

from market_ai.modules.data_fetch.option_chain_ingestor import OptionChainIngestor, ChainConfig
from market_ai.modules.data_fetch.dhan_scrip_cache import resolve_option_security_id
from market_ai.modules.analytics import summarize_market_state
from market_ai.modules.agents.strategy_recommender import serialize_recommendations

if TYPE_CHECKING:
    from market_ai.modules.analytics import MarketRegimeAnalyzer, MarketState
    from market_ai.modules.agents.strategy_recommender import StrategyRecommender, StrategyCandidate

# __all__ for explicit exports and to suppress unused import warnings
__all__ = [
    "MonthlyStrangleWithWeeklyHedge",
    "TradeAction",
    "OpenPosition",
    "LiveConfig",
    "BatmanConfig",
    "run_backtest",
    "run_strategy",
    "run_live",
    "_normalize_positions",
]

LOG = logging.getLogger(__name__)

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
]

_BLOTTER_ROTATE_MAX_BYTES = 5 * 1024 * 1024
_FEATURE_ROTATE_MAX_BYTES = 5 * 1024 * 1024
_CSV_ROTATE_KEEP = 5


def _record_trade(cfg: "LiveConfig", event: Dict[str, Any]) -> None:
    path_str = getattr(cfg, "blotter_path", None)
    if not path_str:
        return
    try:
        path = Path(path_str)
        _rotate_csv_file(path, _BLOTTER_ROTATE_MAX_BYTES, _CSV_ROTATE_KEEP)
        path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not path.exists()
        with path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=BLOTTER_FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow(event)
    except Exception:
        LOG.exception("[live] failed writing blotter row")
        return
    _update_blotter_summary(cfg)


def _update_blotter_summary(cfg: "LiveConfig") -> None:
    blotter_path_str = getattr(cfg, "blotter_path", None)
    summary_path_str = getattr(cfg, "blotter_summary_path", None)
    if not blotter_path_str or not summary_path_str:
        return
    blotter_path = Path(blotter_path_str)
    if not blotter_path.exists():
        return
    try:
        df = pd.read_csv(blotter_path)
    except Exception as exc:
        LOG.warning("[live] failed to load blotter summary: %s", exc)
        return
    if df.empty:
        summary = {
            "total_orders": 0,
            "executed_orders": 0,
            "warn_only_orders": 0,
            "credit_value": 0.0,
            "debit_value": 0.0,
            "net_value": 0.0,
            "last_timestamp": None,
        }
    else:
        exec_series = df["executed"] if "executed" in df.columns else pd.Series([0] * len(df))
        warn_series = df["warn_only"] if "warn_only" in df.columns else pd.Series([0] * len(df))
        price_series = df["price"] if "price" in df.columns else pd.Series([0.0] * len(df))
        qty_series = df["quantity"] if "quantity" in df.columns else pd.Series([0.0] * len(df))

        exec_numeric = cast(pd.Series, pd.to_numeric(exec_series, errors="coerce"))
        warn_numeric = cast(pd.Series, pd.to_numeric(warn_series, errors="coerce"))
        price_numeric = cast(pd.Series, pd.to_numeric(price_series, errors="coerce"))
        qty_numeric = cast(pd.Series, pd.to_numeric(qty_series, errors="coerce"))

        df["executed"] = exec_numeric.fillna(0).astype(int)
        df["warn_only"] = warn_numeric.fillna(0).astype(int)
        price = price_numeric.fillna(0.0)
        qty = qty_numeric.fillna(0.0)
        df["notional"] = price * qty
        executed_df = df[df["executed"] == 1]
        credit = executed_df.loc[executed_df["side"] == "SELL", "notional"].sum()
        debit = executed_df.loc[executed_df["side"] == "BUY", "notional"].sum()
        summary = {
            "total_orders": int(len(df)),
            "executed_orders": int(executed_df["executed"].sum()),
            "warn_only_orders": int(df["warn_only"].sum()),
            "credit_value": round(float(credit), 2),
            "debit_value": round(float(debit), 2),
            "net_value": round(float(credit - debit), 2),
            "last_timestamp": df["timestamp"].iloc[-1],
        }
    summary_path = Path(summary_path_str)
    try:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))
    except Exception:
        LOG.exception("[live] failed writing blotter summary")


def _rotate_csv_file(path: Path, max_bytes: int, keep: int) -> None:
    try:
        if not path.exists():
            return
        if path.stat().st_size < max_bytes:
            return
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        archive_path = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
        path.rename(archive_path)
        archives = sorted(
            path.parent.glob(f"{path.stem}_*{path.suffix}"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for stale in archives[keep:]:
            try:
                stale.unlink()
            except Exception:
                LOG.warning("[log_rotate] failed removing archive %s", stale)
    except Exception:
        LOG.warning("[log_rotate] rotation failed for %s", path)


def _append_csv_row(
    path_str: Optional[str],
    fields: List[str],
    row: Dict[str, Any],
    *,
    max_bytes: Optional[int] = None,
) -> None:
    if not path_str:
        return
    try:
        path = Path(path_str)
        if max_bytes is not None:
            _rotate_csv_file(path, max_bytes, _CSV_ROTATE_KEEP)
        path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not path.exists()
        with path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            if is_new:
                writer.writeheader()
            writer.writerow(row)
    except Exception:
        LOG.exception("[live] failed writing csv row for %s", path_str)


def _record_feature(cfg: "LiveConfig", row: Dict[str, Any]) -> None:
    _append_csv_row(
        getattr(cfg, "feature_log_path", None),
        FEATURE_FIELDS,
        row,
        max_bytes=_FEATURE_ROTATE_MAX_BYTES,
    )


def _record_environment_snapshot(
    cfg: "LiveConfig",
    state: "MarketState",
    candidates: List["StrategyCandidate"],
) -> None:
    feature_path = getattr(cfg, "feature_log_path", None)
    if not feature_path:
        return
    context = {
        "market_state": summarize_market_state(state),
        "strategy_candidates": serialize_recommendations(candidates),
    }
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "strategy": "environment",
        "expiry": "",
        "exchange_seg": cfg.trade_mode,
        "spot": state.features.get("spot"),
        "ce_strike": None,
        "pe_strike": None,
        "ce_ltp": None,
        "pe_ltp": None,
        "ce_delta": None,
        "pe_delta": None,
        "ce_entry": None,
        "pe_entry": None,
        "net_credit": None,
        "warn_only": int(bool(getattr(cfg, "warn_only", False))),
        "trade_mode": cfg.trade_mode,
        "context": json.dumps(context, default=str),
    }
    _append_csv_row(feature_path, FEATURE_FIELDS, row, max_bytes=_FEATURE_ROTATE_MAX_BYTES)


# ----------------- utilities -----------------
def _last_tuesday_of_month(y: int, m: int) -> date:
    if m == 12:
        last = date(y + 1, 1, 1) - timedelta(days=1)
    else:
        last = date(y, m + 1, 1) - timedelta(days=1)
    while last.weekday() != 1:  # Tuesday=1
        last -= timedelta(days=1)
    return last

# --- PATCH: Robust normalization and live runner (single version) ---
def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default


def _coerce_dict(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
    return None


def _iter_option_rows(payload: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    stack: List[Any] = [payload]
    seen_ids: set[int] = set()
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            parsed = _coerce_dict(item)
            if parsed is not None:
                stack.append(parsed)
            continue
        if isinstance(item, list):
            stack.extend(item)
            continue
        if isinstance(item, dict):
            obj_id = id(item)
            if obj_id in seen_ids:
                continue
            seen_ids.add(obj_id)
            has_option_fields = any(k in item for k in ("securityId", "security_id", "strike", "strikePrice", "optionType", "type", "token"))
            if has_option_fields:
                rows.append(item)
            stack.extend(item.values())
    return rows

def _normalize_positions(rows: Any) -> List[Dict[str, Any]]:
    """
    Accept whatever the broker wrapper returned and normalize to a list of dicts.
    Silently drop strings/None/etc to avoid 'str' has no attribute 'get'.
    """
    out: List[Dict[str, Any]] = []

    # JSON string? parse once
    if isinstance(rows, str):
        try:
            rows = json.loads(rows)
        except Exception:
            LOG.warning("[Positions] got a string, not JSON; dropping: %r", rows[:120])
            return out

    # Common envelopes
    if isinstance(rows, dict):
        rows = (
            rows.get("data")
            or rows.get("positions")
            or rows.get("netPositions")
            or rows.get("net_positions")
            or rows.get("result")
            or []
        )

    if not isinstance(rows, list):
        LOG.warning("[Positions] unexpected container type: %s", type(rows).__name__)
        return out

    bad = 0
    for r in rows:
        if not isinstance(r, dict):
            if bad < 3:
                LOG.warning("[Positions] dropped non-dict row: %r", r)
                bad += 1
            continue

        symbol  = r.get("symbol") or r.get("tradingSymbol") or r.get("securityName")
        product = r.get("product") or r.get("productType")
        qty     = r.get("qty") or r.get("netQty") or r.get("netQuantity") or 0
        try:
            qty = int(float(qty))
        except Exception:
            qty = 0

        avg_price = _safe_float(r.get("avg_price") or r.get("costPrice"))

        side = (r.get("side") or r.get("positionType") or "").upper()
        # LTP rule you wanted:
        #   LONG  -> buyAvg
        #   SHORT -> sellAvg
        #   else  -> other fallbacks
        ltp = None
        for c in [
            (r.get("buyAvg") if side == "LONG" else None),
            (r.get("sellAvg") if side == "SHORT" else None),
            r.get("ltp"), r.get("last_price"), r.get("LTP")
        ]:
            ltp = _safe_float(c, None)
            if ltp is not None:
                break

        pnl = _safe_float(r.get("pnl"))

        out.append({
            "symbol": symbol,
            "product": product,
            "qty": qty,
            "avg_price": avg_price,
            "ltp": ltp,
            "side": side,
            "pnl": pnl,
            "_raw": r,
        })

    LOG.info("[Positions] normalized %d rows (with side & pnl rule)", len(out))
    return out

def _safe_ltp(opt: Dict[str, Any]) -> float:
    if not isinstance(opt, dict):
        return 0.0
    for k in ("last_price", "close", "ltp", "LTP", "LastTradedPrice", "lastTradedPrice"):
        if k in opt and opt[k] is not None:
            v = _safe_float(opt[k], None)
            if v is not None:
                return v
    return 0.0

def _safe_delta(opt: Dict[str, Any]) -> Optional[float]:
    if not isinstance(opt, dict):
        return None
    for container in (opt.get("greeks"), opt.get("Greeks"), opt):
        if isinstance(container, dict):
            for k in ("delta", "Delta", "optionDelta", "deltaValue"):
                if container.get(k) is not None:
                    return _safe_float(container[k], None)
    return None

def _round_to(lot_size: int, multiple: int) -> int:
    multiple = max(1, int(multiple))
    return max(lot_size, ((lot_size + multiple - 1) // multiple) * multiple)

def _infer_tick(strikes: List[float], default: float = 50.0) -> float:
    diffs = sorted({round(b - a, 6) for a, b in zip(sorted(strikes), sorted(strikes)[1:]) if b - a > 0})
    for d in diffs:
        if d > 0:
            return d
    return default

# Snap a computed target strike to the nearest AVAILABLE OTM strike in the OC snapshot
def _snap_available(oc: dict, spot: float, target: float, side: str) -> Tuple[Optional[float], dict]:
    try:
        strikes = sorted([float(k) for k in oc.keys()])
    except Exception:
        return None, {}
    if side == "CE":
        # nearest OTM ≥ max(spot, target)
        barrier = max(spot, target)
        cands = [K for K in strikes if K >= barrier]
        if not cands:
            return None, {}
        K = cands[0]
        return K, oc.get(str(K), {}).get("ce", {})
    else:  # PE
        # nearest OTM ≤ min(spot, target)
        barrier = min(spot, target)
        cands = [K for K in strikes if K <= barrier]
        if not cands:
            return None, {}
        K = cands[-1]
        return K, oc.get(str(K), {}).get("pe", {})

# ----------------- dataclasses -----------------
@dataclass
class TradeAction:
    date: str
    action: str   # SELL_CE, SELL_PE, BUY_TO_CLOSE, ROLL_CE, ROLL_PE
    strike: float
    expiry: str
    premium: float
    qty: int
    meta: Dict[str, Any]

@dataclass
class OpenPosition:
    month_key: str
    entry_date: date
    expiry: date
    ce_strike: float
    pe_strike: float
    ce_prem: float
    pe_prem: float
    qty: int
    net_credit: float
    is_closed: bool = False

# ----------------- strategy -----------------
class MonthlyStrangleWithWeeklyHedge:
    """
    Enter NEXT-MONTH strangle on THIS-MONTH's last Tuesday (backtest schedule).
    Selection: OTM legs with Δ≥target (default 0.15). If observed deltas remain too high near ATM,
    we force distance iteratively (±2.5%→3.0%→3.5%→4.0%), snapping to the nearest AVAILABLE OTM
    strike in the rolling snapshot. 0-LTP is nudged further OTM; one-time refetch before skipping.
    """
    DEFAULTS = {
        # selection
        "delta_target": 0.15,
        "underlying_id": 13,            # NIFTY: 13 on DHAN (as you've been using)
        "underlying_seg": "NSE_FNO",
        "lot_size": 50,
        "round_multiple": 50,

        # risk / exits
        "profit_target_pct": 0.50,      # take profit when ~50% credit captured (after half DTE and spot between wings)
        "combined_stop_loss_pct": 1.50, # soft combined SL proxy ~150% of entry credit
        "breach_buffer_pts": 100,       # buffer for roll triggers
        "roll_distance_pts": 200,       # single roll distance per side
        "max_rolls": 1,

        "capital": 500_000,
    }

    def __init__(self, cfg: Dict[str, Any], market):
        self.cfg = {**self.DEFAULTS, **(cfg or {})}
        self.market = market
        self.trades: List[TradeAction] = []
        self.positions: Dict[str, OpenPosition] = {}

        oc_cfg = self.cfg.get("option_chain", {})
        underlying_id = int(oc_cfg.get("underlying_id", self.cfg.get("underlying_id", 13)))
        underlying_seg = oc_cfg.get("underlying_seg", self.cfg.get("underlying_seg", "IDX_I"))
        try:
            self.chain_ingestor = OptionChainIngestor(ChainConfig(underlying_id, underlying_seg))
        except Exception as exc:
            LOG.warning("OptionChainIngestor unavailable: %s", exc)
            self.chain_ingestor = None

    # ---------- entry helpers ----------
    def _enter_on_expiry_for_next_month(self, entry_day: date, sell_expiry: date, spot_hint: Optional[float]) -> None:
        if not getattr(self, "chain_ingestor", None) and not hasattr(self.market, "get_option_chain"):
            raise RuntimeError("No option-chain source available (OptionChainIngestor or market adapter).")

        uid = int(self.cfg["underlying_id"])
        qty = _round_to(int(self.cfg["lot_size"]), int(self.cfg["round_multiple"]))
        expiry_str = sell_expiry.isoformat()

        LOG.info("Fetch chain as_of=%s for sell_expiry=%s (expect next-month)", entry_day, sell_expiry)

        oc: Dict[str, Dict[str, dict]] = {}
        df_chain = None
        ingestor = getattr(self, "chain_ingestor", None)
        if ingestor is not None:
            try:
                df_chain = ingestor.fetch(expiry_str)
            except Exception as exc:
                LOG.warning("OptionChainIngestor fetch failed for %s: %s", expiry_str, exc)
            if df_chain is not None and not df_chain.empty:
                if spot_hint is None and "underlying_ltp" in df_chain.columns:
                    vals = df_chain["underlying_ltp"].dropna()
                    if not vals.empty:
                        spot_hint = _safe_float(vals.iloc[0], spot_hint)
                for _, row in df_chain.iterrows():
                    strike_val = row.get("strike")
                    if strike_val is None or pd.isna(strike_val):
                        continue
                    ce_raw = row.get("raw_call") or {}
                    pe_raw = row.get("raw_put") or {}
                    ce_dict = dict(ce_raw) if isinstance(ce_raw, dict) else {}
                    pe_dict = dict(pe_raw) if isinstance(pe_raw, dict) else {}
                    uv = row.get("underlying_ltp")
                    if uv is not None:
                        ce_dict.setdefault("spot", uv)
                        pe_dict.setdefault("spot", uv)
                    oc[str(strike_val)] = {"ce": ce_dict, "pe": pe_dict}

        if not oc:
            try:
                oc = self.market.get_option_chain(uid, expiry_str, as_of_date=entry_day.isoformat())
            except TypeError:
                oc = self.market.get_option_chain(uid, expiry_str)
            except Exception as e:
                LOG.warning("OC fetch failed on entry %s (expiry %s): %s", entry_day, expiry_str, e)
                return

        # Resolve spot (prefer DF hint; else any spot in OC rows)
        def _spot_from_chain(ocha: dict) -> Optional[float]:
            for _, legs in (ocha or {}).items():
                for leg in ("ce", "pe"):
                    v = legs.get(leg) or {}
                    if not isinstance(v, dict):
                        continue
                    for key in ("spot", "underlying_ltp", "underlyingValue", "underlying_value", "underlyingPrice"):
                        if v.get(key) is not None:
                            s = _safe_float(v.get(key), None)
                            if s is not None:
                                return s
            return None
        spot = spot_hint if spot_hint is not None else _spot_from_chain(oc)
        if spot is None:
            LOG.warning("No spot on %s; skip month to avoid ITM mishaps.", entry_day)
            return

        # Tick detection for strike rounding
        all_strikes = [float(k) for k in (oc or {}).keys() if _safe_float(k) is not None]
        tick = _infer_tick(all_strikes, 50.0)

        # ---------- OTM + delta-thresholded selection ----------
        target = float(self.cfg.get("delta_target", 0.15))

        ce_cands, pe_cands = [], []
        for k, legs in (oc or {}).items():
            K = _safe_float(k, None)
            if K is None:
                continue
            ce_leg = legs.get("ce") or {}
            pe_leg = legs.get("pe") or {}
            ce_d = _safe_delta(ce_leg)
            pe_d = _safe_delta(pe_leg)
            ce_p_tmp = _safe_ltp(ce_leg)
            pe_p_tmp = _safe_ltp(pe_leg)

            # OTM filters
            if K >= spot and ce_p_tmp > 0 and ce_d is not None:
                ce_cands.append((K, ce_leg, ce_d, ce_p_tmp))
            if K <= spot and pe_p_tmp > 0 and pe_d is not None:
                pe_cands.append((K, pe_leg, pe_d, pe_p_tmp))

        def pick_ce(cands, thr):
            if not cands:
                return None, None
            hard = [c for c in cands if c[2] >= thr]
            if hard:
                hard.sort(key=lambda x: (x[2], x[0]))  # smallest Δ above threshold
                return (hard[0][0], hard[0][1])
            # soft fallback: nearest |Δ - thr|
            soft = sorted(cands, key=lambda x: (abs(x[2] - thr), x[0]))
            return (soft[0][0], soft[0][1]) if soft else (None, None)

        def pick_pe(cands, thr):
            if not cands:
                return None, None
            hard = [c for c in cands if c[2] <= -thr]
            if hard:
                hard.sort(key=lambda x: (abs(abs(x[2]) - thr), -x[0]))  # |Δ| nearest to thr, further OTM preferred
                return (hard[0][0], hard[0][1])
            # soft fallback
            soft = sorted(cands, key=lambda x: (abs(abs(x[2]) - thr), -x[0]))
            return (soft[0][0], soft[0][1]) if soft else (None, None)

        ce_strike, ce = pick_ce(ce_cands, target)
        pe_strike, pe = pick_pe(pe_cands, target)

        # If a side is missing, use ±2.5% distance and SNAP to available keys
        if ce_strike is None or pe_strike is None:
            otp = 0.025
            raw_ce = math.ceil((spot * (1 + otp)) / tick) * tick
            raw_pe = math.floor((spot * (1 - otp)) / tick) * tick
            if ce_strike is None:
                ce_strike, ce = _snap_available(oc, spot, raw_ce, "CE")
            if pe_strike is None:
                pe_strike, pe = _snap_available(oc, spot, raw_pe, "PE")

        # Avoid accidental straddle
        if ce_strike is not None and pe_strike is not None and ce_strike == pe_strike:
            ce_strike += tick
            ce = oc.get(str(ce_strike), {}).get("ce", {})

        # Prices
        ce_leg = ce if isinstance(ce, dict) else {}
        pe_leg = pe if isinstance(pe, dict) else {}
        ce_p = _safe_ltp(ce_leg)
        pe_p = _safe_ltp(pe_leg)

        # If a leg has 0 LTP, nudge further OTM on that side (several steps)
        def _nudge_nonzero(strike: Optional[float], side: str, steps: int = 10):
            if strike is None:
                return None, {}, 0.0
            if side == "CE":
                s = strike
                for _ in range(steps):
                    leg = oc.get(str(s), {}).get("ce", {})
                    px = _safe_ltp(leg)
                    if px > 0:
                        return s, leg, px
                    s += tick
            else:
                s = strike
                for _ in range(steps):
                    leg = oc.get(str(s), {}).get("pe", {})
                    px = _safe_ltp(leg)
                    if px > 0:
                        return s, leg, px
                    s -= tick
            return strike, oc.get(str(strike), {}).get("ce" if side == "CE" else "pe", {}), 0.0

        if ce_p <= 0.0:
            ce_strike, ce, ce_p = _nudge_nonzero(ce_strike, "CE")
            ce_leg = ce if isinstance(ce, dict) else {}
        if pe_p <= 0.0:
            pe_strike, pe, pe_p = _nudge_nonzero(pe_strike, "PE")
            pe_leg = pe if isinstance(pe, dict) else {}

        # If still invalid after nudging, one-time refetch (adapter widens automatically), then snap again
        if (ce_p <= 0.0 or pe_p <= 0.0) and hasattr(self.market, "get_option_chain"):
            try:
                oc_wide = self.market.get_option_chain(uid, expiry_str, as_of_date=entry_day.isoformat())
                if ce_p <= 0.0 and ce_strike is not None:
                    ce_strike, ce = _snap_available(oc_wide, spot, ce_strike, "CE")
                    ce_leg = ce if isinstance(ce, dict) else {}
                    ce_p = _safe_ltp(ce_leg)
                if pe_p <= 0.0 and pe_strike is not None:
                    pe_strike, pe = _snap_available(oc_wide, spot, pe_strike, "PE")
                    pe_leg = pe if isinstance(pe, dict) else {}
                    pe_p = _safe_ltp(pe_leg)
            except Exception:
                pass

        # If still invalid, abort safely
        if ce_p <= 0.0 or pe_p <= 0.0 or ce_strike is None or pe_strike is None:
            LOG.warning("Invalid quotes on %s (spot=%.2f): CE %s @ %.2f | PE %s @ %.2f; skip.",
                        entry_day, spot, ce_strike, ce_p, pe_strike, pe_p)
            return

        # Ensure downstream helpers see dict inputs
        ce_leg: Dict[str, Any] = ce if isinstance(ce, dict) else {}
        pe_leg: Dict[str, Any] = pe if isinstance(pe, dict) else {}

        # Read back deltas
        ce_delta = _safe_delta(ce_leg)
        pe_delta = _safe_delta(pe_leg)

        # If chosen deltas are still too high (near ATM), force distance iteratively and SNAP
        def _forced_delta_cycle(target_otp_list: List[float]):
            nonlocal ce_strike, ce, ce_p, ce_delta, pe_strike, pe, pe_p, pe_delta
            for otp in target_otp_list:
                changed = False
                if ce_delta is None or ce_delta > 0.20:
                    ce_target = math.ceil((spot * (1 + otp)) / tick) * tick
                    sK, sLeg = _snap_available(oc, spot, ce_target, "CE")
                    if sK is not None:
                        ce_strike, ce = sK, sLeg
                        ce_leg = ce if isinstance(ce, dict) else {}
                        ce_p = _safe_ltp(ce_leg)
                        ce_delta = _safe_delta(ce_leg)
                        changed = True
                if pe_delta is None or abs(pe_delta) > 0.20:
                    pe_target = math.floor((spot * (1 - otp)) / tick) * tick
                    sK, sLeg = _snap_available(oc, spot, pe_target, "PE")
                    if sK is not None:
                        pe_strike, pe = sK, sLeg
                        pe_leg = pe if isinstance(pe, dict) else {}
                        pe_p = _safe_ltp(pe_leg)
                        pe_delta = _safe_delta(pe_leg)
                        changed = True
                if not changed:
                    break  # nothing more to do

        _forced_delta_cycle([0.025, 0.030, 0.035, 0.040])

        LOG.info(
            "Entry sell_expiry=%s | spot=%.2f | CE %.0f @ %.2f (Δ=%s) | PE %.0f @ %.2f (Δ=%s)",
            expiry_str, spot,
            ce_strike, ce_p, f"{ce_delta:.3f}" if ce_delta is not None else "NA",
            pe_strike, pe_p, f"{pe_delta:.3f}" if pe_delta is not None else "NA"
        )

        # Record entries
        qty = int(qty)
        self.trades.append(TradeAction(entry_day.isoformat(), "SELL_CE", float(ce_strike), expiry_str, float(ce_p), qty,
                                       {"why": "entry", "delta": ce_delta, "spot": spot}))
        self.trades.append(TradeAction(entry_day.isoformat(), "SELL_PE", float(pe_strike), expiry_str, float(pe_p), qty,
                                       {"why": "entry", "delta": pe_delta, "spot": spot}))

        key = f"{sell_expiry.year}-{sell_expiry.month:02d}"
        net_credit = (ce_p + pe_p) * qty
        self.positions[key] = OpenPosition(
            month_key=key, entry_date=entry_day, expiry=sell_expiry,
            ce_strike=float(ce_strike), pe_strike=float(pe_strike),
            ce_prem=float(ce_p), pe_prem=float(pe_p),
            qty=qty, net_credit=float(net_credit)
        )
        LOG.info("Entered cycle (sell_expiry=%s) | CE %.0f @ %.2f, PE %.0f @ %.2f, net_credit=%.2f",
                 expiry_str, ce_strike, ce_p, pe_strike, pe_p, net_credit)

    # ---------- monitoring / adjustments ----------
    def _monitor_and_adjust(self, d: date, spot: Optional[float], pos: OpenPosition) -> None:
        if pos.is_closed or spot is None:
            return
        pt = float(self.cfg["profit_target_pct"])
        sl = float(self.cfg["combined_stop_loss_pct"])
        buffer_pts = float(self.cfg["breach_buffer_pts"])
        roll_dist = float(self.cfg["roll_distance_pts"])
        max_rolls = int(self.cfg["max_rolls"])

        entry_credit_per_lot = pos.ce_prem + pos.pe_prem
        days_total = max(1, (pos.expiry - pos.entry_date).days)
        days_passed = max(0, (d - pos.entry_date).days)

        # Profit booking (only when spot is comfortably between wings and half the cycle is done)
        wide_ok = (spot < pos.ce_strike - buffer_pts) and (spot > pos.pe_strike + buffer_pts)
        if wide_ok and days_passed >= days_total * 0.5:
            self.trades.append(TradeAction(d.isoformat(), "BUY_TO_CLOSE", pos.ce_strike, pos.expiry.isoformat(),
                                           pos.ce_prem * (1 - pt), pos.qty, {"leg": "CE", "why": "PT", "spot": spot}))
            self.trades.append(TradeAction(d.isoformat(), "BUY_TO_CLOSE", pos.pe_strike, pos.expiry.isoformat(),
                                           pos.pe_prem * (1 - pt), pos.qty, {"leg": "PE", "why": "PT", "spot": spot}))
            pos.is_closed = True
            LOG.info("Booked profit for %s on %s ~%.0f%%", pos.month_key, d, pt * 100)
            return

        # Rolls (single per cycle)
        rolls_done = len([t for t in self.trades
                          if t.meta.get("why") == "roll"
                          and t.date >= pos.entry_date.isoformat()
                          and t.expiry == pos.expiry.isoformat()])
        if rolls_done < max_rolls:
            if spot >= pos.ce_strike + buffer_pts:
                prev = pos.ce_strike
                self.trades.append(TradeAction(d.isoformat(), "BUY_TO_CLOSE", pos.ce_strike, pos.expiry.isoformat(),
                                               pos.ce_prem * 1.2, pos.qty, {"leg": "CE", "why": "roll_close", "spot": spot}))
                new_ce = pos.ce_strike + roll_dist
                self.trades.append(TradeAction(d.isoformat(), "ROLL_CE", new_ce, pos.expiry.isoformat(),
                                               pos.ce_prem * 0.6, pos.qty, {"why": "roll", "leg": "CE",
                                                                            "prev_strike": prev, "new_strike": new_ce, "spot": spot}))
                pos.ce_strike = new_ce
                LOG.info("Roll %s | CE: %s → %s | spot=%.2f", d, int(prev), int(new_ce), spot)
            elif spot <= pos.pe_strike - buffer_pts:
                prev = pos.pe_strike
                self.trades.append(TradeAction(d.isoformat(), "BUY_TO_CLOSE", pos.pe_strike, pos.expiry.isoformat(),
                                               pos.pe_prem * 1.2, pos.qty, {"leg": "PE", "why": "roll_close", "spot": spot}))
                new_pe = pos.pe_strike - roll_dist
                self.trades.append(TradeAction(d.isoformat(), "ROLL_PE", new_pe, pos.expiry.isoformat(),
                                               pos.pe_prem * 0.6, pos.qty, {"why": "roll", "leg": "PE",
                                                                            "prev_strike": prev, "new_strike": new_pe, "spot": spot}))
                pos.pe_strike = new_pe
                LOG.info("Roll %s | PE: %s → %s | spot=%.2f", d, int(prev), int(new_pe), spot)

        # Soft combined SL proxy (very approximate — improves with live Greeks)
        near_ce = spot > pos.ce_strike - (buffer_pts * 0.25)
        near_pe = spot < pos.pe_strike + (buffer_pts * 0.25)
        est_combined = entry_credit_per_lot
        if near_ce: est_combined *= 1.3
        if near_pe: est_combined *= 1.3
        if (est_combined / (entry_credit_per_lot + 1e-9)) >= sl:
            self.trades.append(TradeAction(d.isoformat(), "BUY_TO_CLOSE", pos.ce_strike, pos.expiry.isoformat(),
                                           pos.ce_prem * 1.5, pos.qty, {"leg": "CE", "why": "SL", "spot": spot}))
            self.trades.append(TradeAction(d.isoformat(), "BUY_TO_CLOSE", pos.pe_strike, pos.expiry.isoformat(),
                                           pos.pe_prem * 1.5, pos.qty, {"leg": "PE", "why": "SL", "spot": spot}))
            pos.is_closed = True
            LOG.info("Stopped out %s on %s (combined SL %.0f%%)", pos.month_key, d, sl * 100)

    # ---------- main ----------
    def run(self, df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        if cfg:
            self.cfg.update(cfg)
        if not hasattr(self.market, "get_option_chain"):
            raise RuntimeError("Market adapter missing .get_option_chain().")
        if "date" not in df.columns:
            raise RuntimeError("Input df must have a 'date' column")

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.date
        have_spot = "close" in df.columns
        trading_days = sorted(df["date"].unique().tolist())
        if not trading_days:
            return pd.DataFrame(), pd.DataFrame(), {
                "monthly_cycles": 0, "n_trades": 0, "underlying_id": int(self.cfg["underlying_id"]),
                "entry_credit_value": 0.0, "exit_value": 0.0, "net_credit": 0.0, "note": "no data"
            }

        # Plan entries: for each month in dataset, enter on this month's last Tuesday; sell next monthly
        months = []
        first_d, last_d = min(trading_days), max(trading_days)
        y, m = first_d.year, first_d.month
        while date(y, m, 1) <= date(last_d.year, last_d.month, 1):
            months.append((y, m))
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)

        entries_done = 0
        for (y, m) in months:
            entry_day = _last_tuesday_of_month(y, m)
            if entry_day not in trading_days:
                continue
            ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
            sell_expiry = _last_tuesday_of_month(ny, nm)

            spot_hint = None
            if have_spot:
                r = df[df["date"] == entry_day]
                if not r.empty:
                    spot_hint = _safe_float(r.iloc[0]["close"], None)

            self._enter_on_expiry_for_next_month(entry_day, sell_expiry, spot_hint)
            if f"{sell_expiry.year}-{sell_expiry.month:02d}" in self.positions:
                entries_done += 1

        LOG.info("Monthly entries placed (rule = OTM + Δ≥%.2f): %d", float(self.cfg["delta_target"]), entries_done)

        # Monitor daily; auto-expiry close
        for d in trading_days:
            for _, pos in list(self.positions.items()):
                if pos.is_closed or d < pos.entry_date or d > pos.expiry:
                    continue
                spot = None
                if have_spot:
                    r = df[df["date"] == d]
                    if not r.empty:
                        spot = _safe_float(r.iloc[0]["close"], None)
                self._monitor_and_adjust(d, spot, pos)
                if d == pos.expiry and not pos.is_closed:
                    # expiry close proxy (very coarse)
                    between = (spot is not None) and (pos.pe_strike < spot < pos.ce_strike)
                    ce_exit = 0.05 * pos.ce_prem if between else 1.5 * pos.ce_prem
                    pe_exit = 0.05 * pos.pe_prem if between else 1.5 * pos.pe_prem
                    self.trades.append(TradeAction(d.isoformat(), "BUY_TO_CLOSE", pos.ce_strike, pos.expiry.isoformat(),
                                                   ce_exit, pos.qty, {"leg": "CE", "why": "expiry", "spot": spot}))
                    self.trades.append(TradeAction(d.isoformat(), "BUY_TO_CLOSE", pos.pe_strike, pos.expiry.isoformat(),
                                                   pe_exit, pos.qty, {"leg": "PE", "why": "expiry", "spot": spot}))
                    pos.is_closed = True

        trades_df = pd.DataFrame([asdict(t) for t in self.trades])
        if trades_df.empty:
            return trades_df, pd.DataFrame(), {
                "monthly_cycles": 0, "n_trades": 0, "underlying_id": int(self.cfg["underlying_id"]),
                "entry_credit_value": 0.0, "exit_value": 0.0, "net_credit": 0.0,
                "note": "No entries; check market adapter & rolling feed."
            }

        # Cashflow PnL
        entry_mask = trades_df["action"].isin(["SELL_CE", "SELL_PE"])
        exit_mask  = trades_df["action"].isin(["BUY_TO_CLOSE", "ROLL_CE", "ROLL_PE"])
        entry_value = (trades_df.loc[entry_mask, "premium"].astype(float) * trades_df.loc[entry_mask, "qty"].astype(float)).sum()
        exit_value  = (trades_df.loc[exit_mask,  "premium"].astype(float) * trades_df.loc[exit_mask,  "qty"].astype(float)).sum()
        net_credit  = float(entry_value - exit_value)

        summary = {
            "monthly_cycles": entries_done,
            "n_trades": int(len(trades_df)),
            "underlying_id": int(self.cfg["underlying_id"]),
            "entry_credit_value": round(float(entry_value), 2),
            "exit_value": round(float(exit_value), 2),
            "net_credit": round(net_credit, 2),
            "note": "OTM + Δ≥target; next-month strangle; daily monitor with profit/SL/roll; snap-to-available; iterative force-distance; 0-LTP nudged.",
        }
        return trades_df, pd.DataFrame(), summary

# adapter entrypoints
def run_backtest(df: pd.DataFrame, cfg: Dict[str, Any], market: Optional[Any] = None):
    strat = MonthlyStrangleWithWeeklyHedge(cfg, market or cfg.get("market"))
    return strat.run(df, cfg)

def run_strategy(df: pd.DataFrame, cfg: Dict[str, Any], market: Optional[Any] = None):
    return run_backtest(df, cfg, market=market)


# ===== Enhanced live orchestration layer =====
NIFTY_SEG = "IDX_I"
NIFTY_SCRIP = 13
NIFTY_SYMBOL = "NIFTY"
NIFTY_FNO_SEG = "NSE_FNO"


@dataclass
class BatmanConfig:
    enabled: bool = True
    delta_breach: float = 0.30
    premium_hard_x: float = 2.0
    roll_distance: float = 150.0
    hedge_delta_max: float = 0.12
    hedge_price_max: float = 35.0
    salvage_wing_ltp: float = 5.0
    hedge_qty_multiplier: float = 1.0


@dataclass
class BatmanStructure:
    expiry: str
    exchange_seg: str
    short_call: Dict[str, Any]
    short_call_strike: float
    short_put: Dict[str, Any]
    short_put_strike: float
    long_call: Optional[Dict[str, Any]] = None
    long_call_strike: Optional[float] = None
    long_put: Optional[Dict[str, Any]] = None
    long_put_strike: Optional[float] = None

    @property
    def lot_size(self) -> int:
        try:
            return abs(int(float(self.short_call.get("netQty") or 0)))
        except Exception:
            return 0


@dataclass
class LiveConfig:
    max_strangles: int = 1
    lot_size: int = 75
    sl_pct: float = 0.025
    tp_pct: float = 0.0225
    hedge_enable: bool = True
    hedge_price_min: float = 2.5
    hedge_price_max: float = 3.5
    poll_sec: float = 5.0
    delta_breach: float = 0.38
    premium_soft_x: float = 1.8
    premium_hard_x: float = 2.2
    roll_distance: float = 200.0
    warn_only: bool = False
    trade_mode: str = "live"
    blotter_path: Optional[str] = None
    blotter_summary_path: Optional[str] = None
    batman: BatmanConfig = field(default_factory=BatmanConfig)
    feature_log_path: Optional[str] = None
    enable_strangle_entry: bool = True
    regime_refresh_minutes: float = 5.0
    auto_disable_strangle_when_unfavored: bool = True
    last_spot: Optional[float] = field(default=None, repr=False, compare=False)
    _last_spot_warn_at: Optional[datetime] = field(default=None, repr=False, compare=False)



def _series_to_dict(series: pd.Series) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for key, val in series.items():
        if isinstance(val, float):
            try:
                if math.isnan(val):
                    continue
            except Exception:
                pass
        data[str(key)] = val
    return data


def _extract_security_id(raw_leg: Dict[str, Any]) -> Optional[int]:
    if not isinstance(raw_leg, dict):
        return None

    def _clean(val: Any) -> Optional[int]:
        if val in (None, "", "NA"):
            return None
        try:
            if isinstance(val, float) and math.isnan(val):
                return None
        except Exception:
            pass
        try:
            return int(str(val).strip())
        except Exception:
            return None

    for key in ("securityId", "securityID", "security_id", "token", "id", "symbolToken", "instrumentToken"):
        candidate = _clean(raw_leg.get(key))
        if candidate is not None:
            return candidate
    return None


def _resolve_security_id(chain_row: Dict[str, Any], opt_type: str) -> Optional[int]:
    if not isinstance(chain_row, dict):
        return None
    sources: List[Dict[str, Any]] = []
    raw = chain_row.get("raw_call") if opt_type == "CALL" else chain_row.get("raw_put")
    if isinstance(raw, dict):
        sources.append(raw)
    elif isinstance(raw, str):
        try:
            decoded = json.loads(raw)
            if isinstance(decoded, dict):
                sources.append(decoded)
        except Exception:
            pass
    sources.append(chain_row)
    for source in sources:
        sec_id = _extract_security_id(source)
        if sec_id is not None:
            return sec_id
    return None


def _select_replacement(
    df: Optional[pd.DataFrame],
    target_strike: float,
    opt_type: str,
    direction: str,
) -> Optional[Dict[str, Any]]:
    if df is None or df.empty or "strike" not in df.columns:
        return None
    working_df = cast(pd.DataFrame, df)
    if opt_type == "CALL":
        candidates = working_df.loc[working_df["strike"] >= target_strike]
        if candidates.empty:
            candidates = working_df.sort_values(by="strike", ascending=False)
        else:
            candidates = candidates.sort_values(by="strike")
    else:
        candidates = working_df.loc[working_df["strike"] <= target_strike]
        if candidates.empty:
            candidates = working_df.sort_values(by="strike")
        else:
            candidates = candidates.sort_values(by="strike", ascending=False)

    for _, row in candidates.iterrows():
        price_key = "ltp_ce" if opt_type == "CALL" else "ltp_pe"
        price_val = _safe_float(row.get(price_key), None)
        if price_val is None or price_val <= 0:
            continue
        cleaned = _series_to_dict(row)
        cleaned[price_key] = price_val
        return cleaned
    return None


def _submit_order(
    dw,
    cfg: LiveConfig,
    *,
    side: str,
    exchange_seg: str,
    security_id: int,
    quantity: int,
    product_type: str,
    order_type: str = "MARKET",
    price: Optional[float] = None,
    context: Optional[Dict[str, Any]] = None,
) -> bool:
    if quantity <= 0:
        return False

    context = context or {}
    price_val: Optional[float]
    try:
        price_val = float(price) if price is not None else None
    except Exception:
        price_val = None

    placed = False
    warn_only = bool(getattr(cfg, "warn_only", False))
    try:
        if warn_only:
            LOG.info(
                "[live][warn-only] would place order side=%s seg=%s security_id=%s qty=%s product=%s order_type=%s",
                side,
                exchange_seg,
                security_id,
                quantity,
                product_type,
                order_type,
            )
        else:
            dw.place_order(
                side=side,
                exchange_seg=exchange_seg,
                security_id=security_id,
                quantity=quantity,
                product_type=product_type,
                order_type=order_type,
            )
            placed = True
    except Exception:
        LOG.exception("[live] place_order failed (side=%s, sec_id=%s)", side, security_id)
        raise
    finally:
        event = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "trade_mode": getattr(cfg, "trade_mode", "live"),
            "warn_only": 1 if warn_only else 0,
            "executed": 1 if placed else 0,
            "side": side,
            "order_type": order_type,
            "exchange_seg": exchange_seg,
            "product_type": product_type,
            "security_id": security_id,
            "quantity": int(quantity),
            "price": price_val if price_val is not None else "",
            "strike": context.get("strike", ""),
            "tag": context.get("tag") or context.get("action") or context.get("reason") or "",
            "notes": json.dumps(context, default=str) if context else "",
        }
        _record_trade(cfg, event)

    return placed


def _place_option_order(
    dw,
    cfg: LiveConfig,
    side: str,
    lot_qty: int,
    chain_row: Dict[str, Any],
    opt_type: str,
    product: str = "MARGIN",
    context: Optional[Dict[str, Any]] = None,
) -> None:
    qty = int(abs(lot_qty))
    if qty <= 0:
        return
    sec_id = _resolve_security_id(chain_row, opt_type)
    if sec_id is None:
        raise RuntimeError(f"security id missing in option chain row for {opt_type} ({chain_row.get('strike')})")

    strike = chain_row.get("strike")
    price_key = "ltp_ce" if opt_type == "CALL" else "ltp_pe"
    ltp = _safe_float(chain_row.get(price_key), None)
    LOG.info("[live] order %s %s strike=%s qty=%s ltp=%s product=%s", side, opt_type, strike, qty, ltp, product)
    ctx: Dict[str, Any] = dict(context or {})
    if strike is not None:
        ctx.setdefault("strike", strike)
    try:
        placed = _submit_order(
            dw,
            cfg,
            side=side,
            exchange_seg=NIFTY_FNO_SEG,
            security_id=sec_id,
            quantity=qty,
            product_type=product,
            order_type="MARKET",
            price=ltp,
            context=ctx,
        )
        if not placed:
            LOG.debug("[live] option order suppressed (warn_only=%s)", cfg.warn_only)
    except Exception:
        LOG.exception("[live] place_order failed (side=%s, opt=%s, strike=%s, sec_id=%s)", side, opt_type, strike, sec_id)
        raise


def _close_leg(dw, cfg: LiveConfig, row: Dict[str, Any]) -> None:
    seg = row.get("exchangeSegment") or row.get("segment") or NIFTY_FNO_SEG
    qty = int(abs(float(row.get("netQty") or row.get("qty") or 0)))
    if qty <= 0:
        return
    sec_id = _extract_security_id(row)
    if sec_id is None:
        LOG.warning("[live] cannot close leg for %s: missing security id", row.get("symbol") or row.get("tradingSymbol"))
        return
    product = row.get("productType") or row.get("product") or "MARGIN"
    side = "BUY" if float(row.get("netQty") or 0) < 0 else "SELL"
    LOG.info("[live] closing leg side=%s sec_id=%s qty=%s seg=%s", side, sec_id, qty, seg)
    _, strike, _ = _parse_symbol_meta(row)
    exit_price = _safe_float(row.get("ltp") or row.get("last_price") or row.get("sellAvg") or row.get("buyAvg"), None)
    context = {
        "tag": "close_leg",
        "action": "EXIT",
        "symbol": row.get("tradingSymbol") or row.get("symbol"),
        "strike": strike,
    }
    placed = _submit_order(
        dw,
        cfg,
        side=side,
        exchange_seg=seg,
        security_id=sec_id,
        quantity=qty,
        product_type=product,
        order_type="MARKET",
        price=exit_price,
        context=context,
    )
    if not placed:
        LOG.debug("[live] close leg suppressed (warn_only=%s)", cfg.warn_only)


def _roll_short_leg(
    dw,
    cfg: LiveConfig,
    row: Dict[str, Any],
    opt_type: str,
    new_row: Dict[str, Any],
    reason: str,
) -> None:
    qty = int(abs(float(row.get("netQty") or 0)))
    if qty <= 0:
        return
    prev_strike = _safe_float(row.get("strike"), None)
    if prev_strike is None:
        prev_strike = _safe_float(row.get("drvStrikePrice"), None)
    if prev_strike is None:
        prev_strike = _safe_float(row.get("strikePrice"), None)
    _close_leg(dw, cfg, row)
    try:
        _place_option_order(
            dw,
            cfg,
            "SELL",
            qty,
            new_row,
            opt_type,
            row.get("productType") or "MARGIN",
            context={
                "tag": f"roll_{opt_type.lower()}",
                "action": "ROLL",
                "reason": reason,
            },
        )
        LOG.info(
            "[live] roll executed leg=%s prev=%s new=%s qty=%s",
            opt_type,
            prev_strike,
            new_row.get("strike"),
            qty,
        )
    except Exception as exc:
        LOG.warning("[live] roll order failed for %s: %s", opt_type, exc)


def _pnl_from_positions_row(row: Dict[str, Any]) -> float:
    qty = float(row.get("netQty") or 0)
    cost = float(row.get("costPrice") or 0)
    buy_avg = float(row.get("buyAvg") or row.get("buyAveragePrice") or 0)
    sell_avg = float(row.get("sellAvg") or row.get("sellAveragePrice") or 0)
    realized = float(row.get("realizedProfit") or 0)
    if realized:
        return realized
    if qty < 0:
        return abs(qty) * (cost - sell_avg)
    return abs(qty) * (cost - buy_avg)


def _deployed_amount(row: Dict[str, Any]) -> float:
    qty = abs(float(row.get("netQty") or 0))
    cost = float(row.get("costPrice") or 0)
    return qty * cost


def _needs_exit(row: Dict[str, Any], sl_pct: float, tp_pct: float) -> Optional[str]:
    deployed = _deployed_amount(row)
    if deployed <= 0:
        return None
    pnl = _pnl_from_positions_row(row)
    if pnl >= tp_pct * deployed:
        return "EXIT"
    if pnl <= -sl_pct * deployed:
        return "EXIT"
    return None


def _parse_symbol_meta(row: Dict[str, Any]) -> Tuple[str, Optional[float], str]:
    sym = str(row.get("tradingSymbol") or row.get("symbol") or "")
    parts = sym.split("-")
    expiry_tag = parts[1] if len(parts) > 1 else ""
    strike = None
    for token in reversed(parts):
        try:
            strike = float(token)
            break
        except Exception:
            continue
    opt_type = "CALL" if sym.endswith("CE") else "PUT" if sym.endswith("PE") else ""
    return expiry_tag, strike, opt_type


def _group_strangles(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    groups: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        expiry_tag, strike, opt_type = _parse_symbol_meta(r)
        if not opt_type or strike is None:
            continue
        bucket = groups.setdefault(expiry_tag or "unknown", {})
        bucket[opt_type] = {"row": r, "strike": strike}
    return groups


def _pick_monthly_expiry(ing: OptionChainIngestor) -> Optional[str]:
    try:
        exp_series = ing.list_expiries()
    except Exception as exc:
        LOG.warning("[live] cannot list expiries: %s", exc)
        return None
    today = date.today()
    future: List[date] = []
    for item in exp_series.dropna().tolist():
        try:
            d = datetime.strptime(str(item), "%Y-%m-%d").date()
            if d >= today:
                future.append(d)
        except Exception:
            continue
    if not future:
        return None
    future.sort()
    for d in future:
        if d.month != today.month or d.day >= 20:
            return d.isoformat()
    return future[-1].isoformat()


def _find_fno_option_by_price(
    oc_payload: Dict[str, Any],
    target_price_min: float,
    target_price_max: float,
    opt_type: str,
    above_spot: bool,
) -> Optional[Tuple[int, int]]:
    data = oc_payload.get("data", oc_payload) if isinstance(oc_payload, dict) else {}
    quotes = data.get("options") or data.get("records") or []

    chosen = None
    best_diff = 1e9
    for row in quotes:
        if not isinstance(row, dict):
            continue
        strike_val = _safe_float(row.get("strike") or row.get("strikePrice"), None)
        ltp_val = _safe_float(row.get("last_price") or row.get("LTP") or row.get("ltp") or row.get("price"), None)
        typ = str(row.get("optionType") or row.get("type") or "").upper()
        sec_id_raw = row.get("securityId") or row.get("token") or row.get("id")
        if strike_val is None or ltp_val is None or sec_id_raw is None:
            continue
        try:
            strike_int = int(strike_val)
            sec_id_int = int(sec_id_raw)
        except Exception:
            continue
        if not isinstance(sec_id_int, int):
            continue
        if target_price_min <= ltp_val <= target_price_max and typ.startswith(opt_type[0]):
            diff = -strike_int if above_spot else strike_int
            if diff < best_diff:
                chosen = (sec_id_int, strike_int)
                best_diff = diff
    return chosen


def _pick_strangle_strikes_simple(nifty_spot: float, step: int = 100) -> Tuple[int, int]:
    atm = int(round(nifty_spot / step) * step)
    return atm - step, atm + step


def _choose_monthly_expiry(payload: Dict[str, Any]) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    data = payload.get("data", payload)
    if isinstance(data, dict):
        for key in ("expiryList", "expiries", "data"):
            expiries = data.get(key)
            if isinstance(expiries, list) and expiries:
                for item in expiries:
                    if item:
                        return str(item)
        if data.get("expiry"):
            return str(data.get("expiry"))
    elif isinstance(data, list) and data:
        return str(data[0])
    return None


def _infer_default_expiry() -> Optional[str]:
    today = date.today()
    next_month = date(today.year + (1 if today.month == 12 else 0), (today.month % 12) + 1, 1)
    guess = next_month - timedelta(days=1)
    return guess.isoformat()


def _match_chain_row(df: Optional[pd.DataFrame], strike: float, opt_type: str) -> Optional[Dict[str, Any]]:
    if df is None or df.empty:
        return None
    mask = (df["strike"] - strike).abs() < 0.1
    if not mask.any():
        return None
    row = df.loc[mask].iloc[0]
    data = _series_to_dict(row)
    if opt_type == "CALL" and not data.get("raw_call"):
        data["raw_call"] = data.get("raw") if isinstance(data.get("raw"), dict) else {}
    if opt_type == "PUT" and not data.get("raw_put"):
        data["raw_put"] = data.get("raw") if isinstance(data.get("raw"), dict) else {}
    return data


def _option_qty(row: Optional[Dict[str, Any]]) -> int:
    if not row:
        return 0
    try:
        return int(round(float(row.get("netQty") or row.get("qty") or 0)))
    except Exception:
        return 0


def _identify_batman_structures(rows: List[Dict[str, Any]]) -> List[BatmanStructure]:
    groups: Dict[Tuple[str, str], List[Tuple[float, str, Dict[str, Any]]]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        seg = (r.get("exchangeSegment") or r.get("segment") or "").upper()
        if not seg.startswith("NSE_FNO"):
            continue
        expiry_tag, strike, opt_type = _parse_symbol_meta(r)
        if not opt_type or strike is None or not expiry_tag:
            continue
        groups.setdefault((seg, expiry_tag), []).append((strike, opt_type, r))

    structures: List[BatmanStructure] = []
    for (seg, expiry), items in groups.items():
        short_call = short_put = None
        short_call_strike = short_put_strike = None
        long_call = long_put = None
        long_call_strike = long_put_strike = None

        # sort calls ascending by strike, puts descending (to find nearest wings/bodies)
        calls = sorted((itm for itm in items if itm[1] == "CALL"), key=lambda x: x[0])
        puts = sorted((itm for itm in items if itm[1] == "PUT"), key=lambda x: x[0], reverse=True)

        for strike, _, row in calls:
            qty = _option_qty(row)
            if qty < 0:
                if short_call is None or short_call_strike is None or strike < short_call_strike:
                    short_call = row
                    short_call_strike = strike
            elif qty > 0:
                if long_call is None or long_call_strike is None or strike < long_call_strike:
                    long_call = row
                    long_call_strike = strike

        for strike, _, row in puts:
            qty = _option_qty(row)
            if qty < 0:
                if short_put is None or short_put_strike is None or strike > short_put_strike:
                    short_put = row
                    short_put_strike = strike
            elif qty > 0:
                if long_put is None or long_put_strike is None or strike > long_put_strike:
                    long_put = row
                    long_put_strike = strike

        if short_call and short_put and short_call_strike is not None and short_put_strike is not None:
            structures.append(
                BatmanStructure(
                    expiry=expiry,
                    exchange_seg=seg,
                    short_call=short_call,
                    short_call_strike=float(short_call_strike),
                    short_put=short_put,
                    short_put_strike=float(short_put_strike),
                    long_call=long_call,
                    long_call_strike=float(long_call_strike) if long_call_strike is not None else None,
                    long_put=long_put,
                    long_put_strike=float(long_put_strike) if long_put_strike is not None else None,
                )
            )

    return structures


def _ensure_batman_hedges(
    dw,
    cfg: LiveConfig,
    bat_cfg: BatmanConfig,
    structure: BatmanStructure,
    chain_df: pd.DataFrame,
    spot: Optional[float],
) -> None:
    if chain_df is None or chain_df.empty:
        return

    target_delta = bat_cfg.hedge_delta_max
    short_call_qty = abs(_option_qty(structure.short_call))
    short_put_qty = abs(_option_qty(structure.short_put))
    long_call_qty = max(0, _option_qty(structure.long_call))
    long_put_qty = max(0, _option_qty(structure.long_put))

    delta_ce_col = chain_df["delta_ce"] if "delta_ce" in chain_df.columns else None
    delta_pe_col = chain_df["delta_pe"] if "delta_pe" in chain_df.columns else None
    if delta_ce_col is None or delta_pe_col is None:
        LOG.debug("[batman] greeks missing in option chain snapshot; using premium triggers only")

    if short_call_qty > long_call_qty:
        subset = chain_df.loc[chain_df["strike"] > structure.short_call_strike].copy()
        if delta_ce_col is not None:
            mask = delta_ce_col.reindex(subset.index).abs() <= target_delta
            subset = subset.loc[mask]
        if spot is not None:
            subset = subset.loc[subset["strike"] >= spot]
        if not subset.empty:
            row = subset.sort_values(by=["strike"]).iloc[0]
            row_dict = _series_to_dict(row)
            price = _safe_float(row_dict.get("ltp_ce"), None)
            if price is None or price <= bat_cfg.hedge_price_max:
                qty_needed = short_call_qty - long_call_qty
                _place_option_order(
                    dw,
                    cfg,
                    "BUY",
                    qty_needed,
                    row_dict,
                    "CALL",
                    context={"tag": "batman_hedge_call", "action": "HEDGE"},
                )

    if short_put_qty > long_put_qty:
        subset = chain_df.loc[chain_df["strike"] < structure.short_put_strike].copy()
        if delta_pe_col is not None:
            mask = delta_pe_col.reindex(subset.index).abs() <= target_delta
            subset = subset.loc[mask]
        if spot is not None:
            subset = subset.loc[subset["strike"] <= spot]
        if not subset.empty:
            row = subset.sort_values(by=["strike"], ascending=False).iloc[0]
            row_dict = _series_to_dict(row)
            price = _safe_float(row_dict.get("ltp_pe"), None)
            if price is None or price <= bat_cfg.hedge_price_max:
                qty_needed = short_put_qty - long_put_qty
                _place_option_order(
                    dw,
                    cfg,
                    "BUY",
                    qty_needed,
                    row_dict,
                    "PUT",
                    context={"tag": "batman_hedge_put", "action": "HEDGE"},
                )


def monitor_batman_positions(
    dw,
    cfg: LiveConfig,
    bat_cfg: BatmanConfig,
    positions: List[Dict[str, Any]],
    ingestor: Optional[OptionChainIngestor],
) -> None:
    if not bat_cfg.enabled or not positions:
        return
    structures = _identify_batman_structures(positions)
    if not structures:
        LOG.debug("[batman] no eligible Batman structures found")
        return
    if ingestor is None:
        LOG.warning("[batman] OptionChainIngestor unavailable; monitor skipped")
        return

    chain_cache: Dict[str, Optional[pd.DataFrame]] = {}

    for structure in structures:
        expiry = structure.expiry
        if expiry not in chain_cache:
            try:
                chain_cache[expiry] = ingestor.fetch(expiry)
            except Exception as exc:
                LOG.warning("[batman] chain fetch failed for %s: %s", expiry, exc)
                chain_cache[expiry] = None
        chain_df = chain_cache.get(expiry)
        if chain_df is None or chain_df.empty:
            continue

        call_chain = _match_chain_row(chain_df, structure.short_call_strike, "CALL")
        put_chain = _match_chain_row(chain_df, structure.short_put_strike, "PUT")
        if call_chain is None or put_chain is None:
            continue

        call_ltp = _safe_float(call_chain.get("ltp_ce"), None)
        put_ltp = _safe_float(put_chain.get("ltp_pe"), None)
        call_delta = _safe_float(call_chain.get("delta_ce"), None)
        put_delta = _safe_float(put_chain.get("delta_pe"), None)
        spot = _safe_float(call_chain.get("underlying_ltp")) or _safe_float(put_chain.get("underlying_ltp"))

        cost_call = _safe_float(structure.short_call.get("costPrice"), 0.0) or 0.0
        cost_put = _safe_float(structure.short_put.get("costPrice"), 0.0) or 0.0

        if call_ltp is not None and cost_call > 0:
            ratio = call_ltp / cost_call
            if ratio >= bat_cfg.premium_hard_x or (call_delta is not None and call_delta > bat_cfg.delta_breach):
                target = structure.short_call_strike + bat_cfg.roll_distance
                replacement = _select_replacement(chain_df, target, "CALL", "UP")
                if replacement:
                    _roll_short_leg(dw, cfg, structure.short_call, "CALL", replacement, "call_roll")

        if put_ltp is not None and cost_put > 0:
            ratio = put_ltp / cost_put
            if ratio >= bat_cfg.premium_hard_x or (put_delta is not None and abs(put_delta) > bat_cfg.delta_breach):
                target = structure.short_put_strike - bat_cfg.roll_distance
                replacement = _select_replacement(chain_df, target, "PUT", "DOWN")
                if replacement:
                    _roll_short_leg(dw, cfg, structure.short_put, "PUT", replacement, "put_roll")

        _ensure_batman_hedges(dw, cfg, bat_cfg, structure, chain_df, spot)

        feature_row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "strategy": "batman",
            "expiry": structure.expiry,
            "exchange_seg": structure.exchange_seg,
            "spot": spot,
            "ce_strike": structure.short_call_strike,
            "pe_strike": structure.short_put_strike,
            "ce_ltp": call_ltp,
            "pe_ltp": put_ltp,
            "ce_delta": call_delta,
            "pe_delta": put_delta,
            "ce_entry": cost_call,
            "pe_entry": cost_put,
            "net_credit": (cost_call + cost_put),
            "warn_only": int(bool(getattr(cfg, "warn_only", False))),
            "trade_mode": getattr(cfg, "trade_mode", "live"),
            "context": json.dumps({
                "hedge_long_call_qty": _option_qty(structure.long_call),
                "hedge_long_put_qty": _option_qty(structure.long_put),
            }, default=str),
        }
        _record_feature(cfg, feature_row)

def _place_strangle_and_hedge(dw, cfg: LiveConfig, nifty_spot: float) -> None:
    try:
        picker = globals().get("pick_atm_strangle") or _pick_strangle_strikes_simple
        pe_strike, ce_strike = picker(nifty_spot)
        if pe_strike > ce_strike:
            pe_strike, ce_strike = ce_strike, pe_strike
    except Exception:
        pe_strike, ce_strike = _pick_strangle_strikes_simple(nifty_spot)

    try:
        expiry_payload = dw.get_expiry_list(NIFTY_SCRIP, NIFTY_FNO_SEG)
        expiry = _choose_monthly_expiry(expiry_payload) if isinstance(expiry_payload, dict) else None
    except Exception:
        expiry = None
    if not expiry:
        expiry = _infer_default_expiry()

    if expiry is None:
        LOG.info("[live] could not determine expiry; skip entry")
        return

    oc_raw = dw.get_option_chain(underlying_security_id=NIFTY_SCRIP, underlying_seg=NIFTY_FNO_SEG, expiry=expiry)
    oc_dict = _coerce_dict(oc_raw) or {}
    data = _coerce_dict(oc_dict.get("data")) or oc_dict

    chain_sources: List[Dict[str, Any]] = []

    def _push_source(node: Any) -> None:
        if isinstance(node, dict):
            chain_sources.append(node)

    _push_source(data)
    if isinstance(data, dict):
        inner = _coerce_dict(data.get("data"))
        _push_source(inner)
        if isinstance(inner, dict):
            _push_source(inner.get("oc"))
        _push_source(data.get("oc"))
    if not chain_sources:
        chain_sources.append(data if isinstance(data, dict) else {})

    def _find_by_strike(strike: int, typ: str) -> Optional[Tuple[int, Optional[float]]]:
        key_candidates = [str(int(strike)), str(float(strike)), int(strike)]
        for source in chain_sources:
            if not isinstance(source, dict):
                continue
            for raw_key in key_candidates:
                key = str(raw_key)
                direct = source.get(key)
                if not isinstance(direct, dict):
                    continue
                leg_key = "ce" if typ == "CALL" else "pe"
                leg_dict = _coerce_dict(direct.get(leg_key)) or {}
                if not leg_dict:
                    continue
                sec_id = _extract_security_id(leg_dict) or _extract_security_id(direct)
                if sec_id is None:
                    sec_candidate = _safe_float(
                        leg_dict.get("securityId")
                        or direct.get("securityId")
                        or direct.get("id"),
                        None,
                    )
                    if sec_candidate:
                        try:
                            sec_id = int(sec_candidate)
                        except Exception:
                            sec_id = None
                if sec_id is None:
                    sec_id = resolve_option_security_id(NIFTY_SYMBOL, expiry, strike, typ)
                price_val = None
                for price_key in ("last_price", "LTP", "ltp", "price", "lastPrice"):
                    if leg_dict.get(price_key) is not None:
                        price_val = _safe_float(leg_dict.get(price_key), None)
                        break
                if sec_id is not None:
                    return sec_id, price_val

        series = _iter_option_rows(chain_sources)
        for row_dict in series:
            row_strike_val = row_dict.get("strike", row_dict.get("strikePrice"))
            row_strike = _safe_float(row_strike_val, None)
            if row_strike is None:
                continue
            if abs(row_strike - float(strike)) > 0.1:
                continue
            opt = str(
                row_dict.get("optionType")
                or row_dict.get("type")
                or row_dict.get("optiontype")
                or ""
            ).upper()
            if not opt.startswith(typ[0]):
                continue
            sec_id = _extract_security_id(row_dict)
            if sec_id is None:
                raw_sec = (
                    row_dict.get("securityId")
                    or row_dict.get("symbol")
                    or row_dict.get("instrument")
                    or row_dict.get("symbolToken")
                    or row_dict.get("token")
                )
                sec_candidate = _safe_float(raw_sec, None)
                if sec_candidate is None:
                    sec_id = resolve_option_security_id(NIFTY_SYMBOL, expiry, strike, typ)
                else:
                    sec_id = int(sec_candidate)
            else:
                sec_id = int(sec_id)
            if sec_id is None:
                continue
            price_val = None
            for key in ("last_price", "LTP", "ltp", "price", "lastPrice"):
                if row_dict.get(key) is not None:
                    price_val = _safe_float(row_dict.get(key), None)
                    break
            return sec_id, price_val
        return None

    pe_info = _find_by_strike(pe_strike, "PUT")
    ce_info = _find_by_strike(ce_strike, "CALL")
    if not pe_info or not ce_info:
        LOG.warning("[live] could not resolve security ids for strangle entry (pe=%s ce=%s)", pe_strike, ce_strike)
        return

    pe_id, pe_price = pe_info
    ce_id, ce_price = ce_info

    qty = cfg.lot_size
    _submit_order(
        dw,
        cfg,
        side="SELL",
        exchange_seg=NIFTY_FNO_SEG,
        security_id=pe_id,
        quantity=qty,
        product_type="MARGIN",
        price=pe_price,
        context={"tag": "entry_put", "action": "ENTRY", "strike": pe_strike},
    )
    _submit_order(
        dw,
        cfg,
        side="SELL",
        exchange_seg=NIFTY_FNO_SEG,
        security_id=ce_id,
        quantity=qty,
        product_type="MARGIN",
        price=ce_price,
        context={"tag": "entry_call", "action": "ENTRY", "strike": ce_strike},
    )

    if cfg.hedge_enable:
        hedge = _find_fno_option_by_price(oc_dict, cfg.hedge_price_min, cfg.hedge_price_max, "PUT", above_spot=False)
        if hedge:
            hedge_price = None
            lookup = _find_by_strike(hedge[1], "PUT")
            if lookup:
                _, hedge_price = lookup
            _submit_order(
                dw,
                cfg,
                side="BUY",
                exchange_seg=NIFTY_FNO_SEG,
                security_id=hedge[0],
                quantity=qty,
                product_type="MARGIN",
                price=hedge_price,
                context={"tag": "hedge_put", "action": "HEDGE", "strike": hedge[1]},
            )
        hedge = _find_fno_option_by_price(oc_dict, cfg.hedge_price_min, cfg.hedge_price_max, "CALL", above_spot=True)
        if hedge:
            hedge_price = None
            lookup = _find_by_strike(hedge[1], "CALL")
            if lookup:
                _, hedge_price = lookup
            _submit_order(
                dw,
                cfg,
                side="BUY",
                exchange_seg=NIFTY_FNO_SEG,
                security_id=hedge[0],
                quantity=qty,
                product_type="MARGIN",
                price=hedge_price,
                context={"tag": "hedge_call", "action": "HEDGE", "strike": hedge[1]},
            )


def monitor_and_manage_positions(
    dw,
    cfg: LiveConfig,
    positions: List[Dict[str, Any]],
    ingestor: Optional[OptionChainIngestor],
) -> None:
    for row in positions:
        try:
            decision = _needs_exit(row, cfg.sl_pct, cfg.tp_pct)
            if decision == "EXIT":
                _close_leg(dw, cfg, row)
        except Exception:
            LOG.exception("[live] exit check failed for %s", row.get("symbol"))

    monthly_chain: Optional[pd.DataFrame] = None
    weekly_chain: Optional[pd.DataFrame] = None
    spot_hint: Optional[float] = None

    if ingestor:
        monthly_expiry = _pick_monthly_expiry(ingestor)
        if monthly_expiry:
            try:
                monthly_chain = ingestor.fetch(monthly_expiry)
                if not monthly_chain.empty and "underlying_ltp" in monthly_chain.columns:
                    vals = monthly_chain["underlying_ltp"].dropna()
                    if not vals.empty:
                        spot_hint = float(vals.iloc[0])
            except Exception as exc:
                LOG.warning("[live] failed to fetch monthly chain (%s): %s", monthly_expiry, exc)
        try:
            weekly_chain = ingestor.fetch("weekly")
        except Exception as exc:
            LOG.warning("[live] failed to fetch weekly chain: %s", exc)

    groups = _group_strangles(positions)
    roll_step = float(cfg.roll_distance)

    for expiry_tag, legs in groups.items():
        ce_leg = (legs.get("CALL") or {}).get("row")
        pe_leg = (legs.get("PUT") or {}).get("row")
        if not ce_leg or not pe_leg:
            continue

        ce_short = float(ce_leg.get("netQty") or 0) < 0
        pe_short = float(pe_leg.get("netQty") or 0) < 0
        if not (ce_short and pe_short):
            continue

        ce_strike = (legs.get("CALL") or {}).get("strike")
        pe_strike = (legs.get("PUT") or {}).get("strike")

        ce_entry = _safe_float(ce_leg.get("costPrice"), 0.0) or 0.0
        pe_entry = _safe_float(pe_leg.get("costPrice"), 0.0) or 0.0

        ce_row = None
        pe_row = None
        if monthly_chain is not None and not monthly_chain.empty:
            if ce_strike is not None:
                match = monthly_chain.loc[(monthly_chain["strike"] - ce_strike).abs() < 0.1]
                if not match.empty:
                    ce_row = _series_to_dict(match.iloc[0])
            if pe_strike is not None:
                match = monthly_chain.loc[(monthly_chain["strike"] - pe_strike).abs() < 0.1]
                if not match.empty:
                    pe_row = _series_to_dict(match.iloc[0])

        ce_ltp = _safe_float((ce_row or {}).get("ltp_ce"), None)
        pe_ltp = _safe_float((pe_row or {}).get("ltp_pe"), None)
        ce_delta = _safe_float((ce_row or {}).get("delta_ce"), None)
        pe_delta = _safe_float((pe_row or {}).get("delta_pe"), None)

        soft_x = float(cfg.premium_soft_x)
        hard_x = float(cfg.premium_hard_x)
        delta_limit = float(cfg.delta_breach)

        ce_roll_needed = False
        ce_roll_reason = "roll"
        if ce_ltp is not None and ce_entry > 0:
            ratio = ce_ltp / ce_entry
            if ratio >= hard_x:
                LOG.info("[live] CALL %s premium %.2f >= %.1fx entry %.2f", ce_strike, ce_ltp, hard_x, ce_entry)
                ce_roll_needed = True
                ce_roll_reason = "premium_hard"
            elif ratio >= soft_x:
                LOG.info("[live] CALL %s premium hit soft threshold (x%.2f)", ce_strike, ratio)
        if not ce_roll_needed and ce_delta is not None and ce_delta > delta_limit:
            LOG.info("[live] CALL %s delta %.3f > %.3f -> roll", ce_strike, ce_delta, delta_limit)
            ce_roll_needed = True
            ce_roll_reason = "delta_breach"

        if ce_roll_needed and ce_strike is not None:
            target = ce_strike + roll_step
            replacement = _select_replacement(monthly_chain, target, "CALL", "UP")
            if replacement:
                _roll_short_leg(dw, cfg, ce_leg, "CALL", replacement, ce_roll_reason)
            else:
                LOG.warning("[live] no replacement CALL found for %.0f -> %.0f", ce_strike, target)

        pe_roll_needed = False
        pe_roll_reason = "roll"
        if pe_ltp is not None and pe_entry > 0:
            ratio = pe_ltp / pe_entry
            if ratio >= hard_x:
                LOG.info("[live] PUT %s premium %.2f >= %.1fx entry %.2f", pe_strike, pe_ltp, hard_x, pe_entry)
                pe_roll_needed = True
                pe_roll_reason = "premium_hard"
            elif ratio >= soft_x:
                LOG.info("[live] PUT %s premium hit soft threshold (x%.2f)", pe_strike, ratio)
        if not pe_roll_needed and pe_delta is not None and abs(pe_delta) > delta_limit:
            LOG.info("[live] PUT %s |delta| %.3f > %.3f -> roll", pe_strike, abs(pe_delta), delta_limit)
            pe_roll_needed = True
            pe_roll_reason = "delta_breach"

        if pe_roll_needed and pe_strike is not None:
            target = pe_strike - roll_step
            replacement = _select_replacement(monthly_chain, target, "PUT", "DOWN")
            if replacement:
                _roll_short_leg(dw, cfg, pe_leg, "PUT", replacement, pe_roll_reason)
            else:
                LOG.warning("[live] no replacement PUT found for %.0f -> %.0f", pe_strike, target)

        feature_row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "strategy": "strangle",
            "expiry": expiry_tag,
            "exchange_seg": ce_leg.get("exchangeSegment") or ce_leg.get("segment") or "",
            "spot": spot_hint if spot_hint is not None else _safe_float((ce_row or {}).get("underlying_ltp")) or _safe_float((pe_row or {}).get("underlying_ltp")),
            "ce_strike": ce_strike,
            "pe_strike": pe_strike,
            "ce_ltp": ce_ltp,
            "pe_ltp": pe_ltp,
            "ce_delta": ce_delta,
            "pe_delta": pe_delta,
            "ce_entry": ce_entry,
            "pe_entry": pe_entry,
            "net_credit": (ce_entry + pe_entry),
            "warn_only": int(bool(getattr(cfg, "warn_only", False))),
            "trade_mode": getattr(cfg, "trade_mode", "live"),
            "context": json.dumps({
                "ce_roll_needed": ce_roll_needed,
                "pe_roll_needed": pe_roll_needed,
                "ce_ratio": ce_ltp / ce_entry if ce_entry and ce_ltp is not None else None,
                "pe_ratio": pe_ltp / pe_entry if pe_entry and pe_ltp is not None else None,
            }, default=str),
        }
        _record_feature(cfg, feature_row)

    if cfg.hedge_enable and weekly_chain is not None and not weekly_chain.empty:
        hedge_present = False
        for row in positions:
            qty = float(row.get("netQty") or 0)
            if qty > 0:
                price = _safe_float(row.get("costPrice"), None)
                if price is not None and cfg.hedge_price_min <= price <= cfg.hedge_price_max:
                    hedge_present = True
                    break
        if not hedge_present:
            spot = spot_hint
            if spot is None and "underlying_ltp" in weekly_chain.columns:
                vals = weekly_chain["underlying_ltp"].dropna()
                if not vals.empty:
                    spot = float(vals.iloc[0])
            if spot is None:
                LOG.info("[live] hedge placement skipped; no spot reference")
            else:
                pe_mask = weekly_chain["ltp_pe"].between(cfg.hedge_price_min, cfg.hedge_price_max, inclusive="both")
                pe_mask &= weekly_chain["strike"] < spot
                ce_mask = weekly_chain["ltp_ce"].between(cfg.hedge_price_min, cfg.hedge_price_max, inclusive="both")
                ce_mask &= weekly_chain["strike"] > spot

                pe_candidates = weekly_chain.loc[pe_mask].copy()
                ce_candidates = weekly_chain.loc[ce_mask].copy()

                if not pe_candidates.empty:
                    row_dict = _series_to_dict(pe_candidates.sort_values(by="strike", ascending=False).iloc[0])
                    try:
                        _place_option_order(
                            dw,
                            cfg,
                            "BUY",
                            cfg.lot_size,
                            row_dict,
                            "PUT",
                            context={"tag": "hedge_put", "action": "HEDGE"},
                        )
                        LOG.info("[live] hedge PUT bought strike %.0f premium %.2f", row_dict.get("strike"), row_dict.get("ltp_pe"))
                    except Exception as exc:
                        LOG.warning("[live] failed to buy hedge PUT: %s", exc)

                if not ce_candidates.empty:
                    row_dict = _series_to_dict(ce_candidates.sort_values(by="strike").iloc[0])
                    try:
                        _place_option_order(
                            dw,
                            cfg,
                            "BUY",
                            cfg.lot_size,
                            row_dict,
                            "CALL",
                            context={"tag": "hedge_call", "action": "HEDGE"},
                        )
                        LOG.info("[live] hedge CALL bought strike %.0f premium %.2f", row_dict.get("strike"), row_dict.get("ltp_ce"))
                    except Exception as exc:
                        LOG.warning("[live] failed to buy hedge CALL: %s", exc)


def execute_new_strangle(dw, cfg: LiveConfig) -> None:
    spot = _safe_float(dw.get_ltp_once(NIFTY_SEG, NIFTY_SCRIP), None)
    if spot is None or spot <= 0:
        cached = _safe_float(getattr(cfg, "last_spot", None), None)
        if cached and cached > 0:
            LOG.debug("[live] using cached spot %.2f before fallbacks", cached)
            spot = cached
        else:
            spot = None
    if spot is None or spot <= 0:
        try:
            oc_preview = dw.get_option_chain(
                underlying_security_id=NIFTY_SCRIP,
                underlying_seg=NIFTY_FNO_SEG,
                expiry=_infer_default_expiry() or "",
            )
        except Exception:
            oc_preview = None
        fallback_spot = None
        oc_dict = _coerce_dict(oc_preview) or {}
        if oc_dict:
            containers: List[Dict[str, Any]] = []
            raw_data = oc_dict.get("data")
            raw_data_dict = _coerce_dict(raw_data)
            if raw_data_dict:
                containers.append(raw_data_dict)
            elif isinstance(raw_data, list):
                for item in raw_data:
                    item_dict = _coerce_dict(item)
                    if item_dict:
                        containers.append(item_dict)
            normalized_containers: List[Dict[str, Any]] = list(containers)
            if oc_dict:
                normalized_containers.append(oc_dict)

            for container in normalized_containers:
                if not container:
                    continue
                for key in ("underlying_ltp", "underlyingValue", "underlying_value", "underlyingPrice", "spot"):
                    if container.get(key) is not None:
                        fallback_spot = _safe_float(container.get(key), None)
                        if fallback_spot:
                            break
                if fallback_spot:
                    break

            if not fallback_spot:
                for row_dict in _iter_option_rows([raw_data_dict or {}, raw_data, oc_dict]):
                    for key in ("underlying_ltp", "underlyingValue", "underlying_value", "underlyingPrice", "spot"):
                        if row_dict.get(key) is not None:
                            fallback_spot = _safe_float(row_dict.get(key), None)
                            if fallback_spot:
                                break
                    if fallback_spot:
                        break
        if fallback_spot:
            LOG.info("[live] using option-chain spot fallback %.2f", fallback_spot)
        spot = fallback_spot

    if spot is None or spot <= 0:
        try:
            positions = dw.get_positions_live() or []
        except Exception:
            positions = []
        for row in positions:
            if not isinstance(row, dict):
                continue
            base = (
                row.get("underlyingValue")
                or row.get("underlying_value")
                or row.get("underlyingPrice")
                or row.get("indexPrice")
            )
            fallback = _safe_float(base, None)
            if fallback and fallback > 0:
                spot = fallback
                LOG.info("[live] using spot from positions fallback %.2f", spot)
                break

    now = datetime.now()
    if spot is None or spot <= 0:
        last_warn = getattr(cfg, "_last_spot_warn_at", None)
        if not last_warn or (now - last_warn).total_seconds() >= 60:
            LOG.info("[live] no NIFTY spot; entry deferred")
            cfg._last_spot_warn_at = now
        return

    cfg.last_spot = spot
    cfg._last_spot_warn_at = None
    _place_strangle_and_hedge(dw, cfg, spot)


def run_live(
    dw,
    cfg: Optional[LiveConfig] = None,  # type: ignore[override]
    *,
    regime_analyzer: Optional["MarketRegimeAnalyzer"] = None,
    recommender: Optional["StrategyRecommender"] = None,
    feature_history_path: Optional[Path] = None,
) -> None:
    cfg = cfg or LiveConfig()
    try:
        chain_ingestor = OptionChainIngestor(ChainConfig(NIFTY_SCRIP, NIFTY_FNO_SEG))
    except Exception as exc:
        LOG.warning("[live] OptionChainIngestor unavailable: %s", exc)
        chain_ingestor = None

    bat_cfg = getattr(cfg, "batman", BatmanConfig())

    if feature_history_path is None and cfg.feature_log_path:
        feature_history_path = Path(cfg.feature_log_path)

    refresh_iters: Optional[int] = None
    if regime_analyzer:
        interval_seconds = max(60.0, cfg.regime_refresh_minutes * 60.0)
        refresh_iters = max(1, int(interval_seconds / max(float(getattr(cfg, "poll_sec", 5.0)), 1.0)))

    loop_iter = 0

    while True:
        try:
            rows = dw.get_positions_live() or []
            open_deriv_rows = [
                r
                for r in rows
                if isinstance(r, dict)
                and (r.get("exchangeSegment") or "").upper().startswith("NSE_FNO")
                and int(float(r.get("netQty") or 0)) != 0
            ]

            if regime_analyzer and (refresh_iters is None or loop_iter % refresh_iters == 0):
                history_df = pd.DataFrame()
                if feature_history_path and feature_history_path.exists():
                    try:
                        history_df = pd.read_csv(feature_history_path)
                        if len(history_df.index) > 600:
                            history_df = history_df.tail(600)
                    except Exception:
                        history_df = pd.DataFrame()
                if history_df.empty:
                    spot_val = _safe_float(dw.get_ltp_once(NIFTY_SEG, NIFTY_SCRIP), None)
                    if spot_val is None or spot_val <= 0:
                        spot_val = cfg.last_spot
                    if (spot_val is None or spot_val <= 0) and rows:
                        for row in rows:
                            if not isinstance(row, dict):
                                continue
                            base = (
                                row.get("underlyingValue")
                                or row.get("underlying_value")
                                or row.get("underlyingPrice")
                                or row.get("indexPrice")
                            )
                            spot_val = _safe_float(base, None)
                            if spot_val and spot_val > 0:
                                break
                    if spot_val and spot_val > 0:
                        history_df = pd.DataFrame([
                            {
                                "timestamp": datetime.now().isoformat(timespec="seconds"),
                                "spot": spot_val,
                            }
                        ])
                last_state = getattr(cfg, "_last_market_state", None)
                state = regime_analyzer.analyze_feature_history(history_df, fallback_state=last_state, min_required=1)
                if state:
                    cfg._last_market_state = state
                    cfg.last_spot = state.features.get("spot")
                    candidates: List["StrategyCandidate"] = []
                    if recommender:
                        candidates = recommender.recommend(state)
                        cfg._last_strategy_candidates = candidates
                    _record_environment_snapshot(cfg, state, candidates)
                    top = candidates[0] if candidates else None
                    if top:
                        LOG.info(
                            "[env] trend=%s vol=%s top=%s score=%.2f conf=%.2f",
                            state.trend,
                            state.volatility,
                            top.name,
                            top.score,
                            top.confidence,
                        )
                        if not open_deriv_rows and cfg.auto_disable_strangle_when_unfavored:
                            cfg.enable_strangle_entry = top.name == "monthly_strangle_with_weekly_hedge"
                    else:
                        LOG.info(
                            "[env] trend=%s vol=%s (no recommendation)",
                            state.trend,
                            state.volatility,
                        )
                else:
                    LOG.debug("[env] insufficient data for market state analysis")

            if open_deriv_rows and bat_cfg.enabled:
                monitor_batman_positions(dw, cfg, bat_cfg, open_deriv_rows, chain_ingestor)

            active = 1 if open_deriv_rows else 0
            if active >= cfg.max_strangles:
                monitor_and_manage_positions(dw, cfg, open_deriv_rows, chain_ingestor)
            else:
                if open_deriv_rows:
                    monitor_and_manage_positions(dw, cfg, open_deriv_rows, chain_ingestor)
                else:
                    if getattr(cfg, "enable_strangle_entry", True):
                        execute_new_strangle(dw, cfg)
                    else:
                        LOG.debug("[live] strangle entry disabled by selector")
        except Exception as exc:
            LOG.exception("[live] loop error: %s", exc)

        time.sleep(max(1.0, float(getattr(cfg, "poll_sec", 5.0))))
        loop_iter += 1
