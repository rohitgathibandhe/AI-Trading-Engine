from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

from market_ai.strategies.strategy_router import decide_weekly_strategy, RouterConfig
from market_ai.strategies.weekly_regime_classifier import RegimeConfig


@dataclass
class SpreadConfig:
    lot_size: int = 75
    qty: int = 1
    bear_qty_factor: float = 0.5
    min_leg_premium: float = 30.0
    min_combined_premium: float = 120.0
    min_combined_premium_pct: float = 0.012
    max_combined_premium_pct: float = 0.030
    min_distance_points: float = 200.0
    hedge_min_price: float = 2.0
    hedge_max_price: float = 25.0
    tp_credit_pct: float = 0.60
    sl_credit_pct: float = 1.20
    max_hold_days: int = 4
    max_open_baskets: int = 2
    max_daily_loss: float = -5000.0
    max_weekly_loss: float = -15000.0
    router: RouterConfig = RouterConfig(RegimeConfig(), min_score=0.6)


def validate_candidate(short_legs: List[Dict[str, Any]], hedge_legs: List[Dict[str, Any]], spot: float, cfg: SpreadConfig) -> bool:
    if any(l.get("ltp") is None or l["ltp"] < cfg.min_leg_premium for l in short_legs):
        return False
    combined = sum(l.get("ltp", 0.0) for l in short_legs)
    if combined < cfg.min_combined_premium:
        return False
    pct = combined / max(spot, 1.0)
    if not (cfg.min_combined_premium_pct <= pct <= cfg.max_combined_premium_pct):
        return False
    for leg in short_legs:
        if abs(leg.get("strike", 0.0) - spot) < cfg.min_distance_points:
            return False
    if hedge_legs:
        for h in hedge_legs:
            ltp = h.get("ltp")
            if ltp is None or ltp <= 0:
                return False
            if not (cfg.hedge_min_price <= ltp <= cfg.hedge_max_price):
                return False
    return True


def _pick_leg(row: pd.Series, opt: str, offsets: List[int]) -> Optional[Tuple[float, float]]:
    for off in offsets:
        field_ltp = f"{opt}_off{off}_ltp"
        field_strike = f"{opt}_off{off}_strike"
        ltp = row.get(field_ltp)
        strike = row.get(field_strike)
        if pd.isna(ltp) or pd.isna(strike):
            continue
        return float(strike), float(ltp)
    return None


class WeeklySpreadsEngine:
    def __init__(self, cfg: Optional[SpreadConfig] = None):
        self.cfg = cfg or SpreadConfig()
        self.trades: List[Dict[str, Any]] = []

    def _build_bull_put(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        short = _pick_leg(row, "put", [250, 300])
        hedge = _pick_leg(row, "put", [400, 350])
        if short is None or hedge is None:
            return None
        sp_strike, sp_ltp = short
        lp_strike, lp_ltp = hedge
        if sp_strike <= lp_strike:
            return None
        if not validate_candidate(
            [{"strike": sp_strike, "ltp": sp_ltp}],
            [{"strike": lp_strike, "ltp": lp_ltp}],
            row.get("spot", 0.0),
            self.cfg,
        ):
            return None
        credit = sp_ltp - lp_ltp
        return {
            "type": "BULL_PUT_SPREAD",
            "short_leg": ("PUT", sp_strike, sp_ltp),
            "long_leg": ("PUT", lp_strike, lp_ltp),
            "credit": credit,
        }

    def _build_bear_call(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        short = _pick_leg(row, "call", [250, 300])
        hedge = _pick_leg(row, "call", [400, 350])
        if short is None or hedge is None:
            return None
        sc_strike, sc_ltp = short
        lc_strike, lc_ltp = hedge
        if lc_strike <= sc_strike:
            return None
        qty = max(1, int(self.cfg.qty * self.cfg.bear_qty_factor))
        if not validate_candidate(
            [{"strike": sc_strike, "ltp": sc_ltp}],
            [{"strike": lc_strike, "ltp": lc_ltp}],
            row.get("spot", 0.0),
            self.cfg,
        ):
            return None
        credit = sc_ltp - lc_ltp
        return {
            "type": "BEAR_CALL_SPREAD",
            "short_leg": ("CALL", sc_strike, sc_ltp),
            "long_leg": ("CALL", lc_strike, lc_ltp),
            "credit": credit,
            "qty": qty,
        }

    def _mtm(self, spread: Dict[str, Any], row: pd.Series, qty: int) -> float:
        lot = self.cfg.lot_size
        s_leg = spread["short_leg"]
        l_leg = spread["long_leg"]
        opt_s, k_s, entry_s = s_leg
        opt_l, k_l, entry_l = l_leg
        # pick the right columns by strike proximity
        def _ltp(opt: str, strike: float) -> float:
            best = None
            for off in (250, 300, 350, 400):
                col_s = f"{opt.lower()}_off{off}_strike"
                col_ltp = f"{opt.lower()}_off{off}_ltp"
                s = row.get(col_s)
                ltp = row.get(col_ltp)
                if pd.isna(s) or pd.isna(ltp):
                    continue
                if best is None or abs(float(s) - strike) < abs(best[0] - strike):
                    best = (float(s), float(ltp))
            if best:
                return best[1]
            return float(row.get(f"atm_{opt.lower()}_ltp", entry_s if opt == opt_s else entry_l))

        ltp_s = _ltp(opt_s, k_s)
        ltp_l = _ltp(opt_l, k_l)
        pnl_short = (entry_s - ltp_s) * qty * lot
        pnl_long = (ltp_l - entry_l) * qty * lot
        return pnl_short + pnl_long

    def run_backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["trade_date"] = df["timestamp"].dt.date
        daily_pnl = 0.0
        weekly_pnl = 0.0
        open_baskets = 0
        trades: List[Dict[str, Any]] = []

        dates = sorted(df["trade_date"].unique())
        for i, d in enumerate(dates):
            day_rows = df[df["trade_date"] == d].sort_values("timestamp")
            if day_rows.empty:
                continue
            # reset daily pnl per new date
            if i == 0 or dates[i - 1] != d:
                daily_pnl = 0.0
            # weekly reset on Monday
            if day_rows["timestamp"].dt.weekday.iloc[0] == 0:
                weekly_pnl = 0.0
                open_baskets = 0

            if daily_pnl <= self.cfg.max_daily_loss or weekly_pnl <= self.cfg.max_weekly_loss:
                continue
            if open_baskets >= self.cfg.max_open_baskets:
                continue

            first = day_rows.iloc[0]
            prev_close = first.get("prev_day_close") or first.get("spot")
            gap_pct = (first.get("spot", 0.0) - (prev_close or 0.0)) / max(prev_close or 1.0, 1.0) * 100 if prev_close else 0.0
            features = {
                "gap_pct": gap_pct,
                "spot_intraday_range_pct": (day_rows["spot"].max() - day_rows["spot"].min()) / max(first.get("spot", 1.0), 1.0) * 100,
                "spot_trend_20": first.get("spot_trend_20", 0.0),
            }
            routing = decide_weekly_strategy(features, self.cfg.router)
            spread = None
            qty = self.cfg.qty
            if routing["strategy"] == "BULL_PUT_SPREAD":
                spread = self._build_bull_put(first)
            elif routing["strategy"] == "BEAR_CALL_SPREAD":
                spread = self._build_bear_call(first)
                if spread:
                    qty = spread.get("qty", max(1, int(self.cfg.qty * self.cfg.bear_qty_factor)))
            if spread is None:
                continue

            credit = spread["credit"] * qty * self.cfg.lot_size
            tp = credit * self.cfg.tp_credit_pct
            sl = credit * self.cfg.sl_credit_pct

            hold_days = 0
            exit_reason = None
            pnl = 0.0
            # iterate over up to max_hold_days
            for j in range(0, self.cfg.max_hold_days + 1):
                if i + j >= len(dates):
                    break
                day_j = dates[i + j]
                rows_j = df[df["trade_date"] == day_j]
                if rows_j.empty:
                    continue
                last_row = rows_j.iloc[-1]
                pnl = self._mtm(spread, last_row, qty)
                hold_days = j
                if pnl >= tp:
                    exit_reason = "TP"
                    break
                if pnl <= -sl:
                    exit_reason = "SL"
                    break
            if exit_reason is None:
                exit_reason = "TIME"

            trades.append(
                {
                    "entry_date": pd.to_datetime(first["timestamp"]),
                    "exit_date": pd.to_datetime(dates[min(i + hold_days, len(dates) - 1)]),
                    "strategy": routing["strategy"],
                    "regime": routing["regime"],
                    "entry_spot": first.get("spot"),
                    "short_leg": spread["short_leg"] if spread else None,
                    "long_leg": spread["long_leg"] if spread else None,
                    "credit": credit,
                    "pnl": pnl,
                    "exit_reason": exit_reason,
                }
            )
            open_baskets += 1
            daily_pnl += pnl
            weekly_pnl += pnl
        return pd.DataFrame(trades)
