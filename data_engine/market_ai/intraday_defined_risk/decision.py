from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import floor, inf
from typing import Any

from .data_models import OptionType, StrategyType


DIRECTIONAL_CREDIT_CAPTURE = 0.70
CONDOR_CREDIT_CAPTURE = 0.50
TIME_EXIT_IST = "15:15"


@dataclass(frozen=True)
class Candle5m:
    t: str
    o: float
    h: float
    l: float
    c: float
    v: float


@dataclass(frozen=True)
class ChainStrike:
    k: int
    ce_ltp: float
    pe_ltp: float
    ce_bid: float
    ce_ask: float
    pe_bid: float
    pe_ask: float
    ce_delta: float | None = None
    pe_delta: float | None = None
    ce_iv: float | None = None
    pe_iv: float | None = None
    oi_ce: int | None = None
    oi_pe: int | None = None


class DecisionInputError(ValueError):
    pass


class IntradayDecisionAgent:
    def decide(self, payload: dict[str, Any]) -> dict[str, Any]:
        errors = _validate_top_level(payload)
        if errors:
            return _no_trade(errors)

        try:
            time_ist = payload["time_ist"]
            spot = float(payload["underlying"]["spot"])
            vwap = float(payload["underlying"]["vwap"])
            or15_high = float(payload["underlying"]["or15_high"])
            or15_low = float(payload["underlying"]["or15_low"])
            bars = [_parse_bar(item) for item in payload["underlying"]["ohlcv_5m"]]
            strikes = [_parse_strike(item) for item in payload["options_chain"]["strikes"]]
            rv30_pct = float(payload["volatility"]["rv30_pct"])
            atr_5m = float(payload["volatility"]["atr_5m"])
            risk = payload["risk"]
            lot_size = int(risk["lot_size"])
            max_risk_rupees = int(risk["max_risk_rupees"])
            slippage = float(risk["slippage_points_assumption"])
        except (TypeError, ValueError, DecisionInputError) as exc:
            return _no_trade([f"Input parsing failed: {exc}"])

        freshness_reasons = self._validate_freshness(payload, bars)
        if freshness_reasons:
            return _no_trade(freshness_reasons)

        regime, regime_reasons = self._classify_regime(
            now=time_ist,
            spot=spot,
            vwap=vwap,
            or15_high=or15_high,
            or15_low=or15_low,
            bars=bars,
            rv30_pct=rv30_pct,
        )
        if regime == StrategyType.NO_TRADE.value:
            return _no_trade(regime_reasons)

        strategy = {
            "DOWN_TREND": StrategyType.BEAR_CALL_CREDIT_SPREAD,
            "UP_TREND": StrategyType.BULL_PUT_CREDIT_SPREAD,
            "RANGE": StrategyType.IRON_CONDOR,
        }[regime]

        if strategy == StrategyType.IRON_CONDOR and time_ist < "10:00":
            return _no_trade(["Iron condor is allowed only after 10:00 IST on confirmed range days."])

        structure, selection_reasons = self._select_structure(
            strategy=strategy,
            spot=spot,
            strikes=strikes,
            rv30_pct=rv30_pct,
            atr_5m=atr_5m,
        )
        if structure is None:
            return _no_trade(selection_reasons)

        max_loss_rupees_per_lot = round(structure["max_loss_points"] * lot_size, 2)
        if max_loss_rupees_per_lot <= 0:
            return _no_trade(["Computed max loss per lot is non-positive."])
        recommended_lots = floor(max_risk_rupees / max_loss_rupees_per_lot)
        if recommended_lots < 1:
            return _no_trade([
                f"Max loss per lot {max_loss_rupees_per_lot:.2f} exceeds the allowed risk budget {max_risk_rupees}.",
                "Increase risk budget or choose a narrower defined-risk structure.",
            ])

        credit_points = structure["credit_points"]
        tp_capture = DIRECTIONAL_CREDIT_CAPTURE if strategy != StrategyType.IRON_CONDOR else CONDOR_CREDIT_CAPTURE
        tp_value = round(credit_points * (1.0 - tp_capture), 2)
        sl_value = round(credit_points * 2.0, 2)
        rationale = regime_reasons + selection_reasons + [
            f"Net credit after slippage assumption is {max(credit_points - slippage, 0.0):.2f} points.",
            f"Recommended lots are capped at {recommended_lots} to keep per-trade risk within {max_risk_rupees} rupees.",
        ]
        risks = [
            "Exit if spread/condor value reaches 2.0x initial credit.",
            "Exit on regime invalidation: spot crosses VWAP opposite direction and holds for 2 bars.",
            "Force close all legs by 15:15 IST.",
        ]

        return {
            "decision": "TRADE",
            "strategy": strategy.value,
            "legs": structure["legs"],
            "entry_price_targets": {
                "method": "LIMIT_AROUND_MID",
                "net_credit_points": round(credit_points, 2),
                "expected_fill_credit_points_after_slippage": round(max(credit_points - slippage, 0.0), 2),
                "slippage_points_assumption": slippage,
            },
            "tp_rule": {
                "credit_capture_pct": round(tp_capture, 2),
                "target_value_points": tp_value,
            },
            "sl_rule": {
                "premium_stop_multiple": 2.0,
                "stop_value_points": sl_value,
                "regime_invalidation": _regime_invalidation_text(strategy),
            },
            "time_exit": TIME_EXIT_IST,
            "max_loss_rupees_per_lot": max_loss_rupees_per_lot,
            "recommended_lots": recommended_lots,
            "rationale": rationale,
            "risks": risks,
        }

    def _validate_freshness(self, payload: dict[str, Any], bars: list[Candle5m]) -> list[str]:
        reasons: list[str] = []
        if not bars:
            reasons.append("Missing 5-minute OHLCV bars.")
            return reasons
        if bars[-1].t != payload["time_ist"]:
            reasons.append("time_ist does not match the last 5-minute bar; inputs are stale.")
        if payload["underlying"].get("symbol") != "NIFTY":
            reasons.append("Only NIFTY underlying is supported.")
        if int(payload["risk"].get("lot_size", 0)) != 65:
            reasons.append("Lot size must be exactly 65.")
        expiry = payload["options_chain"].get("expiry")
        if not expiry:
            reasons.append("Option chain expiry is missing.")
        else:
            try:
                date.fromisoformat(expiry)
            except ValueError:
                reasons.append("Option chain expiry is not a valid YYYY-MM-DD date.")
        return reasons

    def _classify_regime(
        self,
        now: str,
        spot: float,
        vwap: float,
        or15_high: float,
        or15_low: float,
        bars: list[Candle5m],
        rv30_pct: float,
    ) -> tuple[str, list[str]]:
        reasons: list[str] = []
        if len(bars) < 2:
            return StrategyType.NO_TRADE.value, ["At least two 5-minute candles are required to classify regime."]
        last_two = bars[-2:]
        if spot < vwap and all(bar.c < or15_low for bar in last_two) and rv30_pct >= 0.35:
            reasons.append("Spot is below VWAP and the last two 5-minute closes are below OR15 low with RV30% >= 0.35.")
            return "DOWN_TREND", reasons
        if spot > vwap and all(bar.c > or15_high for bar in last_two) and rv30_pct >= 0.35:
            reasons.append("Spot is above VWAP and the last two 5-minute closes are above OR15 high with RV30% >= 0.35.")
            return "UP_TREND", reasons
        if now >= "10:00" and rv30_pct < 0.35 and self._is_range(spot, vwap, or15_high, or15_low, bars):
            reasons.append("Price has stayed inside OR15 and oscillated around VWAP for at least 30 minutes after 09:30 with RV30% < 0.35.")
            return "RANGE", reasons
        reasons.append("Regime is unclear under the conservative OR15/VWAP rules.")
        return StrategyType.NO_TRADE.value, reasons

    def _is_range(self, spot: float, vwap: float, or15_high: float, or15_low: float, bars: list[Candle5m]) -> bool:
        qualifying = [bar for bar in bars if bar.t >= "09:30"]
        if len(qualifying) < 6:
            return False
        recent = qualifying[-6:]
        inside = all(bar.h <= or15_high and bar.l >= or15_low for bar in recent)
        around_vwap = abs(spot - vwap) <= max((or15_high - or15_low) * 0.15, 10.0)
        above = sum(1 for bar in recent if bar.c > vwap)
        below = sum(1 for bar in recent if bar.c < vwap)
        return inside and around_vwap and above >= 2 and below >= 2

    def _select_structure(
        self,
        strategy: StrategyType,
        spot: float,
        strikes: list[ChainStrike],
        rv30_pct: float,
        atr_5m: float,
    ) -> tuple[dict[str, Any] | None, list[str]]:
        if strategy == StrategyType.BEAR_CALL_CREDIT_SPREAD:
            return self._select_vertical(strategy, spot, strikes, rv30_pct, atr_5m, side=OptionType.CALL)
        if strategy == StrategyType.BULL_PUT_CREDIT_SPREAD:
            return self._select_vertical(strategy, spot, strikes, rv30_pct, atr_5m, side=OptionType.PUT)
        if strategy == StrategyType.IRON_CONDOR:
            return self._select_condor(spot, strikes, rv30_pct, atr_5m)
        return None, ["No permitted structure for this regime."]

    def _select_vertical(
        self,
        strategy: StrategyType,
        spot: float,
        strikes: list[ChainStrike],
        rv30_pct: float,
        atr_5m: float,
        side: OptionType,
    ) -> tuple[dict[str, Any] | None, list[str]]:
        target_short = (0.18, 0.25)
        target_long = (0.05, 0.10)
        liquid_rows = [row for row in strikes if self._is_liquid(row, side)]
        if not liquid_rows:
            return None, [f"No liquid {side.value} strikes passed the bid/ask quality filter."]

        best: dict[str, Any] | None = None
        best_score = -inf
        use_delta = any(self._delta_for(row, side) is not None for row in liquid_rows)
        distance_points = max(spot * rv30_pct / 100.0, atr_5m)

        # Widen minimum distance using the ATM straddle price as the market's own
        # implied expected-move forecast. Short strikes inside 55% of the straddle
        # are inside the market's 1-sigma range — a high-risk sell zone on trend days.
        atm_candidates = [row for row in strikes if row.ce_ltp > 0 and row.pe_ltp > 0]
        if atm_candidates:
            atm_row = min(atm_candidates, key=lambda r: abs(r.k - spot))
            expected_move_pts = atm_row.ce_ltp + atm_row.pe_ltp
            distance_points = max(distance_points, expected_move_pts * 0.55)

        for short_row in liquid_rows:
            short_strike = short_row.k
            if side == OptionType.CALL and short_strike <= spot:
                continue
            if side == OptionType.PUT and short_strike >= spot:
                continue
            for long_row in liquid_rows:
                long_strike = long_row.k
                width = abs(long_strike - short_strike)
                if side == OptionType.CALL and long_strike <= short_strike:
                    continue
                if side == OptionType.PUT and long_strike >= short_strike:
                    continue
                if width < 100 or width > 200:
                    continue
                credit = self._mid(short_row, side) - self._mid(long_row, side)
                if credit <= 0:
                    continue
                if credit < 0.25 * width:
                    continue

                if use_delta:
                    short_delta = abs(self._delta_for(short_row, side) or 0.0)
                    long_delta = abs(self._delta_for(long_row, side) or 0.0)
                    if not (target_short[0] <= short_delta <= target_short[1]):
                        continue
                    if not (target_long[0] <= long_delta <= target_long[1]):
                        continue
                    score = (credit / width) - abs(short_delta - 0.215) - abs(long_delta - 0.075)
                else:
                    if side == OptionType.CALL and (short_strike - spot) < distance_points:
                        continue
                    if side == OptionType.PUT and (spot - short_strike) < distance_points:
                        continue
                    # In distance mode (no delta data) enforce a minimum credit so we
                    # don't select near-worthless far-OTM strikes with negligible theta
                    if credit < 0.75:
                        continue
                    score = (credit / width) - (width / 1000.0)

                if score > best_score:
                    best_score = score
                    best = {
                        "strategy": strategy.value,
                        "legs": [
                            {"action": "SELL", "strike": short_strike, "type": side.value, "qty_lots": 1},
                            {"action": "BUY", "strike": long_strike, "type": side.value, "qty_lots": 1},
                        ],
                        "credit_points": round(credit, 2),
                        "width_points": width,
                        "max_loss_points": round(width - credit, 2),
                        "selection_mode": "DELTA" if use_delta else "DISTANCE",
                        "selection_reasons": [
                            f"Short {side.value} strike {short_strike} chosen under {'delta' if use_delta else 'distance'} targeting.",
                            f"Protective long {side.value} wing at {long_strike} keeps the trade defined-risk with width {width} points.",
                        ],
                    }

        if best is None:
            return None, [f"No {strategy.value} satisfied delta/distance, liquidity, and credit-efficiency constraints."]
        return best, best["selection_reasons"]

    def _select_condor(self, spot: float, strikes: list[ChainStrike], rv30_pct: float, atr_5m: float) -> tuple[dict[str, Any] | None, list[str]]:
        call_side, call_reasons = self._select_condor_side(spot, strikes, rv30_pct, atr_5m, OptionType.CALL)
        put_side, put_reasons = self._select_condor_side(spot, strikes, rv30_pct, atr_5m, OptionType.PUT)
        if call_side is None or put_side is None:
            return None, call_reasons + put_reasons + ["Iron condor requires both sides to be selectable under conservative filters."]
        total_credit = round(call_side["credit_points"] + put_side["credit_points"], 2)
        max_width = max(call_side["width_points"], put_side["width_points"])
        if total_credit < 0.30 * max_width:
            return None, ["Iron condor total credit is below the 30% width-efficiency requirement."]
        return {
            "strategy": StrategyType.IRON_CONDOR.value,
            "legs": call_side["legs"] + put_side["legs"],
            "credit_points": total_credit,
            "width_points": max_width,
            "max_loss_points": round(max(call_side["width_points"] - total_credit, put_side["width_points"] - total_credit), 2),
            "selection_mode": f"{call_side['selection_mode']}+{put_side['selection_mode']}",
            "selection_reasons": call_reasons + put_reasons + [f"Combined condor credit is {total_credit:.2f} points on max width {max_width} points."],
        }, call_reasons + put_reasons

    def _select_condor_side(self, spot: float, strikes: list[ChainStrike], rv30_pct: float, atr_5m: float, side: OptionType) -> tuple[dict[str, Any] | None, list[str]]:
        target_short = (0.10, 0.15)
        target_long = (0.05, 0.10)
        liquid_rows = [row for row in strikes if self._is_liquid(row, side)]
        if not liquid_rows:
            return None, [f"No liquid {side.value} strikes available for condor selection."]
        best: dict[str, Any] | None = None
        best_score = -inf
        use_delta = any(self._delta_for(row, side) is not None for row in liquid_rows)
        distance_points = max(spot * rv30_pct / 100.0, atr_5m)
        for short_row in liquid_rows:
            for long_row in liquid_rows:
                short_strike = short_row.k
                long_strike = long_row.k
                width = abs(long_strike - short_strike)
                if side == OptionType.CALL and short_strike <= spot:
                    continue
                if side == OptionType.CALL and long_strike <= short_strike:
                    continue
                if side == OptionType.PUT and short_strike >= spot:
                    continue
                if side == OptionType.PUT and long_strike >= short_strike:
                    continue
                if width < 100 or width > 200:
                    continue
                credit = self._mid(short_row, side) - self._mid(long_row, side)
                if credit <= 0:
                    continue
                if use_delta:
                    short_delta = abs(self._delta_for(short_row, side) or 0.0)
                    long_delta = abs(self._delta_for(long_row, side) or 0.0)
                    if not (target_short[0] <= short_delta <= target_short[1]):
                        continue
                    if not (target_long[0] <= long_delta <= target_long[1]):
                        continue
                    score = (credit / width) - abs(short_delta - 0.125) - abs(long_delta - 0.075)
                else:
                    if side == OptionType.CALL and (short_strike - spot) < distance_points:
                        continue
                    if side == OptionType.PUT and (spot - short_strike) < distance_points:
                        continue
                    score = credit / width
                if score > best_score:
                    best_score = score
                    best = {
                        "legs": [
                            {"action": "SELL", "strike": short_strike, "type": side.value, "qty_lots": 1},
                            {"action": "BUY", "strike": long_strike, "type": side.value, "qty_lots": 1},
                        ],
                        "credit_points": round(credit, 2),
                        "width_points": width,
                        "selection_mode": "DELTA" if use_delta else "DISTANCE",
                    }
        if best is None:
            return None, [f"No {side.value} side satisfied conservative condor filters."]
        return best, [f"{side.value} side of condor selected with {best['selection_mode'].lower()} targeting and width {best['width_points']} points."]

    def _is_liquid(self, row: ChainStrike, side: OptionType) -> bool:
        ltp = row.ce_ltp if side == OptionType.CALL else row.pe_ltp
        bid = row.ce_bid if side == OptionType.CALL else row.pe_bid
        ask = row.ce_ask if side == OptionType.CALL else row.pe_ask
        oi = row.oi_ce if side == OptionType.CALL else row.oi_pe
        if ltp <= 0 or bid <= 0 or ask <= 0 or ask < bid:
            return False
        # Minimum viable premium — strikes below 0.5 pts have no meaningful theta to capture
        if ltp < 0.5:
            return False
        spread = ask - bid
        # Percentage spread check — tighter for small premiums to avoid execution slippage
        pct = spread / ltp
        if pct > 0.12:  # 12% max spread (was 15%)
            return False
        # Absolute spread cap: max 1.5 pts — prevents wide markets even if % looks ok
        if spread > 1.5:
            return False
        # OI filter: require meaningful open interest to ensure we can get fills
        if oi is not None and oi < 5_000:
            return False
        return True

    def _mid(self, row: ChainStrike, side: OptionType) -> float:
        bid = row.ce_bid if side == OptionType.CALL else row.pe_bid
        ask = row.ce_ask if side == OptionType.CALL else row.pe_ask
        return (bid + ask) / 2.0

    def _delta_for(self, row: ChainStrike, side: OptionType) -> float | None:
        return row.ce_delta if side == OptionType.CALL else row.pe_delta


def make_decision(payload: dict[str, Any]) -> dict[str, Any]:
    return IntradayDecisionAgent().decide(payload)


def _parse_bar(item: dict[str, Any]) -> Candle5m:
    required = {"t", "o", "h", "l", "c", "v"}
    missing = sorted(required - set(item))
    if missing:
        raise DecisionInputError(f"5m OHLCV bar missing fields: {', '.join(missing)}")
    return Candle5m(
        t=str(item["t"]),
        o=float(item["o"]),
        h=float(item["h"]),
        l=float(item["l"]),
        c=float(item["c"]),
        v=float(item["v"]),
    )


def _parse_strike(item: dict[str, Any]) -> ChainStrike:
    required = {"k", "ce_ltp", "pe_ltp", "ce_bid", "ce_ask", "pe_bid", "pe_ask"}
    missing = sorted(required - set(item))
    if missing:
        raise DecisionInputError(f"Option chain strike missing fields: {', '.join(missing)}")
    return ChainStrike(
        k=int(item["k"]),
        ce_ltp=float(item["ce_ltp"]),
        pe_ltp=float(item["pe_ltp"]),
        ce_bid=float(item["ce_bid"]),
        ce_ask=float(item["ce_ask"]),
        pe_bid=float(item["pe_bid"]),
        pe_ask=float(item["pe_ask"]),
        ce_delta=float(item["ce_delta"]) if item.get("ce_delta") is not None else None,
        pe_delta=float(item["pe_delta"]) if item.get("pe_delta") is not None else None,
        ce_iv=float(item["ce_iv"]) if item.get("ce_iv") is not None else None,
        pe_iv=float(item["pe_iv"]) if item.get("pe_iv") is not None else None,
        oi_ce=int(item["oi_ce"]) if item.get("oi_ce") is not None else None,
        oi_pe=int(item["oi_pe"]) if item.get("oi_pe") is not None else None,
    )


def _validate_top_level(payload: dict[str, Any]) -> list[str]:
    required = {"date", "time_ist", "underlying", "volatility", "options_chain", "risk"}
    missing = sorted(required - set(payload))
    reasons: list[str] = []
    if missing:
        reasons.append(f"Missing top-level fields: {', '.join(missing)}")
        return reasons
    for key in ["spot", "vwap", "or15_high", "or15_low", "ohlcv_5m"]:
        if key not in payload["underlying"]:
            reasons.append(f"underlying.{key} is required.")
    for key in ["rv30_pct", "atr_5m"]:
        if key not in payload["volatility"]:
            reasons.append(f"volatility.{key} is required.")
    if "strikes" not in payload["options_chain"]:
        reasons.append("options_chain.strikes is required.")
    for key in ["lot_size", "max_risk_rupees", "slippage_points_assumption"]:
        if key not in payload["risk"]:
            reasons.append(f"risk.{key} is required.")
    return reasons


def _no_trade(reasons: list[str]) -> dict[str, Any]:
    return {
        "decision": "NO_TRADE",
        "reasons": reasons,
        "data_required": [
            "Fresh 5-minute OHLCV with the latest bar matching time_ist",
            "Valid OR15 and VWAP values",
            "Weekly-expiry option chain with bid/ask/LTP and preferably deltas",
            "Risk budget with lot_size=65 and max_risk_rupees",
        ],
    }


def _regime_invalidation_text(strategy: StrategyType) -> str:
    if strategy == StrategyType.BEAR_CALL_CREDIT_SPREAD:
        return "Exit if spot crosses above VWAP and holds there for 2 consecutive 5-minute bars."
    if strategy == StrategyType.BULL_PUT_CREDIT_SPREAD:
        return "Exit if spot crosses below VWAP and holds there for 2 consecutive 5-minute bars."
    return "Exit if price breaks out of the OR15 range and holds outside for 2 consecutive 5-minute bars."
