from typing import Dict, Any, Optional
import datetime
import pandas as pd
import types

# === Import your thin wrapper (has no order endpoints) ===
try:
    from modules.data_fetch import dhan_wrapper as _dw
except Exception as e:
    raise ImportError("Failed to import modules/data_fetch/dhan_wrapper.py") from e

# === Option-chain adapter: your module exports class 'OptionChain' (confirmed) ===
OptionChainAdapter = None
try:
    from modules.data_fetch import dhan_option_chain as _doc
    for _cand in ("OptionChain", "OptionChainAdapter", "DhanOptionChainAdapter", "DhanOptionChain"):
        if hasattr(_doc, _cand):
            OptionChainAdapter = getattr(_doc, _cand)
            break
except Exception:
    OptionChainAdapter = None
if OptionChainAdapter is None:
    raise ImportError(
        "Could not resolve OptionChain adapter from modules/data_fetch/dhan_option_chain.py. "
        "Export one of ['OptionChain','OptionChainAdapter','DhanOptionChainAdapter','DhanOptionChain']."
    )

# -----------------------------------------------------------------------------
# Broker resolution:
# 1) If wrapper exposes a class Broker-like: use it.
# 2) Else, auto-build a PaperBroker that uses get_option_price / get_option_chain.
# -----------------------------------------------------------------------------

# Try to find a class-y broker first
_BROKER_CLASS = None
for _cand in ("Broker", "DhanBroker", "Wrapper", "DhanWrapper", "BrokerClient"):
    if hasattr(_dw, _cand):
        _BROKER_CLASS = getattr(_dw, _cand)
        break

def _call_get_option_price(
    symbol: str, side: str, strike: float, expiry_kind: str
) -> Optional[float]:
    """
    Try common signatures for your get_option_price:
      - get_option_price(symbol, strike, side, expiry='monthly'|'weekly')
      - get_option_price(symbol=symbol, strike=strike, side=side, expiry=expiry_kind)
      - get_option_price(tradingsymbol=..., ...)  -> if you use token mapping, adapt here
    """
    if not hasattr(_dw, "get_option_price"):
        return None
    fn = _dw.get_option_price
    # Try positional
    try:
        return float(fn(symbol, strike, side, expiry=expiry_kind))
    except TypeError:
        pass
    # Try kwargs
    try:
        return float(fn(symbol=symbol, strike=strike, side=side, expiry=expiry_kind))
    except TypeError:
        pass
    # Last resort: maybe function ignores expiry
    try:
        return float(fn(symbol, strike, side))
    except Exception:
        return None

def _paper_refresh_legs(position: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use get_option_price if available; else fall back to entry prices (keeps loop alive).
    """
    sym = position["symbol"]
    ceK = float(position["short_ce"]["strike"])
    peK = float(position["short_pe"]["strike"])

    ce_ltp = _call_get_option_price(sym, "CE", ceK, position["short_ce"].get("expiry", "monthly"))
    pe_ltp = _call_get_option_price(sym, "PE", peK, position["short_pe"].get("expiry", "monthly"))

    if ce_ltp is None or pe_ltp is None:
        # hard fallback: echo entry
        ce_ltp = float(position["short_ce"]["entry"])
        pe_ltp = float(position["short_pe"]["entry"])

    return {"short_ce": {"ltp": float(ce_ltp)}, "short_pe": {"ltp": float(pe_ltp)}}

class _PaperBroker:
    """
    Minimal broker that simulates order placement and uses get_option_price for LTP.
    This unblocks the agent without needing live order endpoints.
    """
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    # "Place" a short leg — we just return an order dict
    def sell_option(self, symbol: str, side: str, strike: float, qty: int, expiry: str = "monthly") -> Dict[str, Any]:
        # capture entry LTP as reference if available
        ltp = _call_get_option_price(symbol, side, strike, expiry) or 0.0
        return {
            "order_id": f"PAPER-{symbol}-{int(strike)}{side}-{expiry}",
            "symbol": symbol, "side": side, "strike": float(strike),
            "qty": int(qty), "expiry": expiry, "entry_ltp": float(ltp)
        }

    # Buy protective wings (simulated)
    def add_wings(self, symbol: str, ce_row: dict, pe_row: dict, qty: int,
                  hedge_df=None, expiry: str = "weekly",
                  ce_wing_strike: float = None, pe_wing_strike: float = None) -> Dict[str, Any]:
        ce_ws = float(ce_wing_strike) if ce_wing_strike is not None else float(ce_row["strike"] + 300)
        pe_ws = float(pe_wing_strike) if pe_wing_strike is not None else float(pe_row["strike"] - 300)
        return {
            "ce_wing": {"order_id": f"PAPER-WING-CE-{int(ce_ws)}-{expiry}", "strike": ce_ws, "expiry": expiry, "qty": int(qty)},
            "pe_wing": {"order_id": f"PAPER-WING-PE-{int(pe_ws)}-{expiry}", "strike": pe_ws, "expiry": expiry, "qty": int(qty)},
        }

    def roll_wings(self, position: Dict[str, Any], weekday: str = "Tuesday") -> Dict[str, Any]:
        # no-op in paper
        return {"rolled": True, "weekday": weekday, "mode": "paper"}

    def refresh_legs(self, position: Dict[str, Any]) -> Dict[str, Any]:
        return _paper_refresh_legs(position)

    def close_all(self, position: Dict[str, Any]) -> None:
        # no-op; in live you should place BUY for shorts & SELL for wings
        return None

# Choose the broker implementation
Broker = _BROKER_CLASS or _PaperBroker

# =============================================================================
#                            HedgedStrangle
# =============================================================================
class HedgedStrangle:
    """
    - Enters monthly short CE/PE (picked by StrategySelector).
    - Buys weekly (Tuesday) wings within ₹2.5–₹3 band (strict by default).
    - Manages rolls/adjustments via optional HedgeRoller.
    - Computes breakevens and evaluates risk-based exits.
    """

    def __init__(self, config: Dict[str, Any], risk_manager, logger=None):
        self.cfg = config
        self.risk = risk_manager
        self.log = logger or print
        self.broker = Broker(config)
        self.chain = OptionChainAdapter(config)

    # ---------- helpers ----------
    def _calc_breakevens(self, ce_price: float, pe_price: float, spot: float) -> Dict[str, float]:
        credit = ce_price + pe_price
        return {"upper": spot + credit, "lower": spot - credit, "credit": credit}

    def _pick_wing_by_price(self, hedge_df: pd.DataFrame, side: str, short_strike: float,
                            pmin: float, pmax: float) -> Optional[dict]:
        """
        Pick a weekly long wing (CE/PE) priced within [pmin, pmax].
        For CE -> strike > short_strike; For PE -> strike < short_strike.
        Preference: price close to mid, then strike proximity.
        """
        if hedge_df is None or len(hedge_df) == 0:
            return None
        col_price = "ltp_ce" if side == "CE" else "ltp_pe"
        if side == "CE":
            cands = hedge_df[hedge_df["strike"] > short_strike].copy()
        else:
            cands = hedge_df[hedge_df["strike"] < short_strike].copy()
        cands = cands[(cands[col_price] >= pmin) & (cands[col_price] <= pmax)].copy()
        if cands.empty:
            if not self.cfg.get("wings", {}).get("enforce_strict", True):
                fmax = float(self.cfg.get("wings", {}).get("fallback_max", pmax))
                cands = hedge_df.copy()
                if side == "CE":
                    cands = cands[cands["strike"] > short_strike]
                else:
                    cands = cands[cands["strike"] < short_strike]
                cands = cands[(cands[col_price] >= pmin) & (cands[col_price] <= fmax)]
                if cands.empty:
                    return None
            else:
                return None
        mid = (pmin + pmax) / 2.0
        cands["score_price"] = (cands[col_price] - mid).abs()
        cands["score_strike_dist"] = (cands["strike"] - short_strike).abs()
        cands = cands.sort_values(["score_price", "score_strike_dist"])
        return cands.iloc[0].to_dict()

    # ---------- life-cycle ----------
    def enter(self, symbol: str, ce_row: dict, pe_row: dict, hedge_df=None) -> Dict[str, Any]:
        qty = self.risk.position_size_for_structure(symbol, ce_row, pe_row)

        # Monthly shorts (paper or live)
        ce_order = self.broker.sell_option(symbol, "CE", ce_row["strike"], qty, expiry="monthly")
        pe_order = self.broker.sell_option(symbol, "PE", pe_row["strike"], qty, expiry="monthly")

        wings = None
        if self.cfg["general"].get("use_wings", True):
            pmin = float(self.cfg["wings"]["price_target_min"])
            pmax = float(self.cfg["wings"]["price_target_max"])
            ce_wing = self._pick_wing_by_price(hedge_df, "CE", float(ce_row["strike"]), pmin, pmax)
            pe_wing = self._pick_wing_by_price(hedge_df, "PE", float(pe_row["strike"]), pmin, pmax)

            if self.cfg["wings"].get("enforce_strict", True) and (ce_wing is None or pe_wing is None):
                raise RuntimeError("Wing selection failed: could not find weekly wings in ₹2.5–₹3 band.")

            if ce_wing and pe_wing:
                wings = self.broker.add_wings(
                    symbol, ce_row, pe_row, qty, hedge_df=hedge_df, expiry="weekly",
                    ce_wing_strike=ce_wing["strike"], pe_wing_strike=pe_wing["strike"]
                )

        position = {
            "symbol": symbol, "qty": qty,
            "short_ce": {"strike": float(ce_row["strike"]), "entry": float(ce_row["ltp_ce"]),
                         "expiry": "monthly", "order": ce_order},
            "short_pe": {"strike": float(pe_row["strike"]), "entry": float(pe_row["ltp_pe"]),
                         "expiry": "monthly", "order": pe_order},
            "wings": wings,
            "ts_enter": datetime.datetime.now().isoformat()
        }
        return position

    def manage(self, position: Dict[str, Any], ofx: Dict[str, Any], spot: float) -> Dict[str, Any]:
        leg = self.broker.refresh_legs(position)
        ce_p = float(leg["short_ce"]["ltp"])
        pe_p = float(leg["short_pe"]["ltp"])
        be = self._calc_breakevens(ce_p, pe_p, spot)

        cfg_adj = self.cfg["adjustments"]
        ce_mult = ce_p / max(1e-6, position["short_ce"]["entry"])
        pe_mult = pe_p / max(1e-6, position["short_pe"]["entry"])
        must_act = (ce_mult >= cfg_adj["premium_x_hard"]) or (pe_mult >= cfg_adj["premium_x_hard"])
        consider = (ce_mult >= cfg_adj["premium_x_soft"]) or (pe_mult >= cfg_adj["premium_x_soft"])

        df = ofx["short_df"]
        row_ce = df.loc[df["strike"] == position["short_ce"]["strike"]].iloc[0]
        row_pe = df.loc[df["strike"] == position["short_pe"]["strike"]].iloc[0]
        delta_breach = (abs(float(row_ce["delta_ce"])) >= cfg_adj["delta_breach"]) \
                       or (abs(float(row_pe["delta_pe"])) >= cfg_adj["delta_breach"])

        credit = be["credit"]
        near_be = (spot >= be["upper"] - cfg_adj["breakeven_buffer_credit_x"] * credit) or \
                  (spot <= be["lower"] + cfg_adj["breakeven_buffer_credit_x"] * credit)

        # Simple option-tape proxy via PCR volume
        pcr = ofx["pcr"]
        option_tape_bias = (pcr["pcr_vol"] < 0.8 or pcr["pcr_vol"] > 1.2)

        action = "HOLD"
        if must_act or delta_breach or (near_be and option_tape_bias) or (consider and option_tape_bias):
            try:
                from .hedge_roller import HedgeRoller
                roller = HedgeRoller(self.cfg, self.risk, self.broker)
                action = roller.roll_tested_side(position, ofx, spot)
            except Exception as e:
                self.log(f"[MANAGE] HedgeRoller unavailable or failed: {e}")
                action = "HOLD"
        return {"action": action, "breakevens": be, "leg_prices": {"ce": ce_p, "pe": pe_p}}

    def should_exit(self, position: Dict[str, Any]) -> Optional[str]:
        if self.cfg["exits"].get("time_exit", True):
            now = datetime.datetime.now().strftime("%H:%M")
            if now >= self.cfg["schedules"]["mandatory_square_off"]:
                return "TIME"
        decision = self.risk.evaluate_exit(position)
        return decision
