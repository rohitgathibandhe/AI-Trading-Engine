"""
modules/trade_logic.py

Provides trading decision helpers used by ai_strategy_backtest.

Exports:
 - decide_trade(row, next_row=None, config=None)
 - simulate_trade_row(row, next_row=None, config=None)

Both return either a trade dict:
 {
   "signal": "sell_PUT"/"sell_CALL"/"buy_CALL"/"buy_PUT",
   "entry_price": float,
   "exit_price": float,
   "hold_days": int,
   "meta": {...}
 }
or None to indicate "no trade".

This module is intentionally conservative and defensive so it integrates
with your existing backtest harness without additional changes.
"""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def decide_trade(row, next_row=None, config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Conservative decision rule.

    Parameters:
      row: pandas.Series like object containing at least price fields and
           prediction fields (e.g. 'pred' and/or 'prob_pos'/'prob').
      next_row: optional pandas.Series representing the next day (for exit)
      config: dictionary with options:
         - prob_threshold     (float) default 0.6
         - mode               (str) 'sell' or 'buy'  (default 'sell')
         - hold_days          (int) default 1
         - min_premium        (float) rupee min premium to allow selling (default 0.0)
    """
    try:
        cfg = dict(config or {})
        prob_threshold = float(cfg.get("prob_threshold", 0.6))
        mode = str(cfg.get("mode", "sell")).lower()
        hold_days = int(cfg.get("hold_days", 1))
        min_premium = float(cfg.get("min_premium", 0.0))

        # read probability (try several common names)
        prob = None
        for k in ("prob_pos", "prob", "probability", "prob_pos_raw"):
            if k in row.index:
                try:
                    prob = float(row.get(k))
                except Exception:
                    prob = None
                break
        if prob is None:
            # no probability -> cannot decide safely
            logger.debug("decide_trade: no probability in row -> skip")
            return None

        # pred if present
        pred = None
        for k in ("pred", "prediction", "pred_raw"):
            if k in row.index:
                pred = row.get(k)
                break
        pred_val = None
        if pred is not None:
            try:
                pred_val = int(pred)
            except Exception:
                pred_val = None

        # Mode-based mapping (sell default)
        side = None
        if mode == "sell":
            # mapping: model UP -> sell PUT, model DOWN -> sell CALL
            if (pred_val == 1) or (pred_val is None and prob >= prob_threshold):
                side = "PUT"
            elif (pred_val == 0) or (pred_val is None and prob <= (1.0 - prob_threshold)):
                side = "CALL"
            else:
                logger.debug("model not decisive (prob in middle band) -> skip")
                return None
            signal = f"sell_{side}"
        elif mode == "buy":
            # buy logic: model UP -> buy_CALL, model DOWN -> buy_PUT
            if (pred_val == 1) or (pred_val is None and prob >= prob_threshold):
                signal = "buy_CALL"
            elif (pred_val == 0) or (pred_val is None and prob <= (1.0 - prob_threshold)):
                signal = "buy_PUT"
            else:
                logger.debug("model not decisive (prob in middle band) -> skip")
                return None
        else:
            logger.debug("unknown mode=%s -> skip", mode)
            return None

        # entry price (prefer open, then close)
        entry_price = None
        for k in ("open", "Open", "entry_price", "close"):
            if k in row.index and row.get(k) is not None:
                try:
                    entry_price = float(row.get(k))
                    break
                except Exception:
                    entry_price = None
        if entry_price is None:
            logger.debug("decide_trade: no price available in row -> skip")
            return None

        # exit price: prefer next_row.close if provided
        exit_price = None
        if next_row is not None and "close" in next_row.index and next_row.get("close") is not None:
            try:
                exit_price = float(next_row.get("close"))
            except Exception:
                exit_price = None
        if exit_price is None:
            # safe fallback - assume flat (no move)
            exit_price = entry_price

        # cheap premium guard
        if min_premium and entry_price < float(min_premium):
            logger.debug("decide_trade: entry_price %.3f below min_premium %.3f -> skip", entry_price, min_premium)
            return None

        trade = {
            "signal": signal,
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "hold_days": int(hold_days),
            "meta": {"prob": prob, "pred": pred}
        }
        return trade

    except Exception:
        logger.exception("decide_trade encountered exception, skipping trade")
        return None


def simulate_trade_row(row, next_row=None, config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Backtest harness wrapper expected by older/backtest code.

    It calls decide_trade then performs small normalization so the
    backtest engine receives the fields it expects.

    Kept intentionally minimal.
    """
    t = decide_trade(row, next_row=next_row, config=config)
    if t is None:
        return None

    # normalize field names the backtest expects (conservative)
    # some engines expect 'signal', 'entry', 'exit', 'hold_days' etc.
    out = {
        "signal": t.get("signal"),
        "entry": t.get("entry_price"),
        "exit": t.get("exit_price"),
        "hold_days": int(t.get("hold_days", 1)),
        "meta": t.get("meta", {})
    }
    # For compatibility, also provide 'entry_price' and 'exit_price'
    out["entry_price"] = out["entry"]
    out["exit_price"] = out["exit"]
    return out


# Export list
__all__ = ["decide_trade", "simulate_trade_row"]
