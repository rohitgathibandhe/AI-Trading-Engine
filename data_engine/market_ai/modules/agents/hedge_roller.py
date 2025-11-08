"""
modules/agent/hedge_roller.py

Purpose:
    Safely roll weekly hedges for short strangle/straddle strategies.
    Ensures "buy new first, then sell old" sequence to avoid naked risk.

Supports:
    • Market/limit hybrid hedge buys (auto-detects low premium)
    • Configurable retries, slippage, and backoff
    • Risk-aware projected margin validation
    • Dry-run mode (for testing)
"""

import time
import logging
from typing import Dict, Any, Tuple, Optional

from market_ai.modules.utils.config_loader import load_all_configs
from market_ai.modules.risk_manager import RiskManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# -----------------------------
# Execution Adapter Interface
# -----------------------------
class ExecutionAdapter:
    """
    Interface adapter for broker order execution.
    Replace these with your Dhan REST/SDK calls.
    """

    def place_limit_buy(self, symbol_id: int, quantity: int, limit_price: float, timeout_sec: int = 30) -> Dict:
        raise NotImplementedError

    def place_market_buy(self, symbol_id: int, quantity: int) -> Dict:
        raise NotImplementedError

    def place_market_sell(self, symbol_id: int, quantity: int) -> Dict:
        raise NotImplementedError

    def get_quote(self, symbol_id: int) -> Dict:
        raise NotImplementedError


# -----------------------------
# Hedge Roller Implementation
# -----------------------------
class HedgeRoller:
    def __init__(
        self,
        configs: Optional[Dict[str, Any]] = None,
        risk_manager: Optional[RiskManager] = None,
        executor: Optional[ExecutionAdapter] = None,
    ):
        self.configs = configs or load_all_configs()
        self.strategy_cfg = (
            self.configs.get("strategies", {}).get("monthly_strangle_with_weekly_hedge")
            or self.configs.get("strategies", {}).get("monthly_strangle", {})
        )

        self.executor = executor
        if executor is None:
            logger.warning("⚠️  No ExecutionAdapter provided — running in DRY-RUN mode.")
        self.risk_manager = risk_manager or RiskManager(self.configs)

        self.hedge_cfg = self.strategy_cfg.get("hedge", {})
        self.buy_timeout = int(self.hedge_cfg.get("limit_order_timeout_sec", 30))
        self.retry_attempts = int(self.hedge_cfg.get("retry_attempts", 3))
        self.slippage_pct = float(self.hedge_cfg.get("price_slippage_pct", 0.10))
        self.hedge_target_price = float(self.hedge_cfg.get("hedge_target_price", 3.0))
        self.hedge_max_price = float(self.hedge_cfg.get("hedge_max_price", 5.0))
        self.market_buy_threshold = 4.0  # custom threshold for switching to market buy
        self.buy_first_then_sell_old = bool(self.hedge_cfg.get("buy_first_then_sell_old", True))

    # -----------------------------
    # Main entry point
    # -----------------------------
    def roll_weekly_hedge(
        self,
        core_trade: Dict[str, Any],
        current_hedge: Dict[str, Any],
        next_week_legs: Dict[str, Any],
        per_strategy_cap: float,
    ) -> Tuple[bool, str]:
        """
        Safely roll weekly hedge (buy new → sell old).
        """
        logger.info("🌀 Starting hedge roll sequence")

        if not next_week_legs or not current_hedge:
            return False, "Missing hedge leg data."

        est_core_margin = float(
            self.strategy_cfg.get("sizing", {}).get("estimate_margin_per_core_lot", 150000)
        )
        est_hedge_margin = float(
            self.strategy_cfg.get("sizing", {}).get("estimate_margin_per_hedge_lot", 20000)
        )

        projected_ok = self.risk_manager.projected_margin_ok(
            core_margin_est=est_core_margin,
            hedge_margin_est=est_hedge_margin,
            usable_cap=per_strategy_cap,
        )
        if not projected_ok:
            msg = "❌ Projected margin too high after roll."
            logger.warning(msg)
            return False, msg

        if self.buy_first_then_sell_old:
            ok, msg = self._buy_new_hedges(next_week_legs)
            if not ok:
                return False, f"Failed to buy new hedges: {msg}"
            ok2, msg2 = self._sell_old_hedges(current_hedge)
            if not ok2:
                return False, f"Failed to sell old hedges: {msg2}"
            return True, "✅ Hedge rolled successfully (buy-first)."
        else:
            ok2, msg2 = self._sell_old_hedges(current_hedge)
            if not ok2:
                return False, f"Failed to sell old hedges: {msg2}"
            ok, msg = self._buy_new_hedges(next_week_legs)
            if not ok:
                return False, f"Failed to buy new hedges: {msg}"
            return True, "✅ Hedge rolled successfully (sell-first)."

    # -----------------------------
    # Buy new hedges logic
    # -----------------------------
    def _buy_new_hedges(self, next_week_legs: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Place hedge buys for CE & PE legs.
        Uses market buy if current quote <= market_buy_threshold, else tries limit first.
        Retries with small price bump on partial fills.
        """
        ce_id = next_week_legs.get("ce_symbol")
        pe_id = next_week_legs.get("pe_symbol")
        qty = int(next_week_legs.get("qty", 1))

        if not ce_id or not pe_id:
            return False, "Missing CE/PE symbol IDs."

        for attempt in range(1, self.retry_attempts + 1):
            try:
                # ---- Get live quote ----
                if self.executor:
                    ce_quote = self.executor.get_quote(ce_id)
                    pe_quote = self.executor.get_quote(pe_id)
                    ce_price = float(ce_quote.get("last_price", self.hedge_target_price))
                    pe_price = float(pe_quote.get("last_price", self.hedge_target_price))
                else:
                    ce_price = pe_price = self.hedge_target_price

                logger.info(
                    f"Attempt {attempt}: CE price={ce_price:.2f}, PE price={pe_price:.2f}"
                )

                # ---- Determine buy mode ----
                avg_price = (ce_price + pe_price) / 2.0
                use_market = avg_price <= self.market_buy_threshold

                if use_market:
                    logger.info("Using market buy (avg price %.2f ≤ %.2f)", avg_price, self.market_buy_threshold)
                    buy_func = self.executor.place_market_buy if self.executor else None
                else:
                    logger.info("Using limit buy (avg price %.2f > %.2f)", avg_price, self.market_buy_threshold)
                    buy_func = self.executor.place_limit_buy if self.executor else None

                price_factor = 1.0 + (attempt - 1) * self.slippage_pct
                limit_price = round(self.hedge_target_price * price_factor, 2)

                # ---- Safety check ----
                if limit_price > self.hedge_max_price:
                    return False, f"Limit price {limit_price} > hedge_max_price {self.hedge_max_price}"

                # ---- Execute buys ----
                if not self.executor:
                    logger.info(f"(DRY-RUN) Buying CE & PE hedges at {limit_price}")
                    return True, "Simulated hedge buys (dry-run)."

                if use_market:
                    o1 = buy_func(ce_id, qty)
                    o2 = buy_func(pe_id, qty)
                else:
                    o1 = buy_func(ce_id, qty, limit_price, self.buy_timeout)
                    o2 = buy_func(pe_id, qty, limit_price, self.buy_timeout)

                if o1.get("filled_qty", 0) >= qty and o2.get("filled_qty", 0) >= qty:
                    logger.info("✅ Hedge buy filled successfully.")
                    return True, "Hedge buy filled"
                else:
                    logger.warning("Partial fills on attempt %d", attempt)

            except Exception as e:
                logger.exception("Error during hedge buy attempt %d: %s", attempt, e)

            # Backoff
            backoff = attempt * 2
            time.sleep(backoff)

        return False, "All hedge buy attempts failed."

    # -----------------------------
    # Sell old hedges logic
    # -----------------------------
    def _sell_old_hedges(self, current_hedge: Dict[str, Any]) -> Tuple[bool, str]:
        qty = int(current_hedge.get("qty", 1))
        ce_id = current_hedge.get("ce_symbol")
        pe_id = current_hedge.get("pe_symbol")

        if not ce_id or not pe_id:
            return False, "Missing CE/PE symbol IDs in current_hedge."

        try:
            if not self.executor:
                logger.info("(DRY-RUN) Selling old CE/PE hedges at market.")
                return True, "Simulated hedge sells (dry-run)."

            r1 = self.executor.place_market_sell(ce_id, qty)
            r2 = self.executor.place_market_sell(pe_id, qty)

            if r1.get("filled_qty", 0) >= qty and r2.get("filled_qty", 0) >= qty:
                logger.info("✅ Old hedges sold successfully.")
                return True, "Old hedges sold"
            else:
                logger.warning("Partial hedge sell detected: CE=%s, PE=%s", r1, r2)
                return False, "Partial fills during sell."

        except Exception as e:
            logger.exception("Error selling old hedges: %s", e)
            return False, f"Exception: {e}"
