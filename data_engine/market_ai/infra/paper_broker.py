# data_engine/market_ai/infra/paper_broker.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = PROJECT_ROOT / "data_engine" / "market_ai" / "runtime"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = RUNTIME_DIR / "paper_state.json"

DEFAULT_VIRTUAL_CASH = 50_000.0

@dataclass
class PaperPosition:
    id: str                 # unique id per position
    symbol: str             # e.g., "NIFTY", "BANKNIFTY", "RELIANCE"
    qty: int                # +long, -short
    avg_price: float
    side: str               # "BUY" | "SELL"
    opened_at: float        # epoch seconds

@dataclass
class PaperState:
    cash: float
    positions: List[PaperPosition]

class PaperBroker:
    """
    Minimal paper broker:
      - Starts with DEFAULT_VIRTUAL_CASH (₹50,000)
      - Place/close 'orders' by adjusting cash & positions
      - Persist state to paper_state.json so UI/agent can share it
    """

    def __init__(self, initial_cash: float = DEFAULT_VIRTUAL_CASH, state_file: Path = STATE_FILE):
        self.state_file = Path(state_file)
        if self.state_file.exists():
            self.state = self._load()
        else:
            self.state = PaperState(cash=float(initial_cash), positions=[])
            self._save()

    # ---------- persistence ----------
    def _load(self) -> PaperState:
        data = json.loads(self.state_file.read_text() or "{}")
        cash = float(data.get("cash", DEFAULT_VIRTUAL_CASH))
        positions = [
            PaperPosition(**p) for p in data.get("positions", [])
        ]
        return PaperState(cash=cash, positions=positions)

    def _save(self) -> None:
        data = {
            "cash": self.state.cash,
            "positions": [asdict(p) for p in self.state.positions],
        }
        self.state_file.write_text(json.dumps(data, indent=2))

    # ---------- funds ----------
    def get_available_funds(self) -> float:
        return float(self.state.cash)

    # ---------- simple order emulation ----------
    def place_market_order(self, symbol: str, side: str, qty: int, price: float) -> str:
        """
        Very simple fill model: fills at 'price' immediately.
        Assumes no fees/slippage for now.
        """
        assert side in ("BUY", "SELL")
        assert qty > 0
        notional = qty * price

        if side == "BUY":
            if self.state.cash < notional:
                raise RuntimeError(f"Insufficient virtual cash: need {notional}, have {self.state.cash}")
            self.state.cash -= notional
            pos = PaperPosition(
                id=f"P{int(time.time()*1000)}",
                symbol=symbol, qty=qty, avg_price=price, side="BUY", opened_at=time.time()
            )
            self.state.positions.append(pos)
        else:  # SELL
            # For simplicity: allow naked short, credit cash
            self.state.cash += notional
            pos = PaperPosition(
                id=f"P{int(time.time()*1000)}",
                symbol=symbol, qty=-qty, avg_price=price, side="SELL", opened_at=time.time()
            )
            self.state.positions.append(pos)

        self._save()
        return pos.id

    def close_position(self, pos_id: str, exit_price: float) -> None:
        idx = next((i for i,p in enumerate(self.state.positions) if p.id == pos_id), None)
        if idx is None:
            return
        p = self.state.positions[idx]
        pnl = (exit_price - p.avg_price) * (p.qty)  # qty already signed
        self.state.cash += pnl
        self.state.positions.pop(idx)
        self._save()

    # ---------- PnL ----------
    def unrealized_pnl(self, last_prices: Dict[str, float]) -> float:
        pnl = 0.0
        for p in self.state.positions:
            if p.symbol in last_prices:
                pnl += (last_prices[p.symbol] - p.avg_price) * (p.qty)
        return float(pnl)
