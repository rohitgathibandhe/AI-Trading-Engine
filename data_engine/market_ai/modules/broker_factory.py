# market_ai/modules/broker_factory.py
from __future__ import annotations
import os
from typing import Literal
from market_ai.modules.paper_broker import PaperBroker
from market_ai.modules.dhan_broker import DhanBroker

def get_broker(mode: Literal["paper","live"], lot_size: int, client_id: str | None, token: str | None):
    mode = (mode or os.environ.get("TRADE_MODE","paper")).lower()
    if mode == "live":
        return DhanBroker(client_id=client_id, access_token=token, lot_size=lot_size)
    return PaperBroker(lot_size=lot_size)
