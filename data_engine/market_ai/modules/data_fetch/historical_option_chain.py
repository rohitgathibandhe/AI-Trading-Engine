# market_ai/modules/data_fetch/historical_option_chain.py
from __future__ import annotations
from typing import Dict, Any, Optional
from market_ai.modules.data_fetch.optionchain_repo import OptionChainRepo

class HistoricalOptionChain:
    """Adapter with the same surface as the live OptionChain, but needs as_of_date."""
    def __init__(self, repo: OptionChainRepo, default_seg: str = "IDX_I"):
        self.repo = repo
        self.default_seg = default_seg
        # NB: We keep the same method name but accept an extra kw 'as_of_date'
    def get_option_chain(self, underlying_id: int, expiry_or_tag: str, underlying_seg: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        if expiry_or_tag.lower() == "weekly":
            return {}  # historical 'weekly' requires mapping; keep it simple for now
        as_of = kwargs.get("as_of_date")
        if not as_of:
            # Backtests must pass as_of_date to fetch the correct snapshot
            return {}
        seg = (underlying_seg or self.default_seg) or "IDX_I"
        return self.repo.get_chain(underlying_id, seg, as_of, expiry_or_tag)
