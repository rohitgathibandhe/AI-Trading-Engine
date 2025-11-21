from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..strategies.models import LegSide, OptionLeg, OptionType, StrategyType
from .orb_engine import ORBBreakoutSignal
from .sr_engine import SRLevels


@dataclass
class LegInstruction:
    action: str  # "OPEN" or "CLOSE"
    leg: OptionLeg
    reason: str


class ExecutionEngine:
    def __init__(self):
        self.open_positions: List[OptionLeg] = []

    def plan_entries(
        self,
        strategy: StrategyType,
        option_chain: List[dict],
        expiry,
        lot_size: int,
        builder,
    ) -> List[LegInstruction]:
        legs = builder(option_chain, expiry, lot_size)
        return [LegInstruction("OPEN", leg, f"Deploy {strategy.name}") for leg in legs]

    def apply_orb_exit(self, breakout: ORBBreakoutSignal) -> List[LegInstruction]:
        if not breakout or not breakout.active:
            return []
        losing_type = OptionType.CALL if breakout.direction == StrategyType.BULL_PUT_SPREAD else OptionType.PUT
        instructions = []
        for leg in self.open_positions:
            if leg.side == LegSide.SELL and leg.option_type == losing_type:
                instructions.append(LegInstruction("CLOSE", leg, breakout.reason))
        return instructions

    def plan_reentry(self, side: OptionType, sr_levels: SRLevels) -> Optional[str]:
        if side == OptionType.CALL and sr_levels.resistances:
            return f"Watch resistance {sr_levels.resistances[0]} for CALL re-entry"
        if side == OptionType.PUT and sr_levels.supports:
            return f"Watch support {sr_levels.supports[-1]} for PUT re-entry"
        return None
