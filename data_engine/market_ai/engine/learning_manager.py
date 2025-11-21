from __future__ import annotations

from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Deque, Dict, Tuple


@dataclass
class PerfStats:
    count: int
    win_rate: float
    avg_pnl: float
    loss_streak: int


class LearningManager:
    """Keeps rolling performance per regime and strategy."""

    def __init__(self, window: int = 30):
        self.window = window
        self.regime_perf: Dict[str, Deque[float]] = defaultdict(deque)
        self.strategy_perf: Dict[str, Deque[float]] = defaultdict(deque)

    def _update_bucket(self, bucket: Deque[float], pnl: float):
        bucket.append(pnl)
        if len(bucket) > self.window:
            bucket.popleft()

    def update(self, regime: str, strategy: str, pnl: float):
        self._update_bucket(self.regime_perf[regime], pnl)
        self._update_bucket(self.strategy_perf[strategy], pnl)

    def _stats(self, bucket: Deque[float]) -> PerfStats:
        if not bucket:
            return PerfStats(0, 0.0, 0.0, 0)
        wins = sum(1 for x in bucket if x > 0)
        loss_streak = 0
        for x in reversed(bucket):
            if x <= 0:
                loss_streak += 1
            else:
                break
        return PerfStats(
            count=len(bucket),
            win_rate=wins / len(bucket),
            avg_pnl=sum(bucket) / len(bucket),
            loss_streak=loss_streak,
        )

    def stats(self, regime: str, strategy: str) -> Tuple[PerfStats, PerfStats]:
        return self._stats(self.regime_perf[regime]), self._stats(self.strategy_perf[strategy])

    def should_downweight_bear(self) -> bool:
        _, strat_stats = self.stats("BEAR", "BEAR_CALL_SPREAD")
        return strat_stats.count >= 5 and strat_stats.avg_pnl < 0 and strat_stats.win_rate < 0.4

    def size_multiplier(self, regime: str, strategy: str, confidence: float) -> float:
        regime_stats, strat_stats = self.stats(regime, strategy)
        size = 1.0
        if confidence < 0.6:
            size *= 0.5
        elif confidence >= 0.75:
            size *= 1.25
        if strat_stats.loss_streak >= 3:
            size *= 0.5
        if regime_stats.win_rate >= 0.65 and strat_stats.win_rate >= 0.65 and strat_stats.avg_pnl > 0:
            size = min(1.5, size * 1.25)
        return max(0.25, size)
