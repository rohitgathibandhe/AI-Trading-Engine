# -*- coding: utf-8 -*-
"""
Strategy selection helper.

Loads the trained selector model (JSON produced by
`scripts/strategy_selector_scaffold.py`) and scores candidate
strategies based on the latest feature snapshot.  The scoring is
currently a simple linear model fitted via ridge regression.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from market_ai.modules.training.feature_dataset import (
    engineer_features,
    load_feature_history,
    build_feature_matrix,
    DEFAULT_FEATURE_COLUMNS,
)


@dataclass
class StrategyScore:
    name: str
    expected_credit: float
    samples: int
    confidence: float


class StrategySelector:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.model: Dict[str, Any] = {}
        if model_path.exists():
            self.model = json.loads(model_path.read_text())

    def available(self) -> bool:
        return bool(self.model.get("strategies"))

    def score_latest(self, feature_path: Path) -> Dict[str, StrategyScore]:
        df = load_feature_history(feature_path)
        if df.empty or not self.available():
            return {}

        df = engineer_features(df)
        scores: Dict[str, StrategyScore] = {}

        for strat_name, payload in self.model["strategies"].items():
            strat_rows = df[df["strategy"] == strat_name]
            if strat_rows.empty:
                continue
            latest = strat_rows.sort_values("timestamp").iloc[-1]

            feature_vector = self._build_feature_vector(latest)
            if feature_vector is None:
                continue
            weights = np.array(payload["weights"], dtype=float)
            intercept = float(payload["intercept"])
            expected = float(intercept + np.dot(weights, feature_vector))
            mean_target = float(payload.get("mean_target", expected))
            samples = int(payload.get("samples", len(strat_rows)))

            confidence = min(1.0, max(0.0, samples / 200.0))
            expected = 0.5 * expected + 0.5 * mean_target  # blend point prediction with historical mean

            scores[strat_name] = StrategyScore(
                name=strat_name,
                expected_credit=expected,
                samples=samples,
                confidence=confidence,
            )

        return scores

    def _build_feature_vector(self, row: pd.Series) -> Optional[np.ndarray]:
        feature_cols = self.model.get("feature_columns", DEFAULT_FEATURE_COLUMNS)
        try:
            vector = row[feature_cols].fillna(0.0).to_numpy(dtype=float)
        except KeyError:
            return None
        return vector


def select_preferred_strategy(model_path: Path, feature_path: Path) -> Optional[StrategyScore]:
    selector = StrategySelector(model_path)
    if not selector.available():
        return None
    scores = selector.score_latest(feature_path)
    if not scores:
        return None
    return max(scores.values(), key=lambda s: s.expected_credit)
