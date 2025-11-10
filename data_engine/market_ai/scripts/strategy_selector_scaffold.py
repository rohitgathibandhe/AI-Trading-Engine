#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scaffold CLI for the strategy selector training pipeline.

This script loads `state/feature_history.csv`, engineers simple features,
and prints dataset diagnostics.  Model training is intentionally left as
a TODO until sufficient paper/live data accumulates.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from market_ai.modules.training.feature_dataset import (
    FEATURE_HISTORY_DEFAULT,
    engineer_features,
    load_feature_history,
    make_training_target,
    summarise_dataset,
    build_feature_matrix,
)
from market_ai.modules.training.rolling_option_features import build_dataset
from market_ai.modules.data_fetch.dhan_rolling_option import RollingOptionIngestor, RollingOptionConfig
from market_ai.modules.data_fetch.dhan_api import make_client

MODEL_PATH_DEFAULT = Path(__file__).resolve().parents[2] / "state" / "strategy_selector_model.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strategy selector training scaffold")
    parser.add_argument(
        "--history",
        type=Path,
        default=FEATURE_HISTORY_DEFAULT,
        help="Path to feature_history.csv (defaults to state/feature_history.csv)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Shift horizon for provisional target creation (in samples)",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=MODEL_PATH_DEFAULT,
        help="Where to store the trained selector model",
    )
    parser.add_argument(
        "--rolling-option-dir",
        type=Path,
        help="Directory containing rolling_option parquet folders",
    )
    parser.add_argument(
        "--fetch-rolling-option",
        action="store_true",
        help="Fetch Dhan rolling option history before training",
    )
    parser.add_argument("--fetch-underlying", help="Underlying symbol for rolling option fetch (e.g., NIFTY)")
    parser.add_argument("--fetch-segment", default="NSE_FNO", help="Underlying segment for fetch (default NSE_FNO)")
    parser.add_argument("--fetch-security-id", type=int, help="Numeric securityId required for rolling option fetch")
    parser.add_argument("--fetch-start", help="Fetch start date YYYY-MM-DD")
    parser.add_argument("--fetch-end", help="Fetch end date YYYY-MM-DD")
    parser.add_argument("--fetch-expiry", help="Optional expiry filter YYYY-MM-DD")
    parser.add_argument("--fetch-interval", default="1", help="Rolling option interval (default 1)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rolling_dir = args.rolling_option_dir

    if args.fetch_rolling_option:
        if not all([args.fetch_underlying, args.fetch_security_id, args.fetch_start, args.fetch_end]):
            raise SystemExit("--fetch-underlying, --fetch-security-id, --fetch-start and --fetch-end are required when --fetch-rolling-option is used")
        rolling_dir = rolling_dir or Path(__file__).resolve().parents[2] / "state" / "rolling_option"
        rolling_dir.mkdir(parents=True, exist_ok=True)
        cfg = RollingOptionConfig(
            underlying=args.fetch_underlying,
            segment=args.fetch_segment,
            security_id=args.fetch_security_id,
            start=datetime.strptime(args.fetch_start, "%Y-%m-%d"),
            end=datetime.strptime(args.fetch_end, "%Y-%m-%d"),
            expiry=args.fetch_expiry,
            interval=args.fetch_interval,
        )
        ingestor = RollingOptionIngestor(client=make_client(), out_dir=rolling_dir)
        written = ingestor.fetch_range(cfg)
        print(f"Fetched {len(written)} rolling-option parquet files into {rolling_dir}")

    df_raw = load_feature_history(args.history)
    if rolling_dir:
        ro_features = build_dataset(rolling_dir)
        if not ro_features.empty:
            df_raw = pd.concat([df_raw, ro_features], ignore_index=True)
    if df_raw.empty:
        print("No feature history available yet. Run the agent in paper/live mode to collect data.")
        return

    print("Raw dataset summary:")
    print(summarise_dataset(df_raw))

    df_features = engineer_features(df_raw)
    df_train, target = make_training_target(df_features, horizon=args.horizon)
    X, y, columns = build_feature_matrix(df_train)

    print("\nEngineered feature columns:", columns)
    print("Training samples:", len(df_train))

    if len(df_train) < 50:
        print("Not enough samples (<50) to train a meaningful selector yet. Aborting.")
        return

    model_data = {
        "trained_at": datetime.utcnow().isoformat(timespec="seconds"),
        "feature_columns": columns,
        "strategies": {},
    }

    for strategy in sorted(df_train["strategy"].unique()):
        strat_mask = df_train["strategy"] == strategy
        Xs = X[strat_mask.values]
        ys = y[strat_mask.values]
        if len(Xs) < 10:
            print(f"Skipping {strategy}: insufficient samples ({len(Xs)})")
            continue

        weights, intercept = _fit_linear_model(Xs, ys.to_numpy())
        model_data["strategies"][strategy] = {
            "intercept": float(intercept),
            "weights": [float(w) for w in weights],
            "samples": int(len(Xs)),
            "mean_target": float(ys.mean()),
        }
        print(f"Trained model for {strategy}: mean target={ys.mean():.2f} samples={len(Xs)}")

    if not model_data["strategies"]:
        print("No strategy models trained.")
        return

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.write_text(json.dumps(model_data, indent=2))
    print(f"\nSaved selector model to {args.model_out}")


def _fit_linear_model(X: np.ndarray, y: np.ndarray, l2: float = 1e-6) -> tuple[np.ndarray, float]:
    """Simple ridge regression via normal equations."""
    X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
    ridge = l2 * np.eye(X_aug.shape[1])
    beta = np.linalg.pinv(X_aug.T @ X_aug + ridge) @ X_aug.T @ y
    intercept = beta[0]
    weights = beta[1:]
    return weights, intercept


if __name__ == "__main__":
    main()
